"""Five-block modern seasonal stability test for Duan et al. (2019)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("BEA", "CRU")
PRIMARY_REGIONS = ("hma", "central_80_90", "east_ge90")
ALL_REGIONS = ("hma", "west_lt80", "central_80_90", "east_ge90")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if stat.st_size <= 100_000_000 else None
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def weighted_region(values: np.ndarray, latitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    use = np.asarray(mask, dtype=bool)
    if int(use.sum()) < 3:
        raise RuntimeError("insufficient-region-support")
    subset = values[:, use]
    weights = np.cos(np.deg2rad(latitudes[use]))
    if not np.isfinite(subset).all():
        raise RuntimeError("nonfinite-seasonal-region-input")
    return np.average(subset, axis=1, weights=weights)


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_157_duan2019_seasonal_stability_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round157_duan2019_seasonal_stability_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    r118 = load_module(
        "round157_r118",
        driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    r156 = loop / "round_156_duan2019_temperature_identity_v2"
    r116 = loop / "round_116_temperature_low_rank_residual_oof_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "r156_summary": r156 / "summary.json",
        "duan_allowed": r156 / "duan2019_allowed.csv",
        "support": r116 / "temperature_low_rank_residual_oof_fields.npz",
        "bea": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    r156_summary = json.loads(paths["r156_summary"].read_text(encoding="utf-8"))
    if r156_summary.get("decision") != "advance_to_source_material_overlap_and_modern_seasonal_stability_audit":
        raise RuntimeError("round156-decision-mismatch")

    frame = pd.read_csv(paths["duan_allowed"])
    if ((frame.year >= 1921) & (frame.year <= 1939)).any():
        raise RuntimeError("locked-year-in-allowed-export")
    lookup = dict(zip(frame.year.astype(int), frame.value_1.astype(float)))
    candidate = np.asarray([lookup.get(int(year), np.nan) for year in YEARS])
    if not np.isfinite(candidate).all():
        raise RuntimeError("candidate-modern-gap")

    with np.load(paths["support"]) as support:
        grid_lat = np.asarray(support["grid_lat"], dtype=float)
        grid_lon = np.asarray(support["grid_lon"], dtype=float)
        hma_mask = np.asarray(support["hma_mask"], dtype=bool).reshape(-1)
        observed_support = np.asarray(support["observed_support"], dtype=bool)
    lat_grid, lon_grid = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    flat_lat = lat_grid.reshape(-1)
    flat_lon = lon_grid.reshape(-1)
    common = hma_mask & observed_support
    latitudes = flat_lat[common]
    longitudes = flat_lon[common]
    if len(latitudes) != 99:
        raise RuntimeError(f"support-count-mismatch:{len(latitudes)}")

    bea_monthly = r118.read_berkeley_points(paths["bea"], latitudes, longitudes)
    cru_monthly = r118.read_netcdf3_points(paths["cru"], "tmp", 1901, latitudes, longitudes)
    months = [(8, 9)] * len(latitudes)
    seasonal = {
        "BEA": r118.aggregate_temperature(bea_monthly, months),
        "CRU": r118.aggregate_temperature(cru_monthly, months),
    }
    region_masks = {
        "hma": np.ones(len(longitudes), dtype=bool),
        "west_lt80": longitudes < 80,
        "central_80_90": (longitudes >= 80) & (longitudes < 90),
        "east_ge90": longitudes >= 90,
    }
    targets = {
        (product, region): weighted_region(values, latitudes, region_masks[region])
        for product, values in seasonal.items()
        for region in ALL_REGIONS
    }

    predictions = {(product, region): np.full(len(YEARS), np.nan) for product in PRODUCTS for region in ALL_REGIONS}
    baselines = {(product, region): np.full(len(YEARS), np.nan) for product in PRODUCTS for region in ALL_REGIONS}
    fold_rows = []
    for product in PRODUCTS:
        for region in ALL_REGIONS:
            target = targets[(product, region)]
            for fold, (start, end) in enumerate(BLOCKS, start=1):
                test = (YEARS >= start) & (YEARS <= end)
                train = ~test
                coefficients = np.linalg.lstsq(
                    np.column_stack((np.ones(int(train.sum())), candidate[train])),
                    target[train],
                    rcond=None,
                )[0]
                prediction = coefficients[0] + coefficients[1] * candidate[test]
                predictions[(product, region)][test] = prediction
                baselines[(product, region)][test] = np.mean(target[train])
                fold_rows.append({
                    "product": product,
                    "region": region,
                    "fold": fold,
                    "test_years": f"{start}-{end}",
                    "fit_intercept": float(coefficients[0]),
                    "fit_slope": float(coefficients[1]),
                    "test_correlation": correlation(target[test], prediction),
                    "test_rmse_degC": float(np.sqrt(np.mean((target[test] - prediction) ** 2))),
                })
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output / "blocked_oof_fold_metrics.csv", index=False)

    summary_rows = []
    for product in PRODUCTS:
        for region in ALL_REGIONS:
            target = targets[(product, region)]
            prediction = predictions[(product, region)]
            baseline = baselines[(product, region)]
            rows = fold_frame[(fold_frame["product"] == product) & (fold_frame.region == region)]
            candidate_sse = float(np.sum((target - prediction) ** 2))
            baseline_sse = float(np.sum((target - baseline) ** 2))
            early = YEARS <= 1969
            late = YEARS >= 1970
            summary_rows.append({
                "product": product,
                "region": region,
                "oof_correlation": correlation(target, prediction),
                "oof_skill_vs_training_fold_climatology": 1 - candidate_sse / baseline_sse,
                "oof_rmse_degC": float(np.sqrt(candidate_sse / len(YEARS))),
                "positive_fold_correlation_count": int((rows.test_correlation > 0).sum()),
                "positive_fit_slope_count": int((rows.fit_slope > 0).sum()),
                "early_1940_1969_correlation": correlation(target[early], prediction[early]),
                "late_1970_2000_correlation": correlation(target[late], prediction[late]),
            })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output / "blocked_oof_summary.csv", index=False)

    regional_product_prediction_correlation = {
        region: correlation(predictions[("BEA", region)], predictions[("CRU", region)])
        for region in ALL_REGIONS
    }
    primary = summary_frame[summary_frame.region.isin(PRIMARY_REGIONS)]
    gates = {
        "all_primary_oof_correlations_ge_0_30": bool((primary.oof_correlation >= 0.30).all()),
        "all_primary_oof_skills_positive": bool((primary.oof_skill_vs_training_fold_climatology > 0).all()),
        "all_primary_paths_four_of_five_fold_correlations_positive": bool((primary.positive_fold_correlation_count >= 4).all()),
        "all_primary_fit_slopes_positive": bool((primary.positive_fit_slope_count == 5).all()),
        "all_primary_early_and_late_correlations_nonnegative": bool(
            (primary.early_1940_1969_correlation >= 0).all()
            and (primary.late_1970_2000_correlation >= 0).all()
        ),
        "all_primary_bea_cru_prediction_correlations_ge_0_80": bool(
            all(regional_product_prediction_correlation[region] >= 0.80 for region in PRIMARY_REGIONS)
        ),
    }
    passes = bool(all(gates.values()))
    decision = "advance_as_source_derived_seasonal_operator_candidate" if passes else "close_duan2019_seasonal_operator_candidate"

    curves = pd.DataFrame({"year": YEARS, "duan2019_temperature": candidate})
    for product in PRODUCTS:
        for region in ALL_REGIONS:
            curves[f"{product}_{region}_observed"] = targets[(product, region)]
            curves[f"{product}_{region}_oof"] = predictions[(product, region)]
    curves.to_csv(output / "modern_oof_curves.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    colors = {"BEA": "#0072B2", "CRU": "#D55E00"}
    for axis, region in zip(axes, PRIMARY_REGIONS):
        for product in PRODUCTS:
            row = summary_frame[(summary_frame["product"] == product) & (summary_frame.region == region)].iloc[0]
            axis.plot(YEARS, targets[(product, region)], color=colors[product], linewidth=1.3, label=f"{product} AS observed")
            axis.plot(YEARS, predictions[(product, region)], color=colors[product], linestyle="--", linewidth=1.0, label=f"{product} OOF r={row.oof_correlation:.2f}, skill={row.oof_skill_vs_training_fold_climatology:.2f}")
        for _, end in BLOCKS[:-1]:
            axis.axvline(end + 0.5, color="#999999", linestyle=":", linewidth=0.7)
        axis.set_title(region)
        axis.set_ylabel("Aug-Sep T (degC)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("year")
    fig.savefig(output / "duan2019_seasonal_stability_oof.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_157_duan2019_seasonal_stability_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "common_hma_support_cell_count": int(len(latitudes)),
        "primary_regions": list(PRIMARY_REGIONS),
        "regional_product_prediction_correlation": regional_product_prediction_correlation,
        "metrics": summary_frame.to_dict(orient="records"),
        "gates": gates,
        "passes_all_primary_gates": passes,
        "decision": decision,
        "candidate_role_if_passed": "source_derived_seasonal_operator_candidate_not_raw_proxy_or_independent_validation",
        "locked_1921_1939_accessed": False,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 157 Duan 2019 Seasonal Stability\n\n"
        + "\n".join(
            f"- {row.region} {row.product}: OOF r=`{row.oof_correlation:.3f}`, skill=`{row.oof_skill_vs_training_fold_climatology:.3f}`, positive folds=`{row.positive_fold_correlation_count}/5`, early/late r=`{row.early_1940_1969_correlation:.3f}/{row.late_1970_2000_correlation:.3f}`."
            for row in summary_frame.itertuples()
            if row.region in PRIMARY_REGIONS
        )
        + f"\n- Passes all primary gates: `{passes}`.\n"
        + f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
