"""Five-block modern OOF validation of the new Haizi and Hengduan chronologies."""

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


def candidate_series(path: Path, year_column: str, value_column: str) -> np.ndarray:
    frame = pd.read_csv(path)
    years = frame[year_column].to_numpy(dtype=int)
    if ((years >= 1921) & (years <= 1939)).any():
        raise RuntimeError("locked-year-in-allowed-export")
    lookup = dict(zip(years, frame[value_column].to_numpy(dtype=float)))
    values = np.asarray([lookup.get(int(year), np.nan) for year in YEARS])
    if not np.isfinite(values).all():
        raise RuntimeError(f"candidate-modern-gap:{path.name}")
    return values


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_146_new_temperature_proxy_modern_oof_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round146_new_temperature_proxy_modern_oof_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    r118 = load_module(
        "round146_r118",
        driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    r144 = loop / "round_144_temperature_candidate_boundary_preflight_v3"
    r145 = loop / "round_145_hengduan_haizi_sequence_identity_closure_v1"
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    raw = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_external_proxy_discovery_20260818_v1/raw_unopened"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "r145_summary": r145 / "summary.json",
        "haizi": r144 / "haizi__sheet1__allowed.csv",
        "hengduan": r144 / "hengduan_info__tree_ring_width_chronology__allowed.csv",
        "bea": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
        "jionglacuo_sealed": raw / "jionglacuo_temperature_reconstruction.xlsx",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    r145_summary = json.loads(paths["r145_summary"].read_text(encoding="utf-8"))
    if r145_summary.get("decision") != "advance_hengduan_haizi_to_site_measurement_and_modern_oof":
        raise RuntimeError("round145-decision-mismatch")

    proxies = {
        "Haizi": candidate_series(paths["haizi"], "year", "std"),
        "Hengduan": candidate_series(paths["hengduan"], "Year", "SsfStb"),
    }
    latitudes = np.asarray([29.03, 27 + 47 / 60, 27 + 11 / 60, 28 + 4 / 60, 26 + 26 / 60, 27 + 10 / 60])
    longitudes = np.asarray([100.18, 98 + 28 / 60, 98 + 43 / 60, 98 + 45 / 60, 98 + 46 / 60, 98 + 43 / 60])
    bea_monthly = r118.read_berkeley_points(paths["bea"], latitudes, longitudes)
    cru_monthly = r118.read_netcdf3_points(paths["cru"], "tmp", 1901, latitudes, longitudes)
    months = [(tuple(range(1, 13)))] + [tuple(range(5, 10))] * 5
    climate = {}
    for product, monthly in (("BEA", bea_monthly), ("CRU", cru_monthly)):
        cells = r118.aggregate_temperature(monthly, months)
        if cells.shape != (61, 6) or not np.isfinite(cells).all():
            raise RuntimeError(f"temperature-target-failed:{product}")
        climate[("Haizi", product)] = cells[:, 0]
        site_weights = np.cos(np.deg2rad(latitudes[1:]))
        climate[("Hengduan", product)] = np.average(cells[:, 1:], axis=1, weights=site_weights)

    fold_rows = []
    predictions = {(candidate, product): np.full(len(YEARS), np.nan) for candidate in proxies for product in PRODUCTS}
    climatologies = {(candidate, product): np.full(len(YEARS), np.nan) for candidate in proxies for product in PRODUCTS}
    for candidate, proxy in proxies.items():
        for product in PRODUCTS:
            target = climate[(candidate, product)]
            for fold_index, (start, end) in enumerate(BLOCKS, start=1):
                test = (YEARS >= start) & (YEARS <= end)
                train = ~test
                design = np.column_stack((np.ones(int(train.sum())), proxy[train]))
                coefficients = np.linalg.lstsq(design, target[train], rcond=None)[0]
                prediction = coefficients[0] + coefficients[1] * proxy[test]
                predictions[(candidate, product)][test] = prediction
                climatologies[(candidate, product)][test] = np.mean(target[train])
                fold_rows.append({
                    "candidate": candidate,
                    "product": product,
                    "fold": fold_index,
                    "test_years": f"{start}-{end}",
                    "test_count": int(test.sum()),
                    "fit_intercept": float(coefficients[0]),
                    "fit_slope": float(coefficients[1]),
                    "test_correlation": correlation(target[test], prediction),
                    "test_rmse_degC": float(np.sqrt(np.mean((target[test] - prediction) ** 2))),
                })
    fold_metrics = pd.DataFrame(fold_rows)

    summary_rows = []
    candidate_gates = {}
    for candidate in proxies:
        for product in PRODUCTS:
            target = climate[(candidate, product)]
            prediction = predictions[(candidate, product)]
            baseline = climatologies[(candidate, product)]
            candidate_sse = float(np.sum((target - prediction) ** 2))
            baseline_sse = float(np.sum((target - baseline) ** 2))
            folds = fold_metrics[(fold_metrics.candidate == candidate) & (fold_metrics["product"] == product)]
            summary_rows.append({
                "candidate": candidate,
                "product": product,
                "oof_correlation": correlation(target, prediction),
                "oof_skill_vs_training_fold_climatology": 1 - candidate_sse / baseline_sse,
                "oof_rmse_degC": float(np.sqrt(candidate_sse / len(YEARS))),
                "positive_fold_correlation_count": int((folds.test_correlation > 0).sum()),
                "positive_fit_slope_count": int((folds.fit_slope > 0).sum()),
            })
        product_prediction_correlation = correlation(predictions[(candidate, "BEA")], predictions[(candidate, "CRU")])
        rows = pd.DataFrame(summary_rows)
        use = rows[rows.candidate == candidate]
        gates = {
            "both_oof_correlations_ge_0_30": bool((use.oof_correlation >= 0.30).all()),
            "both_oof_skills_positive": bool((use.oof_skill_vs_training_fold_climatology > 0).all()),
            "both_products_four_of_five_fold_correlations_positive": bool((use.positive_fold_correlation_count >= 4).all()),
            "all_ten_fit_slopes_positive": bool((use.positive_fit_slope_count == 5).all()),
            "bea_cru_oof_prediction_correlation_ge_0_80": bool(product_prediction_correlation >= 0.80),
        }
        candidate_gates[candidate] = {"product_prediction_correlation": product_prediction_correlation, "gates": gates, "passes": bool(all(gates.values()))}
    summary_frame = pd.DataFrame(summary_rows)
    passing = [candidate for candidate, item in candidate_gates.items() if item["passes"]]
    decision = "advance_" + "_and_".join(name.lower() for name in passing) + "_to_paleo_robustness" if passing else "close_both_new_temperature_linear_psm_candidates"

    fold_metrics.to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    summary_frame.to_csv(output / "blocked_oof_summary.csv", index=False)
    curves = pd.DataFrame({"year": YEARS})
    for candidate in proxies:
        curves[f"{candidate}_proxy"] = proxies[candidate]
        for product in PRODUCTS:
            curves[f"{candidate}_{product}_observed"] = climate[(candidate, product)]
            curves[f"{candidate}_{product}_oof"] = predictions[(candidate, product)]
    curves.to_csv(output / "modern_oof_curves.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    colors = {"BEA": "#0072B2", "CRU": "#D55E00"}
    for axis, candidate in zip(axes, proxies):
        for product in PRODUCTS:
            row = summary_frame[(summary_frame.candidate == candidate) & (summary_frame["product"] == product)].iloc[0]
            axis.plot(YEARS, climate[(candidate, product)], color=colors[product], linewidth=1.4, label=f"{product} observed")
            axis.plot(YEARS, predictions[(candidate, product)], color=colors[product], linestyle="--", linewidth=1.0, label=f"{product} OOF r={row.oof_correlation:.3f}")
        for _, end in BLOCKS[:-1]:
            axis.axvline(end + 0.5, color="#999999", linestyle=":", linewidth=0.7)
        axis.set_title(f"{candidate}: frozen-season five-block OOF")
        axis.set_ylabel("temperature (degC)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("year")
    fig.savefig(output / "round146_new_temperature_proxy_oof_curves.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_146_new_temperature_proxy_modern_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "target_contract": {
            "Haizi": "annual_mean_at_Daocheng_29.03N_100.18E",
            "Hengduan": "May_September_mean_across_five_published_sites",
        },
        "oof": summary_frame.to_dict(orient="records"),
        "candidate_gates": candidate_gates,
        "passing_candidates": passing,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 146 New Temperature Proxy Modern OOF\n\n"
        + "\n".join(
            f"- {row.candidate} {row.product}: OOF r=`{row.oof_correlation:.3f}`, skill=`{row.oof_skill_vs_training_fold_climatology:.3f}`, positive folds=`{row.positive_fold_correlation_count}/5`."
            for row in summary_frame.itertuples()
        )
        + f"\n- Passing candidates: `{passing}`.\n"
        + f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
