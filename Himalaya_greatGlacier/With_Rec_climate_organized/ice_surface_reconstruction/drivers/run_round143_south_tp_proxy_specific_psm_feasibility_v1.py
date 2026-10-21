"""Blocked-OOF feasibility of a proxy-specific reliability-weighted seasonal PSM."""

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 100_000_000:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def weighted_composite(base, values, target, fit, groups):
    z, center, scale = base.standardize_columns(values, fit)
    correlations = np.asarray([base.correlation(z[fit, column], target[fit]) for column in range(z.shape[1])])
    signs = np.sign(correlations)
    weights = np.abs(correlations)
    oriented = z * signs[None, :]
    group_values = np.full((len(values), len(groups)), np.nan)
    for group_index, members in enumerate(groups):
        member_values = oriented[:, members]
        member_weights = weights[members]
        finite = np.isfinite(member_values)
        numerator = np.nansum(member_values * member_weights[None, :], axis=1)
        denominator = np.sum(finite * member_weights[None, :], axis=1)
        group_values[:, group_index] = np.divide(
            numerator,
            denominator,
            out=np.full(len(values), np.nan),
            where=denominator > 0,
        )
    active = np.isfinite(group_values).sum(axis=1)
    composite = np.divide(
        np.nansum(group_values, axis=1),
        active,
        out=np.full(len(values), np.nan),
        where=active > 0,
    )
    return composite, correlations, weights, active


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_143_south_tp_proxy_specific_psm_feasibility_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round143_south_tp_proxy_specific_psm_feasibility_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    base = load_module("round143_base", driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py")
    r141 = load_module("round143_r141", driver / "run_round141_south_tp_crossyear_precipitation_external_validation_v1.py")
    r118 = load_module(
        "round143_r118",
        driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    audit = load_module("round143_archive", driver / "audit_positive_b2_event_seasonal_psm_v1.py")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    r141_output = loop / "round_141_south_tp_crossyear_precipitation_external_validation_v1"
    r142_output = loop / "round_142_south_tp_transfer_failure_attribution_v2"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "proxy6": audit.archive_proxy_path(),
        "metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "r141_summary": r141_output / "summary.json",
        "r141_oof": r141_output / "blocked_oof_summary.csv",
        "r142_summary": r142_output / "summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    r142 = json.loads(paths["r142_summary"].read_text(encoding="utf-8"))
    if r142.get("decision") != "advance_to_proxy_specific_cross_validated_seasonal_psm_redesign":
        raise RuntimeError("round142-decision-mismatch")

    metadata = pd.read_csv(paths["metadata"])
    selected = metadata[
        metadata.latitude.between(27.0, 35.0)
        & metadata.longitude.between(90.0, 102.0)
        & (metadata.first_observed_year <= 1500)
        & (metadata.last_observed_year >= 2000)
    ].copy()
    proxy_ids = tuple(sorted(selected.proxy_index.astype(int)))
    _, group_keys, groups = r141.generic_groups(metadata, proxy_ids)
    if len(proxy_ids) != 38 or len(groups) != 19:
        raise RuntimeError("frozen-inventory-mismatch")
    proxy = base.read_proxy_values(paths["proxy6"], proxy_ids, r141.CAL_YEARS)

    lat = np.tile(np.asarray([27, 29, 31, 33, 35], dtype=float), 6)
    lon = np.repeat(np.asarray([91, 93, 95, 97, 99, 101], dtype=float), 5)
    weights = np.cos(np.deg2rad(lat))
    targets = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, lat, lon)
        for season, (previous, current) in r141.SEASONS.items():
            cells = r141.aggregate_crossyear(monthly, previous, current)
            targets[(season, product)] = np.average(cells, axis=1, weights=weights)

    fold_rows = []
    weight_rows = []
    predictions = {}
    observed = {}
    climatology = {}
    for season in r141.SEASONS:
        for product in r141.PRODUCTS:
            target = targets[(season, product)]
            prediction_all = np.full(len(r141.CAL_YEARS), np.nan)
            observed_all = np.full(len(r141.CAL_YEARS), np.nan)
            climatology_all = np.full(len(r141.CAL_YEARS), np.nan)
            for fold_name, fit_start, fit_end, test_start, test_end in r141.FOLDS:
                fit_mask = (r141.CAL_YEARS >= fit_start) & (r141.CAL_YEARS <= fit_end)
                test_mask = (r141.CAL_YEARS >= test_start) & (r141.CAL_YEARS <= test_end)
                fold_proxy = np.fmax(base.smooth_inside_mask(proxy, fit_mask), base.smooth_inside_mask(proxy, test_mask))
                fold_target = np.fmax(base.smooth_inside_mask(target, fit_mask), base.smooth_inside_mask(target, test_mask))
                fit_valid = fit_mask & np.isfinite(fold_target)
                test_valid = test_mask & np.isfinite(fold_target)
                comp, correlations, reliability, active = weighted_composite(base, fold_proxy, fold_target, fit_valid, groups)
                prediction, coefficients = base.linear_fit_predict(comp, fold_target, fit_valid, test_valid)
                prediction_all[test_valid] = prediction[test_valid]
                observed_all[test_valid] = fold_target[test_valid]
                climatology_all[test_valid] = np.nanmean(fold_target[fit_valid])
                fold_rows.append({
                    "season": season,
                    "product": product,
                    "fold": fold_name,
                    "test_count": int(np.isfinite(prediction[test_valid]).sum()),
                    "test_correlation": base.correlation(fold_target[test_valid], prediction[test_valid]),
                    "test_rmse_mm_season": float(np.sqrt(np.nanmean((fold_target[test_valid] - prediction[test_valid]) ** 2))),
                    "fit_slope": float(coefficients[1]),
                    "fit_mean_absolute_proxy_correlation": float(np.nanmean(reliability)),
                    "test_minimum_active_group_count": int(np.min(active[test_valid])),
                })
                for proxy_id, correlation_value, weight_value in zip(proxy_ids, correlations, reliability):
                    weight_rows.append({
                        "season": season,
                        "product": product,
                        "fold": fold_name,
                        "proxy_index": proxy_id,
                        "fit_correlation": correlation_value,
                        "fit_reliability_weight": weight_value,
                    })
            predictions[(season, product)] = prediction_all
            observed[(season, product)] = observed_all
            climatology[(season, product)] = climatology_all

    fold_metrics = pd.DataFrame(fold_rows)
    baseline = pd.read_csv(paths["r141_oof"]).set_index(["season", "product"])
    summary_rows = []
    for season in r141.SEASONS:
        for product in r141.PRODUCTS:
            key = (season, product)
            valid = np.isfinite(predictions[key]) & np.isfinite(observed[key])
            candidate_sse = float(np.sum((observed[key][valid] - predictions[key][valid]) ** 2))
            climatology_sse = float(np.sum((observed[key][valid] - climatology[key][valid]) ** 2))
            corr = base.correlation(observed[key][valid], predictions[key][valid])
            summary_rows.append({
                "season": season,
                "product": product,
                "oof_count": int(valid.sum()),
                "oof_correlation": corr,
                "round141_equal_weight_oof_correlation": float(baseline.loc[key, "oof_correlation"]),
                "correlation_change_vs_round141": corr - float(baseline.loc[key, "oof_correlation"]),
                "oof_skill_vs_fold_climatology": 1.0 - candidate_sse / climatology_sse,
                "oof_rmse_mm_season": float(np.sqrt(candidate_sse / valid.sum())),
            })
    summary_frame = pd.DataFrame(summary_rows)
    product_correlations = {}
    for season in r141.SEASONS:
        valid = np.isfinite(predictions[(season, "GPCC")]) & np.isfinite(predictions[(season, "CRU")])
        product_correlations[season] = base.correlation(
            predictions[(season, "GPCC")][valid], predictions[(season, "CRU")][valid]
        )

    gates = {
        "all_eight_fold_correlations_positive": bool((fold_metrics.test_correlation > 0).all()),
        "all_four_oof_correlations_ge_0_50": bool((summary_frame.oof_correlation >= 0.50).all()),
        "all_four_oof_skills_positive": bool((summary_frame.oof_skill_vs_fold_climatology > 0).all()),
        "both_oof_product_correlations_ge_0_80": bool(all(value >= 0.80 for value in product_correlations.values())),
    }
    passed = bool(all(gates.values()))
    decision = (
        "advance_proxy_specific_psm_to_paleo_external_validation"
        if passed
        else "close_south_tp_modern_correlation_seasonal_psm_family"
    )

    fold_metrics.to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    summary_frame.to_csv(output / "blocked_oof_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "fit_proxy_reliability_weights.csv", index=False)
    curves = pd.DataFrame({"year": r141.CAL_YEARS})
    for season in r141.SEASONS:
        for product in r141.PRODUCTS:
            curves[f"{season}_{product}_observed"] = observed[(season, product)]
            curves[f"{season}_{product}_weighted_oof"] = predictions[(season, product)]
    curves.to_csv(output / "modern_oof_curves.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    colors = {"GPCC": "#0072B2", "CRU": "#D55E00"}
    for column, season in enumerate(r141.SEASONS):
        for product in r141.PRODUCTS:
            key = (season, product)
            row = summary_frame[(summary_frame.season == season) & (summary_frame["product"] == product)].iloc[0]
            axes[column].plot(r141.CAL_YEARS, observed[key], color=colors[product], linewidth=1.5, label=f"{product} observed")
            axes[column].plot(r141.CAL_YEARS, predictions[key], color=colors[product], linewidth=1.0, linestyle="--", label=f"{product} OOF r={row.oof_correlation:.3f}")
        axes[column].axvline(1970.5, color="#777777", linestyle=":", linewidth=1)
        axes[column].set_title(f"{season}: reliability-weighted blocked OOF")
        axes[column].set_xlabel("year")
        axes[column].set_ylabel("precipitation (mm/season)")
        axes[column].grid(axis="y", color="#dddddd", linewidth=0.5)
        axes[column].spines[["top", "right"]].set_visible(False)
        axes[column].legend(frameon=False, fontsize=8)
    fig.savefig(output / "round143_proxy_specific_psm_oof_curves.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_143_south_tp_proxy_specific_psm_feasibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "method": "fit_period_absolute_correlation_reliability_with_equal_spatial_group_weight",
        "oof": summary_frame.to_dict(orient="records"),
        "oof_product_correlations": product_correlations,
        "gates": gates,
        "passes_all_gates": passed,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "external_validation_values_opened": False,
        "external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 143 Proxy-Specific Seasonal PSM Feasibility\n\n"
        + "\n".join(
            f"- {row.season} {row.product}: weighted OOF r=`{row.oof_correlation:.3f}`, change=`{row.correlation_change_vs_round141:+.3f}`, skill=`{row.oof_skill_vs_fold_climatology:.3f}`."
            for row in summary_frame.itertuples()
        )
        + f"\n- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
