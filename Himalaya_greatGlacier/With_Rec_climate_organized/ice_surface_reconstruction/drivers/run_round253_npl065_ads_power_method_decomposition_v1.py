from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1960, 2014, dtype=int)
MATRIX_YEARS = np.arange(1941, 2014, dtype=int)
BLOCKS = ((1960, 1968), (1969, 1977), (1978, 1986), (1987, 1995), (1996, 2004), (2005, 2013))
BRANCHES = ("ads30_ratio", "cookpow_ads30_ratio", "cookpow_ads30_ratio_nonpos")
PUBLISHED_R = 0.781


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cook_power_transform(series: np.ndarray, precision: float) -> tuple[np.ndarray, float]:
    values = np.asarray(series, dtype=float)
    if np.any(values < 0):
        raise ValueError("negative-ring-width")
    adjacent_right = values[1:]
    adjacent_left = values[:-1]
    level = (adjacent_right + adjacent_left) / 2.0
    spread = np.abs(adjacent_right - adjacent_left)
    level[level == 0] = precision
    spread[spread == 0] = precision
    beta = np.linalg.lstsq(
        np.column_stack((np.ones(len(level)), np.log(level))), np.log(spread), rcond=None
    )[0]
    power = abs(1.0 - float(beta[1]))
    return values ** power, power


def build_rwi(raw: pd.DataFrame, branch: str, ads_module) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rwi = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
    details = []
    precision = 1.0
    for sample in raw.columns:
        source = raw[sample].dropna()
        values = source.to_numpy(float)
        transformed = values
        power = None
        if branch.startswith("cookpow"):
            transformed, power = cook_power_transform(values, precision)
        pos_slope = branch != "cookpow_ads30_ratio_nonpos"
        curve = ads_module.ads(transformed, nyrs0=30, pos_slope=pos_slope)
        if not np.isfinite(curve).all() or np.any(curve <= 0):
            raise RuntimeError(f"invalid-ads-curve:{branch}:{sample}")
        indices = transformed / curve
        rwi.loc[source.index, sample] = indices
        details.append({
            "sample": str(sample),
            "length": len(values),
            "cook_power": power,
            "positive_end_slope_allowed": pos_slope,
            "curve_last_slope": float(curve[-1] - curve[-2]),
        })
    return rwi, details


def rolling(round246, proxy: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for start in range(1960, 1985):
        end = start + 29
        mask = (YEARS >= start) & (YEARS <= end)
        rows.append({"window_start": start, "window_end": end, "correlation": round246.correlation(proxy[mask], target[mask])})
    return rows


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_253_npl065_ads_power_method_decomposition_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round253_npl065_ads_power_method_decomposition_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round246_driver": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "round252_driver": drivers / "run_round252_dplr_ads_python_port_validation_v3.py",
        "round252_validation": loop / "round_252_dplr_ads_python_port_validation_v3/validation_summary.json",
        "round250_metrics": loop / "round_250_npl065_spline_method_decomposition_v1/method_metrics.csv",
        "round250_curves": loop / "round_250_npl065_spline_method_decomposition_v1/method_curves.csv",
        "rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "pre2001_manifest": loop / "round_208_setp_cru_tmin_annual_target_audit_v1/cru_tmn_acquisition_manifest.json",
        "post2000_manifest": loop / "round_245_cru_ts401_tmin_post2000_acquisition_v1/acquisition_manifest.json",
        "dplr_ads_fortran": loop / "round_251_dplr179_method_source_acquisition_v1/vendor_source/src/adsf.f95",
        "dplr_power_source": loop / "round_251_dplr179_method_source_acquisition_v1/vendor_source/R/capPOWTseries.R",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    validation = json.loads(paths["round252_validation"].read_text(encoding="ascii"))
    if validation.get("decision") != "admit_ads_python_port_to_modern_only_method_experiment":
        raise RuntimeError("ads-port-not-admitted")
    round246 = load_module("round253_round246", paths["round246_driver"])
    ads_module = load_module("round253_ads", paths["round252_driver"])
    raw, parse_metadata = round246.parse_modern_rwl(paths["rwl"])
    target, station_support, cru_files, box = round246.read_cru_box([paths["pre2001_manifest"], paths["post2000_manifest"]])
    paths.update({f"cru_{index:02d}": path for index, path in enumerate(cru_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})

    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy

    year_indices = pd.Index(MATRIX_YEARS).get_indexer(YEARS)
    metrics_rows = []
    fold_rows = []
    rolling_rows = []
    curve_columns = {"year": YEARS, "cru_box_djf_tmin": target}
    branch_details = {}
    for branch in BRANCHES:
        rwi, details = build_rwi(raw, branch, ads_module)
        chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
        proxy_all = chronology["Mean RWI"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
        depth_all = chronology["Sample depth"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
        if not np.isfinite(proxy_all).all() or not np.isfinite(depth_all).all():
            raise RuntimeError(f"chronology-gap:{branch}")
        proxy = proxy_all[year_indices]
        metrics, prediction, folds = round246.blocked_oof(YEARS, proxy, target, BLOCKS)
        rolling_values = rolling(round246, proxy, target)
        correlations = np.asarray([row["correlation"] for row in rolling_values], dtype=float)
        metrics_rows.append({
            "branch": branch,
            **metrics,
            "rolling30_min_correlation": float(np.min(correlations)),
            "rolling30_median_correlation": float(np.median(correlations)),
            "rolling30_max_correlation": float(np.max(correlations)),
            "rolling30_positive_window_count": int(np.sum(correlations > 0)),
            "rolling30_ge_0_5_window_count": int(np.sum(correlations >= 0.5)),
            "gap_to_published_in_sample_r": float(PUBLISHED_R - metrics["in_sample_correlation"]),
        })
        fold_rows.extend({"branch": branch, **row} for row in folds)
        rolling_rows.extend({"branch": branch, **row} for row in rolling_values)
        curve_columns[f"{branch}_chronology"] = proxy
        curve_columns[f"{branch}_oof"] = prediction
        powers = np.asarray([row["cook_power"] for row in details if row["cook_power"] is not None], dtype=float)
        branch_details[branch] = {
            "minimum_depth": int(np.min(depth_all[year_indices])),
            "positive_end_slope_allowed": details[0]["positive_end_slope_allowed"],
            "positive_final_curve_slope_series_count": int(sum(row["curve_last_slope"] > 0 for row in details)),
            "cook_power_min": float(np.min(powers)) if len(powers) else None,
            "cook_power_median": float(np.median(powers)) if len(powers) else None,
            "cook_power_max": float(np.max(powers)) if len(powers) else None,
        }

    metrics = pd.DataFrame(metrics_rows).set_index("branch")
    baseline_metrics = pd.read_csv(paths["round250_metrics"]).set_index("branch").loc["modnegex"].to_dict()
    baseline_curves = pd.read_csv(paths["round250_curves"])
    if not np.array_equal(baseline_curves.year.to_numpy(int), YEARS):
        raise RuntimeError("round250-year-axis-changed")
    pd.DataFrame(metrics_rows).to_csv(output / "ads_power_method_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "ads_power_fold_metrics.csv", index=False)
    pd.DataFrame(rolling_rows).to_csv(output / "ads_power_rolling30_correlations.csv", index=False)
    pd.DataFrame(curve_columns).to_csv(output / "ads_power_method_curves.csv", index=False)
    pd.DataFrame(
        [
            {"branch": branch, **row}
            for branch in BRANCHES
            for row in build_rwi(raw, branch, ads_module)[1]
        ]
    ).to_csv(output / "series_method_parameters.csv", index=False)

    colors = {
        "modnegex": "#555555",
        "ads30_ratio": "#0072B2",
        "cookpow_ads30_ratio": "#D55E00",
        "cookpow_ads30_ratio_nonpos": "#009E73",
    }
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.4), constrained_layout=True)
    target_z = (target - np.mean(target)) / np.std(target, ddof=1)
    axes[0].plot(YEARS, target_z, color="#111111", linewidth=1.7, label="CRU DJF Tmin")
    baseline = baseline_curves.modnegex_chronology.to_numpy(float)
    axes[0].plot(YEARS, (baseline - baseline.mean()) / baseline.std(ddof=1), color=colors["modnegex"], linewidth=1.1, label=f"modnegex, r={float(baseline_metrics['in_sample_correlation']):.3f}")
    for branch in BRANCHES:
        values = np.asarray(curve_columns[f"{branch}_chronology"], dtype=float)
        axes[0].plot(YEARS, (values - values.mean()) / values.std(ddof=1), color=colors[branch], linewidth=1.1, label=f"{branch}, r={metrics.at[branch, 'in_sample_correlation']:.3f}")
    axes[0].set_ylabel("standardized anomaly")
    axes[0].set_title("NPL065 age-dependent spline and power-transform comparison")
    axes[0].legend(frameon=False, ncol=2, fontsize=8.5)
    baseline_roll = []
    for start in range(1960, 1985):
        mask = (YEARS >= start) & (YEARS <= start + 29)
        baseline_roll.append(round246.correlation(baseline[mask], target[mask]))
    axes[1].plot(np.arange(1989, 2014), baseline_roll, color=colors["modnegex"], linewidth=1.3, label="modnegex")
    rolling_frame = pd.DataFrame(rolling_rows)
    for branch in BRANCHES:
        subset = rolling_frame[rolling_frame.branch == branch]
        axes[1].plot(subset.window_end, subset.correlation, color=colors[branch], linewidth=1.45, label=branch)
    axes[1].axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axes[1].set_ylabel("30-year correlation")
    axes[1].set_xlabel("rolling-window end year")
    axes[1].set_title("Rolling-window stability against published Figure S4")
    axes[1].legend(frameon=False, ncol=2, fontsize=8.5)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "npl065_ads_power_method_decomposition.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    best_branch = str(metrics.in_sample_correlation.idxmax())
    best_gap = float(metrics.at[best_branch, "gap_to_published_in_sample_r"])
    if best_gap <= 0.05 and metrics.at[best_branch, "oof_correlation"] > 0:
        decision = "ads_and_power_transform_explain_most_published_skill_gap"
    elif metrics.at[best_branch, "in_sample_correlation"] > float(baseline_metrics["in_sample_correlation"]):
        decision = "ads_or_power_partially_improves_but_signal_free_iteration_still_required"
    else:
        decision = "ads_and_power_do_not_explain_gap_signal_free_or_full_length_context_required"
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "baseline_modnegex": baseline_metrics,
        "branches": branch_details,
        "metrics": metrics_rows,
        "best_branch": best_branch,
        "best_branch_gap_to_published_r": best_gap,
        "decision": decision,
        "parse_metadata": parse_metadata,
        "target": {"dataset": "CRU_TS4.01_tmn", "season": "DJF", "box": box, "station_support": station_support},
        "signal_free_iteration_implemented": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "paleo_proxy_values_decoded": False,
        "jionglacuo_opened_or_hashed": False,
        "rcsigfree_binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    lines = ["# Round 253 NPL065 ADS/power method decomposition", "", f"- ModNegEx baseline: in-sample r `{float(baseline_metrics['in_sample_correlation']):.3f}`, OOF r `{float(baseline_metrics['oof_correlation']):.3f}`."]
    for branch in BRANCHES:
        lines.append(f"- {branch}: in-sample r `{metrics.at[branch, 'in_sample_correlation']:.3f}`, OOF r `{metrics.at[branch, 'oof_correlation']:.3f}`, rolling-30 median `{metrics.at[branch, 'rolling30_median_correlation']:.3f}`.")
    lines.extend(["", f"Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
