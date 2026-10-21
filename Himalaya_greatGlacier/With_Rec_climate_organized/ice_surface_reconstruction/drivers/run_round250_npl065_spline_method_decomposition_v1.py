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
BRANCHES = ("modnegex", "spline30", "spline50")


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


def build_chronology(dplpy, raw: pd.DataFrame, branch: str):
    if branch == "modnegex":
        columns = []
        fallbacks = []
        for sample in raw.columns:
            try:
                series = dplpy.detrend(raw[sample], fit="ModNegEx", method="residual", plot=False)
            except RuntimeError:
                series = dplpy.detrend(raw[sample], fit="linear", method="residual", plot=False)
                fallbacks.append(str(sample))
            columns.append(series.rename(sample))
        rwi = pd.concat(columns, axis=1).reindex(raw.index)
    else:
        period = 30 if branch == "spline30" else 50
        rwi = dplpy.detrend(raw, fit="spline", method="residual", plot=False, period=period)
        fallbacks = []
    chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
    proxy = chronology["Mean RWI"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
    depth = chronology["Sample depth"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
    if not np.isfinite(proxy).all() or not np.isfinite(depth).all():
        raise RuntimeError(f"chronology-gap:{branch}")
    return proxy, depth, fallbacks


def rolling_correlations(round246, proxy: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
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
    output = loop / "round_250_npl065_spline_method_decomposition_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round250_npl065_spline_method_decomposition_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round246_driver": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "round246_summary": loop / "round_246_npl065_calibration_endpoint_attribution_v2/summary.json",
        "rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "pre2001_manifest": loop / "round_208_setp_cru_tmin_annual_target_audit_v1/cru_tmn_acquisition_manifest.json",
        "post2000_manifest": loop / "round_245_cru_ts401_tmin_post2000_acquisition_v1/acquisition_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    round246 = load_module("round250_round246", paths["round246_driver"])
    baseline_summary = json.loads(paths["round246_summary"].read_text(encoding="ascii"))
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
        proxy_all, depth_all, fallbacks = build_chronology(dplpy, raw, branch)
        proxy = proxy_all[year_indices]
        metrics, prediction, folds = round246.blocked_oof(YEARS, proxy, target, BLOCKS)
        rolling = rolling_correlations(round246, proxy, target)
        rolling_values = np.asarray([row["correlation"] for row in rolling], dtype=float)
        metrics_rows.append({
            "branch": branch,
            **metrics,
            "rolling30_min_correlation": float(np.min(rolling_values)),
            "rolling30_median_correlation": float(np.median(rolling_values)),
            "rolling30_max_correlation": float(np.max(rolling_values)),
            "rolling30_positive_window_count": int(np.sum(rolling_values > 0)),
            "rolling30_ge_0_5_window_count": int(np.sum(rolling_values >= 0.5)),
        })
        fold_rows.extend({"branch": branch, **row} for row in folds)
        rolling_rows.extend({"branch": branch, **row} for row in rolling)
        curve_columns[f"{branch}_chronology"] = proxy
        curve_columns[f"{branch}_oof"] = prediction
        branch_details[branch] = {"linear_fallbacks": fallbacks, "minimum_depth": int(np.min(depth_all[year_indices]))}

    metrics = pd.DataFrame(metrics_rows).set_index("branch")
    expected_baseline = baseline_summary["metrics"]["1960_2013"]
    closure = {
        "in_sample_correlation": float(abs(metrics.at["modnegex", "in_sample_correlation"] - expected_baseline["in_sample_correlation"])),
        "oof_correlation": float(abs(metrics.at["modnegex", "oof_correlation"] - expected_baseline["oof_correlation"])),
        "oof_skill": float(abs(metrics.at["modnegex", "oof_skill_vs_training_fold_climatology"] - expected_baseline["oof_skill_vs_training_fold_climatology"])),
    }
    if max(closure.values()) > 1e-12:
        raise RuntimeError(f"round246-baseline-closure-failed:{closure}")

    pd.DataFrame(metrics_rows).to_csv(output / "method_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "method_fold_metrics.csv", index=False)
    pd.DataFrame(rolling_rows).to_csv(output / "method_rolling30_correlations.csv", index=False)
    pd.DataFrame(curve_columns).to_csv(output / "method_curves.csv", index=False)

    colors = {"modnegex": "#555555", "spline30": "#C44E52", "spline50": "#006D77"}
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.3), constrained_layout=True)
    target_z = (target - np.mean(target)) / np.std(target, ddof=1)
    axes[0].plot(YEARS, target_z, color="#111111", linewidth=1.7, label="CRU DJF Tmin")
    for branch in BRANCHES:
        values = np.asarray(curve_columns[f"{branch}_chronology"], dtype=float)
        values = (values - np.mean(values)) / np.std(values, ddof=1)
        axes[0].plot(YEARS, values, color=colors[branch], linewidth=1.15, label=f"{branch}, r={metrics.at[branch, 'in_sample_correlation']:.3f}")
    axes[0].set_ylabel("standardized anomaly")
    axes[0].set_title("NPL065 chronology-method comparison")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    for branch in BRANCHES:
        subset = pd.DataFrame(rolling_rows)
        subset = subset[subset.branch == branch]
        axes[1].plot(subset.window_end, subset.correlation, color=colors[branch], linewidth=1.5, label=branch)
    axes[1].axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axes[1].set_ylabel("30-year correlation")
    axes[1].set_xlabel("rolling-window end year")
    axes[1].set_title("Rolling 30-year relation with CRU DJF Tmin")
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "npl065_spline_method_decomposition.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    spline30_gain = float(metrics.at["spline30", "oof_correlation"] - metrics.at["modnegex", "oof_correlation"])
    if spline30_gain >= 0.10:
        decision = "thirty_year_spline_materially_explains_npl065_skill_gap"
    elif spline30_gain > 0:
        decision = "thirty_year_spline_partially_improves_but_signal_free_or_power_transform_still_needed"
    else:
        decision = "thirty_year_spline_does_not_explain_npl065_skill_gap"
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "branches": branch_details,
        "target": {"dataset": "CRU_TS4.01_tmn", "season": "DJF", "box": box, "station_support": station_support},
        "metrics": metrics_rows,
        "round246_baseline_closure_absolute_error": closure,
        "spline30_minus_modnegex_oof_correlation": spline30_gain,
        "decision": decision,
        "parse_metadata": parse_metadata,
        "signal_free_iteration_implemented": False,
        "power_transform_implemented": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "paleo_proxy_values_decoded": False,
        "jionglacuo_opened_or_hashed": False,
        "binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    lines = ["# Round 250 NPL065 spline-method decomposition", ""]
    for branch in BRANCHES:
        lines.append(
            f"- {branch}: in-sample r `{metrics.at[branch, 'in_sample_correlation']:.3f}`, "
            f"OOF r `{metrics.at[branch, 'oof_correlation']:.3f}`, rolling-30 median `{metrics.at[branch, 'rolling30_median_correlation']:.3f}`."
        )
    lines.extend(["", f"Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
