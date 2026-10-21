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
BRANCHES = ("ssf_positive_end_allowed", "ssf_positive_end_constrained")
MAX_ITERATIONS = 25
MAD_THRESHOLD = 1e-4
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


def robust_chronology(dplpy, rwi: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
    mean = chronology["Mean RWI"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
    depth = chronology["Sample depth"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
    if not np.isfinite(mean).all() or not np.isfinite(depth).all() or np.any(mean == 0):
        raise RuntimeError("invalid-chronology")
    return mean, depth


def get_curves(data: pd.DataFrame, positive_end_allowed: bool, ads_module) -> pd.DataFrame:
    curves = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
    for sample in data.columns:
        series = data[sample].dropna()
        if len(series) < 4 or np.any(np.diff(series.index.to_numpy(int)) != 1):
            raise RuntimeError(f"noncontiguous-series:{sample}")
        values = series.to_numpy(float)
        curve = ads_module.ads(values, nyrs0=30, pos_slope=positive_end_allowed)
        if not np.isfinite(curve).all() or np.any(curve <= 0):
            raise RuntimeError(f"invalid-signal-free-curve:{sample}")
        curves.loc[series.index, sample] = curve
    return curves


def power_transform_frame(raw: pd.DataFrame, round253) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    transformed = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
    rows = []
    for sample in raw.columns:
        series = raw[sample].dropna()
        values, power = round253.cook_power_transform(series.to_numpy(float), 1.0)
        transformed.loc[series.index, sample] = values
        rows.append({"sample": str(sample), "cook_power": power, "length": len(values)})
    return transformed, rows


def signal_free(data: pd.DataFrame, positive_end_allowed: bool, dplpy, ads_module, spline_function):
    initial_curves = get_curves(data, positive_end_allowed, ads_module)
    initial_rwi = data / initial_curves
    initial_chronology, sample_depth = robust_chronology(dplpy, initial_rwi)
    normalized_depth = np.sqrt(sample_depth - 1.0) / np.sqrt(np.max(sample_depth - 1.0))
    segment_lengths = data.notna().sum(axis=0).to_numpy(int)
    median_segment_length = int(np.floor(np.median(segment_lengths)))
    original_means = data.mean(axis=0, skipna=True)
    last_chronology = initial_chronology
    previous_high_frequency = None
    trace = []
    final_rwi = None
    converged = False
    for iteration in range(1, MAX_ITERATIONS + 1):
        signal_free_measurements = data.div(last_chronology, axis=0)
        rescaled = signal_free_measurements.subtract(signal_free_measurements.mean(axis=0, skipna=True), axis=1)
        rescaled = rescaled.add(original_means, axis=1)
        if (rescaled <= 0).any(axis=None, skipna=True):
            raise RuntimeError(f"nonpositive-rescaled-signal-free-measurement:iteration={iteration}")
        curves = get_curves(rescaled, positive_end_allowed, ads_module)
        final_rwi = data / curves
        chronology, depth = robust_chronology(dplpy, final_rwi)
        if not np.array_equal(depth, sample_depth):
            raise RuntimeError("sample-depth-changed")
        low_frequency = np.asarray(
            spline_function(np.arange(len(chronology), dtype=float), chronology, median_segment_length), dtype=float
        )
        high_frequency = chronology - low_frequency
        median_abs_diff = None
        if previous_high_frequency is not None:
            median_abs_diff = float(np.median(np.abs(
                high_frequency * normalized_depth - previous_high_frequency * normalized_depth
            )))
        trace.append({"iteration": iteration, "median_abs_diff": median_abs_diff})
        last_chronology = chronology
        previous_high_frequency = high_frequency
        if median_abs_diff is not None and median_abs_diff <= MAD_THRESHOLD:
            converged = True
            break
    if final_rwi is None:
        raise RuntimeError("signal-free-no-iteration")
    return {
        "initial_chronology": initial_chronology,
        "final_chronology": last_chronology,
        "sample_depth": sample_depth,
        "trace": trace,
        "iterations": len(trace),
        "converged": converged,
        "final_mad": trace[-1]["median_abs_diff"],
        "median_segment_length": median_segment_length,
    }


def rolling(round246, proxy: np.ndarray, target: np.ndarray):
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
    output = loop / "round_256_npl065_modern_signal_free_sensitivity_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round256_npl065_modern_signal_free_sensitivity_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round246_driver": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "round252_driver": drivers / "run_round252_dplr_ads_python_port_validation_v3.py",
        "round253_driver": drivers / "run_round253_npl065_ads_power_method_decomposition_v1.py",
        "round253_summary": loop / "round_253_npl065_ads_power_method_decomposition_v1/summary.json",
        "dplr_ssf_source": loop / "round_251_dplr179_method_source_acquisition_v1/vendor_source/R/ssf.R",
        "rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "pre2001_manifest": loop / "round_208_setp_cru_tmin_annual_target_audit_v1/cru_tmn_acquisition_manifest.json",
        "post2000_manifest": loop / "round_245_cru_ts401_tmin_post2000_acquisition_v1/acquisition_manifest.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    round246 = load_module("round256_round246", paths["round246_driver"])
    ads_module = load_module("round256_ads", paths["round252_driver"])
    round253 = load_module("round256_round253", paths["round253_driver"])
    raw, parse_metadata = round246.parse_modern_rwl(paths["rwl"])
    transformed, power_rows = power_transform_frame(raw, round253)
    target, station_support, cru_files, box = round246.read_cru_box([paths["pre2001_manifest"], paths["post2000_manifest"]])
    paths.update({f"cru_{index:02d}": path for index, path in enumerate(cru_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})

    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy
    from smoothingspline import spline

    year_indices = pd.Index(MATRIX_YEARS).get_indexer(YEARS)
    round253_summary = json.loads(paths["round253_summary"].read_text(encoding="ascii"))
    expected_by_branch = {row["branch"]: row for row in round253_summary["metrics"]}
    expected_names = {
        "ssf_positive_end_allowed": "cookpow_ads30_ratio",
        "ssf_positive_end_constrained": "cookpow_ads30_ratio_nonpos",
    }
    metrics_rows = []
    fold_rows = []
    rolling_rows = []
    trace_rows = []
    curve_columns = {"year": YEARS, "cru_box_djf_tmin": target}
    closures = {}
    branch_info = {}
    for branch in BRANCHES:
        positive = branch == "ssf_positive_end_allowed"
        result = signal_free(transformed, positive, dplpy, ads_module, spline)
        initial_proxy = result["initial_chronology"][year_indices]
        final_proxy = result["final_chronology"][year_indices]
        initial_metrics, _, _ = round246.blocked_oof(YEARS, initial_proxy, target, BLOCKS)
        expected = expected_by_branch[expected_names[branch]]
        closure = {
            "in_sample_correlation": float(abs(initial_metrics["in_sample_correlation"] - expected["in_sample_correlation"])),
            "oof_correlation": float(abs(initial_metrics["oof_correlation"] - expected["oof_correlation"])),
        }
        if max(closure.values()) > 1e-12:
            raise RuntimeError(f"iteration-zero-closure-failed:{branch}:{closure}")
        closures[branch] = closure
        metrics, prediction, folds = round246.blocked_oof(YEARS, final_proxy, target, BLOCKS)
        roll = rolling(round246, final_proxy, target)
        roll_values = np.asarray([row["correlation"] for row in roll], dtype=float)
        metrics_rows.append({
            "branch": branch,
            **metrics,
            "initial_in_sample_correlation": initial_metrics["in_sample_correlation"],
            "initial_oof_correlation": initial_metrics["oof_correlation"],
            "signal_free_minus_initial_in_sample_correlation": metrics["in_sample_correlation"] - initial_metrics["in_sample_correlation"],
            "signal_free_minus_initial_oof_correlation": metrics["oof_correlation"] - initial_metrics["oof_correlation"],
            "rolling30_min_correlation": float(roll_values.min()),
            "rolling30_median_correlation": float(np.median(roll_values)),
            "rolling30_max_correlation": float(roll_values.max()),
            "rolling30_ge_0_5_window_count": int(np.sum(roll_values >= 0.5)),
            "gap_to_published_in_sample_r": PUBLISHED_R - metrics["in_sample_correlation"],
        })
        fold_rows.extend({"branch": branch, **row} for row in folds)
        rolling_rows.extend({"branch": branch, **row} for row in roll)
        trace_rows.extend({"branch": branch, **row} for row in result["trace"])
        curve_columns[f"{branch}_initial"] = initial_proxy
        curve_columns[f"{branch}_final"] = final_proxy
        curve_columns[f"{branch}_oof"] = prediction
        branch_info[branch] = {
            "converged": result["converged"],
            "iterations": result["iterations"],
            "final_mad": result["final_mad"],
            "median_segment_length": result["median_segment_length"],
        }

    metrics = pd.DataFrame(metrics_rows).set_index("branch")
    pd.DataFrame(metrics_rows).to_csv(output / "signal_free_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "signal_free_fold_metrics.csv", index=False)
    pd.DataFrame(rolling_rows).to_csv(output / "signal_free_rolling30_correlations.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(output / "signal_free_convergence_trace.csv", index=False)
    pd.DataFrame(power_rows).to_csv(output / "cook_power_parameters.csv", index=False)
    pd.DataFrame(curve_columns).to_csv(output / "signal_free_curves.csv", index=False)

    colors = {"ssf_positive_end_allowed": "#0072B2", "ssf_positive_end_constrained": "#D55E00"}
    figure, axes = plt.subplots(2, 1, figsize=(11.8, 7.2), constrained_layout=True)
    target_z = (target - target.mean()) / target.std(ddof=1)
    axes[0].plot(YEARS, target_z, color="#111111", linewidth=1.7, label="CRU DJF Tmin")
    for branch in BRANCHES:
        values = np.asarray(curve_columns[f"{branch}_final"], dtype=float)
        axes[0].plot(YEARS, (values - values.mean()) / values.std(ddof=1), color=colors[branch], linewidth=1.4, label=f"{branch}, r={metrics.at[branch, 'in_sample_correlation']:.3f}")
    axes[0].set_ylabel("standardized anomaly")
    axes[0].set_title("Modern-only signal-free chronology sensitivity")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    rolling_frame = pd.DataFrame(rolling_rows)
    for branch in BRANCHES:
        subset = rolling_frame[rolling_frame.branch == branch]
        axes[1].plot(subset.window_end, subset.correlation, color=colors[branch], linewidth=1.6, label=branch)
    axes[1].axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axes[1].set_ylabel("30-year correlation")
    axes[1].set_xlabel("rolling-window end year")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "npl065_modern_signal_free_sensitivity.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    converged_all = all(info["converged"] for info in branch_info.values())
    best_branch = str(metrics.in_sample_correlation.idxmax())
    best_gap = float(metrics.at[best_branch, "gap_to_published_in_sample_r"])
    if not converged_all:
        decision = "modern_signal_free_failed_convergence"
    elif best_gap <= 0.05 and metrics.at[best_branch, "oof_correlation"] > 0:
        decision = "modern_signal_free_explains_most_published_skill_gap"
    elif metrics.at[best_branch, "signal_free_minus_initial_in_sample_correlation"] > 0:
        decision = "modern_signal_free_partially_improves_full_length_context_still_required"
    else:
        decision = "modern_signal_free_does_not_explain_gap_full_length_context_or_exact_rcsigfree_required"
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "branches": branch_info,
        "metrics": metrics_rows,
        "iteration_zero_closure_absolute_error": closures,
        "best_branch": best_branch,
        "best_gap_to_published_r": best_gap,
        "decision": decision,
        "parse_metadata": parse_metadata,
        "target": {"dataset": "CRU_TS4.01_tmn", "season": "DJF", "box": box, "station_support": station_support},
        "implementation_scope": "modern_only_port_of_dplr_1.7.9_ssf_ratio_algorithm",
        "pre1941_numeric_values_used": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "paleo_proxy_numeric_values_decoded": False,
        "jionglacuo_opened_or_hashed": False,
        "rcsigfree_binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    lines = ["# Round 256 NPL065 modern signal-free sensitivity", ""]
    for branch in BRANCHES:
        lines.append(f"- {branch}: final in-sample r `{metrics.at[branch, 'in_sample_correlation']:.3f}`, OOF r `{metrics.at[branch, 'oof_correlation']:.3f}`, iterations `{branch_info[branch]['iterations']}`, converged `{str(branch_info[branch]['converged']).lower()}`.")
    lines.extend(["", f"Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
