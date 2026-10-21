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
BRANCHES = ("reset_at_modern_start", "inherit_structural_age")
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


def ads_with_start_stiffness(values: np.ndarray, start_stiffness: int, ads_module) -> np.ndarray:
    stiffness = np.arange(start_stiffness, start_stiffness + len(values), dtype=int)
    fitted = ads_module.ads_core(values, stiffness)
    differences = np.r_[0.0, np.diff(fitted)]
    eligible = np.flatnonzero(differences <= 0)
    cutoff = int(eligible[-1]) if len(eligible) else 0
    constrained = fitted.copy()
    constrained[cutoff:] = constrained[cutoff]
    return ads_module.ads_core(constrained, stiffness)


def build_rwi(raw: pd.DataFrame, first_years: dict[str, int], branch: str, ads_module, round253):
    rwi = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
    rows = []
    for sample in raw.columns:
        source = raw[sample].dropna()
        modern_first = int(source.index[0])
        original_first = int(first_years[str(sample)])
        if modern_first < original_first:
            raise RuntimeError(f"modern-before-original:{sample}")
        start_stiffness = 30 if branch == "reset_at_modern_start" else 30 + modern_first - original_first
        transformed, power = round253.cook_power_transform(source.to_numpy(float), 1.0)
        curve = ads_with_start_stiffness(transformed, start_stiffness, ads_module)
        if np.any(curve <= 0) or not np.isfinite(curve).all():
            raise RuntimeError(f"invalid-curve:{branch}:{sample}")
        rwi.loc[source.index, sample] = transformed / curve
        rows.append({
            "branch": branch,
            "sample": str(sample),
            "original_first_year": original_first,
            "modern_first_year": modern_first,
            "start_stiffness": start_stiffness,
            "cook_power": power,
            "curve_last_slope": float(curve[-1] - curve[-2]),
        })
    return rwi, rows


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
    output = loop / "round_255_npl065_ads_age_offset_sensitivity_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round255_npl065_ads_age_offset_sensitivity_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round246_driver": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "round252_driver": drivers / "run_round252_dplr_ads_python_port_validation_v3.py",
        "round253_driver": drivers / "run_round253_npl065_ads_power_method_decomposition_v1.py",
        "round253_summary": loop / "round_253_npl065_ads_power_method_decomposition_v1/summary.json",
        "age_offsets": loop / "round_254_npl065_structural_age_offset_audit_v1/sample_structural_age_offsets.csv",
        "round254_summary": loop / "round_254_npl065_structural_age_offset_audit_v1/summary.json",
        "rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "pre2001_manifest": loop / "round_208_setp_cru_tmin_annual_target_audit_v1/cru_tmn_acquisition_manifest.json",
        "post2000_manifest": loop / "round_245_cru_ts401_tmin_post2000_acquisition_v1/acquisition_manifest.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    if json.loads(paths["round254_summary"].read_text(encoding="ascii")).get("passes") is not True:
        raise RuntimeError("structural-age-offsets-not-admitted")
    round246 = load_module("round255_round246", paths["round246_driver"])
    ads_module = load_module("round255_ads", paths["round252_driver"])
    round253 = load_module("round255_round253", paths["round253_driver"])
    raw, parse_metadata = round246.parse_modern_rwl(paths["rwl"])
    age_frame = pd.read_csv(paths["age_offsets"])
    first_years = dict(zip(age_frame["sample"].astype(str), age_frame.first_occupied_year.astype(int)))
    if set(first_years) != set(map(str, raw.columns)):
        raise RuntimeError("age-offset-sample-set-mismatch")
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
    parameter_rows = []
    curve_columns = {"year": YEARS, "cru_box_djf_tmin": target}
    for branch in BRANCHES:
        rwi, parameters = build_rwi(raw, first_years, branch, ads_module, round253)
        parameter_rows.extend(parameters)
        chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
        proxy = chronology["Mean RWI"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)[year_indices]
        depth = chronology["Sample depth"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)[year_indices]
        if not np.isfinite(proxy).all() or not np.isfinite(depth).all():
            raise RuntimeError(f"chronology-gap:{branch}")
        metrics, prediction, folds = round246.blocked_oof(YEARS, proxy, target, BLOCKS)
        roll = rolling(round246, proxy, target)
        values = np.asarray([row["correlation"] for row in roll], dtype=float)
        starts = np.asarray([row["start_stiffness"] for row in parameters], dtype=int)
        metrics_rows.append({
            "branch": branch,
            **metrics,
            "rolling30_min_correlation": float(values.min()),
            "rolling30_median_correlation": float(np.median(values)),
            "rolling30_max_correlation": float(values.max()),
            "rolling30_ge_0_5_window_count": int(np.sum(values >= 0.5)),
            "start_stiffness_min": int(starts.min()),
            "start_stiffness_median": float(np.median(starts)),
            "start_stiffness_max": int(starts.max()),
            "gap_to_published_in_sample_r": float(PUBLISHED_R - metrics["in_sample_correlation"]),
        })
        fold_rows.extend({"branch": branch, **row} for row in folds)
        rolling_rows.extend({"branch": branch, **row} for row in roll)
        curve_columns[f"{branch}_chronology"] = proxy
        curve_columns[f"{branch}_oof"] = prediction

    metrics = pd.DataFrame(metrics_rows).set_index("branch")
    round253_summary = json.loads(paths["round253_summary"].read_text(encoding="ascii"))
    expected = next(row for row in round253_summary["metrics"] if row["branch"] == "cookpow_ads30_ratio_nonpos")
    closure = {
        "in_sample_correlation": float(abs(metrics.at["reset_at_modern_start", "in_sample_correlation"] - expected["in_sample_correlation"])),
        "oof_correlation": float(abs(metrics.at["reset_at_modern_start", "oof_correlation"] - expected["oof_correlation"])),
    }
    if max(closure.values()) > 1e-12:
        raise RuntimeError(f"round253-reset-closure-failed:{closure}")
    pd.DataFrame(metrics_rows).to_csv(output / "age_offset_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "age_offset_fold_metrics.csv", index=False)
    pd.DataFrame(rolling_rows).to_csv(output / "age_offset_rolling30_correlations.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "age_offset_series_parameters.csv", index=False)
    pd.DataFrame(curve_columns).to_csv(output / "age_offset_curves.csv", index=False)

    colors = {"reset_at_modern_start": "#777777", "inherit_structural_age": "#D55E00"}
    figure, axes = plt.subplots(2, 1, figsize=(11.8, 7.1), constrained_layout=True)
    target_z = (target - target.mean()) / target.std(ddof=1)
    axes[0].plot(YEARS, target_z, color="#111111", linewidth=1.7, label="CRU DJF Tmin")
    for branch in BRANCHES:
        values = np.asarray(curve_columns[f"{branch}_chronology"], dtype=float)
        axes[0].plot(YEARS, (values - values.mean()) / values.std(ddof=1), color=colors[branch], linewidth=1.3, label=f"{branch}, r={metrics.at[branch, 'in_sample_correlation']:.3f}")
    axes[0].set_ylabel("standardized anomaly")
    axes[0].set_title("Effect of inheriting structural biological age")
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
    figure.savefig(output / "npl065_ads_age_offset_sensitivity.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    delta_in = float(metrics.at["inherit_structural_age", "in_sample_correlation"] - metrics.at["reset_at_modern_start", "in_sample_correlation"])
    delta_oof = float(metrics.at["inherit_structural_age", "oof_correlation"] - metrics.at["reset_at_modern_start", "oof_correlation"])
    if delta_in >= 0.05 and delta_oof >= 0.05:
        decision = "structural_age_reset_materially_suppressed_modern_skill"
    elif delta_in > 0 or delta_oof > 0:
        decision = "structural_age_reset_partially_suppressed_modern_skill"
    else:
        decision = "structural_age_reset_did_not_explain_skill_gap"
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "metrics": metrics_rows,
        "round253_reset_closure_absolute_error": closure,
        "inherit_minus_reset_in_sample_correlation": delta_in,
        "inherit_minus_reset_oof_correlation": delta_oof,
        "decision": decision,
        "parse_metadata": parse_metadata,
        "target": {"dataset": "CRU_TS4.01_tmn", "season": "DJF", "box": box, "station_support": station_support},
        "pre1941_numeric_values_used": False,
        "signal_free_iteration_implemented": False,
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
    (output / "FINAL_RESULT.md").write_text(
        "# Round 255 NPL065 ADS age-offset sensitivity\n\n"
        f"- Reset: in-sample r `{metrics.at['reset_at_modern_start', 'in_sample_correlation']:.3f}`, OOF r `{metrics.at['reset_at_modern_start', 'oof_correlation']:.3f}`.\n"
        f"- Inherited age: in-sample r `{metrics.at['inherit_structural_age', 'in_sample_correlation']:.3f}`, OOF r `{metrics.at['inherit_structural_age', 'oof_correlation']:.3f}`.\n"
        f"- Inherited-minus-reset delta: in-sample `{delta_in:+.3f}`, OOF `{delta_oof:+.3f}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
