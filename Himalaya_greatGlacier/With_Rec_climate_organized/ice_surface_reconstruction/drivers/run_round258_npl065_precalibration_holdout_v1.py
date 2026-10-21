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
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1952, 2014, dtype=int)
CALIBRATION = YEARS >= 1960
HOLDOUT = YEARS <= 1959
LAT_RANGE = (27.5, 28.5)
LON_RANGE = (86.0, 87.0)


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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def area_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def read_target(manifests: list[Path]):
    monthly = {}
    support = {}
    files = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            start, end = [int(value) for value in record["allowed_year_range"].split("-")]
            if end < 1951 or start > 2013:
                continue
            if start <= 1939:
                raise RuntimeError(f"forbidden-range:{start}-{end}")
            path = Path(record["netcdf_path"])
            if fingerprint(path)["sha256"] != record["netcdf_sha256"]:
                raise RuntimeError(f"cru-hash-changed:{path.name}")
            files.append(path)
            with netcdf_file(path, "r", mmap=True) as dataset:
                lats = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
                lons = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
                iy = np.flatnonzero((lats >= LAT_RANGE[0]) & (lats <= LAT_RANGE[1]))
                ix = np.flatnonzero((lons >= LON_RANGE[0]) & (lons <= LON_RANGE[1]))
                tmn = np.asarray(dataset.variables["tmn"].data[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], dtype=float).copy()
                stn = np.asarray(dataset.variables["stn"].data[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], dtype=float).copy()
                fill = float(dataset.variables["tmn"]._attributes["_FillValue"])
                tmn[np.isclose(tmn, fill) | (tmn > 1e20)] = np.nan
                selected_lats = lats[iy]
            periods = pd.period_range(f"{start}-01", periods=12 * (end - start + 1), freq="M")
            values = area_mean(tmn, selected_lats)
            fractions = np.mean(stn > 0, axis=(1, 2))
            for period, value, fraction in zip(periods, values, fractions):
                key = (int(period.year), int(period.month))
                if key in monthly:
                    raise RuntimeError(f"duplicate-month:{key}")
                monthly[key] = float(value)
                support[key] = float(fraction)
    target = np.full(len(YEARS), np.nan)
    support_fraction = np.full(len(YEARS), np.nan)
    for index, year in enumerate(YEARS):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        values = np.asarray([monthly.get(key, np.nan) for key in keys], dtype=float)
        fractions = np.asarray([support.get(key, np.nan) for key in keys], dtype=float)
        if np.isfinite(values).all():
            target[index] = float(values.mean())
            support_fraction[index] = float(fractions.mean())
    if not np.isfinite(target).all():
        raise RuntimeError(f"incomplete-target:{YEARS[~np.isfinite(target)].tolist()}")
    return target, support_fraction, files


def evaluate(name: str, proxy: np.ndarray, target: np.ndarray):
    beta = np.linalg.lstsq(
        np.column_stack((np.ones(int(CALIBRATION.sum())), proxy[CALIBRATION])),
        target[CALIBRATION], rcond=None,
    )[0]
    prediction = beta[0] + beta[1] * proxy
    calibration_mean = float(target[CALIBRATION].mean())
    holdout_sse = float(np.sum((target[HOLDOUT] - prediction[HOLDOUT]) ** 2))
    holdout_baseline_sse = float(np.sum((target[HOLDOUT] - calibration_mean) ** 2))
    holdout_diff_observed = np.diff(target[HOLDOUT])
    holdout_diff_predicted = np.diff(prediction[HOLDOUT])
    metrics = {
        "method": name,
        "fit_intercept": float(beta[0]),
        "fit_slope": float(beta[1]),
        "calibration_correlation": correlation(proxy[CALIBRATION], target[CALIBRATION]),
        "holdout_correlation": correlation(prediction[HOLDOUT], target[HOLDOUT]),
        "holdout_rmse_degC": float(np.sqrt(holdout_sse / HOLDOUT.sum())),
        "holdout_mae_degC": float(np.mean(np.abs(target[HOLDOUT] - prediction[HOLDOUT]))),
        "holdout_bias_degC": float(np.mean(prediction[HOLDOUT] - target[HOLDOUT])),
        "holdout_skill_vs_calibration_climatology": 1.0 - holdout_sse / holdout_baseline_sse,
        "holdout_first_difference_correlation": correlation(holdout_diff_predicted, holdout_diff_observed),
        "holdout_first_difference_sign_agreement": float(np.mean(np.sign(holdout_diff_predicted) == np.sign(holdout_diff_observed))),
    }
    return metrics, prediction


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_258_npl065_precalibration_holdout_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "round246_driver": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "round252_driver": drivers / "run_round252_dplr_ads_python_port_validation_v3.py",
        "round253_driver": drivers / "run_round253_npl065_ads_power_method_decomposition_v1.py",
        "round256_driver": drivers / "run_round256_npl065_modern_signal_free_sensitivity_v1.py",
        "round256_summary": loop / "round_256_npl065_modern_signal_free_sensitivity_v1/summary.json",
        "modnegex_chronology": loop / "round_246_npl065_calibration_endpoint_attribution_v2/modern_only_chronology_1941_2013.csv",
        "rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "pre2001_manifest": loop / "round_208_setp_cru_tmin_annual_target_audit_v1/cru_tmn_acquisition_manifest.json",
        "post2000_manifest": loop / "round_245_cru_ts401_tmin_post2000_acquisition_v1/acquisition_manifest.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    round246 = load_module("round258_round246", paths["round246_driver"])
    ads_module = load_module("round258_ads", paths["round252_driver"])
    round253 = load_module("round258_round253", paths["round253_driver"])
    round256 = load_module("round258_round256", paths["round256_driver"])
    raw, parse_metadata = round246.parse_modern_rwl(paths["rwl"])
    transformed, _ = round256.power_transform_frame(raw, round253)

    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy
    from smoothingspline import spline

    sf = round256.signal_free(transformed, False, dplpy, ads_module, spline)
    if not sf["converged"]:
        raise RuntimeError("signal-free-did-not-converge")
    sf_full = sf["final_chronology"]
    sf_proxy = sf_full[pd.Index(MATRIX_YEARS).get_indexer(YEARS)]
    mod_frame = pd.read_csv(paths["modnegex_chronology"])
    mod_proxy = mod_frame.set_index("year").loc[YEARS, "modern_only_std_chronology"].to_numpy(float)
    target, support, cru_files = read_target([paths["pre2001_manifest"], paths["post2000_manifest"]])
    paths.update({f"cru_{index:02d}": path for index, path in enumerate(cru_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})
    metrics_rows = []
    predictions = {}
    for name, proxy in (("signal_free_constrained", sf_proxy), ("modnegex", mod_proxy)):
        metrics, prediction = evaluate(name, proxy, target)
        metrics_rows.append(metrics)
        predictions[name] = prediction
    metrics = pd.DataFrame(metrics_rows).set_index("method")
    pd.DataFrame(metrics_rows).to_csv(output / "precalibration_holdout_metrics.csv", index=False)
    pd.DataFrame({
        "year": YEARS,
        "cru_box_djf_tmin": target,
        "station_supported_cell_fraction": support,
        "signal_free_chronology": sf_proxy,
        "signal_free_prediction": predictions["signal_free_constrained"],
        "modnegex_chronology": mod_proxy,
        "modnegex_prediction": predictions["modnegex"],
        "period": np.where(HOLDOUT, "holdout", "calibration"),
    }).to_csv(output / "precalibration_holdout_curves.csv", index=False)

    figure, axis = plt.subplots(figsize=(11.8, 5.1), constrained_layout=True)
    axis.plot(YEARS, target, color="#111111", linewidth=1.7, label="CRU DJF Tmin")
    axis.plot(YEARS, predictions["signal_free_constrained"], color="#D55E00", linewidth=1.45, label=f"Signal-free; holdout r={metrics.at['signal_free_constrained', 'holdout_correlation']:.3f}")
    axis.plot(YEARS, predictions["modnegex"], color="#0072B2", linewidth=1.2, linestyle="--", label=f"ModNegEx; holdout r={metrics.at['modnegex', 'holdout_correlation']:.3f}")
    axis.axvspan(1951.5, 1959.5, color="#E9C46A", alpha=0.25, label="1952-1959 holdout")
    axis.axvline(1959.5, color="#777777", linestyle=":", linewidth=1)
    axis.set_xlabel("Ring year")
    axis.set_ylabel("degC")
    axis.set_title("NPL065 pre-calibration temporal holdout")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=9)
    figure.savefig(output / "npl065_precalibration_holdout.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    sf_skill = float(metrics.at["signal_free_constrained", "holdout_skill_vs_calibration_climatology"])
    sf_r = float(metrics.at["signal_free_constrained", "holdout_correlation"])
    mod_r = float(metrics.at["modnegex", "holdout_correlation"])
    if sf_skill > 0 and sf_r > mod_r:
        decision = "signal_free_skill_transports_to_short_precalibration_holdout"
    elif sf_skill > 0:
        decision = "signal_free_positive_short_holdout_skill_without_method_advantage"
    else:
        decision = "signal_free_does_not_transport_to_short_precalibration_holdout"
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "holdout_years": [1952, 1959],
        "calibration_years": [1960, 2013],
        "holdout_year_count": int(HOLDOUT.sum()),
        "holdout_station_supported_cell_fraction_mean": float(np.mean(support[HOLDOUT])),
        "metrics": metrics_rows,
        "signal_free_iterations": sf["iterations"],
        "signal_free_final_mad": sf["final_mad"],
        "decision": decision,
        "interpretation_boundary": "eight_year_holdout_is_supporting_not_decisive_evidence",
        "parse_metadata": parse_metadata,
        "locked_1921_1939_values_accessed": False,
        "paleo_proxy_values_accessed": False,
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
        "# Round 258 NPL065 pre-calibration holdout\n\n"
        f"- Signal-free holdout r: `{sf_r:.3f}`; skill: `{sf_skill:.3f}`; RMSE: `{metrics.at['signal_free_constrained', 'holdout_rmse_degC']:.3f} degC`.\n"
        f"- ModNegEx holdout r: `{mod_r:.3f}`; skill: `{metrics.at['modnegex', 'holdout_skill_vs_calibration_climatology']:.3f}`.\n"
        f"- Holdout station-supported cell fraction: `{np.mean(support[HOLDOUT]):.1%}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
