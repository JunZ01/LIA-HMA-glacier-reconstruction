"""Run fixed-input EnKF branches and validate the stitched blocked-CV reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-load-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v73(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name]) for name in handle.keys() if not name.startswith("#")}


def anomaly(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    selected = (years >= period[0]) & (years <= period[1])
    return values - np.nanmean(values[..., selected], axis=-1, keepdims=True)


def curve_metrics(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> dict[str, float]:
    selected = (years >= period[0]) & (years <= period[1])
    return {
        "mean_c": float(np.nanmean(values[selected])),
        "trend_c_per_century": float(100.0 * np.polyfit(years[selected], values[selected], 1)[0]),
        "std_c": float(np.nanstd(values[selected])),
    }


def plot_curves(years: np.ndarray, curves: dict[str, np.ndarray], observations: dict[str, np.ndarray], output: Path) -> None:
    selected = (years >= 1850) & (years <= 2000)
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 8.5), sharex=True, constrained_layout=True)
    styles = {
        "archived_1940_2000": ("#111111", 2.2),
        "full_1901_2000": ("#b2182b", 2.2),
        "blocked_crossfit": ("#007f8b", 2.5),
    }
    for name, values in curves.items():
        color, width = styles[name]
        rolled = pd.Series(values).rolling(15, center=True, min_periods=15).mean().to_numpy()
        axes[0].plot(years[selected], values[selected], color=color, lw=width, label=name)
        axes[1].plot(years[selected], rolled[selected], color=color, lw=width, label=name)
    for name, values in observations.items():
        rolled = pd.Series(values).rolling(15, center=True, min_periods=15).mean().to_numpy()
        axes[0].plot(years[selected], values[selected], color="#888888", lw=0.7, alpha=0.35)
        axes[1].plot(years[selected], rolled[selected], color="#888888", lw=0.8, alpha=0.5, label=name)
    for axis in axes:
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.axvline(1950.5, color="#777777", lw=0.8, ls="--")
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA temperature under full-period and blocked cross-fitted PSM calibration")
    axes[1].set_title("Centered 15-year mean")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].legend(frameon=False, ncol=4, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    result = root / cfg["result_root"]
    operators_path = result / "blocked_cv_operators.mat"
    if not operators_path.exists():
        raise FileNotFoundError(f"missing-matlab-output:{operators_path}")
    archive = Path(cfg["archive_root"])
    source = cfg["sources"]
    paths = {
        "operators": operators_path,
        "proxy9": archive / source["proxy9"],
        "window_v4": root / source["window_v4"],
        "baseline_driver": root / source["baseline_driver"],
        "evaluation_driver": root / source["evaluation_driver"],
        "prior": root / source["prior"],
        "indices": archive / source["indices"],
        "archived_regem": root / source["archived_regem"],
        "baseline_arrays": root / source["baseline_arrays"],
        "hma_mask": root / source["hma_mask"],
        "instrumental_netcdf": Path(source["instrumental_netcdf"]),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("blocked_cv_base", paths["baseline_driver"])
    evaluation = load_module("blocked_cv_eval", paths["evaluation_driver"])
    saved = load_v73(paths["operators"])
    hxb = np.asarray(saved["hxb"], dtype=float).transpose(2, 1, 0)
    # MATLAB 297 x 7 x 848 becomes 297 x 7 x 848 after h5py reversal and transpose.
    if hxb.shape != (297, 7, 848):
        raise RuntimeError(f"unexpected-hxb-shape:{hxb.shape}")
    baseline_hxb_closure = float(np.asarray(saved["baseline_hxb_max_abs"]).squeeze())
    baseline_coefficient_closure = float(np.asarray(saved["baseline_coefficient_max_abs"]).squeeze())
    if baseline_hxb_closure > float(cfg["gates"]["baseline_hxb_max_abs"]) or baseline_coefficient_closure > float(cfg["gates"]["baseline_coefficient_max_abs"]):
        raise RuntimeError(f"matlab-baseline-closure-failed:{baseline_hxb_closure}:{baseline_coefficient_closure}")

    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    branch_names = ["archived_1940_2000", "full_1901_2000"] + [f"fold_{i}" for i in range(1, 6)]
    proxies = {"archived_1940_2000": {key: np.asarray(value).copy() for key, value in proxy.items()}}
    for index, name in enumerate(branch_names[1:], start=1):
        item = {key: np.asarray(value).copy() for key, value in proxy.items()}
        item["hxb"] = proxy["hxb"] + (hxb[:, index, :] - hxb[:, 0, :])
        proxies[name] = item

    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"])
    mask2 = evaluation.source_grid(mask05[None].astype(float), source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    state_mask = mask2.T.reshape(-1)
    regional_weights = np.cos(np.deg2rad(state_lat)) * state_mask
    regional_weights /= regional_weights.sum()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)
    results = {
        name: base._run_branch(prior, samples, item, residuals, target_weights, regional_weights)
        for name, item in proxies.items()
    }
    with np.load(paths["baseline_arrays"]) as baseline_saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(results["archived_1940_2000"]["analysis"]) - baseline_saved["proxy9_analysis"])))
    enkf_closure = max(
        float(result_item[key])
        for result_item in results.values()
        for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c")
    )
    if baseline_state_closure > float(cfg["gates"]["baseline_state_closure_max_abs_c"]) or enkf_closure > float(cfg["gates"]["enkf_closure_max_abs_c"]):
        raise RuntimeError(f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    reference = tuple(cfg["periods"]["reference"])
    regional_raw = {name: np.asarray(item["regional_analysis"]).copy() for name, item in results.items()}
    analysis_raw = {name: np.asarray(item["analysis"]).copy() for name, item in results.items()}
    crossfit_regional = regional_raw["archived_1940_2000"].copy()
    crossfit_analysis = analysis_raw["archived_1940_2000"].copy()
    for fold, (start, end) in enumerate(cfg["test_blocks"], start=1):
        selected = (years >= start) & (years <= end)
        crossfit_regional[selected] = regional_raw[f"fold_{fold}"][selected]
        crossfit_analysis[:, selected] = analysis_raw[f"fold_{fold}"][:, selected]

    regional = {
        "archived_1940_2000": anomaly(regional_raw["archived_1940_2000"], years, reference),
        "full_1901_2000": anomaly(regional_raw["full_1901_2000"], years, reference),
        "blocked_crossfit": anomaly(crossfit_regional, years, reference),
    }
    cubes = {
        "archived_1940_2000": evaluation.anomaly(evaluation.to_cube(analysis_raw["archived_1940_2000"], len(grid_lat), len(grid_lon)), years, reference),
        "full_1901_2000": evaluation.anomaly(evaluation.to_cube(analysis_raw["full_1901_2000"], len(grid_lat), len(grid_lon)), years, reference),
        "blocked_crossfit": evaluation.anomaly(evaluation.to_cube(crossfit_analysis, len(grid_lat), len(grid_lon)), years, reference),
    }

    observations = {}
    observation_cubes = {}
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        for label, variable in {
            "Berkeley Earth": "tas_obs_bea_05deg",
            "CRU": "tas_obs_cru_tmp_05deg",
            "GISTEMP": "tas_obs_GIST_05deg",
            "HadCRUT5": "tas_obs_had5_05deg",
            "NOAA": "tas_obs_NOAA_05deg",
            "rec_ref": "tas_ensemble_mean_rec_ref",
        }.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, reference)
            observation_cubes[label] = field
            observations[label] = evaluation.hma_mean(field, grid_lat, mask2)

    rows = []
    curve_rows = []
    for branch, field in cubes.items():
        for period_name in ("early", "late", "full"):
            start, end = cfg["periods"][period_name]
            selected = (years >= start) & (years <= end)
            curve_rows.append({"branch": branch, "period": f"{start}-{end}", **curve_metrics(regional[branch], years, (start, end))})
            for observation_name in ("Berkeley Earth", "CRU", "GISTEMP", "HadCRUT5", "NOAA"):
                row = {"branch": branch, "observation": observation_name, "period": f"{start}-{end}"}
                row.update(evaluation.metrics(field[selected], observation_cubes[observation_name][selected], years[selected], mask2))
                rows.append(row)
    skill = pd.DataFrame(rows)
    skill.to_csv(result / "instrumental_skill_summary_2deg.csv", index=False)
    principal = (
        skill[~skill["observation"].eq("NOAA")]
        .groupby(["branch", "period"])[["median_lowpass_correlation_15y", "median_rmse", "median_abs_trend_difference"]]
        .mean()
        .reset_index()
    )
    principal.to_csv(result / "instrumental_skill_principal_mean.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(result / "hma_period_summary.csv", index=False)
    pd.DataFrame({"year": years, **regional, **observations}).to_csv(result / "hma_regional_series.csv", index=False)

    holdout_count = np.asarray(saved["holdout_count"], dtype=float).T
    holdout_rmse = np.asarray(saved["holdout_rmse"], dtype=float).T
    holdout_correlation = np.asarray(saved["holdout_correlation"], dtype=float).T
    proxy_rows = []
    for record in range(297):
        valid_rmse = holdout_rmse[record][np.isfinite(holdout_rmse[record])]
        valid_corr = holdout_correlation[record][np.isfinite(holdout_correlation[record])]
        proxy_rows.append({
            "proxy9_record_1based": record + 1,
            "total_holdout_observations": int(np.nansum(holdout_count[record])),
            "valid_fold_rmse_count": int(len(valid_rmse)),
            "median_holdout_rmse": float(np.median(valid_rmse)) if len(valid_rmse) else np.nan,
            "median_holdout_correlation": float(np.median(valid_corr)) if len(valid_corr) else np.nan,
        })
    proxy_cv = pd.DataFrame(proxy_rows)
    proxy_cv.to_csv(result / "proxy9_blocked_cv_diagnostics.csv", index=False)

    plot_curves(
        years,
        regional,
        {key: observations[key] for key in ("Berkeley Earth", "CRU", "GISTEMP", "HadCRUT5")},
        result / "figure_hma_blocked_cv_curves_1850_2000.png",
    )

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if pure else "fail",
        "baseline_hxb_max_abs": baseline_hxb_closure,
        "baseline_coefficient_max_abs": baseline_coefficient_closure,
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "source_hashes_unchanged": pure,
        "proxy_cv": {
            "records_with_all_five_valid_rmse_folds": int((proxy_cv["valid_fold_rmse_count"] == 5).sum()),
            "median_record_holdout_rmse": float(proxy_cv["median_holdout_rmse"].median()),
            "median_record_holdout_correlation": float(proxy_cv["median_holdout_correlation"].median()),
        },
        "hma_period_summary": curves.to_dict(orient="records"),
        "principal_instrumental_skill": principal.to_dict(orient="records"),
        "interpretation_boundary": "The stitched blocked_crossfit branch is temporally out of sample for PSM calibration. The full_1901_2000 branch is a production hypothesis only and is not accepted from its in-period fit.",
    }
    (result / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (result / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (result / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (result / "FINAL_RESULT.md").write_text(
        "# Proxy9 Five-Fold Blocked Calibration Validation v1\n\n"
        f"Status: **{summary['status']}**. Interpret the stitched blocked-crossfit branch as the independent temporal validation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
