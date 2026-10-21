"""Compare temperature-only and bivariate blocked-CV reconstructions."""

from __future__ import annotations

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


def anomaly(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    selected = (years >= period[0]) & (years <= period[1])
    return values - np.nanmean(values[..., selected], axis=-1, keepdims=True)


def period_metrics(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> dict[str, float]:
    selected = (years >= period[0]) & (years <= period[1])
    return {
        "mean_c": float(np.nanmean(values[selected])),
        "trend_c_per_century": float(100.0 * np.polyfit(years[selected], values[selected], 1)[0]),
        "std_c": float(np.nanstd(values[selected])),
    }


def main() -> int:
    root = project_root()
    config_path = Path(__file__).with_name("experiment_config.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    result = root / cfg["result_root"]
    archive = Path(cfg["archive_root"])
    source = cfg["sources"]
    paths = {
        "temperature_only": result / "temperature_only_operators.mat",
        "bivariate": root / source["bivariate_operators"],
        "proxy9": archive / source["proxy9"],
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
    base = load_module("temp_only_base", paths["baseline_driver"])
    evaluation = load_module("temp_only_eval", paths["evaluation_driver"])
    with h5py.File(paths["temperature_only"], "r") as handle:
        temp_hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
        temp_holdout_rmse = np.asarray(handle["holdout_rmse"], dtype=float).T
        temp_holdout_corr = np.asarray(handle["holdout_correlation"], dtype=float).T
        temp_training_count = np.asarray(handle["training_count"], dtype=float).T
    with h5py.File(paths["bivariate"], "r") as handle:
        bi_hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
        bi_holdout_rmse = np.asarray(handle["holdout_rmse"], dtype=float).T
        bi_holdout_corr = np.asarray(handle["holdout_correlation"], dtype=float).T
    if temp_hxb.shape != (297, 6, 848):
        raise RuntimeError(f"unexpected-temperature-only-shape:{temp_hxb.shape}")

    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
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

    proxies = {"archived_1940_2000": {key: np.asarray(value).copy() for key, value in proxy.items()}}
    full = {key: np.asarray(value).copy() for key, value in proxy.items()}
    full["hxb"] = proxy["hxb"] + (temp_hxb[:, 0, :] - bi_hxb[:, 0, :])
    proxies["temperature_only_full_1901_2000"] = full
    for fold in range(1, 6):
        item = {key: np.asarray(value).copy() for key, value in proxy.items()}
        item["hxb"] = proxy["hxb"] + (temp_hxb[:, fold, :] - bi_hxb[:, 0, :])
        proxies[f"temperature_only_fold_{fold}"] = item
    results = {
        name: base._run_branch(prior, samples, item, residuals, target_weights, regional_weights)
        for name, item in proxies.items()
    }
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(results["archived_1940_2000"]["analysis"]) - saved["proxy9_analysis"])))
    enkf_closure = max(
        float(result_item[key])
        for result_item in results.values()
        for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c")
    )
    if baseline_state_closure > float(cfg["gates"]["baseline_state_closure_max_abs_c"]) or enkf_closure > float(cfg["gates"]["enkf_closure_max_abs_c"]):
        raise RuntimeError(f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    reference = tuple(cfg["periods"]["reference"])
    archived_regional_raw = np.asarray(results["archived_1940_2000"]["regional_analysis"])
    archived_analysis_raw = np.asarray(results["archived_1940_2000"]["analysis"])
    crossfit_regional_raw = archived_regional_raw.copy()
    crossfit_analysis_raw = archived_analysis_raw.copy()
    for fold, (start, end) in enumerate(cfg["test_blocks"], start=1):
        selected = (years >= start) & (years <= end)
        crossfit_regional_raw[selected] = np.asarray(results[f"temperature_only_fold_{fold}"]["regional_analysis"])[selected]
        crossfit_analysis_raw[:, selected] = np.asarray(results[f"temperature_only_fold_{fold}"]["analysis"])[:, selected]
    regional = {
        "archived_1940_2000": anomaly(archived_regional_raw, years, reference),
        "temperature_only_full_1901_2000": anomaly(np.asarray(results["temperature_only_full_1901_2000"]["regional_analysis"]), years, reference),
        "temperature_only_blocked_crossfit": anomaly(crossfit_regional_raw, years, reference),
    }
    cubes = {
        "archived_1940_2000": evaluation.anomaly(evaluation.to_cube(archived_analysis_raw, len(grid_lat), len(grid_lon)), years, reference),
        "temperature_only_full_1901_2000": evaluation.anomaly(evaluation.to_cube(np.asarray(results["temperature_only_full_1901_2000"]["analysis"]), len(grid_lat), len(grid_lon)), years, reference),
        "temperature_only_blocked_crossfit": evaluation.anomaly(evaluation.to_cube(crossfit_analysis_raw, len(grid_lat), len(grid_lon)), years, reference),
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
        }.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, reference)
            observation_cubes[label] = field
            observations[label] = evaluation.hma_mean(field, grid_lat, mask2)

    skill_rows = []
    period_rows = []
    for branch, field in cubes.items():
        for period_name in ("early", "late", "full"):
            start, end = cfg["periods"][period_name]
            selected = (years >= start) & (years <= end)
            period_rows.append({"branch": branch, "period": f"{start}-{end}", **period_metrics(regional[branch], years, (start, end))})
            for observation_name, observation_field in observation_cubes.items():
                row = {"branch": branch, "observation": observation_name, "period": f"{start}-{end}"}
                row.update(evaluation.metrics(field[selected], observation_field[selected], years[selected], mask2))
                skill_rows.append(row)
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(result / "instrumental_skill_summary_2deg.csv", index=False)
    principal = (
        skill[~skill["observation"].eq("NOAA")]
        .groupby(["branch", "period"])[["median_lowpass_correlation_15y", "median_rmse", "median_abs_trend_difference"]]
        .mean()
        .reset_index()
    )
    principal.to_csv(result / "instrumental_skill_principal_mean.csv", index=False)
    periods = pd.DataFrame(period_rows)
    periods.to_csv(result / "hma_period_summary.csv", index=False)
    pd.DataFrame({"year": years, **regional, **observations}).to_csv(result / "hma_regional_series.csv", index=False)

    proxy_cv = pd.DataFrame({
        "proxy9_record_1based": np.arange(1, 298),
        "temperature_only_median_holdout_rmse": np.nanmedian(temp_holdout_rmse, axis=1),
        "bivariate_median_holdout_rmse": np.nanmedian(bi_holdout_rmse, axis=1),
        "temperature_only_median_holdout_correlation": np.nanmedian(temp_holdout_corr, axis=1),
        "bivariate_median_holdout_correlation": np.nanmedian(bi_holdout_corr, axis=1),
        "temperature_only_fold5_rmse": temp_holdout_rmse[:, 4],
        "bivariate_fold5_rmse": bi_holdout_rmse[:, 4],
        "temperature_only_fold5_correlation": temp_holdout_corr[:, 4],
        "bivariate_fold5_correlation": bi_holdout_corr[:, 4],
        "minimum_temperature_only_training_count": np.nanmin(temp_training_count, axis=1),
    })
    proxy_cv["median_rmse_improvement"] = proxy_cv["bivariate_median_holdout_rmse"] - proxy_cv["temperature_only_median_holdout_rmse"]
    proxy_cv["fold5_rmse_improvement"] = proxy_cv["bivariate_fold5_rmse"] - proxy_cv["temperature_only_fold5_rmse"]
    proxy_cv.to_csv(result / "proxy9_temperature_only_cv_comparison.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(12.8, 8.4), sharex=True, constrained_layout=True)
    selected = (years >= 1850) & (years <= 2000)
    styles = {
        "archived_1940_2000": ("#111111", 2.2),
        "temperature_only_full_1901_2000": ("#b2182b", 2.2),
        "temperature_only_blocked_crossfit": ("#007f8b", 2.5),
    }
    for name, values in regional.items():
        color, width = styles[name]
        rolled = pd.Series(values).rolling(15, center=True, min_periods=15).mean().to_numpy()
        axes[0].plot(years[selected], values[selected], color=color, lw=width, label=name)
        axes[1].plot(years[selected], rolled[selected], color=color, lw=width, label=name)
    for axis in axes:
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.axvline(1950.5, color="#777777", lw=0.8, ls="--")
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA temperature with annual temperature-only proxy9 PSMs")
    axes[1].set_title("Centered 15-year mean")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=3)
    figure.savefig(result / "figure_temperature_only_blocked_cv_curves.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if pure else "fail",
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "source_hashes_unchanged": pure,
        "proxy_cv": {
            "records_temperature_only_lower_median_rmse": int((proxy_cv["median_rmse_improvement"] > 0).sum()),
            "records_temperature_only_lower_fold5_rmse": int((proxy_cv["fold5_rmse_improvement"] > 0).sum()),
            "median_median_rmse_improvement": float(proxy_cv["median_rmse_improvement"].median()),
            "median_fold5_rmse_improvement": float(proxy_cv["fold5_rmse_improvement"].median()),
        },
        "hma_period_summary": periods.to_dict(orient="records"),
        "principal_instrumental_skill": principal.to_dict(orient="records"),
        "interpretation_boundary": "Temperature-only operator sensitivity under blocked validation. The full-period branch is not accepted from its in-period behavior.",
    }
    (result / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (result / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (result / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (result / "FINAL_RESULT.md").write_text(
        "# Proxy9 Temperature-Only Blocked Validation v1\n\n"
        f"Status: **{summary['status']}**. Judge the model change from the blocked-crossfit branch, not the full-period fit.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
