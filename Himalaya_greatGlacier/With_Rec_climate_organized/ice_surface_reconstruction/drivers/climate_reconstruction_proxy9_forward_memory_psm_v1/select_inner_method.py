"""Select temporal memory using only forward forecasts ending by 1980."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io


METHODS = ("expanding_ols", "memory_10y", "memory_20y", "memory_40y")
TEST_BLOCKS = ((1921, 1940), (1941, 1960), (1961, 1980))


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


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_forward_memory_psm_v1"
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    paths = {
        "operators": output / "forward_memory_operators.mat",
        "blocked_cv": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "baseline_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py",
        "evaluation_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_jja_temperature_network_repair_v1/run_experiment.py",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archived_regem": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("forward_memory_inner_base", paths["baseline_driver"])
    evaluation = load_module("forward_memory_inner_eval", paths["evaluation_driver"])
    with h5py.File(paths["operators"], "r") as handle:
        hxb = np.asarray(handle["hxb"], dtype=float).transpose(3, 2, 1, 0)
        eligible = np.asarray(handle["eligible"], dtype=bool).transpose(2, 1, 0)
        effective_sample_size = np.asarray(handle["effective_sample_size"], dtype=float).transpose(2, 1, 0)
    with h5py.File(paths["blocked_cv"], "r") as handle:
        blocked_hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
    fresh_baseline = blocked_hxb[:, 0, :]

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

    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        observations = []
        for variable in ("tas_obs_bea_05deg", "tas_obs_cru_tmp_05deg", "tas_obs_GIST_05deg", "tas_obs_had5_05deg"):
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, (1961, 1990))
            observations.append(evaluation.hma_mean(field, grid_lat, mask2))
    observation_mean = np.mean(np.stack(observations), axis=0)

    stitched = {method: np.full(60, np.nan) for method in METHODS}
    stitched_obs = np.full(60, np.nan)
    fallback_rows = []
    closure = 0.0
    cursor = 0
    for branch, (start, end) in enumerate(TEST_BLOCKS):
        expanding = hxb[:, 0, branch, :].copy()
        expanding_missing = ~eligible[:, 0, branch]
        expanding[expanding_missing] = fresh_baseline[expanding_missing]
        selected = (years >= start) & (years <= end)
        observed = observation_mean[selected]
        stitched_obs[cursor:cursor + selected.sum()] = observed - observed.mean()
        for method_index, method in enumerate(METHODS):
            candidate = hxb[:, method_index, branch, :].copy()
            method_missing = ~eligible[:, method_index, branch]
            candidate[method_missing] = expanding[method_missing]
            item = {key: np.asarray(value).copy() for key, value in proxy.items()}
            item["hxb"] = proxy["hxb"] + (candidate - fresh_baseline)
            result = base._run_branch(prior, samples, item, residuals, target_weights, regional_weights)
            closure = max(closure, *(float(result[key]) for key in (
                "state_increment_closure_max_abs_c",
                "target_increment_closure_max_abs_c",
                "regional_increment_closure_max_abs_c",
            )))
            regional = np.asarray(result["regional_analysis"])[selected]
            stitched[method][cursor:cursor + selected.sum()] = regional - regional.mean()
            fallback_rows.append({
                "forecast": branch + 1,
                "period": f"{start}-{end}",
                "method": method,
                "method_to_expanding_records": int(method_missing.sum()) if method_index > 0 else 0,
                "expanding_to_archived_records": int(expanding_missing.sum()),
                "median_effective_sample_size": float(np.nanmedian(effective_sample_size[:, method_index, branch])),
            })
        cursor += selected.sum()

    if closure > 1e-10 or np.any(~np.isfinite(stitched_obs)):
        raise RuntimeError(f"inner-forward-gate-failed:{closure}")
    rows = []
    for method in METHODS:
        values = stitched[method]
        rows.append({
            "method": method,
            "centered_rmse_c": float(np.sqrt(np.mean((values - stitched_obs) ** 2))),
            "correlation": float(np.corrcoef(values, stitched_obs)[0, 1]),
        })
    metrics = pd.DataFrame(rows).sort_values(["centered_rmse_c", "correlation"], ascending=[True, False]).reset_index(drop=True)
    selected_method = str(metrics.loc[0, "method"])
    metrics.to_csv(output / "inner_forward_selection_metrics.csv", index=False)
    pd.DataFrame(fallback_rows).to_csv(output / "inner_forward_fallback_summary.csv", index=False)
    pd.DataFrame({"year": np.arange(1921, 1981), "four_product_mean_centered_by_forecast": stitched_obs, **{f"{method}_centered_by_forecast": stitched[method] for method in METHODS}}).to_csv(output / "inner_forward_hma_series.csv", index=False)
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    frozen = {
        "experiment_id": "climate_reconstruction_proxy9_forward_memory_psm_v1",
        "selection_date": "2026-08-13",
        "selection_forecast_years": [1921, 1980],
        "selection_training_data_end_year": 1960,
        "outer_test_not_read_for_selection": True,
        "selected_method": selected_method,
        "selection_rule": "minimum stitched forecast-centered RMSE; maximum correlation tie-breaker",
        "methods": metrics.to_dict(orient="records"),
        "enkf_closure_max_abs_c": closure,
        "source_hashes_unchanged": pure,
    }
    (output / "FROZEN_SELECTION.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (output / "inner_input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "inner_input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
