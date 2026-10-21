"""Extend the lower instrument-error scale using only 1901-1980."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io


PRODUCTS = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}
BLOCKS = ((1901, 1920), (1921, 1940), (1941, 1960), (1961, 1980))
MULTIPLIERS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


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
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_instrument_anchor_nested_scale_v2"
    output.mkdir(parents=True, exist_ok=True)
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    paths = {
        "pilot_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_proxy9_instrument_anchor_loo_v1/run_experiment.py",
        "v1_selection": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_instrument_anchor_nested_scale_v1/FROZEN_SELECTION.json",
        "fold5_operators": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "baseline_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py",
        "evaluation_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_jja_temperature_network_repair_v1/run_experiment.py",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archived_regem": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
    }
    v1 = json.loads(paths["v1_selection"].read_text(encoding="utf-8"))
    if v1["selected_multiplier"] != 0.125 or not v1["outer_test_not_read_for_selection"]:
        raise RuntimeError("v1-boundary-precondition-failed")
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    pilot = load_module("nested_scale_v2_pilot", paths["pilot_driver"])
    base = load_module("nested_scale_v2_base", paths["baseline_driver"])
    evaluation = load_module("nested_scale_v2_eval", paths["evaluation_driver"])
    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    proxy_error_variance = np.var(residuals, axis=1, ddof=1)
    with h5py.File(paths["fold5_operators"], "r") as handle:
        blocked_hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
    fold5_hxb = proxy["hxb"] + (blocked_hxb[:, 6, :] - blocked_hxb[:, 0, :])

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
    joint_hxb = np.vstack([fold5_hxb, (regional_weights @ prior).reshape(1, -1)])

    product_series: dict[str, np.ndarray] = {}
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        for label, variable in PRODUCTS.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, (1961, 1990))
            product_series[label] = evaluation.hma_mean(field, grid_lat, mask2)

    cells = []
    for heldout in PRODUCTS:
        training_names = [name for name in PRODUCTS if name != heldout]
        training_mean = np.nanmean(np.stack([product_series[name] for name in training_names]), axis=0)
        for block_index, (start, end) in enumerate(BLOCKS, start=1):
            test = (years >= start) & (years <= end)
            variance_training = (years >= 1901) & (years <= 1980) & ~test
            valid_variance = variance_training & np.isfinite(training_mean) & np.isfinite(product_series[heldout])
            base_variance = float(np.mean((training_mean[valid_variance] - product_series[heldout][valid_variance]) ** 2))
            instrument_observations = np.full_like(training_mean, np.nan)
            instrument_observations[test] = training_mean[test]
            joint_observations = np.vstack([proxy["observations"], instrument_observations[None, :]])
            for multiplier in MULTIPLIERS:
                result = pilot.run_explicit_enkf(
                    prior, samples, joint_hxb, joint_observations,
                    np.r_[proxy_error_variance, base_variance * multiplier],
                    target_weights, regional_weights,
                )
                analysis = np.asarray(result["regional_analysis"])
                valid_test = test & np.isfinite(analysis) & np.isfinite(product_series[heldout])
                centered_analysis = analysis[valid_test] - np.mean(analysis[valid_test])
                centered_heldout = product_series[heldout][valid_test] - np.mean(product_series[heldout][valid_test])
                cells.append({
                    "heldout_product": heldout,
                    "block": block_index,
                    "period": f"{start}-{end}",
                    "multiplier": multiplier,
                    "base_variance_c2": base_variance,
                    "n_test_years": int(valid_test.sum()),
                    "squared_error_sum": float(np.sum((centered_analysis - centered_heldout) ** 2)),
                    "correlation": float(np.corrcoef(centered_analysis, centered_heldout)[0, 1]),
                    "enkf_closure_max_abs_c": float(result["state_increment_closure_max_abs_c"]),
                })
    cell_table = pd.DataFrame(cells)
    cell_table.to_csv(output / "inner_product_time_cells.csv", index=False)
    rows = []
    for multiplier in MULTIPLIERS:
        selected = cell_table.loc[cell_table.multiplier.eq(multiplier)]
        total_years = int(selected.n_test_years.sum())
        rows.append({
            "multiplier": multiplier,
            "pooled_centered_rmse_c": float(np.sqrt(selected.squared_error_sum.sum() / total_years)),
            "mean_cell_correlation": float(selected.correlation.mean()),
            "max_enkf_closure_abs_c": float(selected.enkf_closure_max_abs_c.max()),
            "validation_cells": int(len(selected)),
            "validation_year_values": total_years,
        })
    metrics = pd.DataFrame(rows).sort_values(["pooled_centered_rmse_c", "mean_cell_correlation"], ascending=[True, False]).reset_index(drop=True)
    selected_multiplier = float(metrics.loc[0, "multiplier"])
    selected_at_boundary = selected_multiplier in (min(MULTIPLIERS), max(MULTIPLIERS))
    metrics.to_csv(output / "inner_scale_metrics.csv", index=False)
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    frozen = {
        "experiment_id": "climate_reconstruction_proxy9_instrument_anchor_nested_scale_v2",
        "selection_date": "2026-08-13",
        "selection_years": [1901, 1980],
        "outer_test_not_read_for_selection": True,
        "candidate_multipliers": list(MULTIPLIERS),
        "selected_multiplier": selected_multiplier,
        "selected_at_candidate_boundary": selected_at_boundary,
        "selection_rule": "minimum pooled centered RMSE over 16 product-by-time cells; maximum mean cell correlation tie-breaker",
        "methods": metrics.to_dict(orient="records"),
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
