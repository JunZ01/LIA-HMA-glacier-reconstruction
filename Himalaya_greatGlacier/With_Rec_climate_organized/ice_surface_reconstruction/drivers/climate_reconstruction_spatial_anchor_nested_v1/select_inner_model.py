"""Select spatial covariance shrinkage and scale using only 1901-1980."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg


PRODUCTS = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}
BLOCKS = ((1901, 1920), (1921, 1940), (1941, 1960), (1961, 1980))
SHRINKAGES = (0.1, 0.25, 0.5, 0.75, 1.0)
SCALES = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)


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


def flatten_state(field: np.ndarray) -> np.ndarray:
    return np.asarray(field).transpose(0, 2, 1).reshape(field.shape[0], -1)


def prepare_covariances(prior: np.ndarray, samples: np.ndarray, hxb: np.ndarray):
    background = prior[:, samples]
    projected = hxb[:, samples]
    background_mean = np.mean(background, axis=1)
    background_prime = background - background_mean[:, None]
    projected_mean = np.mean(projected, axis=1)
    projected_prime = projected - projected_mean[:, None]
    denominator = background.shape[1] - 1
    return (
        background_mean,
        projected_mean,
        background_prime @ projected_prime.T / denominator,
        projected_prime @ projected_prime.T / denominator,
    )


def run_years(prepared, observations: np.ndarray, error_covariance: np.ndarray):
    background_mean, projected_mean, pbht, hpbh = prepared
    analysis = np.repeat(background_mean[:, None], observations.shape[1], axis=1)
    closure = 0.0
    groups: dict[tuple[int, ...], list[int]] = {}
    for year_index in range(observations.shape[1]):
        active = tuple(np.flatnonzero(np.isfinite(observations[:, year_index])))
        groups.setdefault(active, []).append(year_index)
    for active_tuple, year_list in groups.items():
        if not active_tuple:
            continue
        active = np.asarray(active_tuple, dtype=int)
        year_indices = np.asarray(year_list, dtype=int)
        covariance = hpbh[np.ix_(active, active)] + error_covariance[np.ix_(active, active)]
        innovation = observations[np.ix_(active, year_indices)] - projected_mean[active, None]
        factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
        weights = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
        increment = pbht[:, active] @ weights
        analysis[:, year_indices] += increment
        explicit = np.sum(pbht[:, active, None] * weights[None, :, :], axis=1)
        closure = max(closure, float(np.max(np.abs(increment - explicit))))
    return analysis, closure


def temporal_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.full(a.shape[1], np.nan)
    for index in range(a.shape[1]):
        valid = np.isfinite(a[:, index]) & np.isfinite(b[:, index])
        if valid.sum() >= 3 and np.std(a[valid, index]) > 0 and np.std(b[valid, index]) > 0:
            result[index] = np.corrcoef(a[valid, index], b[valid, index])[0, 1]
    return result


def spatial_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.full(a.shape[0], np.nan)
    for index in range(a.shape[0]):
        valid = np.isfinite(a[index]) & np.isfinite(b[index])
        if valid.sum() >= 3:
            av = a[index, valid] - np.mean(a[index, valid])
            bv = b[index, valid] - np.mean(b[index, valid])
            if np.std(av) > 0 and np.std(bv) > 0:
                result[index] = np.corrcoef(av, bv)[0, 1]
    return result


def trend_error(a: np.ndarray, b: np.ndarray, years: np.ndarray) -> np.ndarray:
    result = np.full(a.shape[1], np.nan)
    for index in range(a.shape[1]):
        valid = np.isfinite(a[:, index]) & np.isfinite(b[:, index])
        if valid.sum() >= 3:
            ta = 100.0 * np.polyfit(years[valid], a[valid, index], 1)[0]
            tb = 100.0 * np.polyfit(years[valid], b[valid, index], 1)[0]
            result[index] = abs(ta - tb)
    return result


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_anchor_nested_v1"
    output.mkdir(parents=True, exist_ok=True)
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    paths = {
        "spatial_audit": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_error_audit_v1/spatial_error_audit_arrays.npz",
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
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("spatial_anchor_inner_base", paths["baseline_driver"])
    evaluation = load_module("spatial_anchor_inner_eval", paths["evaluation_driver"])
    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    proxy_variance = np.var(residuals, axis=1, ddof=1)
    with h5py.File(paths["fold5_operators"], "r") as handle:
        blocked_hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
    fold5_hxb = proxy["hxb"] + (blocked_hxb[:, 6, :] - blocked_hxb[:, 0, :])
    with np.load(paths["spatial_audit"]) as audit:
        common_state = np.asarray(audit["common_state_mask"], dtype=bool)
        hma_state = np.asarray(audit["hma_state_mask"], dtype=bool)
    observed_state_indices = np.flatnonzero(common_state & hma_state)
    if len(observed_state_indices) != 86:
        raise RuntimeError(f"unexpected-observed-state-count:{len(observed_state_indices)}")
    joint_hxb = np.vstack([fold5_hxb, prior[observed_state_indices]])
    prepared = prepare_covariances(prior, samples, joint_hxb)

    grid_lat, grid_lon, _, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        product_fields = {}
        for label, variable in PRODUCTS.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            mapped = evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon)
            product_fields[label] = flatten_state(evaluation.anomaly(mapped, years, (1961, 1990)))[:, observed_state_indices]

    rows = []
    for heldout in PRODUCTS:
        training_names = [name for name in PRODUCTS if name != heldout]
        training_mean = np.nanmean(np.stack([product_fields[name] for name in training_names]), axis=0)
        for block_index, (start, end) in enumerate(BLOCKS, start=1):
            test = (years >= start) & (years <= end)
            train = (years >= 1901) & (years <= 1980) & ~test
            residual_train = training_mean[train] - product_fields[heldout][train]
            if not np.all(np.isfinite(residual_train)):
                raise RuntimeError(f"nonfinite-training-residual:{heldout}:{start}")
            centered = residual_train - np.mean(residual_train, axis=0, keepdims=True)
            sample_covariance = centered.T @ centered / (centered.shape[0] - 1)
            diagonal = np.diag(np.diag(sample_covariance))
            selected_years = years[test]
            observations = np.vstack([proxy["observations"][:, test], training_mean[test].T])
            heldout_values = product_fields[heldout][test]
            for shrinkage in SHRINKAGES:
                covariance_shape = (1.0 - shrinkage) * sample_covariance + shrinkage * diagonal
                minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_shape)[0])
                if minimum_eigenvalue <= 0:
                    raise RuntimeError(f"nonpositive-shrunk-covariance:{heldout}:{start}:{shrinkage}:{minimum_eigenvalue}")
                for scale in SCALES:
                    error_covariance = scipy.linalg.block_diag(np.diag(proxy_variance), scale * covariance_shape)
                    analysis, closure = run_years(prepared, observations, error_covariance)
                    analysis_values = analysis[observed_state_indices].T
                    difference = analysis_values - heldout_values
                    temporal = temporal_correlations(analysis_values, heldout_values)
                    spatial = spatial_correlations(analysis_values, heldout_values)
                    trends = trend_error(analysis_values, heldout_values, selected_years)
                    rows.append({
                        "heldout_product": heldout,
                        "block": block_index,
                        "period": f"{start}-{end}",
                        "shrinkage": shrinkage,
                        "scale": scale,
                        "minimum_covariance_eigenvalue_c2": minimum_eigenvalue * scale,
                        "grid_year_squared_error_sum": float(np.sum(difference ** 2)),
                        "grid_year_count": int(difference.size),
                        "median_grid_temporal_correlation": float(np.nanmedian(temporal)),
                        "median_annual_spatial_correlation": float(np.nanmedian(spatial)),
                        "median_abs_grid_trend_error_c_per_century": float(np.nanmedian(trends)),
                        "enkf_closure_max_abs_c": closure,
                    })
    cells = pd.DataFrame(rows)
    cells.to_csv(output / "inner_product_time_model_cells.csv", index=False)
    aggregate_rows = []
    for shrinkage in SHRINKAGES:
        for scale in SCALES:
            selected = cells.loc[cells.shrinkage.eq(shrinkage) & cells.scale.eq(scale)]
            aggregate_rows.append({
                "shrinkage": shrinkage,
                "scale": scale,
                "pooled_grid_year_rmse_c": float(np.sqrt(selected.grid_year_squared_error_sum.sum() / selected.grid_year_count.sum())),
                "mean_cell_median_grid_temporal_correlation": float(selected.median_grid_temporal_correlation.mean()),
                "mean_cell_median_annual_spatial_correlation": float(selected.median_annual_spatial_correlation.mean()),
                "mean_cell_median_abs_grid_trend_error_c_per_century": float(selected.median_abs_grid_trend_error_c_per_century.mean()),
                "max_enkf_closure_abs_c": float(selected.enkf_closure_max_abs_c.max()),
                "validation_cells": int(len(selected)),
            })
    metrics = pd.DataFrame(aggregate_rows).sort_values(
        ["pooled_grid_year_rmse_c", "mean_cell_median_grid_temporal_correlation"],
        ascending=[True, False],
    ).reset_index(drop=True)
    metrics.to_csv(output / "inner_model_metrics.csv", index=False)
    selected_shrinkage = float(metrics.loc[0, "shrinkage"])
    selected_scale = float(metrics.loc[0, "scale"])
    scale_at_boundary = selected_scale in (min(SCALES), max(SCALES))
    shrinkage_at_boundary = selected_shrinkage in (min(SHRINKAGES), max(SHRINKAGES))
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    frozen = {
        "experiment_id": "climate_reconstruction_spatial_anchor_nested_v1",
        "selection_date": "2026-08-13",
        "selection_years": [1901, 1980],
        "outer_test_not_read_for_selection": True,
        "observed_hma_state_count": int(len(observed_state_indices)),
        "candidate_shrinkages": list(SHRINKAGES),
        "candidate_scales": list(SCALES),
        "selected_shrinkage": selected_shrinkage,
        "selected_scale": selected_scale,
        "selected_scale_at_boundary": scale_at_boundary,
        "selected_shrinkage_at_boundary": shrinkage_at_boundary,
        "selection_rule": "minimum pooled grid-year RMSE over 16 product-by-time cells; maximum mean cell median grid temporal correlation tie-breaker",
        "models": metrics.to_dict(orient="records"),
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
