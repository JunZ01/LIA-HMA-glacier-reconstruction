"""Test spatial propagation into completely unassimilated HMA sectors."""

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
SECTORS = (
    ("west_69_77E", 69.0, 77.0),
    ("central_west_79_87E", 79.0, 87.0),
    ("central_east_89_95E", 89.0, 95.0),
    ("east_97_103E", 97.0, 103.0),
)
REGIONAL_SCALE = 0.0625
SPATIAL_SHRINKAGE = 0.25
SPATIAL_SCALE = 0.00390625


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


def temporal_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.full(a.shape[1], np.nan)
    for index in range(a.shape[1]):
        if np.std(a[:, index]) > 0 and np.std(b[:, index]) > 0:
            result[index] = np.corrcoef(a[:, index], b[:, index])[0, 1]
    return result


def spatial_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.full(a.shape[0], np.nan)
    for index in range(a.shape[0]):
        av = a[index] - np.mean(a[index])
        bv = b[index] - np.mean(b[index])
        if np.std(av) > 0 and np.std(bv) > 0:
            result[index] = np.corrcoef(av, bv)[0, 1]
    return result


def trend_errors(a: np.ndarray, b: np.ndarray, years: np.ndarray) -> np.ndarray:
    return np.abs(
        100.0 * np.polyfit(years, a, 1)[0]
        - 100.0 * np.polyfit(years, b, 1)[0]
    )


def score(branch: str, analysis: np.ndarray, heldout: np.ndarray, years: np.ndarray) -> dict[str, float | str | int]:
    difference = analysis - heldout
    temporal = temporal_correlations(analysis, heldout)
    spatial = spatial_correlations(analysis, heldout)
    trends = trend_errors(analysis, heldout, years)
    return {
        "branch": branch,
        "grid_year_squared_error_sum": float(np.sum(difference ** 2)),
        "grid_year_count": int(difference.size),
        "rmse_c": float(np.sqrt(np.mean(difference ** 2))),
        "median_grid_temporal_correlation": float(np.nanmedian(temporal)),
        "median_annual_spatial_correlation": float(np.nanmedian(spatial)),
        "median_abs_grid_trend_error_c_per_century": float(np.nanmedian(trends)),
    }


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_sector_holdout_v1"
    output.mkdir(parents=True, exist_ok=True)
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    helper_path = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_spatial_anchor_nested_v1/select_inner_model.py"
    paths = {
        "helper": helper_path,
        "regional_selection": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_instrument_anchor_nested_scale_v2/FROZEN_SELECTION.json",
        "spatial_selection": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_anchor_nested_v2/FROZEN_SELECTION.json",
        "spatial_audit": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_error_audit_v1/spatial_error_audit_arrays.npz",
        "fold5_operators": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "baseline_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py",
        "evaluation_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_jja_temperature_network_repair_v1/run_experiment.py",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archived_regem": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
    }
    regional_selection = json.loads(paths["regional_selection"].read_text(encoding="utf-8"))
    spatial_selection = json.loads(paths["spatial_selection"].read_text(encoding="utf-8"))
    if regional_selection["selected_multiplier"] != REGIONAL_SCALE:
        raise RuntimeError("regional-scale-precondition-failed")
    if spatial_selection["selected_shrinkage"] != SPATIAL_SHRINKAGE or spatial_selection["selected_scale"] != SPATIAL_SCALE:
        raise RuntimeError("spatial-model-precondition-failed")
    if not spatial_selection["selected_scale_at_boundary"]:
        raise RuntimeError("spatial-pilot-no-longer-boundary-labeled")
    hashes_before = {name: sha256(path) for name, path in paths.items()}

    helper = load_module("sector_holdout_helper", paths["helper"])
    base = load_module("sector_holdout_base", paths["baseline_driver"])
    evaluation = load_module("sector_holdout_eval", paths["evaluation_driver"])
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
    proxy_prepared = helper.prepare_covariances(prior, samples, fold5_hxb)

    with np.load(paths["spatial_audit"]) as audit:
        common_state = np.asarray(audit["common_state_mask"], dtype=bool)
        hma_state = np.asarray(audit["hma_state_mask"], dtype=bool)
        state_lat = np.asarray(audit["state_lat"], dtype=float)
        state_lon = np.asarray(audit["state_lon"], dtype=float)
    common_indices = np.flatnonzero(common_state & hma_state)
    common_lon = state_lon[common_indices]
    common_lat = state_lat[common_indices]
    if len(common_indices) != 86:
        raise RuntimeError(f"unexpected-common-count:{len(common_indices)}")

    grid_lat, grid_lon, _, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        product_fields = {}
        for label, variable in PRODUCTS.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            mapped = evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon)
            product_fields[label] = flatten_state(
                evaluation.anomaly(mapped, years, (1961, 1990))
            )[:, common_indices]

    sector_rows = []
    sector_masks = {}
    for sector_name, lon_min, lon_max in SECTORS:
        mask = (common_lon >= lon_min) & (common_lon <= lon_max)
        if mask.sum() < 10:
            raise RuntimeError(f"sector-too-small:{sector_name}:{mask.sum()}")
        sector_masks[sector_name] = mask
        sector_rows.append({
            "sector": sector_name,
            "longitude_min": lon_min,
            "longitude_max": lon_max,
            "state_count": int(mask.sum()),
            "latitude_min": float(common_lat[mask].min()),
            "latitude_max": float(common_lat[mask].max()),
        })
    if np.any(np.sum(np.stack(list(sector_masks.values())), axis=0) != 1):
        raise RuntimeError("sector-partition-not-exclusive-complete")
    pd.DataFrame(sector_rows).to_csv(output / "spatial_sector_definition.csv", index=False)

    rows = []
    max_closure = 0.0
    for heldout_product in PRODUCTS:
        training_names = [name for name in PRODUCTS if name != heldout_product]
        training_mean = np.nanmean(np.stack([product_fields[name] for name in training_names]), axis=0)
        heldout_field = product_fields[heldout_product]
        for block_index, (start, end) in enumerate(BLOCKS, start=1):
            test = (years >= start) & (years <= end)
            train = (years >= 1901) & (years <= 1980) & ~test
            test_years = years[test]

            proxy_observations = proxy["observations"][:, test]
            proxy_analysis, proxy_closure = helper.run_years(
                proxy_prepared,
                proxy_observations,
                np.diag(proxy_variance),
            )
            max_closure = max(max_closure, proxy_closure)

            for sector_name, _, _ in SECTORS:
                validation_mask = sector_masks[sector_name]
                observed_mask = ~validation_mask
                observed_indices = common_indices[observed_mask]
                validation_indices = common_indices[validation_mask]
                observed_training_mean = training_mean[:, observed_mask]
                heldout_validation = heldout_field[test][:, validation_mask]

                residual_train = observed_training_mean[train] - heldout_field[train][:, observed_mask]
                centered = residual_train - np.mean(residual_train, axis=0, keepdims=True)
                sample_covariance = centered.T @ centered / (centered.shape[0] - 1)
                diagonal = np.diag(np.diag(sample_covariance))
                spatial_covariance = SPATIAL_SCALE * (
                    (1.0 - SPATIAL_SHRINKAGE) * sample_covariance
                    + SPATIAL_SHRINKAGE * diagonal
                )
                if np.linalg.eigvalsh(spatial_covariance)[0] <= 0:
                    raise RuntimeError(f"spatial-covariance-nonpositive:{heldout_product}:{start}:{sector_name}")

                spatial_hxb = np.vstack([fold5_hxb, prior[observed_indices]])
                spatial_prepared = helper.prepare_covariances(prior, samples, spatial_hxb)
                spatial_observations = np.vstack([proxy_observations, observed_training_mean[test].T])
                spatial_error = scipy.linalg.block_diag(np.diag(proxy_variance), spatial_covariance)
                spatial_analysis, spatial_closure = helper.run_years(
                    spatial_prepared, spatial_observations, spatial_error
                )
                max_closure = max(max_closure, spatial_closure)

                weights = np.cos(np.deg2rad(common_lat[observed_mask]))
                weights /= weights.sum()
                regional_observation = observed_training_mean[test] @ weights
                heldout_observed_mean = heldout_field[:, observed_mask] @ weights
                training_observed_mean = training_mean[:, observed_mask] @ weights
                valid_train = train & np.isfinite(training_observed_mean) & np.isfinite(heldout_observed_mean)
                regional_base_variance = float(np.mean(
                    (training_observed_mean[valid_train] - heldout_observed_mean[valid_train]) ** 2
                ))
                regional_hxb = np.vstack([fold5_hxb, (weights @ prior[observed_indices]).reshape(1, -1)])
                regional_prepared = helper.prepare_covariances(prior, samples, regional_hxb)
                regional_observations = np.vstack([proxy_observations, regional_observation[None, :]])
                regional_error = scipy.linalg.block_diag(
                    np.diag(proxy_variance),
                    np.asarray([[REGIONAL_SCALE * regional_base_variance]]),
                )
                regional_analysis, regional_closure = helper.run_years(
                    regional_prepared, regional_observations, regional_error
                )
                max_closure = max(max_closure, regional_closure)

                branch_values = {
                    "proxy_only": proxy_analysis[validation_indices].T,
                    "observed_sector_regional_anchor": regional_analysis[validation_indices].T,
                    "spatial_anchor_boundary_model": spatial_analysis[validation_indices].T,
                    "three_product_mean_unassimilated_benchmark": training_mean[test][:, validation_mask],
                }
                for branch, values in branch_values.items():
                    row = {
                        "heldout_product": heldout_product,
                        "block": block_index,
                        "period": f"{start}-{end}",
                        "heldout_sector": sector_name,
                        "assimilated_state_count": int(observed_mask.sum()),
                        "validation_state_count": int(validation_mask.sum()),
                        "enkf_closure_max_abs_c": 0.0 if branch.startswith("three_product") else max(
                            proxy_closure,
                            regional_closure if branch.startswith("observed_sector") else 0.0,
                            spatial_closure if branch.startswith("spatial") else 0.0,
                        ),
                    }
                    row.update(score(branch, values, heldout_validation, test_years))
                    rows.append(row)

    cells = pd.DataFrame(rows)
    cells.to_csv(output / "product_time_sector_cells.csv", index=False)
    aggregate_rows = []
    for branch in cells.branch.unique():
        selected = cells.loc[cells.branch.eq(branch)]
        aggregate_rows.append({
            "branch": branch,
            "pooled_unassimilated_grid_year_rmse_c": float(np.sqrt(
                selected.grid_year_squared_error_sum.sum() / selected.grid_year_count.sum()
            )),
            "mean_cell_median_grid_temporal_correlation": float(selected.median_grid_temporal_correlation.mean()),
            "mean_cell_median_annual_spatial_correlation": float(selected.median_annual_spatial_correlation.mean()),
            "mean_cell_median_abs_grid_trend_error_c_per_century": float(selected.median_abs_grid_trend_error_c_per_century.mean()),
            "max_enkf_closure_abs_c": float(selected.enkf_closure_max_abs_c.max()),
            "validation_cells": int(len(selected)),
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values("pooled_unassimilated_grid_year_rmse_c").reset_index(drop=True)
    aggregate.to_csv(output / "sector_holdout_aggregate.csv", index=False)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    proxy_row = aggregate.loc[aggregate.branch.eq("proxy_only")].iloc[0]
    regional_row = aggregate.loc[aggregate.branch.eq("observed_sector_regional_anchor")].iloc[0]
    spatial_row = aggregate.loc[aggregate.branch.eq("spatial_anchor_boundary_model")].iloc[0]
    spatial_advances = bool(
        spatial_row.pooled_unassimilated_grid_year_rmse_c < proxy_row.pooled_unassimilated_grid_year_rmse_c
        and spatial_row.pooled_unassimilated_grid_year_rmse_c < regional_row.pooled_unassimilated_grid_year_rmse_c
        and spatial_row.mean_cell_median_annual_spatial_correlation > proxy_row.mean_cell_median_annual_spatial_correlation
    )
    summary = {
        "experiment_id": "climate_reconstruction_spatial_sector_holdout_v1",
        "date": "2026-08-13",
        "validation_years": [1901, 1980],
        "outer_1981_2000_not_read": True,
        "heldout_products": list(PRODUCTS),
        "heldout_time_blocks": [list(block) for block in BLOCKS],
        "heldout_spatial_sectors": [item[0] for item in SECTORS],
        "validation_cell_count_per_branch": 64,
        "regional_scale": REGIONAL_SCALE,
        "spatial_shrinkage": SPATIAL_SHRINKAGE,
        "spatial_scale_boundary_candidate": SPATIAL_SCALE,
        "max_enkf_closure_abs_c": max_closure,
        "source_hashes_unchanged": pure,
        "spatial_candidate_advances": spatial_advances,
        "aggregate": aggregate.to_dict(orient="records"),
        "decision_boundary": "The spatial boundary model advances only if it improves completely unassimilated-sector RMSE over both proxy-only and visible-sector regional anchoring, and improves spatial correlation over proxy-only. No 1981-2000 field was read.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if max_closure > 1e-9:
        raise RuntimeError(f"enkf-closure-failed:{max_closure}")
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
