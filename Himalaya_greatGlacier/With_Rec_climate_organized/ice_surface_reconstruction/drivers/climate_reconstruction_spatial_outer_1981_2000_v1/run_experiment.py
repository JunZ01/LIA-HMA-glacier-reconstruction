"""Evaluate the frozen HMA spatial temperature anchor on 1981-2000."""

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
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRODUCTS = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}
CALIBRATION_YEARS = np.arange(1901, 1981)
OUTER_YEARS = np.arange(1981, 2001)
READ_YEARS = np.arange(1901, 2001)
READ_SLICE = slice(901, 1001)
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


def temporal_center(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values, axis=0, keepdims=True)


def temporal_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.full(a.shape[1], np.nan)
    for index in range(a.shape[1]):
        av = a[:, index] - np.mean(a[:, index])
        bv = b[:, index] - np.mean(b[:, index])
        if np.std(av) > 0 and np.std(bv) > 0:
            result[index] = np.corrcoef(av, bv)[0, 1]
    return result


def spatial_anomaly_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.full(a.shape[0], np.nan)
    for index in range(a.shape[0]):
        av = a[index] - np.mean(a[index])
        bv = b[index] - np.mean(b[index])
        if np.std(av) > 0 and np.std(bv) > 0:
            result[index] = np.corrcoef(av, bv)[0, 1]
    return result


def grid_trends(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    return 100.0 * np.polyfit(years, values, 1)[0]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return values @ weights


def score(branch: str, analysis: np.ndarray, heldout: np.ndarray) -> tuple[dict, pd.DataFrame]:
    analysis_centered = temporal_center(analysis)
    heldout_centered = temporal_center(heldout)
    difference = analysis_centered - heldout_centered
    temporal = temporal_correlations(analysis, heldout)
    spatial = spatial_anomaly_correlations(analysis, heldout)
    trend_analysis = grid_trends(analysis, OUTER_YEARS)
    trend_heldout = grid_trends(heldout, OUTER_YEARS)
    metrics = {
        "branch": branch,
        "centered_grid_year_squared_error_sum": float(np.sum(difference ** 2)),
        "grid_year_count": int(difference.size),
        "centered_rmse_c": float(np.sqrt(np.mean(difference ** 2))),
        "median_grid_temporal_correlation": float(np.nanmedian(temporal)),
        "median_annual_spatial_anomaly_correlation": float(np.nanmedian(spatial)),
        "median_abs_grid_trend_error_c_per_century": float(np.nanmedian(np.abs(trend_analysis - trend_heldout))),
    }
    grid = pd.DataFrame({
        "branch": branch,
        "grid_centered_rmse_c": np.sqrt(np.mean(difference ** 2, axis=0)),
        "grid_temporal_correlation": temporal,
        "analysis_trend_c_per_century": trend_analysis,
        "heldout_trend_c_per_century": trend_heldout,
        "abs_trend_error_c_per_century": np.abs(trend_analysis - trend_heldout),
    })
    return metrics, grid


def plot_aggregate(aggregate: pd.DataFrame, output: Path) -> None:
    order = [
        "proxy_only",
        "regional_anchor",
        "spatial_anchor_frozen",
        "three_product_mean_benchmark",
    ]
    labels = {
        "proxy_only": "Proxy only",
        "regional_anchor": "Regional anchor",
        "spatial_anchor_frozen": "Spatial anchor",
        "three_product_mean_benchmark": "3-product benchmark",
    }
    colors = {
        "proxy_only": "#777777",
        "regional_anchor": "#2b6cb0",
        "spatial_anchor_frozen": "#b8322a",
        "three_product_mean_benchmark": "#2f855a",
    }
    table = aggregate.set_index("branch").loc[order]
    x = np.arange(len(order))
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), constrained_layout=True)
    axes[0].bar(x, table.pooled_centered_grid_year_rmse_c, color=[colors[name] for name in order])
    axes[0].set_ylabel("Centered RMSE (degC)")
    axes[0].set_title("1981-2000 field error")
    axes[1].bar(x, table.mean_product_median_grid_temporal_correlation, color=[colors[name] for name in order])
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Grid-cell temporal skill")
    axes[2].bar(x, table.mean_product_median_annual_spatial_anomaly_correlation, color=[colors[name] for name in order])
    axes[2].set_ylabel("Correlation")
    axes[2].set_title("Annual spatial-anomaly skill")
    for axis in axes:
        axis.set_xticks(x, [labels[name] for name in order], rotation=23, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "figure_outer_aggregate.png", dpi=220)
    plt.close(figure)


def plot_product_curves(series: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), sharex=True, sharey=True, constrained_layout=True)
    for axis, heldout in zip(axes.flat, PRODUCTS):
        subset = series.loc[series.heldout_product.eq(heldout)]
        for branch, color, width in (
            ("heldout", "#222222", 2.4),
            ("proxy_only", "#777777", 1.5),
            ("regional_anchor", "#2b6cb0", 1.8),
            ("spatial_anchor_frozen", "#b8322a", 2.0),
            ("three_product_mean_benchmark", "#2f855a", 1.7),
        ):
            values = subset.loc[subset.branch.eq(branch), "regional_mean_c"].to_numpy()
            values = values - values.mean()
            axis.plot(OUTER_YEARS, values, color=color, linewidth=width, label=branch.replace("_", " "))
        axis.axhline(0, color="#888888", linewidth=0.7)
        axis.grid(alpha=0.18)
        axis.set_title(f"Held out: {heldout}")
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].set_ylabel("Centered HMA mean (degC)")
    axes[1, 0].set_ylabel("Centered HMA mean (degC)")
    axes[0, 0].legend(frameon=False, fontsize=7.3, ncol=2)
    figure.savefig(output / "figure_outer_regional_curves.png", dpi=220)
    plt.close(figure)


def plot_grid_improvement(grid_metrics: pd.DataFrame, output: Path) -> None:
    proxy = grid_metrics.loc[grid_metrics.branch.eq("proxy_only")].copy()
    spatial = grid_metrics.loc[grid_metrics.branch.eq("spatial_anchor_frozen")].copy()
    keys = ["heldout_product", "state_index", "longitude", "latitude"]
    merged = proxy.merge(spatial, on=keys, suffixes=("_proxy", "_spatial"))
    merged["rmse_improvement_c"] = merged.grid_centered_rmse_c_proxy - merged.grid_centered_rmse_c_spatial
    summary = merged.groupby(["state_index", "longitude", "latitude"], as_index=False).rmse_improvement_c.mean()
    summary.to_csv(output / "grid_mean_spatial_rmse_improvement.csv", index=False)
    limit = float(np.max(np.abs(summary.rmse_improvement_c)))
    figure, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    scatter = axis.scatter(
        summary.longitude,
        summary.latitude,
        c=summary.rmse_improvement_c,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        s=72,
        edgecolor="#333333",
        linewidth=0.35,
    )
    axis.set_xlabel("Longitude (degE)")
    axis.set_ylabel("Latitude (degN)")
    axis.set_title("Spatial-anchor RMSE improvement over proxy-only")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("RMSE reduction (degC)")
    axis.grid(alpha=0.16)
    figure.savefig(output / "figure_outer_grid_rmse_improvement.png", dpi=220)
    plt.close(figure)


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_outer_1981_2000_v1"
    output.mkdir(parents=True, exist_ok=True)
    archive = Path("D:/Users/JUNZ/Desktop/\u6587\u4ef6\u6574\u7406/1 \u8bba\u6587/work1/compared_result_all")
    helper_path = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_spatial_anchor_nested_v1/select_inner_model.py"
    paths = {
        "helper": helper_path,
        "sector_v2_summary": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_sector_holdout_v2/summary.json",
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
    sector_v2 = json.loads(paths["sector_v2_summary"].read_text(encoding="utf-8"))
    regional_selection = json.loads(paths["regional_selection"].read_text(encoding="utf-8"))
    spatial_selection = json.loads(paths["spatial_selection"].read_text(encoding="utf-8"))
    if not sector_v2["spatial_candidate_advances"] or not sector_v2["outer_1981_2000_not_read"]:
        raise RuntimeError("sector-v2-advance-precondition-failed")
    if regional_selection["selected_multiplier"] != REGIONAL_SCALE:
        raise RuntimeError("regional-scale-precondition-failed")
    if spatial_selection["selected_shrinkage"] != SPATIAL_SHRINKAGE or spatial_selection["selected_scale"] != SPATIAL_SCALE:
        raise RuntimeError("spatial-model-precondition-failed")
    if not spatial_selection["selected_scale_at_boundary"]:
        raise RuntimeError("spatial-boundary-label-precondition-failed")
    hashes_before = {name: sha256(path) for name, path in paths.items()}

    helper = load_module("spatial_outer_helper", paths["helper"])
    base = load_module("spatial_outer_base", paths["baseline_driver"])
    evaluation = load_module("spatial_outer_eval", paths["evaluation_driver"])
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
    common_lat = state_lat[common_indices]
    common_lon = state_lon[common_indices]
    if len(common_indices) != 86:
        raise RuntimeError(f"unexpected-common-count:{len(common_indices)}")
    area_weights = np.cos(np.deg2rad(common_lat))
    area_weights /= area_weights.sum()

    grid_lat, grid_lon, _, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        product_fields = {}
        for label, variable in PRODUCTS.items():
            values = np.asarray(handle[variable][READ_SLICE], dtype=float)
            values[values < -9000] = np.nan
            mapped = evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon)
            product_fields[label] = flatten_state(mapped)[:, common_indices]
    if any(field.shape != (100, 86) for field in product_fields.values()):
        raise RuntimeError("instrument-read-shape-failed")
    if not all(np.all(np.isfinite(field)) for field in product_fields.values()):
        raise RuntimeError("nonfinite-common-instrument-field")
    calibration = READ_YEARS <= 1980
    outer = READ_YEARS >= 1981
    referenced = {
        name: field - np.mean(field[calibration], axis=0, keepdims=True)
        for name, field in product_fields.items()
    }
    proxy_observations = proxy["observations"][:, OUTER_YEARS - 1000]
    proxy_analysis, proxy_closure = helper.run_years(proxy_prepared, proxy_observations, np.diag(proxy_variance))
    max_closure = proxy_closure

    product_rows = []
    grid_tables = []
    regional_series_rows = []
    arrays = {}
    for heldout_product in PRODUCTS:
        training_names = [name for name in PRODUCTS if name != heldout_product]
        training_mean = np.mean(np.stack([referenced[name] for name in training_names]), axis=0)
        heldout_field = referenced[heldout_product]
        residual_calibration = training_mean[calibration] - heldout_field[calibration]
        centered_residual = residual_calibration - np.mean(residual_calibration, axis=0, keepdims=True)
        sample_covariance = centered_residual.T @ centered_residual / (centered_residual.shape[0] - 1)
        diagonal = np.diag(np.diag(sample_covariance))
        spatial_covariance = SPATIAL_SCALE * (
            (1.0 - SPATIAL_SHRINKAGE) * sample_covariance
            + SPATIAL_SHRINKAGE * diagonal
        )
        minimum_eigenvalue = float(np.linalg.eigvalsh(spatial_covariance)[0])
        if minimum_eigenvalue <= 0:
            raise RuntimeError(f"spatial-covariance-nonpositive:{heldout_product}:{minimum_eigenvalue}")

        spatial_hxb = np.vstack([fold5_hxb, prior[common_indices]])
        spatial_prepared = helper.prepare_covariances(prior, samples, spatial_hxb)
        spatial_observations = np.vstack([proxy_observations, training_mean[outer].T])
        spatial_error = scipy.linalg.block_diag(np.diag(proxy_variance), spatial_covariance)
        spatial_analysis, spatial_closure = helper.run_years(spatial_prepared, spatial_observations, spatial_error)
        max_closure = max(max_closure, spatial_closure)

        regional_training = weighted_mean(training_mean, area_weights)
        regional_heldout = weighted_mean(heldout_field, area_weights)
        regional_base_variance = float(np.mean((regional_training[calibration] - regional_heldout[calibration]) ** 2))
        regional_hxb = np.vstack([fold5_hxb, (area_weights @ prior[common_indices]).reshape(1, -1)])
        regional_prepared = helper.prepare_covariances(prior, samples, regional_hxb)
        regional_observations = np.vstack([proxy_observations, regional_training[outer][None, :]])
        regional_error = scipy.linalg.block_diag(
            np.diag(proxy_variance), np.asarray([[REGIONAL_SCALE * regional_base_variance]])
        )
        regional_analysis, regional_closure = helper.run_years(regional_prepared, regional_observations, regional_error)
        max_closure = max(max_closure, regional_closure)

        branches = {
            "proxy_only": proxy_analysis[common_indices].T,
            "regional_anchor": regional_analysis[common_indices].T,
            "spatial_anchor_frozen": spatial_analysis[common_indices].T,
            "three_product_mean_benchmark": training_mean[outer],
        }
        heldout_outer = heldout_field[outer]
        for branch, values in branches.items():
            metrics, grid = score(branch, values, heldout_outer)
            metrics.update({
                "heldout_product": heldout_product,
                "calibration_years": "1901-1980",
                "outer_years": "1981-2000",
                "minimum_spatial_covariance_eigenvalue_c2": minimum_eigenvalue,
                "regional_base_variance_c2": regional_base_variance,
                "enkf_closure_max_abs_c": 0.0 if branch.startswith("three_product") else max(
                    proxy_closure,
                    regional_closure if branch == "regional_anchor" else 0.0,
                    spatial_closure if branch == "spatial_anchor_frozen" else 0.0,
                ),
            })
            product_rows.append(metrics)
            grid.insert(0, "heldout_product", heldout_product)
            grid.insert(2, "state_index", common_indices)
            grid.insert(3, "longitude", common_lon)
            grid.insert(4, "latitude", common_lat)
            grid_tables.append(grid)
            regional_values = weighted_mean(values, area_weights)
            for year, value in zip(OUTER_YEARS, regional_values):
                regional_series_rows.append({
                    "heldout_product": heldout_product,
                    "branch": branch,
                    "year": int(year),
                    "regional_mean_c": float(value),
                })
            arrays[f"{heldout_product}_{branch}"] = values
        for year, value in zip(OUTER_YEARS, weighted_mean(heldout_outer, area_weights)):
            regional_series_rows.append({
                "heldout_product": heldout_product,
                "branch": "heldout",
                "year": int(year),
                "regional_mean_c": float(value),
            })
        arrays[f"{heldout_product}_heldout"] = heldout_outer

    product_metrics = pd.DataFrame(product_rows)
    product_metrics.to_csv(output / "outer_product_branch_metrics.csv", index=False)
    grid_metrics = pd.concat(grid_tables, ignore_index=True)
    grid_metrics.to_csv(output / "outer_grid_metrics.csv", index=False)
    regional_series = pd.DataFrame(regional_series_rows)
    regional_series.to_csv(output / "outer_regional_series.csv", index=False)
    np.savez_compressed(output / "outer_branch_arrays.npz", years=OUTER_YEARS, state_indices=common_indices, state_lat=common_lat, state_lon=common_lon, **arrays)

    aggregate_rows = []
    for branch in product_metrics.branch.unique():
        selected = product_metrics.loc[product_metrics.branch.eq(branch)]
        aggregate_rows.append({
            "branch": branch,
            "pooled_centered_grid_year_rmse_c": float(np.sqrt(
                selected.centered_grid_year_squared_error_sum.sum() / selected.grid_year_count.sum()
            )),
            "mean_product_median_grid_temporal_correlation": float(selected.median_grid_temporal_correlation.mean()),
            "mean_product_median_annual_spatial_anomaly_correlation": float(selected.median_annual_spatial_anomaly_correlation.mean()),
            "mean_product_median_abs_grid_trend_error_c_per_century": float(selected.median_abs_grid_trend_error_c_per_century.mean()),
            "max_enkf_closure_abs_c": float(selected.enkf_closure_max_abs_c.max()),
            "heldout_products": int(len(selected)),
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values("pooled_centered_grid_year_rmse_c").reset_index(drop=True)
    aggregate.to_csv(output / "outer_aggregate.csv", index=False)
    plot_aggregate(aggregate, output)
    plot_product_curves(regional_series, output)
    plot_grid_improvement(grid_metrics, output)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    proxy_row = aggregate.loc[aggregate.branch.eq("proxy_only")].iloc[0]
    regional_row = aggregate.loc[aggregate.branch.eq("regional_anchor")].iloc[0]
    spatial_row = aggregate.loc[aggregate.branch.eq("spatial_anchor_frozen")].iloc[0]
    benchmark_row = aggregate.loc[aggregate.branch.eq("three_product_mean_benchmark")].iloc[0]
    outer_supports_spatial = bool(
        spatial_row.pooled_centered_grid_year_rmse_c < proxy_row.pooled_centered_grid_year_rmse_c
        and spatial_row.pooled_centered_grid_year_rmse_c < regional_row.pooled_centered_grid_year_rmse_c
        and spatial_row.mean_product_median_annual_spatial_anomaly_correlation > proxy_row.mean_product_median_annual_spatial_anomaly_correlation
    )
    summary = {
        "experiment_id": "climate_reconstruction_spatial_outer_1981_2000_v1",
        "date": "2026-08-13",
        "calibration_years": [1901, 1980],
        "outer_years": [1981, 2000],
        "heldout_products": list(PRODUCTS),
        "common_hma_state_count": int(len(common_indices)),
        "regional_scale_frozen": REGIONAL_SCALE,
        "spatial_shrinkage_frozen": SPATIAL_SHRINKAGE,
        "spatial_scale_frozen_boundary_candidate": SPATIAL_SCALE,
        "no_model_or_scale_selection_on_outer": True,
        "rec_ref_not_used": True,
        "precipitation_not_tested": True,
        "max_enkf_closure_abs_c": max_closure,
        "source_hashes_unchanged": pure,
        "outer_supports_spatial_mechanism": outer_supports_spatial,
        "spatial_to_benchmark_rmse_ratio": float(
            spatial_row.pooled_centered_grid_year_rmse_c / benchmark_row.pooled_centered_grid_year_rmse_c
        ),
        "aggregate": aggregate.to_dict(orient="records"),
        "product_metrics": product_metrics.to_dict(orient="records"),
        "interpretation_boundary": "This is frozen temporal outer validation across four related instrumental products. Passing supports the temperature spatial mechanism but does not establish product independence, precipitation validity, or final production-field status.",
        "canonical_sync_status": "Pending because apply_patch cannot edit any existing file on this Windows workspace; completed sector-v2 facts remain in PENDING_FACTS_UPDATE.md.",
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
