"""Audit instrumental product disagreement on the 336-state EnKF grid."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRODUCTS = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}


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


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_spatial_error_audit_v1"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py",
        "evaluation_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_jja_temperature_network_repair_v1/run_experiment.py",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("spatial_error_base", paths["baseline_driver"])
    evaluation = load_module("spatial_error_eval", paths["evaluation_driver"])
    years = np.arange(1000, 2001)
    calibration = (years >= 1901) & (years <= 1980)
    grid_lat, grid_lon, state_lat, state_lon = base._state_coordinates()
    state_count = len(state_lat)
    if state_count != 336:
        raise RuntimeError(f"unexpected-state-count:{state_count}")

    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        fields = {}
        for label, variable in PRODUCTS.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            mapped = evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon)
            fields[label] = evaluation.anomaly(mapped, years, (1961, 1990))
    mask05 = np.load(paths["hma_mask"])
    mask2 = evaluation.source_grid(mask05[None].astype(float), source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    state_hma_mask = mask2.T.reshape(-1)

    state_fields = {label: flatten_state(field) for label, field in fields.items()}
    coverage_rows = []
    for index in range(state_count):
        row = {
            "state_index_0based": index,
            "latitude": state_lat[index],
            "longitude": state_lon[index],
            "inside_hma_mask": bool(state_hma_mask[index]),
        }
        for label, values in state_fields.items():
            row[f"{label}_finite_1901_1980"] = int(np.isfinite(values[calibration, index]).sum())
            row[f"{label}_finite_1981_2000"] = int(np.isfinite(values[(years >= 1981) & (years <= 2000), index]).sum())
        coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output / "state_grid_coverage.csv", index=False)

    residual_blocks = []
    residual_labels = []
    for heldout in PRODUCTS:
        others = [state_fields[name] for name in PRODUCTS if name != heldout]
        training_mean = np.nanmean(np.stack(others), axis=0)
        residual = training_mean[calibration] - state_fields[heldout][calibration]
        residual_blocks.append(residual)
        residual_labels.extend([heldout] * calibration.sum())
    residuals = np.vstack(residual_blocks)
    common_state = np.all(np.isfinite(residuals), axis=0)
    if common_state.sum() < 2:
        raise RuntimeError("insufficient-common-spatial-coverage")
    common_residuals = residuals[:, common_state]
    centered = common_residuals - np.mean(common_residuals, axis=0, keepdims=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    variance = np.diag(covariance)
    if np.any(variance <= 0) or np.any(~np.isfinite(variance)):
        raise RuntimeError("invalid-spatial-residual-variance")
    correlation = covariance / np.sqrt(variance[:, None] * variance[None, :])
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    effective_rank = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))

    thresholds = {}
    for threshold in (0.80, 0.90, 0.95, 0.99):
        thresholds[str(threshold)] = int(np.searchsorted(cumulative, threshold) + 1)
    common_indices = np.flatnonzero(common_state)
    grid_variance = np.full(state_count, np.nan)
    grid_variance[common_state] = variance
    pd.DataFrame({
        "state_index_0based": np.arange(state_count),
        "latitude": state_lat,
        "longitude": state_lon,
        "inside_hma_mask": state_hma_mask,
        "common_across_all_product_year_residuals": common_state,
        "product_disagreement_variance_c2": grid_variance,
    }).to_csv(output / "spatial_error_variance.csv", index=False)
    pd.DataFrame({
        "mode_1based": np.arange(1, len(eigenvalues) + 1),
        "eigenvalue_c2": eigenvalues,
        "explained_fraction": explained,
        "cumulative_fraction": cumulative,
    }).to_csv(output / "spatial_error_eigenspectrum.csv", index=False)
    np.savez_compressed(
        output / "spatial_error_audit_arrays.npz",
        residuals=residuals,
        residual_heldout_product=np.asarray(residual_labels),
        common_state_mask=common_state,
        common_state_indices=common_indices,
        covariance=covariance,
        correlation=correlation,
        eigenvalues=eigenvalues,
        state_lat=state_lat,
        state_lon=state_lon,
        hma_state_mask=state_hma_mask,
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    variance_cube = grid_variance.reshape(len(grid_lon), len(grid_lat)).T
    image = axes[0].pcolormesh(grid_lon, grid_lat, variance_cube, shading="auto", cmap="magma")
    axes[0].set_title("1901-1980 product-disagreement variance")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    figure.colorbar(image, ax=axes[0], label="degC²")
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, color="#2166ac", lw=2)
    for threshold in (0.8, 0.9, 0.95, 0.99):
        axes[1].axhline(threshold, color="#777777", lw=0.7)
    axes[1].set_xlim(1, min(120, len(cumulative)))
    axes[1].set_ylim(0, 1.01)
    axes[1].grid(alpha=0.18)
    axes[1].set_xlabel("Spatial covariance mode")
    axes[1].set_ylabel("Cumulative variance fraction")
    axes[1].set_title("Spatial-error eigenspectrum")
    axes[2].hist(off_diagonal, bins=50, color="#2f8f46", alpha=0.82)
    axes[2].axvline(0, color="#777777", lw=0.7)
    axes[2].set_xlabel("Pairwise residual correlation")
    axes[2].set_ylabel("Grid-cell pairs")
    axes[2].set_title("Off-diagonal error correlation")
    figure.savefig(output / "figure_spatial_error_audit.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": "climate_reconstruction_spatial_error_audit_v1",
        "date": "2026-08-13",
        "state_grid_count": state_count,
        "hma_mask_state_count": int(state_hma_mask.sum()),
        "common_residual_state_count": int(common_state.sum()),
        "common_hma_residual_state_count": int(np.sum(common_state & state_hma_mask)),
        "residual_field_samples": int(residuals.shape[0]),
        "median_off_diagonal_correlation": float(np.median(off_diagonal)),
        "median_absolute_off_diagonal_correlation": float(np.median(np.abs(off_diagonal))),
        "p90_absolute_off_diagonal_correlation": float(np.quantile(np.abs(off_diagonal), 0.9)),
        "participation_ratio_effective_rank": effective_rank,
        "modes_for_variance_fraction": thresholds,
        "minimum_grid_variance_c2": float(np.min(variance)),
        "median_grid_variance_c2": float(np.median(variance)),
        "maximum_grid_variance_c2": float(np.max(variance)),
        "source_hashes_unchanged": pure,
        "decision_boundary": "This audit does not select or run a spatial EnKF. It determines whether diagonal, shrinkage-full, or reduced-rank instrumental error models are scientifically admissible for the next pre-1981 validation stage.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
