from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
ARCHIVE_ROOT = (
    Path(r"D:\Users\JUNZ\Desktop")
    / "\u6587\u4ef6\u6574\u7406"
    / "1 \u8bba\u6587"
    / "work1"
    / "compared_result_all"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "enkf_reconstruction_source_recovery_audit_v2"
    / "replay_proxy6_enkf_exp1_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interp_trailing(
    src_y: np.ndarray,
    src_x: np.ndarray,
    values: np.ndarray,
    dst_y: np.ndarray,
    dst_x: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (src_y, src_x), values, method="linear", bounds_error=False, fill_value=np.nan
    )
    xx, yy = np.meshgrid(dst_x, dst_y)
    points = np.column_stack([yy.ravel(), xx.ravel()])
    out = interpolator(points)
    return out.reshape(len(dst_y), len(dst_x), *values.shape[2:])


def _berkeley_annual_climatology(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, decode_times=False) as dataset:
        climate = np.asarray(dataset["climatology"].values, dtype=np.float64)
    climate = np.transpose(climate, (1, 2, 0))
    midpoint = climate.shape[1] // 2
    climate = np.concatenate([climate[:, midpoint:, :], climate[:, :midpoint, :]], axis=1)

    src_lat = np.linspace(0.0, 1.0, climate.shape[0])
    src_lon = np.linspace(0.0, 1.0, climate.shape[1])
    new_lat = np.linspace(-89.0, 89.0, 90)
    new_lon = np.linspace(-179.0, 179.0, 180)
    full = _interp_trailing(
        src_lat,
        src_lon,
        climate,
        np.linspace(0.0, 1.0, len(new_lat)),
        np.linspace(0.0, 1.0, len(new_lon)),
    )
    lat_ids = np.flatnonzero((new_lat >= 24.0) & (new_lat <= 51.0))
    lon_ids = np.flatnonzero((new_lon >= 62.0) & (new_lon <= 110.0))
    shifted_lon_ids = lon_ids - 90
    hma = full[lat_ids][:, shifted_lon_ids, :]
    return np.mean(hma, axis=2), new_lat[lat_ids], new_lon[lon_ids]


def build_temperature_prior(netcdf_path: Path, climatology_path: Path) -> np.ndarray:
    annual_clim, dst_lat, dst_lon = _berkeley_annual_climatology(climatology_path)
    with xr.open_dataset(netcdf_path, decode_times=False) as dataset:
        lat = np.asarray(dataset["lat"].values, dtype=np.float64)
        lon = np.asarray(dataset["lon"].values, dtype=np.float64)
        lon_signed = lon.copy()
        lon_signed[lon_signed > 180.0] -= 360.0
        lon_order = np.argsort(lon_signed)
        lon_sorted = lon_signed[lon_order]
        lat_ids = np.flatnonzero((lat >= 23.0) & (lat <= 53.0))
        lon_sorted_ids = np.flatnonzero((lon_sorted >= 61.0) & (lon_sorted <= 111.0))
        raw = np.asarray(
            dataset["TREFHT"].isel(
                lat=lat_ids, lon=lon_order[lon_sorted_ids]
            ).values,
            dtype=np.float64,
        )
    if raw.shape[0] != 12000:
        raise ValueError(f"Expected 12000 monthly fields, got {raw.shape[0]}")
    annual = raw.reshape(1000, 12, raw.shape[1], raw.shape[2]).mean(axis=1)
    annual = np.transpose(annual, (1, 2, 0))
    annual_hma = _interp_trailing(
        lat[lat_ids], lon_sorted[lon_sorted_ids], annual, dst_lat, dst_lon
    )
    corrected = annual_hma - np.mean(annual_hma, axis=2, keepdims=True)
    corrected += annual_clim[:, :, None]
    selected = corrected[:, :, 151:999]
    if selected.shape != (14, 24, 848):
        raise ValueError(f"Unexpected selected prior shape: {selected.shape}")
    return np.reshape(selected, (336, 848), order="F")


def _load_indices(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        indices = np.asarray(handle["all_Xb_indices"], dtype=np.int64)
    return indices[0] - 1


def _load_proxy6(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = np.arange(1000, 2001)
    with h5py.File(path, "r") as handle:
        group = handle["proxy_data"]
        references = np.asarray(group["full_proxy"]).reshape(-1, order="F")
        hxb = np.empty((len(references), 848), dtype=np.float64)
        residuals = np.empty((len(references), 61), dtype=np.float64)
        observations = np.full((len(references), len(years)), np.nan)
        for index, reference in enumerate(references):
            proxy = handle[reference]
            hxb[index] = np.asarray(proxy["HXb"]).reshape(-1, order="F")
            residuals[index] = np.asarray(proxy["Rvec"]).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"]).reshape(-1, order="F")
            data = np.asarray(proxy["data"]).reshape(-1, order="F")
            by_year = {int(age): data[i] for i, age in enumerate(ages) if np.isfinite(age)}
            observations[index] = [by_year.get(int(year), np.nan) for year in years]
    return hxb, residuals, observations


def _matrix_update(
    background: np.ndarray,
    modeled_observations: np.ndarray,
    observations: np.ndarray,
    error_covariance: np.ndarray,
) -> np.ndarray:
    members = background.shape[1]
    background_mean = np.mean(background, axis=1)
    background_prime = background - background_mean[:, None]
    modeled_mean = np.mean(modeled_observations, axis=1)
    modeled_prime = modeled_observations - modeled_mean[:, None]
    pbht = background_prime @ modeled_prime.T / (members - 1)
    hpbhtr = modeled_prime @ modeled_prime.T / (members - 1) + error_covariance
    gain = pbht @ np.linalg.inv(hpbhtr)
    root = np.real_if_close(scipy.linalg.sqrtm(hpbhtr)).real
    root_r = np.sqrt(error_covariance)
    reduced_gain = pbht @ np.linalg.inv(root).T @ np.linalg.inv(root + root_r)
    analysis_mean = background_mean + gain @ (observations - modeled_mean)
    analysis_prime = background_prime - reduced_gain @ modeled_prime
    return analysis_prime + analysis_mean[:, None]


def _percentile_candidates(analysis: np.ndarray) -> dict[str, np.ndarray]:
    methods = ["linear", "hazen", "weibull", "median_unbiased", "normal_unbiased"]
    return {
        method: np.percentile(analysis, [5, 50, 95], axis=1, method=method).T
        for method in methods
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[1000, 1200, 1400])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(year < 1000 or year >= 1450 for year in args.years):
        raise ValueError("v1 preflight supports years 1000-1449, before proxy 202 enters")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    netcdf_path = SOURCE_ROOT / "input-recon" / "b.e11.BLMTRC5CN.f19_g16.010.cam.h0.TREFHT.085001-184912.nc"
    climatology_path = SOURCE_ROOT / "input-recon" / "Land_and_Ocean_LatLong1_1850_2016.nc"
    indices_path = ARCHIVE_ROOT / "fixed_proxy_experiments_69" / "tas" / "indices" / "all_Xb_indices.mat"
    proxy_path = ARCHIVE_ROOT / "python_data" / "proxy_data_6.mat"
    result_path = ARCHIVE_ROOT / "fixed_proxy_experiments_69" / "tas" / "results" / "proxy6" / "EnKF_results.mat"

    prior = build_temperature_prior(netcdf_path, climatology_path)
    sample_indices = _load_indices(indices_path)
    background = prior[:, sample_indices]
    hxb, residuals, observations = _load_proxy6(proxy_path)
    sampled_hxb = hxb[:, sample_indices]
    np.save(args.output_dir / "temperature_prior_336x848.npy", prior)

    rows = []
    with h5py.File(result_path, "r") as archived:
        for year in args.years:
            column = year - 1000
            active = np.flatnonzero(np.isfinite(observations[:, column]))
            if np.isnan(residuals[active]).any():
                raise ValueError(f"Year {year} needs RegEM residual filling")
            error = np.diag(np.var(residuals[active], axis=1, ddof=1))
            analysis = _matrix_update(
                background,
                sampled_hxb[active],
                observations[active, column],
                error,
            )
            mean = np.mean(analysis, axis=1)
            sigma = np.std(analysis, axis=1, ddof=1)
            target_mean = np.asarray(archived["Xa_m_all"][0, column, :], dtype=np.float64)
            target_sigma = np.asarray(archived["Xa_sigma_all"][0, column, :], dtype=np.float64)
            target_prct = np.asarray(archived["Xa_prct_all"][0, column, :, :], dtype=np.float64).T
            percentile_diffs = {
                method: float(np.max(np.abs(values - target_prct)))
                for method, values in _percentile_candidates(analysis).items()
            }
            rows.append(
                {
                    "year": year,
                    "active_proxy_count": int(active.size),
                    "mean_max_abs_diff": float(np.max(np.abs(mean - target_mean))),
                    "sigma_max_abs_diff": float(np.max(np.abs(sigma - target_sigma))),
                    "percentile_max_abs_diff_by_method": percentile_diffs,
                    "best_percentile_method": min(percentile_diffs, key=percentile_diffs.get),
                }
            )

    summary = {
        "case": "proxy6 / EnKF / experiment 1",
        "scope": "pre-1450 replay without RegEM-imputed proxy 202",
        "years": args.years,
        "prior_shape": list(prior.shape),
        "sampled_background_shape": list(background.shape),
        "prior_sha256": _sha256(args.output_dir / "temperature_prior_336x848.npy"),
        "source_netcdf": str(netcdf_path),
        "source_netcdf_sha256": _sha256(netcdf_path),
        "climatology": str(climatology_path),
        "comparisons": rows,
    }
    (args.output_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
