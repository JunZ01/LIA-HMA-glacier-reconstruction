from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "enkf_replay_v1", HERE / "replay_enkf_fixed_experiment_v1.py"
)
v1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v1)


def _target_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    new_lat = np.linspace(-89.0, 89.0, 90)
    new_lon = np.linspace(-179.0, 179.0, 180)
    lat_ids = np.flatnonzero((new_lat >= 24.0) & (new_lat <= 51.0))
    lon_ids = np.flatnonzero((new_lon >= 62.0) & (new_lon <= 110.0))
    return new_lat, new_lon, lat_ids, lon_ids


def _cru_monthly_climatology(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(path)
    if raw.shape != (4320, 720):
        raise ValueError(f"Unexpected CRU CL shape: {raw.shape}")
    monthly = np.reshape(raw, (360, 12, 720), order="F")
    monthly = np.transpose(monthly, (1, 0, 2))
    monthly = np.concatenate([monthly[:, :, 360:], monthly[:, :, :360]], axis=2)
    monthly[monthly == -999] = np.nan
    monthly = np.transpose(monthly / 10.0, (1, 2, 0))

    new_lat, new_lon, lat_ids, lon_ids = _target_grid()
    full = v1._interp_trailing(
        np.linspace(0.0, 1.0, 360),
        np.linspace(0.0, 1.0, 720),
        monthly,
        np.linspace(0.0, 1.0, len(new_lat)),
        np.linspace(0.0, 1.0, len(new_lon)),
    )
    return full[lat_ids][:, lon_ids - 90, :], new_lat[lat_ids], new_lon[lon_ids]


def build_cru_temperature_prior(netcdf_path: Path, cru_path: Path) -> np.ndarray:
    monthly_clim, dst_lat, dst_lon = _cru_monthly_climatology(cru_path)
    with xr.open_dataset(netcdf_path, decode_times=False) as dataset:
        lat = np.asarray(dataset["lat"].values, dtype=np.float64)
        lon = np.asarray(dataset["lon"].values, dtype=np.float64)
        lon_signed = lon.copy()
        lon_signed[lon_signed > 180.0] -= 360.0
        lon_order = np.argsort(lon_signed)
        lon_sorted = lon_signed[lon_order]
        lat_ids = np.flatnonzero((lat >= 23.0) & (lat <= 53.0))
        lon_ids = np.flatnonzero((lon_sorted >= 61.0) & (lon_sorted <= 111.0))
        raw = np.asarray(
            dataset["TREFHT"].isel(lat=lat_ids, lon=lon_order[lon_ids]).values,
            dtype=np.float64,
        )
    if raw.shape[0] != 12000:
        raise ValueError(f"Expected 12000 monthly fields, got {raw.shape[0]}")
    monthly = raw.reshape(1000, 12, raw.shape[1], raw.shape[2])
    monthly = np.transpose(monthly, (2, 3, 1, 0))
    monthly_hma = v1._interp_trailing(
        lat[lat_ids], lon_sorted[lon_ids], monthly, dst_lat, dst_lon
    )
    anomalies = monthly_hma - np.mean(monthly_hma, axis=3, keepdims=True)
    annual = np.nansum(anomalies + monthly_clim[:, :, :, None], axis=2) / 12.0
    selected = annual[:, :, 151:999]
    if selected.shape != (14, 24, 848):
        raise ValueError(f"Unexpected selected prior shape: {selected.shape}")
    return np.reshape(selected, (336, 848), order="F")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[1000, 1200, 1400])
    parser.add_argument(
        "--output-dir", type=Path, default=v1.DEFAULT_OUTPUT.parent / "replay_proxy6_enkf_exp1_v2_cru"
    )
    args = parser.parse_args()
    if any(year < 1000 or year >= 1450 for year in args.years):
        raise ValueError("v2 preflight supports years 1000-1449")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    netcdf_path = v1.SOURCE_ROOT / "input-recon" / "b.e11.BLMTRC5CN.f19_g16.010.cam.h0.TREFHT.085001-184912.nc"
    cru_path = v1.SOURCE_ROOT / "input-recon" / "cru_cl_1.1.6190.tmp.dat"
    indices_path = v1.ARCHIVE_ROOT / "fixed_proxy_experiments_69" / "tas" / "indices" / "all_Xb_indices.mat"
    proxy_path = v1.ARCHIVE_ROOT / "python_data" / "proxy_data_6.mat"
    result_path = v1.ARCHIVE_ROOT / "fixed_proxy_experiments_69" / "tas" / "results" / "proxy6" / "EnKF_results.mat"

    prior = build_cru_temperature_prior(netcdf_path, cru_path)
    sample_indices = v1._load_indices(indices_path)
    background = prior[:, sample_indices]
    hxb, residuals, observations = v1._load_proxy6(proxy_path)
    sampled_hxb = hxb[:, sample_indices]
    prior_path = args.output_dir / "temperature_prior_336x848_cru.npy"
    np.save(prior_path, prior)

    rows = []
    with h5py.File(result_path, "r") as archived:
        for year in args.years:
            column = year - 1000
            active = np.flatnonzero(np.isfinite(observations[:, column]))
            if np.isnan(residuals[active]).any():
                raise ValueError(f"Year {year} needs RegEM residual filling")
            error = np.diag(np.var(residuals[active], axis=1, ddof=1))
            analysis = v1._matrix_update(
                background,
                sampled_hxb[active],
                observations[active, column],
                error,
            )
            target_mean = np.asarray(archived["Xa_m_all"][0, column, :], dtype=float)
            target_sigma = np.asarray(archived["Xa_sigma_all"][0, column, :], dtype=float)
            target_prct = np.asarray(archived["Xa_prct_all"][0, column, :, :], dtype=float).T
            percentile_diffs = {
                method: float(np.max(np.abs(values - target_prct)))
                for method, values in v1._percentile_candidates(analysis).items()
            }
            rows.append(
                {
                    "year": year,
                    "active_proxy_count": int(active.size),
                    "mean_max_abs_diff": float(
                        np.max(np.abs(np.mean(analysis, axis=1) - target_mean))
                    ),
                    "sigma_max_abs_diff": float(
                        np.max(np.abs(np.std(analysis, axis=1, ddof=1) - target_sigma))
                    ),
                    "percentile_max_abs_diff_by_method": percentile_diffs,
                    "best_percentile_method": min(percentile_diffs, key=percentile_diffs.get),
                }
            )

    summary = {
        "case": "proxy6 / EnKF / experiment 1",
        "scope": "pre-1450 replay using recovered CRU CL temperature branch",
        "historical_loader_evidence": str(
            Path(r"D:\Users\JUNZ\Desktop\select\PHYDA\load\load_S_720360.m")
        ),
        "years": args.years,
        "prior_shape": list(prior.shape),
        "prior_sha256": v1._sha256(prior_path),
        "comparisons": rows,
    }
    (args.output_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
