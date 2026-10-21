from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "enkf_replay_v2", HERE / "replay_enkf_fixed_experiment_v2.py"
)
v2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v2)


MERGED_TEMPERATURE = Path(r"D:\VM\gcm_7\cesm\cesm_TREFHT_merged.nc")


def build_merged_cru_temperature_prior(_: Path, cru_path: Path) -> np.ndarray:
    monthly_clim, dst_lat, dst_lon = v2._cru_monthly_climatology(cru_path)
    with xr.open_dataset(MERGED_TEMPERATURE, decode_times=False) as dataset:
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
    if raw.shape[0] % 12:
        raise ValueError(f"Monthly count is not divisible by 12: {raw.shape[0]}")
    years = raw.shape[0] // 12
    monthly = raw.reshape(years, 12, raw.shape[1], raw.shape[2])
    monthly = np.transpose(monthly, (2, 3, 1, 0))
    monthly_hma = v2.v1._interp_trailing(
        lat[lat_ids], lon_sorted[lon_ids], monthly, dst_lat, dst_lon
    )
    anomalies = monthly_hma - np.mean(monthly_hma, axis=3, keepdims=True)
    annual = np.nansum(anomalies + monthly_clim[:, :, :, None], axis=2) / 12.0
    selected = annual[:, :, 151:999]
    if selected.shape != (14, 24, 848):
        raise ValueError(f"Unexpected selected prior shape: {selected.shape}")
    return np.reshape(selected, (336, 848), order="F")


if __name__ == "__main__":
    v2.build_cru_temperature_prior = build_merged_cru_temperature_prior
    v2.main()
