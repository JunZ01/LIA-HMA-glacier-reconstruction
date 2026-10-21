"""Run the fixed three-family audit without the optional rasterio dependency."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


def bilinear_points(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
) -> np.ndarray:
    data = np.asarray(field, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat[1] < lat[0]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    if lon[1] < lon[0]:
        lon = lon[::-1]
        data = data[:, :, ::-1]
    fy = (station_lat - lat[0]) / (lat[1] - lat[0])
    fx = (station_lon - lon[0]) / (lon[1] - lon[0])
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    if (
        np.any(y0 < 0)
        or np.any(y0 >= len(lat) - 1)
        or np.any(x0 < 0)
        or np.any(x0 >= len(lon) - 1)
    ):
        raise RuntimeError("station outside product grid")
    wy = fy - y0
    wx = fx - x0
    corners = np.stack(
        [
            data[:, y0, x0],
            data[:, y0, x0 + 1],
            data[:, y0 + 1, x0],
            data[:, y0 + 1, x0 + 1],
        ],
        axis=0,
    )
    weights = np.stack(
        [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx],
        axis=0,
    )
    valid = np.isfinite(corners)
    denominator = np.sum(weights[:, None, :] * valid, axis=0)
    numerator = np.nansum(corners * weights[:, None, :], axis=0)
    return numerator / np.where(denominator > 0, denominator, np.nan)


station_stub = types.ModuleType("run_temperature_station_annual_anomaly_benchmark_v1")
station_stub.bilinear_points = bilinear_points
sys.modules[station_stub.__name__] = station_stub

import audit_temperature_external_three_family_fixed_candidate_v1 as audit


audit.DRIVER = Path(__file__).resolve()
audit.OUTPUT = audit.RESULTS / "climate_reconstruction_temperature_external_three_family_fixed_candidate_audit_v2"


if __name__ == "__main__":
    raise SystemExit(audit.main())
