from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "stage3_initial_state"))

from build_pilot_preflight import nearest_valid_pixel  # noqa: E402


def test_nearest_valid_pixel_skips_closer_nan_cell() -> None:
    temperature = np.ones((2, 2, 2), dtype=float)
    precipitation = np.ones((2, 2, 2), dtype=float)
    precipitation[:, 0, 0] = np.nan
    dataset = xr.Dataset(
        {
            "temp": (("time", "lat", "lon"), temperature),
            "prcp": (("time", "lat", "lon"), precipitation),
        },
        coords={"time": [0, 1], "lat": [30.0, 30.5], "lon": [80.0, 80.5]},
    )
    valid = (
        np.isfinite(dataset["temp"]).all("time")
        & np.isfinite(dataset["prcp"]).all("time")
    ).values
    lat_index, lon_index, distance = nearest_valid_pixel(
        dataset, 80.01, 30.01, valid
    )
    assert (lat_index, lon_index) in {(0, 1), (1, 0)}
    assert distance > 0
