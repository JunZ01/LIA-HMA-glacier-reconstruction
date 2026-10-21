"""Use complete cached official ModE tas/pr files; supersedes v1-v4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

import acquire_temperature_four_prior_state_pools_1500_1848_v4 as v4


v1 = v4.v1
MODE_FILES = {
    "tas": v1.OUTPUT / "source_cache/ModE-Sim_set_1420-3_ensmean_temp2_abs_1420-1849_year.nc",
    "pr": v1.OUTPUT / "source_cache/ModE-Sim_set_1420-3_ensmean_totprec_abs_1420-1849_year.nc",
}


def local_complete_mode_annual(variable: str, points: np.ndarray, request_log):
    path = MODE_FILES[variable]
    name = "temp2" if variable == "tas" else "totprec"
    with Dataset(path) as dataset:
        time = dataset.variables["time"]
        dates = num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"), only_use_cftime_datetimes=True)
        years = np.asarray([int(date.year) for date in dates], dtype=int)
        indices = np.asarray([int(np.flatnonzero(years == year)[0]) for year in v1.YEARS])
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        fields = np.asarray(dataset.variables[name][indices], dtype=float)
        units = str(dataset.variables[name].units)
    if variable == "pr":
        fields, conversion = v1.precip_to_mm_day(fields, units)
    else:
        conversion = "kelvin_centered_anomaly_equals_degC"
    values = v1.interpolate_fields(fields, lat, lon, points)
    return values, {
        "path": str(path), "units": units, "conversion": conversion,
        "sha256": v1.file_state(path)["sha256"],
        "source": "complete_official_file_cached_in_independent_experiment",
    }


v1.mode_annual = local_complete_mode_annual
v1.DRIVER = Path(__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(v1.main())
