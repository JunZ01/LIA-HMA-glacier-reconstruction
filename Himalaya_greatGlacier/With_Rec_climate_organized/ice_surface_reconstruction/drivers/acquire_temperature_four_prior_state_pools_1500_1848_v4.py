"""Use local verified ModE temperature and cached official precipitation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset, num2date

import acquire_temperature_four_prior_state_pools_1500_1848_v3 as v3


v1 = v3.v2.v1
LOCAL_TAS = v1.RESULTS / "climate_reconstruction_improvement_loop_v2/round_76_mode_sim_subset_allowed_temperature_acquisition_v1/mode_sim_three_historical_sets_allowed_hma_bbox.nc"
LOCAL_PR = v1.OUTPUT / "source_cache/ModE-Sim_set_1420-3_ensmean_totprec_abs_1420-1849_year.nc"


def local_mode_annual(variable: str, points: np.ndarray, request_log):
    if variable == "tas":
        with h5py.File(LOCAL_TAS, "r") as handle:
            years = np.asarray(handle["historical_year"], dtype=int)
            lat = np.asarray(handle["lat"], dtype=float)
            lon = np.asarray(handle["lon"], dtype=float)
            source = np.asarray(handle["set_1420_3_temp2_anomaly_1961_1990"], dtype=float)
        indices = np.asarray([int(np.flatnonzero(years == year)[0]) for year in v1.YEARS])
        values = v1.interpolate_fields(source[indices], lat, lon, points)
        return values, {"path": str(LOCAL_TAS), "units": "K anomaly", "conversion": "centered_anomaly_equals_degC", "source": "verified_round76_local_subset"}
    with Dataset(LOCAL_PR) as dataset:
        time = dataset.variables["time"]
        dates = num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"), only_use_cftime_datetimes=True)
        years = np.asarray([int(date.year) for date in dates], dtype=int)
        indices = np.asarray([int(np.flatnonzero(years == year)[0]) for year in v1.YEARS])
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        fields = np.asarray(dataset.variables["totprec"][indices], dtype=float)
        units = str(dataset.variables["totprec"].units)
    fields, conversion = v1.precip_to_mm_day(fields, units)
    values = v1.interpolate_fields(fields, lat, lon, points)
    return values, {"path": str(LOCAL_PR), "units": units, "conversion": conversion, "sha256": v1.file_state(LOCAL_PR)["sha256"], "source": "official_file_cached_in_independent_experiment"}


v1.mode_annual = local_mode_annual
v1.DRIVER = Path(__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(v1.main())
