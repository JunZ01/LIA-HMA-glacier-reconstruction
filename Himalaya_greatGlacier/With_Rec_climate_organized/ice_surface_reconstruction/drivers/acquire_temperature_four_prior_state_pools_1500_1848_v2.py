"""Executable repair of the ModE CDF2 variable-object handling in v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import acquire_temperature_four_prior_state_pools_1500_1848_v1 as v1


def mode_annual(variable: str, points: np.ndarray, request_log: list[dict[str, Any]]):
    url = v1.MODE_URLS[variable]
    label = f"mode_sim_set_1420_3_{variable}"
    source = v1.range_access.ExactRangeSource(url, label, request_log)
    try:
        header = v1.cdf2.parse_header(source.read(0, 65535, "cdf2_header", (1420, 1420)))
        lat_var = v1.range_access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = v1.range_access.coordinate_variable(header, ("lon", "longitude", "x"))
        time_var = v1.range_access.coordinate_variable(header, ("time", "t"))
        data_name = "temp2" if variable == "tas" else "totprec"
        if data_name not in header.variables:
            raise RuntimeError(f"missing-mode-variable:{data_name}:{list(header.variables)}")
        data_var = header.variables[data_name]
        lat = v1.range_access.read_fixed_variable(source, header, lat_var)
        lon = v1.range_access.normalize_longitude(v1.range_access.read_fixed_variable(source, header, lon_var))
        lat_margin = float(np.median(np.abs(np.diff(np.sort(lat))))) * 1.5
        lon_margin = float(np.median(np.abs(np.diff(np.sort(lon))))) * 1.5
        lat_ids = np.flatnonzero((lat >= points[:, 0].min() - lat_margin) & (lat <= points[:, 0].max() + lat_margin))
        lon_ids = np.flatnonzero((lon >= points[:, 1].min() - lon_margin) & (lon <= points[:, 1].max() + lon_margin))
        spatial_dims = list(data_var.dimensions[1:])
        lat_axis = spatial_dims.index(lat_var.dimensions[0])
        lon_axis = spatial_dims.index(lon_var.dimensions[0])
        fields = np.empty((len(v1.YEARS), len(lat_ids), len(lon_ids)), dtype=np.float64)
        decoded: list[int] = []
        units = v1.range_access.text_attr(data_var.attributes.get("units", ""))
        time_units = v1.range_access.text_attr(time_var.attributes.get("units", ""))
        calendar = v1.range_access.text_attr(time_var.attributes.get("calendar", "proleptic_gregorian"))
        for block_start in range(int(v1.YEARS[0]), int(v1.YEARS[-1]) + 1, 25):
            block_end = min(block_start + 24, int(v1.YEARS[-1]))
            first = block_start - 1420
            last = block_end - 1420
            byte_start = header.record_base + first * header.record_size
            byte_end = header.record_base + (last + 1) * header.record_size - 1
            payload = source.read(byte_start, byte_end, f"complete_allowed_annual_{variable}_records", (block_start, block_end))
            for year in range(block_start, block_end + 1):
                local = year - block_start
                raw_time = v1.range_access.record_segment(payload, header, time_var, local)
                time_value = v1.range_access.decode_numeric(raw_time, time_var, header.shape(time_var)[1:] or (1,)).reshape(-1)[0]
                date = v1.range_access.num2date(float(time_value), time_units, calendar)
                decoded.append(int(date.year))
                if int(date.year) != year:
                    raise RuntimeError(f"mode-year-decode:{year}:{date.year}")
                raw = v1.range_access.record_segment(payload, header, data_var, local)
                field = v1.range_access.decode_numeric(raw, data_var, header.shape(data_var)[1:])
                if (lat_axis, lon_axis) != (0, 1):
                    field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                fields[year - v1.YEARS[0]] = field[np.ix_(lat_ids, lon_ids)]
        if decoded != v1.YEARS.tolist():
            raise RuntimeError("mode-year-axis-failed")
        if variable == "pr":
            fields, conversion = v1.precip_to_mm_day(fields, units)
        else:
            conversion = "kelvin_centered_anomaly_equals_degC"
        values = v1.interpolate_fields(fields, lat[lat_ids], lon[lon_ids], points)
        return values, {"url": url, "units": units, "conversion": conversion, "etag": source.etag, "last_modified": source.last_modified}
    finally:
        source.close()


v1.mode_annual = mode_annual
v1.DRIVER = Path(__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(v1.main())
