"""Batching wrapper for the PMIP multi-model prior audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import audit_temperature_pmip_multimodel_prior_v1 as base


def interpolate_monthly_hma(
    dataset,
    time_indices: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    if len(time_indices) == 0 or np.any(np.diff(time_indices) != 1):
        raise RuntimeError("Remote batch must use one contiguous time segment")
    lat = np.asarray(dataset.variables["lat"][:], dtype=float)
    lon = np.mod(np.asarray(dataset.variables["lon"][:], dtype=float), 360.0)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat = lat[lat_order]
    lon = lon[lon_order]
    lat_pad = max(float(np.median(np.diff(lat))) * 1.5, 2.0)
    lon_pad = max(float(np.median(np.diff(lon))) * 1.5, 2.0)
    lat_keep = np.flatnonzero(
        (lat >= target_lat.min() - lat_pad) & (lat <= target_lat.max() + lat_pad)
    )
    lon_keep = np.flatnonzero(
        (lon >= target_lon.min() - lon_pad) & (lon <= target_lon.max() + lon_pad)
    )
    if len(lat_keep) < 2 or len(lon_keep) < 2:
        raise RuntimeError("Insufficient source-grid overlap with HMA")

    original_lat = lat_order[lat_keep]
    original_lon = lon_order[lon_keep]
    lat_slice = slice(int(original_lat.min()), int(original_lat.max()) + 1)
    lon_slice = slice(int(original_lon.min()), int(original_lon.max()) + 1)
    source_lat_raw = np.asarray(dataset.variables["lat"][lat_slice], dtype=float)
    source_lon_raw = np.mod(
        np.asarray(dataset.variables["lon"][lon_slice], dtype=float), 360.0
    )
    lat_sort = np.argsort(source_lat_raw)
    lon_sort = np.argsort(source_lon_raw)
    source_lat = source_lat_raw[lat_sort]
    source_lon = source_lon_raw[lon_sort]

    target_lat_2d, target_lon_2d = np.meshgrid(
        target_lat, target_lon, indexing="ij"
    )
    points = np.column_stack(
        [target_lat_2d[target_mask], target_lon_2d[target_mask]]
    )
    weights = np.cos(np.deg2rad(target_lat_2d[target_mask]))
    tas = dataset.variables["tas"]
    indexers: list[Any] = []
    kept_dims: list[str] = []
    for dimension in tas.dimensions:
        if dimension == "time":
            indexers.append(slice(int(time_indices[0]), int(time_indices[-1]) + 1))
            kept_dims.append("time")
        elif dimension == "lat":
            indexers.append(lat_slice)
            kept_dims.append("lat")
        elif dimension == "lon":
            indexers.append(lon_slice)
            kept_dims.append("lon")
        else:
            indexers.append(0)
    fields = np.asarray(tas[tuple(indexers)], dtype=float).squeeze()
    if kept_dims != ["time", "lat", "lon"] or fields.ndim != 3:
        raise RuntimeError(f"Unexpected tas dimensions: {tas.dimensions} -> {fields.shape}")
    fields = fields[:, lat_sort][:, :, lon_sort]
    output = np.full(len(time_indices), np.nan)
    for output_index, field in enumerate(fields):
        interpolator = base.RegularGridInterpolator(
            (source_lat, source_lon), field, bounds_error=False, fill_value=np.nan
        )
        target_values = interpolator(points)
        valid = np.isfinite(target_values)
        if int(valid.sum()) < int(0.9 * len(target_values)):
            raise RuntimeError("Too few finite interpolated HMA cells")
        output[output_index] = float(
            np.average(target_values[valid], weights=weights[valid])
        )
    return output


def read_remote_series(
    files: list[dict[str, Any]],
    allowed_ranges: list[tuple[int, int]],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, float | int]] = []
    audit: list[dict[str, Any]] = []
    for item in files:
        with base.Dataset(item["url"]) as dataset:
            time = dataset.variables["time"]
            calendar = getattr(time, "calendar", "standard")
            dates = base.num2date(
                time[:],
                units=time.units,
                calendar=calendar,
                only_use_cftime_datetimes=True,
            )
            model_years = np.asarray([int(date.year) for date in dates])
            ce_years = model_years + int(item["offset"])
            allowed = np.zeros(len(ce_years), dtype=bool)
            for start, end in allowed_ranges:
                allowed |= (ce_years >= start) & (ce_years <= end)
            time_indices = np.flatnonzero(allowed)
            if len(time_indices) == 0:
                continue
            if np.any(
                (ce_years[time_indices] >= base.LOCKED_START)
                & (ce_years[time_indices] <= base.LOCKED_END)
            ):
                raise RuntimeError("Locked climate values requested")
            segments = np.split(
                time_indices, np.flatnonzero(np.diff(time_indices) != 1) + 1
            )
            for segment in segments:
                values = interpolate_monthly_hma(
                    dataset, segment, target_lat, target_lon, target_mask
                )
                for index, value in zip(segment, values, strict=True):
                    rows.append(
                        {
                            "year": int(ce_years[index]),
                            "month": int(dates[index].month),
                            "tas_K": float(value),
                        }
                    )
            audit.append(
                {
                    **item,
                    "requested_month_count": int(len(time_indices)),
                    "requested_first_year": int(ce_years[time_indices].min()),
                    "requested_last_year": int(ce_years[time_indices].max()),
                    "requested_contiguous_segments": int(len(segments)),
                    "locked_climate_values_requested": 0,
                }
            )
    monthly = (
        pd.DataFrame(rows)
        .drop_duplicates(["year", "month"])
        .sort_values(["year", "month"])
    )
    counts = monthly.groupby("year")["month"].nunique()
    incomplete = counts[counts != 12]
    if len(incomplete):
        raise RuntimeError(f"Incomplete model years: {incomplete.to_dict()}")
    annual = monthly.groupby("year", as_index=False)["tas_K"].mean()
    return annual, audit


if __name__ == "__main__":
    base.OUTPUT = (
        base.RESULTS
        / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2"
    )
    base.DRIVER = Path(__file__).resolve()
    base.interpolate_monthly_hma = interpolate_monthly_hma
    base.read_remote_series = read_remote_series
    raise SystemExit(base.main())
