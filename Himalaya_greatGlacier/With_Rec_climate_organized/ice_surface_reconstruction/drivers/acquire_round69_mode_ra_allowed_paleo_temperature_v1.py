"""Acquire the modern-validated ModE-RA temperature field for 1421-1920."""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access


FIRST_YEAR = 1421
LAST_YEAR = 1920
BLOCK_YEARS = 10


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def standard_calendar_month_days(year: int, month: int) -> int:
    if year == 1582 and month == 10:
        return 21
    if year < 1582 or (year == 1582 and month < 10):
        if month == 2:
            return 29 if year % 4 == 0 else 28
        return 30 if month in (4, 6, 9, 11) else 31
    return calendar.monthrange(year, month)[1]


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v1"
    round67 = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    round68 = loop / "round_68_mode_ra_modern_independent_validation_v1"
    paths = {
        "round67_remote_metadata": round67 / "remote_source_metadata.json",
        "round68_summary": round68 / "summary.json",
        "round67_access_driver": Path(access.__file__).resolve(),
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    for name in ("summary.json", "mode_ra_temp2_1421_1920_hma_bbox.nc"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    before = {name: state(path) for name, path in paths.items()}
    round68_summary = json.loads(paths["round68_summary"].read_text(encoding="utf-8"))
    if not round68_summary["variables"]["temperature"]["passes_spatial_field"]:
        raise RuntimeError("round68-temperature-spatial-field-did-not-pass")
    if round68_summary["variables"]["precipitation"]["passes_regional_curve"]:
        raise RuntimeError("round68-precipitation-route-unexpectedly-open")

    request_log: list[dict[str, Any]] = []
    source = access.ExactRangeSource(access.SOURCES["temp2"], "temp2", request_log)
    try:
        header_payload = source.read(
            0,
            access.HEADER_BYTES - 1,
            "classic_netcdf_header_and_allowed_1421_prefix",
            (1421, 1421),
        )
        header = access.parse_header(header_payload)
        lat_var = access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = access.coordinate_variable(header, ("lon", "longitude", "x"))
        time_var = access.coordinate_variable(header, ("time", "t"))
        data_var = header.variables["temp2"]
        lat = access.read_fixed_variable(source, header, lat_var)
        lon = access.normalize_longitude(access.read_fixed_variable(source, header, lon_var))
        lat_indices = np.flatnonzero(
            (lat >= access.HMA_BBOX["south"]) & (lat <= access.HMA_BBOX["north"])
        )
        lon_indices = np.flatnonzero(
            (lon >= access.HMA_BBOX["west"]) & (lon <= access.HMA_BBOX["east"])
        )
        spatial_dims = list(data_var.dimensions[1:])
        lat_axis = spatial_dims.index(lat_var.dimensions[0])
        lon_axis = spatial_dims.index(lon_var.dimensions[0])
        if len(spatial_dims) != 2 or (lat_axis, lon_axis) not in ((0, 1), (1, 0)):
            raise RuntimeError(f"unexpected-temperature-spatial-dimensions:{spatial_dims}")
        years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
        annual = np.full((len(years), len(lat_indices), len(lon_indices)), np.nan)
        time_values: list[float] = []
        for block_start in range(FIRST_YEAR, LAST_YEAR + 1, BLOCK_YEARS):
            block_end = min(block_start + BLOCK_YEARS - 1, LAST_YEAR)
            first_record = (block_start - access.START_YEAR) * 12
            last_record = (block_end - access.START_YEAR + 1) * 12 - 1
            byte_start = header.record_base + first_record * header.record_size
            byte_end = header.record_base + (last_record + 1) * header.record_size - 1
            payload = source.read(
                byte_start,
                byte_end,
                "complete_allowed_paleo_temperature_records",
                (block_start, block_end),
            )
            for year in range(block_start, block_end + 1):
                monthly = []
                weights = []
                for month in range(1, 13):
                    local_record = (year - block_start) * 12 + month - 1
                    time_raw = access.record_segment(payload, header, time_var, local_record)
                    time_value = access.decode_numeric(
                        time_raw,
                        time_var,
                        header.shape(time_var)[1:] or (1,),
                    ).reshape(-1)[0]
                    time_values.append(float(time_value))
                    data_raw = access.record_segment(payload, header, data_var, local_record)
                    field = access.decode_numeric(data_raw, data_var, header.shape(data_var)[1:])
                    if (lat_axis, lon_axis) != (0, 1):
                        field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                    monthly.append(field[np.ix_(lat_indices, lon_indices)])
                    weights.append(standard_calendar_month_days(year, month))
                monthly_stack = np.stack(monthly)
                month_weights = np.asarray(weights, dtype=float)
                complete = np.all(np.isfinite(monthly_stack), axis=0)
                annual[year - FIRST_YEAR] = np.where(
                    complete,
                    np.sum(monthly_stack * month_weights[:, None, None], axis=0) / month_weights.sum(),
                    np.nan,
                )
    finally:
        source.close()

    time_values_array = np.asarray(time_values)
    time_steps_days = np.diff(time_values_array) / 86400.0
    if len(time_values_array) != len(years) * 12:
        raise RuntimeError(f"monthly-time-count-failed:{len(time_values_array)}")
    if not np.all(time_steps_days > 0):
        raise RuntimeError("monthly-time-axis-not-strictly-increasing")
    if not (20.0 <= float(np.min(time_steps_days)) <= float(np.max(time_steps_days)) <= 32.0):
        raise RuntimeError(
            f"unexpected-monthly-time-steps:{float(np.min(time_steps_days))}:{float(np.max(time_steps_days))}"
        )
    lat_subset = lat[lat_indices]
    lon_subset = lon[lon_indices]
    lat_order = np.argsort(lat_subset)
    lon_order = np.argsort(lon_subset)
    annual = annual[:, lat_order, :][:, :, lon_order]
    lat_subset = lat_subset[lat_order]
    lon_subset = lon_subset[lon_order]
    baseline = (years >= 1961) & (years <= 1990)
    recentered = annual - np.nanmean(annual[baseline], axis=0)[None]
    target = output / "mode_ra_temp2_1421_1920_hma_bbox.nc"
    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "ModE-RA modern-validated temperature, allowed paleo HMA subset"
        handle.attrs["source_url"] = access.SOURCES["temp2"]
        handle.attrs["access_method"] = "validated HTTP 206 complete-record byte ranges"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.create_dataset("time", data=years.astype(np.int16))
        handle["time"].attrs["units"] = "year CE"
        handle.create_dataset("lat", data=lat_subset.astype(np.float64))
        handle.create_dataset("lon", data=lon_subset.astype(np.float64))
        handle.create_dataset(
            "temp2_anomaly_source_baseline",
            data=annual.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["temp2_anomaly_source_baseline"].attrs["units"] = "degC"
        handle["temp2_anomaly_source_baseline"].attrs["baseline"] = "1901-2000"
        handle.create_dataset(
            "temp2_anomaly_1961_1990",
            data=recentered.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["temp2_anomaly_1961_1990"].attrs["units"] = "degC"
        handle["temp2_anomaly_1961_1990"].attrs["baseline"] = "1961-1990"

    with (output / "http_range_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(request_log[0]))
        writer.writeheader()
        writer.writerows(request_log)
    request_bytes = sum(int(row["byte_count"]) for row in request_log)
    gates = {
        "all_http_responses_are_exact_206_ranges": all(row["http_status"] == 206 for row in request_log),
        "no_requested_year_range_overlaps_1921_1939": all(not row["locked_period_overlap"] for row in request_log),
        "output_axis_is_exact_1421_1920": np.array_equal(years, np.arange(1421, 1921)),
        "monthly_time_count_is_6000": len(time_values_array) == 6000,
        "monthly_time_axis_strictly_increasing": bool(np.all(time_steps_days > 0)),
        "annual_field_is_fully_finite": bool(np.all(np.isfinite(annual))),
        "recentered_1961_1990_closure_below_1e_6": float(
            np.max(np.abs(np.mean(recentered[baseline], axis=0)))
        ) < 1e-6,
        "round68_temperature_spatial_validation_passed": True,
        "round68_precipitation_branch_not_downloaded": True,
    }
    summary = {
        "status": "complete",
        "source_product": "ModE-RA lowres 20-member ensemble-mean temp2",
        "years": [FIRST_YEAR, LAST_YEAR],
        "locked_1921_1939_accessed": False,
        "shape": list(annual.shape),
        "latitude_range": [float(lat_subset.min()), float(lat_subset.max())],
        "longitude_range": [float(lon_subset.min()), float(lon_subset.max())],
        "monthly_time_step_days_range": [float(time_steps_days.min()), float(time_steps_days.max())],
        "finite_fraction": float(np.isfinite(annual).mean()),
        "recentered_1961_1990_max_abs_mean": float(
            np.max(np.abs(np.mean(recentered[baseline], axis=0)))
        ),
        "http_range_request_count": len(request_log),
        "http_range_bytes_read": request_bytes,
        "gates": gates,
        "passes_acquisition_preflight": bool(all(gates.values())),
        "decision": (
            "advance_mode_ra_temperature_to_allowed_paleo_comparison"
            if all(gates.values())
            else "stop_mode_ra_paleo_temperature_on_acquisition_conflict"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-RA Allowed Paleo Temperature Acquisition",
                "",
                f"- Years and shape: `{FIRST_YEAR}-{LAST_YEAR}`, `{annual.shape}`.",
                f"- Exact requests/bytes: `{len(request_log)}` / `{request_bytes}`.",
                f"- Finite fraction: `{np.isfinite(annual).mean():.3%}`.",
                f"- Locked period accessed: `false`.",
                f"- All gates pass: `{str(summary['passes_acquisition_preflight']).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "local_sources_before": before,
                "local_sources_after": after,
                "local_source_states_unchanged": unchanged,
                "remote_source": {
                    "url": access.SOURCES["temp2"],
                    "content_length": source.content_length,
                    "etag": source.etag,
                    "last_modified": source.last_modified,
                },
                "output": state(target),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("local-source-state-changed")
    if not summary["passes_acquisition_preflight"]:
        raise RuntimeError("mode-ra-paleo-temperature-acquisition-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
