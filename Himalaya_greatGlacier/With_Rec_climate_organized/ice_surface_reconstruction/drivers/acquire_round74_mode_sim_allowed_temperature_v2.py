"""Acquire allowed-period ModE-Sim temperature matching the ModE-RA product."""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import mode_sim_cdf2_parser_v1 as cdf2


SOURCE_URL = (
    "https://mode-ra.unibe.ch/data/ModE-Sim/outdata/"
    "lowres_20mem_Set_1420-3_1850-1/anom_wrt_1901-2000/ensstat/by_var/mon/"
    "ModE-Sim_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
)
PERIODS = ((1421, 1920), (1940, 2000))
BLOCK_YEARS = 10
ALLOWED_YEARS = np.concatenate([np.arange(start, end + 1) for start, end in PERIODS])


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


def month_days_standard(year: int, month: int) -> int:
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
    output = loop / "round_74_mode_sim_allowed_temperature_acquisition_v2"
    round69 = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v3"
    round67 = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    round73 = loop / "round_73_mode_ra_hma_source_overlap_v1"
    paths = {
        "mode_ra_paleo": round69 / "mode_ra_temp2_1421_1920_hma_bbox.nc",
        "mode_ra_modern": round67 / "mode_ra_ensmean_1940_2000_hma_bbox.nc",
        "round73_summary": round73 / "summary.json",
        "round67_access_driver": Path(access.__file__).resolve(),
        "cdf2_parser": Path(cdf2.__file__).resolve(),
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "mode_sim_temp2_allowed_hma_bbox.nc"
    if target.exists() or (output / "summary.json").exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    overlap = json.loads(paths["round73_summary"].read_text(encoding="utf-8"))
    if overlap["eligible_as_proxy_evidence_independent_family"]:
        raise RuntimeError("round73-did-not-identify-evidence-overlap")

    request_log: list[dict[str, Any]] = []
    source = access.ExactRangeSource(SOURCE_URL, "mode_sim_temp2", request_log)
    try:
        header_payload = source.read(
            0,
            access.HEADER_BYTES - 1,
            "classic_netcdf_header_and_allowed_1421_prefix",
            (1421, 1421),
        )
        header = cdf2.parse_header(header_payload)
        lat_var = access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = access.coordinate_variable(header, ("lon", "longitude", "x"))
        time_var = access.coordinate_variable(header, ("time", "t"))
        if "temp2" not in header.variables:
            raise RuntimeError(f"temp2-not-found:{list(header.variables)}")
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
            raise RuntimeError(f"unexpected-spatial-dimensions:{spatial_dims}")
        annual = np.full(
            (len(ALLOWED_YEARS), len(lat_indices), len(lon_indices)), np.nan
        )
        all_time_values: list[float] = []
        output_index = 0
        for period_start, period_end in PERIODS:
            period_time_values: list[float] = []
            for block_start in range(period_start, period_end + 1, BLOCK_YEARS):
                block_end = min(block_start + BLOCK_YEARS - 1, period_end)
                first_record = (block_start - access.START_YEAR) * 12
                last_record = (block_end - access.START_YEAR + 1) * 12 - 1
                byte_start = header.record_base + first_record * header.record_size
                byte_end = header.record_base + (last_record + 1) * header.record_size - 1
                payload = source.read(
                    byte_start,
                    byte_end,
                    "complete_allowed_mode_sim_temperature_records",
                    (block_start, block_end),
                )
                for year in range(block_start, block_end + 1):
                    monthly = []
                    weights = []
                    for month in range(1, 13):
                        local_record = (year - block_start) * 12 + month - 1
                        time_raw = access.record_segment(payload, header, time_var, local_record)
                        time_value = access.decode_numeric(
                            time_raw, time_var, header.shape(time_var)[1:] or (1,)
                        ).reshape(-1)[0]
                        period_time_values.append(float(time_value))
                        data_raw = access.record_segment(payload, header, data_var, local_record)
                        field = access.decode_numeric(
                            data_raw, data_var, header.shape(data_var)[1:]
                        )
                        if (lat_axis, lon_axis) != (0, 1):
                            field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                        monthly.append(field[np.ix_(lat_indices, lon_indices)])
                        weights.append(month_days_standard(year, month))
                    monthly_stack = np.stack(monthly)
                    month_weights = np.asarray(weights, dtype=float)
                    complete = np.all(np.isfinite(monthly_stack), axis=0)
                    annual[output_index] = np.where(
                        complete,
                        np.sum(monthly_stack * month_weights[:, None, None], axis=0)
                        / month_weights.sum(),
                        np.nan,
                    )
                    output_index += 1
            if len(period_time_values) != (period_end - period_start + 1) * 12:
                raise RuntimeError(f"period-time-count-failed:{period_start}:{period_end}")
            steps = np.diff(np.asarray(period_time_values)) / 86400.0
            if not np.all(steps > 0) or float(steps.min()) < 20 or float(steps.max()) > 32:
                raise RuntimeError(
                    f"period-time-steps-failed:{period_start}:{float(steps.min())}:{float(steps.max())}"
                )
            all_time_values.extend(period_time_values)
    finally:
        source.close()

    if output_index != len(ALLOWED_YEARS):
        raise RuntimeError(f"annual-output-count-failed:{output_index}:{len(ALLOWED_YEARS)}")
    lat_subset = lat[lat_indices]
    lon_subset = lon[lon_indices]
    lat_order = np.argsort(lat_subset)
    lon_order = np.argsort(lon_subset)
    annual = annual[:, lat_order, :][:, :, lon_order]
    lat_subset = lat_subset[lat_order]
    lon_subset = lon_subset[lon_order]
    with h5py.File(paths["mode_ra_paleo"], "r") as handle:
        ra_lat = np.asarray(handle["lat"], dtype=float)
        ra_lon = np.asarray(handle["lon"], dtype=float)
    if not np.array_equal(lat_subset, ra_lat) or not np.array_equal(lon_subset, ra_lon):
        raise RuntimeError("mode-sim-mode-ra-grid-conflict")
    baseline = (ALLOWED_YEARS >= 1961) & (ALLOWED_YEARS <= 1990)
    center = np.mean(annual[baseline], axis=0)
    recentered = annual - center[None]
    closure = float(np.max(np.abs(np.mean(recentered[baseline], axis=0))))
    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "ModE-Sim ensemble-mean allowed-period HMA temperature"
        handle.attrs["source_url"] = SOURCE_URL
        handle.attrs["matching_mode_ra_configuration"] = "lowres_20mem_Set_1420-3_1850-1"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.create_dataset("time", data=ALLOWED_YEARS.astype(np.int16))
        handle.create_dataset("lat", data=lat_subset)
        handle.create_dataset("lon", data=lon_subset)
        handle.create_dataset(
            "temp2_anomaly_source_baseline",
            data=annual.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["temp2_anomaly_source_baseline"].attrs["units"] = "degC"
        handle["temp2_anomaly_source_baseline"].attrs["reference_period"] = "1901-2000"
        handle.create_dataset(
            "temp2_anomaly_1961_1990",
            data=recentered.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["temp2_anomaly_1961_1990"].attrs["units"] = "degC"
        handle["temp2_anomaly_1961_1990"].attrs["reference_period"] = "1961-1990"

    with (output / "http_range_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(request_log[0]))
        writer.writeheader()
        writer.writerows(request_log)
    request_bytes = sum(int(row["byte_count"]) for row in request_log)
    gates = {
        "all_http_responses_are_exact_206_ranges": all(row["http_status"] == 206 for row in request_log),
        "no_requested_range_overlaps_1921_1939": all(not row["locked_period_overlap"] for row in request_log),
        "output_axis_is_exact_allowed_periods": np.array_equal(ALLOWED_YEARS, np.concatenate((np.arange(1421, 1921), np.arange(1940, 2001)))),
        "annual_field_is_fully_finite": bool(np.all(np.isfinite(annual))),
        "mode_sim_and_mode_ra_grids_identical": True,
        "recentered_1961_1990_closure_below_1e_6": closure < 1e-6,
    }
    summary = {
        "status": "complete",
        "source_product": "ModE-Sim lowres 20-member ensemble-mean temp2",
        "allowed_periods": [list(period) for period in PERIODS],
        "locked_1921_1939_accessed": False,
        "shape": list(annual.shape),
        "finite_fraction": float(np.isfinite(annual).mean()),
        "recentered_1961_1990_max_abs_mean": closure,
        "http_range_request_count": len(request_log),
        "http_range_bytes_read": request_bytes,
        "gates": gates,
        "passes_acquisition_preflight": bool(all(gates.values())),
        "decision": (
            "advance_to_mode_ra_prior_assimilation_decomposition"
            if all(gates.values())
            else "stop_mode_sim_branch_on_acquisition_conflict"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-Sim Allowed Temperature Acquisition",
                "",
                f"- Allowed periods/shape: `{PERIODS}` / `{annual.shape}`.",
                f"- Exact requests/bytes: `{len(request_log)}` / `{request_bytes}`.",
                f"- Finite fraction and recenter closure: `{np.isfinite(annual).mean():.3%}` / `{closure:.3e}` degC.",
                "- Locked period accessed: `false`.",
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
                    "url": SOURCE_URL,
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
        raise RuntimeError("mode-sim-acquisition-preflight-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
