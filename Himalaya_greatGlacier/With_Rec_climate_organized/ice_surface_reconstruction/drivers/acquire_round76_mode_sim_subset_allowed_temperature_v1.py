"""Acquire allowed annual HMA temperature from three ModE-Sim historical sets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import mode_sim_cdf2_parser_v1 as cdf2


HEADER_BYTES = 65536
BLOCK_YEARS = 25
HISTORICAL_START = 1421
HISTORICAL_END = 1849
MODERN_START = 1940
MODERN_END = 2000
OVERVIEW_URL = "https://mode-ra.unibe.ch/data/ModE-Sim/docu/ModE-Sim_overview.csv"
BASE_URL = "https://mode-ra.unibe.ch/data/ModE-Sim/outdata"
SOURCES = {
    "set_1420_1": (
        f"{BASE_URL}/set_1420-1/abs/ensstat/by_var/year/"
        "ModE-Sim_set_1420-1_ensmean_temp2_abs_1420-1849_year.nc",
        1420,
        HISTORICAL_START,
        HISTORICAL_END,
    ),
    "set_1420_2": (
        f"{BASE_URL}/set_1420-2/abs/ensstat/by_var/year/"
        "ModE-Sim_set_1420-2_ensmean_temp2_abs_1420-1849_year.nc",
        1420,
        HISTORICAL_START,
        HISTORICAL_END,
    ),
    "set_1420_3": (
        f"{BASE_URL}/set_1420-3/abs/ensstat/by_var/year/"
        "ModE-Sim_set_1420-3_ensmean_temp2_abs_1420-1849_year.nc",
        1420,
        HISTORICAL_START,
        HISTORICAL_END,
    ),
    "set_1850_1_modern": (
        f"{BASE_URL}/set_1850-1/abs/ensstat/by_var/year/"
        "ModE-Sim_set_1850-1_ensmean_temp2_abs_1850-2009_year.nc",
        1850,
        MODERN_START,
        MODERN_END,
    ),
}


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


def text_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def acquire_source(
    label: str,
    url: str,
    source_start: int,
    selected_start: int,
    selected_end: int,
    request_log: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source = access.ExactRangeSource(url, label, request_log)
    try:
        header_payload = source.read(
            0,
            HEADER_BYTES - 1,
            "cdf2_header_and_safe_leading_records",
            (source_start, source_start),
        )
        header = cdf2.parse_header(header_payload)
        lat_var = access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = access.coordinate_variable(header, ("lon", "longitude", "x"))
        time_var = access.coordinate_variable(header, ("time", "t"))
        if "temp2" not in header.variables:
            raise RuntimeError(f"temp2-not-found:{label}:{list(header.variables)}")
        data_var = header.variables["temp2"]
        if time_var not in header.record_variables or data_var not in header.record_variables:
            raise RuntimeError(f"time-or-temp2-not-record-variable:{label}")
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
            raise RuntimeError(f"unexpected-spatial-dimensions:{label}:{spatial_dims}")
        years = np.arange(selected_start, selected_end + 1)
        fields = np.full((len(years), len(lat_indices), len(lon_indices)), np.nan)
        decoded_years: list[int] = []
        time_units = text_attr(time_var.attributes.get("units", ""))
        time_calendar = text_attr(time_var.attributes.get("calendar", "proleptic_gregorian"))
        for block_start in range(selected_start, selected_end + 1, BLOCK_YEARS):
            block_end = min(block_start + BLOCK_YEARS - 1, selected_end)
            first_record = block_start - source_start
            last_record = block_end - source_start
            if first_record < 0 or last_record >= header.num_records:
                raise RuntimeError(f"record-range-failed:{label}:{first_record}:{last_record}")
            byte_start = header.record_base + first_record * header.record_size
            byte_end = header.record_base + (last_record + 1) * header.record_size - 1
            payload = source.read(
                byte_start,
                byte_end,
                "complete_allowed_annual_temperature_records",
                (block_start, block_end),
            )
            for year in range(block_start, block_end + 1):
                local_record = year - block_start
                time_raw = access.record_segment(payload, header, time_var, local_record)
                time_value = access.decode_numeric(
                    time_raw, time_var, header.shape(time_var)[1:] or (1,)
                ).reshape(-1)[0]
                date = access.num2date(float(time_value), time_units, time_calendar)
                decoded_years.append(date.year)
                if date.year != year:
                    raise RuntimeError(f"decoded-year-failed:{label}:{year}:{date.isoformat()}")
                data_raw = access.record_segment(payload, header, data_var, local_record)
                field = access.decode_numeric(data_raw, data_var, header.shape(data_var)[1:])
                if (lat_axis, lon_axis) != (0, 1):
                    field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                fields[year - selected_start] = field[np.ix_(lat_indices, lon_indices)]
        lat_subset = lat[lat_indices]
        lon_subset = lon[lon_indices]
        lat_order = np.argsort(lat_subset)
        lon_order = np.argsort(lon_subset)
        fields = fields[:, lat_order, :][:, :, lon_order]
        metadata = {
            "url": url,
            "content_length": source.content_length,
            "etag": source.etag,
            "last_modified": source.last_modified,
            "container": "CDF2",
            "record_count": header.num_records,
            "record_size": header.record_size,
            "units": text_attr(data_var.attributes.get("units", "")),
            "time_units": time_units,
            "time_calendar": time_calendar,
            "decoded_years_exact": decoded_years == years.tolist(),
        }
        return years, lat_subset[lat_order], lon_subset[lon_order], fields, metadata
    finally:
        source.close()


def summarize_overview() -> tuple[str, dict[str, Any]]:
    response = requests.get(OVERVIEW_URL, timeout=120)
    response.raise_for_status()
    text = response.text
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    result: dict[str, Any] = {}
    for set_name in ("1420-1", "1420-2", "1420-3"):
        members = [
            row
            for row in rows
            if row.get("final_set") == set_name and row.get("final_member", "").startswith("m")
        ]
        result[set_name] = {
            "unique_members": len({row["final_member"] for row in members}),
            "sst_forcing_types": sorted({row["sst_forc_type"] for row in members}),
            "sic_forcing_types": sorted({row["sic_forc_type"] for row in members}),
            "volcanic_forcing_types": sorted({row["volc_forc_type"] for row in members}),
        }
    return text, result


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_76_mode_sim_subset_allowed_temperature_acquisition_v1"
    round74 = loop / "round_74_mode_sim_allowed_temperature_acquisition_v3"
    paths = {
        "round74_field": round74 / "mode_sim_temp2_allowed_hma_bbox.nc",
        "round74_summary": round74 / "summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
        "cdf1_access": Path(access.__file__).resolve(),
        "cdf2_parser": Path(cdf2.__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "mode_sim_three_historical_sets_allowed_hma_bbox.nc"
    if target.exists() or (output / "summary.json").exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    if not json.loads(paths["round74_summary"].read_text(encoding="utf-8"))[
        "passes_acquisition_preflight"
    ]:
        raise RuntimeError("round74-did-not-pass")

    overview_text, overview = summarize_overview()
    (output / "mode_sim_overview_source_snapshot.csv").write_text(
        overview_text, encoding="utf-8"
    )
    request_log: list[dict[str, Any]] = []
    acquired: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for label, (url, source_start, selected_start, selected_end) in SOURCES.items():
        acquired[label] = acquire_source(
            label,
            url,
            source_start,
            selected_start,
            selected_end,
            request_log,
        )
    first = acquired["set_1420_1"]
    historical_years, lat, lon = first[0], first[1], first[2]
    for label, result in acquired.items():
        if not np.array_equal(result[1], lat) or not np.array_equal(result[2], lon):
            raise RuntimeError(f"grid-conflict:{label}")
    modern_years, _, _, modern_absolute, _ = acquired["set_1850_1_modern"]
    baseline = (modern_years >= 1961) & (modern_years <= 1990)
    if int(baseline.sum()) != 30:
        raise RuntimeError("modern-baseline-count-failed")
    baseline_grid = np.mean(modern_absolute[baseline], axis=0)
    anomaly_fields = {
        label: acquired[label][3] - baseline_grid[None]
        for label in ("set_1420_1", "set_1420_2", "set_1420_3")
    }
    modern_anomaly = modern_absolute - baseline_grid[None]

    with h5py.File(paths["round74_field"], "r") as handle:
        round74_years = np.asarray(handle["time"], dtype=int)
        round74_lat = np.asarray(handle["lat"], dtype=float)
        round74_lon = np.asarray(handle["lon"], dtype=float)
        round74_field = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)
    selected_round74 = (round74_years >= HISTORICAL_START) & (
        round74_years <= HISTORICAL_END
    )
    if not np.array_equal(round74_years[selected_round74], historical_years):
        raise RuntimeError("round74-common-year-axis-failed")
    if not np.array_equal(round74_lat, lat) or not np.array_equal(round74_lon, lon):
        raise RuntimeError("round74-grid-conflict")
    set3_round74_max_abs = float(
        np.max(np.abs(anomaly_fields["set_1420_3"] - round74_field[selected_round74]))
    )
    set3_round74_rmse = float(
        np.sqrt(
            np.mean(
                (anomaly_fields["set_1420_3"] - round74_field[selected_round74]) ** 2
            )
        )
    )

    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "Allowed annual ModE-Sim historical-set HMA temperature"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.attrs["baseline"] = "set_1850-1 grid-cell mean, 1961-1990"
        handle.create_dataset("historical_year", data=historical_years.astype(np.int16))
        handle.create_dataset("modern_year", data=modern_years.astype(np.int16))
        handle.create_dataset("lat", data=lat.astype(np.float64))
        handle.create_dataset("lon", data=lon.astype(np.float64))
        handle.create_dataset("temp2_1961_1990_grid_center_K", data=baseline_grid.astype(np.float32))
        handle.create_dataset(
            "set_1420_1_temp2_anomaly_1961_1990",
            data=anomaly_fields["set_1420_1"].astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        handle.create_dataset(
            "set_1420_2_temp2_anomaly_1961_1990",
            data=anomaly_fields["set_1420_2"].astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        handle.create_dataset(
            "set_1420_3_temp2_anomaly_1961_1990",
            data=anomaly_fields["set_1420_3"].astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        handle.create_dataset(
            "set_1850_1_modern_temp2_anomaly_1961_1990",
            data=modern_anomaly.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
    with (output / "http_range_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(request_log[0]))
        writer.writeheader()
        writer.writerows(request_log)

    gates = {
        "all_http_responses_are_exact_206_ranges": all(
            row["http_status"] == 206 for row in request_log
        ),
        "no_requested_range_overlaps_1921_1939": all(
            not row["locked_period_overlap"] for row in request_log
        ),
        "historical_axis_is_exact_1421_1849": np.array_equal(
            historical_years, np.arange(1421, 1850)
        ),
        "modern_axis_is_exact_1940_2000": np.array_equal(
            modern_years, np.arange(1940, 2001)
        ),
        "all_fields_are_fully_finite": all(
            np.all(np.isfinite(field)) for field in [*anomaly_fields.values(), modern_anomaly]
        ),
        "all_grids_are_identical": True,
        "modern_1961_1990_closure_below_1e_6": float(
            np.max(np.abs(np.mean(modern_anomaly[baseline], axis=0)))
        )
        <= 1e-6,
        "set3_round74_max_abs_closure_below_0_01_degC": set3_round74_max_abs <= 0.01,
        "official_overview_has_20_members_per_set": all(
            overview[name]["unique_members"] == 20 for name in overview
        ),
        "official_forcing_metadata_matches_contract": (
            overview["1420-1"]["volcanic_forcing_types"] == ["PMIP4"]
            and overview["1420-2"]["volcanic_forcing_types"] == ["EVA"]
            and overview["1420-3"]["volcanic_forcing_types"] == ["PMIP4"]
        ),
    }
    summary = {
        "status": "complete",
        "historical_years": [HISTORICAL_START, HISTORICAL_END],
        "modern_baseline_source_years": [MODERN_START, MODERN_END],
        "locked_1921_1939_accessed": False,
        "shape_historical": list(anomaly_fields["set_1420_1"].shape),
        "shape_modern": list(modern_anomaly.shape),
        "source_metadata": {label: value[4] for label, value in acquired.items()},
        "official_experiment_overview": overview,
        "set3_round74_max_abs_closure_degC": set3_round74_max_abs,
        "set3_round74_rmse_closure_degC": set3_round74_rmse,
        "http_range_request_count": len(request_log),
        "http_range_bytes_read": int(sum(row["byte_count"] for row in request_log)),
        "gates": gates,
        "passes_acquisition_preflight": bool(all(gates.values())),
        "decision": (
            "advance_to_historical_set_robustness_adjudication"
            if all(gates.values())
            else "stop_before_historical_set_comparison"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-Sim Historical-Set Acquisition",
                "",
                f"- Historical sets: `3 x {len(historical_years)} x {len(lat)} x {len(lon)}`.",
                f"- Modern baseline: `{len(modern_years)} x {len(lat)} x {len(lon)}`.",
                f"- HTTP ranges: `{len(request_log)}` requests, `{sum(row['byte_count'] for row in request_log):,}` bytes.",
                f"- Set 1420-3 maximum/RMSE closure to Round 74: `{set3_round74_max_abs:.6f}` / `{set3_round74_rmse:.6f}` degC.",
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
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    if not summary["passes_acquisition_preflight"]:
        raise RuntimeError("acquisition-preflight-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
