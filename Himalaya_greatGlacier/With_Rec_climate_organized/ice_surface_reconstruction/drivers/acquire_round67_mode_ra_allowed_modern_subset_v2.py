"""Acquire an allowed-period HMA subset from official ModE-RA classic NetCDF files.

The remote files span the locked 1921-1939 interval.  This driver therefore
uses validated HTTP byte ranges over complete 1940-2000 records and never
opens or downloads either source as a whole.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests
from datetime import datetime, timedelta


START_YEAR = 1421
MODERN_START = 1940
MODERN_END = 2000
LOCKED_START = 1921
LOCKED_END = 1939
HEADER_BYTES = 65536
HMA_BBOX = {"west": 55.0, "east": 116.0, "south": 18.0, "north": 51.0}
BASE_URL = (
    "https://mode-ra.unibe.ch/data/ModE-RA/outdata/"
    "lowres_20mem_Set_1420-3_1850-1/anom_wrt_1901-2000/"
    "ensstat/by_var/mon"
)
SOURCES = {
    "temp2": (
        f"{BASE_URL}/ModE-RA_lowres_20mem_Set_1420-3_1850-1_ensmean_"
        "temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
    ),
    "totprec": (
        f"{BASE_URL}/ModE-RA_lowres_20mem_Set_1420-3_1850-1_ensmean_"
        "totprec_anom_wrt_1901-2000_1421-2008_mon.nc"
    ),
}
TYPE_DTYPES = {
    1: np.dtype("i1"),
    2: np.dtype("S1"),
    3: np.dtype(">i2"),
    4: np.dtype(">i4"),
    5: np.dtype(">f4"),
    6: np.dtype(">f8"),
}
TYPE_SIZES = {code: dtype.itemsize for code, dtype in TYPE_DTYPES.items()}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def file_state(path: Path) -> dict[str, Any]:
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


class Cursor:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def take(self, count: int) -> bytes:
        end = self.offset + count
        if end > len(self.payload):
            raise EOFError(f"header-too-short:{end}>{len(self.payload)}")
        result = self.payload[self.offset:end]
        self.offset = end
        return result

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def name(self) -> str:
        count = self.u32()
        raw = self.take(count)
        self.take((-count) % 4)
        return raw.decode("utf-8")


def decode_values(raw: bytes, type_code: int, count: int) -> Any:
    if type_code == 2:
        return raw[:count].decode("utf-8", errors="replace").rstrip("\x00")
    values = np.frombuffer(raw[: count * TYPE_SIZES[type_code]], dtype=TYPE_DTYPES[type_code])
    native = values.astype(values.dtype.newbyteorder("="), copy=False)
    if count == 1:
        return native[0].item()
    return native.tolist()


def parse_attributes(cursor: Cursor) -> dict[str, Any]:
    tag = cursor.u32()
    count = cursor.u32()
    if tag == 0 and count == 0:
        return {}
    if tag != 12:
        raise RuntimeError(f"unexpected-attribute-tag:{tag}")
    result: dict[str, Any] = {}
    for _ in range(count):
        name = cursor.name()
        type_code = cursor.u32()
        value_count = cursor.u32()
        if type_code not in TYPE_SIZES:
            raise RuntimeError(f"unsupported-attribute-type:{type_code}")
        byte_count = value_count * TYPE_SIZES[type_code]
        raw = cursor.take(byte_count)
        cursor.take((-byte_count) % 4)
        result[name] = decode_values(raw, type_code, value_count)
    return result


@dataclass(frozen=True)
class Variable:
    name: str
    dimensions: tuple[str, ...]
    attributes: dict[str, Any]
    type_code: int
    vsize: int
    begin: int


@dataclass(frozen=True)
class ClassicHeader:
    num_records: int
    dimensions: dict[str, int | None]
    global_attributes: dict[str, Any]
    variables: dict[str, Variable]
    parsed_bytes: int

    def shape(self, variable: Variable) -> tuple[int, ...]:
        return tuple(
            self.num_records if self.dimensions[name] is None else int(self.dimensions[name])
            for name in variable.dimensions
        )

    @property
    def record_variables(self) -> list[Variable]:
        return [
            variable
            for variable in self.variables.values()
            if variable.dimensions and self.dimensions[variable.dimensions[0]] is None
        ]

    @property
    def record_size(self) -> int:
        variables = self.record_variables
        if len(variables) == 1 and variables[0].type_code in (1, 2, 3):
            shape = self.shape(variables[0])[1:]
            return math.prod(shape) * TYPE_SIZES[variables[0].type_code]
        return sum(variable.vsize for variable in variables)

    @property
    def record_base(self) -> int:
        return min(variable.begin for variable in self.record_variables)


def parse_header(payload: bytes) -> ClassicHeader:
    cursor = Cursor(payload)
    if cursor.take(4) != b"CDF\x01":
        raise RuntimeError("source-is-not-classic-netcdf-cdf1")
    num_records = cursor.u32()
    dim_tag = cursor.u32()
    dim_count = cursor.u32()
    if dim_tag == 0 and dim_count == 0:
        dimensions: dict[str, int | None] = {}
    elif dim_tag == 10:
        dimensions = {}
        for _ in range(dim_count):
            name = cursor.name()
            size = cursor.u32()
            dimensions[name] = None if size == 0 else size
    else:
        raise RuntimeError(f"unexpected-dimension-tag:{dim_tag}")
    global_attributes = parse_attributes(cursor)
    var_tag = cursor.u32()
    var_count = cursor.u32()
    if var_tag != 11:
        raise RuntimeError(f"unexpected-variable-tag:{var_tag}")
    dim_names = list(dimensions)
    variables: dict[str, Variable] = {}
    for _ in range(var_count):
        name = cursor.name()
        dimension_count = cursor.u32()
        dim_ids = [cursor.u32() for _ in range(dimension_count)]
        attributes = parse_attributes(cursor)
        type_code = cursor.u32()
        vsize = cursor.u32()
        begin = cursor.u32()
        variables[name] = Variable(
            name=name,
            dimensions=tuple(dim_names[index] for index in dim_ids),
            attributes=attributes,
            type_code=type_code,
            vsize=vsize,
            begin=begin,
        )
    return ClassicHeader(
        num_records=num_records,
        dimensions=dimensions,
        global_attributes=global_attributes,
        variables=variables,
        parsed_bytes=cursor.offset,
    )


class ExactRangeSource:
    def __init__(self, url: str, label: str, log: list[dict[str, Any]]):
        self.url = url
        self.label = label
        self.log = log
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "LIA-HMA-ModE-RA-allowed-subset-audit/1.0"})
        head = self.session.head(url, allow_redirects=True, timeout=120)
        head.raise_for_status()
        self.content_length = int(head.headers["Content-Length"])
        self.etag = head.headers.get("ETag", "")
        self.last_modified = head.headers.get("Last-Modified", "")

    def read(self, start: int, end: int, purpose: str, years: tuple[int, int] | None) -> bytes:
        if start < 0 or end < start or end >= self.content_length:
            raise RuntimeError(f"invalid-byte-range:{self.label}:{start}:{end}")
        if years is not None:
            first, last = years
            if first <= LOCKED_END and last >= LOCKED_START:
                raise RuntimeError(f"locked-year-range-request-blocked:{self.label}:{first}:{last}")
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=180,
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(f"range-status-not-206:{self.label}:{response.status_code}")
            expected_header = f"bytes {start}-{end}/{self.content_length}"
            if response.headers.get("Content-Range") != expected_header:
                raise RuntimeError(
                    f"content-range-mismatch:{self.label}:"
                    f"{response.headers.get('Content-Range')}:{expected_header}"
                )
            expected_size = end - start + 1
            payload = response.raw.read(expected_size + 1)
            if len(payload) != expected_size:
                raise RuntimeError(
                    f"range-size-mismatch:{self.label}:{len(payload)}:{expected_size}"
                )
        finally:
            response.close()
        self.log.append(
            {
                "source": self.label,
                "purpose": purpose,
                "byte_start": start,
                "byte_end": end,
                "byte_count": end - start + 1,
                "first_year": "" if years is None else years[0],
                "last_year": "" if years is None else years[1],
                "locked_period_overlap": False,
                "http_status": 206,
            }
        )
        return payload

    def close(self) -> None:
        self.session.close()


def scalar_attribute(variable: Variable, name: str, default: float) -> float:
    value = variable.attributes.get(name, default)
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if len(array) else default


def decode_numeric(raw: bytes, variable: Variable, shape: tuple[int, ...]) -> np.ndarray:
    dtype = TYPE_DTYPES[variable.type_code]
    count = math.prod(shape)
    expected = count * dtype.itemsize
    if len(raw) != expected:
        raise RuntimeError(f"variable-byte-count-failed:{variable.name}:{len(raw)}:{expected}")
    values = np.frombuffer(raw, dtype=dtype, count=count).astype(float).reshape(shape)
    fill_values = []
    for key in ("_FillValue", "missing_value"):
        if key in variable.attributes:
            fill_values.extend(np.asarray(variable.attributes[key], dtype=float).reshape(-1).tolist())
    for fill in fill_values:
        values[np.isclose(values, fill)] = np.nan
    values = values * scalar_attribute(variable, "scale_factor", 1.0)
    values = values + scalar_attribute(variable, "add_offset", 0.0)
    return values


def coordinate_variable(header: ClassicHeader, candidates: tuple[str, ...]) -> Variable:
    lower = {name.lower(): variable for name, variable in header.variables.items()}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    for variable in header.variables.values():
        standard = str(variable.attributes.get("standard_name", "")).lower()
        axis = str(variable.attributes.get("axis", "")).lower()
        if any(candidate in standard for candidate in candidates) or axis in candidates:
            return variable
    raise RuntimeError(f"coordinate-variable-not-found:{candidates}")


def read_fixed_variable(source: ExactRangeSource, header: ClassicHeader, variable: Variable) -> np.ndarray:
    if variable in header.record_variables:
        raise RuntimeError(f"expected-fixed-variable:{variable.name}")
    shape = header.shape(variable)
    byte_count = math.prod(shape) * TYPE_SIZES[variable.type_code]
    raw = source.read(
        variable.begin,
        variable.begin + byte_count - 1,
        f"coordinate_{variable.name}",
        None,
    )
    return decode_numeric(raw, variable, shape).reshape(-1)


def record_segment(payload: bytes, header: ClassicHeader, variable: Variable, record: int) -> bytes:
    offset = variable.begin - header.record_base + record * header.record_size
    shape = header.shape(variable)[1:]
    byte_count = math.prod(shape) * TYPE_SIZES[variable.type_code]
    return payload[offset : offset + byte_count]


def num2date(value: float, units: str, calendar: str = "standard") -> datetime:
    if calendar.lower() not in ("standard", "gregorian", "proleptic_gregorian"):
        raise RuntimeError(f"unsupported-time-calendar:{calendar}")
    match = re.match(
        r"^\s*(seconds|minutes|hours|days)\s+since\s+"
        r"(\d{1,4})-(\d{1,2})-(\d{1,2})"
        r"(?:[ T](\d{1,2})(?::(\d{1,2})(?::([0-9.]+))?)?)?",
        units,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError(f"unsupported-time-units:{units}")
    unit, year, month, day, hour, minute, second = match.groups()
    second_value = float(second or 0.0)
    origin = datetime(
        int(year), int(month), int(day), int(hour or 0), int(minute or 0),
        int(second_value), int(round((second_value % 1.0) * 1_000_000)),
    )
    scale = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}[unit.lower()]
    return origin + timedelta(seconds=float(value) * scale)

def normalize_longitude(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def acquire_variable(
    source_name: str,
    url: str,
    request_log: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source = ExactRangeSource(url, source_name, request_log)
    try:
        header_payload = source.read(
            0,
            HEADER_BYTES - 1,
            "classic_netcdf_header_and_allowed_1421_prefix",
            (1421, 1421),
        )
        header = parse_header(header_payload)
        if header.parsed_bytes >= HEADER_BYTES:
            raise RuntimeError(f"header-buffer-exhausted:{source_name}:{header.parsed_bytes}")
        lat_var = coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = coordinate_variable(header, ("lon", "longitude", "x"))
        time_var = coordinate_variable(header, ("time", "t"))
        if source_name not in header.variables:
            raise RuntimeError(f"data-variable-not-found:{source_name}:{list(header.variables)}")
        data_var = header.variables[source_name]
        if not data_var.dimensions or header.dimensions[data_var.dimensions[0]] is not None:
            raise RuntimeError(f"data-variable-is-not-record-variable:{source_name}:{data_var.dimensions}")
        if time_var not in header.record_variables:
            raise RuntimeError(f"time-variable-is-not-record-variable:{time_var.name}")
        lat = read_fixed_variable(source, header, lat_var)
        lon_native = read_fixed_variable(source, header, lon_var)
        lon = normalize_longitude(lon_native)
        lat_indices = np.flatnonzero((lat >= HMA_BBOX["south"]) & (lat <= HMA_BBOX["north"]))
        lon_indices = np.flatnonzero((lon >= HMA_BBOX["west"]) & (lon <= HMA_BBOX["east"]))
        if len(lat_indices) < 2 or len(lon_indices) < 2:
            raise RuntimeError(f"hma-bbox-support-failed:{source_name}:{len(lat_indices)}:{len(lon_indices)}")
        spatial_dims = list(data_var.dimensions[1:])
        lat_axis = spatial_dims.index(lat_var.dimensions[0])
        lon_axis = spatial_dims.index(lon_var.dimensions[0])
        if len(spatial_dims) != 2 or lat_axis == lon_axis:
            raise RuntimeError(f"unexpected-spatial-dimensions:{source_name}:{spatial_dims}")

        start_record = (MODERN_START - START_YEAR) * 12
        end_record = (MODERN_END - START_YEAR + 1) * 12 - 1
        if start_record < 0 or end_record >= header.num_records:
            raise RuntimeError(f"modern-record-range-failed:{source_name}:{start_record}:{end_record}")
        annual = np.full((MODERN_END - MODERN_START + 1, len(lat_indices), len(lon_indices)), np.nan)
        decoded_months: list[tuple[int, int]] = []
        time_units = str(time_var.attributes.get("units", ""))
        time_calendar = str(time_var.attributes.get("calendar", "standard"))
        for year in range(MODERN_START, MODERN_END + 1):
            first_record = (year - START_YEAR) * 12
            last_record = first_record + 11
            byte_start = header.record_base + first_record * header.record_size
            byte_end = header.record_base + (last_record + 1) * header.record_size - 1
            payload = source.read(byte_start, byte_end, "complete_modern_year_records", (year, year))
            monthly = []
            dates = []
            for local_record in range(12):
                time_raw = record_segment(payload, header, time_var, local_record)
                time_value = decode_numeric(time_raw, time_var, header.shape(time_var)[1:] or (1,)).reshape(-1)[0]
                date = num2date(time_value, units=time_units, calendar=time_calendar)
                dates.append(date)
                data_raw = record_segment(payload, header, data_var, local_record)
                field = decode_numeric(data_raw, data_var, header.shape(data_var)[1:])
                if (lat_axis, lon_axis) != (0, 1):
                    field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                monthly.append(field[np.ix_(lat_indices, lon_indices)])
            actual = [(int(date.year), int(date.month)) for date in dates]
            expected = [(year, month) for month in range(1, 13)]
            if actual != expected:
                raise RuntimeError(f"monthly-axis-failed:{source_name}:{year}:{actual}")
            decoded_months.extend(actual)
            month_stack = np.stack(monthly)
            weights = np.asarray([calendar.monthrange(year, month)[1] for month in range(1, 13)], dtype=float)
            complete = np.all(np.isfinite(month_stack), axis=0)
            annual[year - MODERN_START] = np.where(
                complete,
                np.sum(month_stack * weights[:, None, None], axis=0) / np.sum(weights),
                np.nan,
            )
        expected_months = [
            (year, month)
            for year in range(MODERN_START, MODERN_END + 1)
            for month in range(1, 13)
        ]
        if decoded_months != expected_months:
            raise RuntimeError(f"combined-month-axis-failed:{source_name}")
        if source_name == "totprec":
            units = str(data_var.attributes.get("units", ""))
            if units not in ("kg m-2 s-1", "kg m**-2 s**-1", "kg/m2/s"):
                raise RuntimeError(f"unexpected-totprec-units:{units}")
            annual *= 86400.0
            output_units = "mm/day"
        else:
            units = str(data_var.attributes.get("units", ""))
            if units not in ("K", "degK", "kelvin"):
                raise RuntimeError(f"unexpected-temp2-units:{units}")
            output_units = "degC anomaly"
        lat_subset = lat[lat_indices]
        lon_subset = lon[lon_indices]
        lat_order = np.argsort(lat_subset)
        lon_order = np.argsort(lon_subset)
        annual = annual[:, lat_order, :][:, :, lon_order]
        lat_subset = lat_subset[lat_order]
        lon_subset = lon_subset[lon_order]
        metadata = {
            "url": url,
            "content_length": source.content_length,
            "etag": source.etag,
            "last_modified": source.last_modified,
            "cdf_version": 1,
            "header_parsed_bytes": header.parsed_bytes,
            "num_records": header.num_records,
            "dimensions": header.dimensions,
            "variables": {
                name: {
                    "dimensions": variable.dimensions,
                    "shape": header.shape(variable),
                    "type_code": variable.type_code,
                    "vsize": variable.vsize,
                    "begin": variable.begin,
                    "attributes": variable.attributes,
                }
                for name, variable in header.variables.items()
            },
            "record_size": header.record_size,
            "record_base": header.record_base,
            "modern_record_indices": [start_record, end_record],
            "native_units": units,
            "output_units": output_units,
            "finite_fraction": float(np.isfinite(annual).mean()),
        }
        return annual, lat_subset, lon_subset, metadata
    finally:
        source.close()


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    contract = output / "ROUND_CONTRACT.md"
    driver = Path(__file__).resolve()
    for path in (output, contract, driver):
        if not path.exists():
            raise FileNotFoundError(path)
    for name in ("summary.json", "mode_ra_ensmean_1940_2000_hma_bbox.nc"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    before = {"contract": file_state(contract), "driver": file_state(driver)}
    request_log: list[dict[str, Any]] = []
    temp, temp_lat, temp_lon, temp_metadata = acquire_variable("temp2", SOURCES["temp2"], request_log)
    precip, precip_lat, precip_lon, precip_metadata = acquire_variable(
        "totprec", SOURCES["totprec"], request_log
    )
    if not np.array_equal(temp_lat, precip_lat) or not np.array_equal(temp_lon, precip_lon):
        raise RuntimeError("temperature-precipitation-grid-conflict")
    years = np.arange(MODERN_START, MODERN_END + 1, dtype=np.int16)
    if np.any((years >= LOCKED_START) & (years <= LOCKED_END)):
        raise RuntimeError("locked-years-in-output-axis")

    target = output / "mode_ra_ensmean_1940_2000_hma_bbox.nc"
    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "ModE-RA ensemble-mean allowed-period HMA modern subset"
        handle.attrs["source_urls"] = json.dumps(SOURCES)
        handle.attrs["source_product"] = "ModE-RA lowres 20-member ensemble mean"
        handle.attrs["source_anomaly_baseline"] = "1901-2000"
        handle.attrs["access_method"] = "validated HTTP 206 complete-record byte ranges"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.create_dataset("time", data=years)
        handle["time"].attrs["units"] = "year CE"
        handle.create_dataset("lat", data=temp_lat.astype(np.float64))
        handle["lat"].attrs["units"] = "degrees_north"
        handle.create_dataset("lon", data=temp_lon.astype(np.float64))
        handle["lon"].attrs["units"] = "degrees_east"
        handle.create_dataset(
            "temp2_anomaly",
            data=temp.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["temp2_anomaly"].attrs["units"] = "degC"
        handle["temp2_anomaly"].attrs["source_variable"] = "temp2"
        handle.create_dataset(
            "totprec_anomaly",
            data=precip.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["totprec_anomaly"].attrs["units"] = "mm/day"
        handle["totprec_anomaly"].attrs["source_variable"] = "totprec"

    with (output / "http_range_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(request_log[0]))
        writer.writeheader()
        writer.writerows(request_log)
    (output / "remote_source_metadata.json").write_text(
        json.dumps({"temp2": temp_metadata, "totprec": precip_metadata}, indent=2),
        encoding="utf-8",
    )
    request_bytes = sum(int(row["byte_count"]) for row in request_log)
    gates = {
        "all_http_responses_are_exact_206_ranges": all(row["http_status"] == 206 for row in request_log),
        "no_requested_year_range_overlaps_1921_1939": all(not row["locked_period_overlap"] for row in request_log),
        "output_axis_is_exact_1940_2000": np.array_equal(years, np.arange(1940, 2001)),
        "temperature_and_precipitation_grids_identical": True,
        "temperature_finite_fraction_above_0_95": float(np.isfinite(temp).mean()) > 0.95,
        "precipitation_finite_fraction_above_0_95": float(np.isfinite(precip).mean()) > 0.95,
        "temperature_units_converted_to_degC_anomaly": temp_metadata["output_units"] == "degC anomaly",
        "precipitation_units_converted_to_mm_day_anomaly": precip_metadata["output_units"] == "mm/day",
    }
    summary = {
        "status": "complete",
        "source_product": "ModE-RA lowres 20-member ensemble mean",
        "source_anomaly_baseline": "1901-2000",
        "years": [MODERN_START, MODERN_END],
        "locked_1921_1939_accessed": False,
        "bbox": HMA_BBOX,
        "shape": list(temp.shape),
        "latitude_range": [float(temp_lat.min()), float(temp_lat.max())],
        "longitude_range": [float(temp_lon.min()), float(temp_lon.max())],
        "temperature_finite_fraction": float(np.isfinite(temp).mean()),
        "precipitation_finite_fraction": float(np.isfinite(precip).mean()),
        "http_range_request_count": len(request_log),
        "http_range_bytes_read": request_bytes,
        "full_remote_bytes": temp_metadata["content_length"] + precip_metadata["content_length"],
        "gates": gates,
        "passes_acquisition_preflight": bool(all(gates.values())),
        "decision": (
            "advance_mode_ra_to_independent_modern_validation"
            if all(gates.values())
            else "stop_mode_ra_branch_on_acquisition_conflict"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# ModE-RA Allowed Modern Subset Acquisition",
        "",
        f"- Years: `{MODERN_START}-{MODERN_END}`; locked period accessed: `false`.",
        f"- HMA-box shape: `{temp.shape}`.",
        f"- Exact HTTP range requests: `{len(request_log)}`; bytes read: `{request_bytes}`.",
        f"- Temperature/precipitation finite fractions: `{np.isfinite(temp).mean():.3%}` / `{np.isfinite(precip).mean():.3%}`.",
        f"- All acquisition gates pass: `{str(summary['passes_acquisition_preflight']).lower()}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {"contract": file_state(contract), "driver": file_state(driver)}
    unchanged = before == after
    output_state = file_state(target)
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "local_sources_before": before,
                "local_sources_after": after,
                "local_source_states_unchanged": unchanged,
                "remote_sources": {
                    name: {
                        "url": SOURCES[name],
                        "content_length": metadata["content_length"],
                        "etag": metadata["etag"],
                        "last_modified": metadata["last_modified"],
                    }
                    for name, metadata in (("temp2", temp_metadata), ("totprec", precip_metadata))
                },
                "output": output_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("local-source-state-changed")
    if not summary["passes_acquisition_preflight"]:
        raise RuntimeError("mode-ra-acquisition-preflight-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
