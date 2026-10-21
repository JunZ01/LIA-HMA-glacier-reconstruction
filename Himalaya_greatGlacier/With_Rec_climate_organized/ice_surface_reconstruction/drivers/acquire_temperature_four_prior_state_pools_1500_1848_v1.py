"""Acquire harmonized 1500-1848 tas/pr state pools for four-prior replay.

The script reads only the independent experiment inputs needed by the
strategy-C background selector. It never reads 1901-1939 model values and does
not run EnKF or modify any production reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1"
STATE_FILE = OUTPUT / "four_prior_state_pools_1500_1848_v1.npz"
YEARS = np.arange(1500, 1849, dtype=int)

ARCHIVE = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
PROXY_PATHS = {6: ARCHIVE / "python_data/proxy_data_6.mat", 9: ARCHIVE / "python_data/proxy_data_9.mat"}
CESM_T = RESULTS / "enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
CESM_P = RESULTS / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat"
CATALOG = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/remote_source_catalog.json"

DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import acquire_round67_mode_ra_allowed_modern_subset_v2 as range_access
import mode_sim_cdf2_parser_v1 as cdf2


MODE_BASE = "https://mode-ra.unibe.ch/data/ModE-Sim/outdata/set_1420-3/abs/ensstat/by_var/year"
MODE_URLS = {
    "tas": MODE_BASE + "/ModE-Sim_set_1420-3_ensmean_temp2_abs_1420-1849_year.nc",
    "pr": MODE_BASE + "/ModE-Sim_set_1420-3_ensmean_totprec_abs_1420-1849_year.nc",
}


def file_state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def native_coordinates() -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-89.0, 89.0, 90)
    lon = np.linspace(-179.0, 179.0, 180)
    return lat[(lat >= 24.0) & (lat <= 51.0)], lon[(lon >= 62.0) & (lon <= 110.0)]


def proxy_inventory() -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    all_points: list[tuple[float, float]] = []
    locations: dict[int, np.ndarray] = {}
    point_indices: dict[int, np.ndarray] = {}
    for network, path in PROXY_PATHS.items():
        with h5py.File(path, "r") as handle:
            refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
            loc = np.stack([np.asarray(handle[ref]["location"], dtype=float).reshape(-1, order="F") for ref in refs])
        locations[network] = loc
        all_points.extend((float(row[0]), float(row[1])) for row in loc)
    unique = np.asarray(sorted(set(all_points)), dtype=float)
    lookup = {tuple(row): index for index, row in enumerate(unique)}
    for network, loc in locations.items():
        point_indices[network] = np.asarray([lookup[tuple(row)] for row in loc], dtype=int)
    return unique, locations, point_indices


def target_points(proxy_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat, lon = native_coordinates()
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    state_points = np.column_stack([lat2.reshape(-1, order="F"), lon2.reshape(-1, order="F")])
    return np.vstack([state_points, proxy_points]), lat, lon


def interpolate_fields(fields: np.ndarray, lat: np.ndarray, lon: np.ndarray, points: np.ndarray) -> np.ndarray:
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat = lat[lat_order]
    lon = np.mod(lon[lon_order], 360.0)
    lon_order2 = np.argsort(lon)
    lon = lon[lon_order2]
    fields = fields[:, lat_order][:, :, lon_order][:, :, lon_order2]
    query = points.copy()
    query[:, 1] = np.mod(query[:, 1], 360.0)
    output = np.empty((len(fields), len(points)), dtype=np.float64)
    for index, field in enumerate(fields):
        output[index] = RegularGridInterpolator((lat, lon), field, bounds_error=False, fill_value=np.nan)(query)
    if not np.all(np.isfinite(output)):
        count = int(np.size(output) - np.isfinite(output).sum())
        raise RuntimeError(f"nonfinite-interpolated-values:{count}")
    return output


def precip_to_mm_day(values: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    normalized = units.lower().replace("**", "^").replace(" ", "")
    if ("kg" in normalized and ("s-1" in normalized or "/s" in normalized)) or normalized in {"kgm-2s-1", "kg/m2/s"}:
        return values * 86400.0, "kg m-2 s-1_to_mm_day"
    if "mm/day" in normalized or "mmd-1" in normalized or "mmday-1" in normalized:
        return values, "already_mm_day"
    if normalized in {"mm", "kgm-2", "kg/m2"}:
        return values / 365.2425, "annual_total_to_mm_day"
    raise RuntimeError(f"unsupported-precipitation-units:{units}")


def mode_annual(variable: str, points: np.ndarray, request_log: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    url = MODE_URLS[variable]
    label = f"mode_sim_set_1420_3_{variable}"
    source = range_access.ExactRangeSource(url, label, request_log)
    try:
        header = cdf2.parse_header(source.read(0, 65535, "cdf2_header", (1420, 1420)))
        lat_name = range_access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_name = range_access.coordinate_variable(header, ("lon", "longitude", "x"))
        time_name = range_access.coordinate_variable(header, ("time", "t"))
        data_name = "temp2" if variable == "tas" else "totprec"
        if data_name not in header.variables:
            raise RuntimeError(f"missing-mode-variable:{data_name}:{list(header.variables)}")
        data_var = header.variables[data_name]
        time_var = header.variables[time_name]
        lat = range_access.read_fixed_variable(source, header, lat_name)
        lon = range_access.normalize_longitude(range_access.read_fixed_variable(source, header, lon_name))
        lat_margin = float(np.median(np.abs(np.diff(np.sort(lat))))) * 1.5
        lon_margin = float(np.median(np.abs(np.diff(np.sort(lon))))) * 1.5
        lat_ids = np.flatnonzero((lat >= points[:, 0].min() - lat_margin) & (lat <= points[:, 0].max() + lat_margin))
        lon_ids = np.flatnonzero((lon >= points[:, 1].min() - lon_margin) & (lon <= points[:, 1].max() + lon_margin))
        spatial_dims = list(data_var.dimensions[1:])
        lat_axis = spatial_dims.index(header.variables[lat_name].dimensions[0])
        lon_axis = spatial_dims.index(header.variables[lon_name].dimensions[0])
        fields = np.empty((len(YEARS), len(lat_ids), len(lon_ids)), dtype=np.float64)
        decoded: list[int] = []
        units = str(data_var.attributes.get("units", ""))
        time_units = str(time_var.attributes.get("units", ""))
        calendar = str(time_var.attributes.get("calendar", "proleptic_gregorian"))
        for block_start in range(int(YEARS[0]), int(YEARS[-1]) + 1, 25):
            block_end = min(block_start + 24, int(YEARS[-1]))
            first = block_start - 1420
            last = block_end - 1420
            byte_start = header.record_base + first * header.record_size
            byte_end = header.record_base + (last + 1) * header.record_size - 1
            payload = source.read(byte_start, byte_end, f"complete_allowed_annual_{variable}_records", (block_start, block_end))
            for year in range(block_start, block_end + 1):
                local = year - block_start
                raw_time = range_access.record_segment(payload, header, time_var, local)
                time_value = range_access.decode_numeric(raw_time, time_var, header.shape(time_var)[1:] or (1,)).reshape(-1)[0]
                date = range_access.num2date(float(time_value), time_units, calendar)
                decoded.append(int(date.year))
                if int(date.year) != year:
                    raise RuntimeError(f"mode-year-decode:{year}:{date.year}")
                raw = range_access.record_segment(payload, header, data_var, local)
                field = range_access.decode_numeric(raw, data_var, header.shape(data_var)[1:])
                if (lat_axis, lon_axis) != (0, 1):
                    field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                fields[year - YEARS[0]] = field[np.ix_(lat_ids, lon_ids)]
        if decoded != YEARS.tolist():
            raise RuntimeError("mode-year-axis-failed")
        if variable == "pr":
            fields, conversion = precip_to_mm_day(fields, units)
        else:
            conversion = "kelvin_centered_anomaly_equals_degC"
        values = interpolate_fields(fields, lat[lat_ids], lon[lon_ids], points)
        return values, {"url": url, "units": units, "conversion": conversion, "etag": source.etag, "last_modified": source.last_modified}
    finally:
        source.close()


def pmip_urls(model: str, variable: str) -> list[dict[str, Any]]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    items = [item for item in catalog if item["model"] == model and item["experiment_role"] == "past" and item["end_year_ce"] >= YEARS[0] and item["start_year_ce"] <= YEARS[-1]]
    output: list[dict[str, Any]] = []
    for item in items:
        url = str(item["url"])
        if variable == "pr":
            url = url.replace("/Amon/tas/", "/Amon/pr/").replace("/tas_Amon_", "/pr_Amon_")
            if model == "MIROC-ES2L":
                url = url.replace("/v20231115/", "/v20200318/")
        output.append({**item, "url": url})
    if not output:
        raise RuntimeError(f"no-pmip-files:{model}:{variable}")
    return output


def source_subset(dataset: Dataset, points: np.ndarray) -> tuple[slice, slice, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
    raw_lon = np.mod(np.asarray(dataset.variables["lon"][:], dtype=float), 360.0)
    lat_step = float(np.median(np.abs(np.diff(np.sort(raw_lat)))))
    lon_step = float(np.median(np.abs(np.diff(np.sort(raw_lon)))))
    lat_ids = np.flatnonzero((raw_lat >= points[:, 0].min() - 1.5 * lat_step) & (raw_lat <= points[:, 0].max() + 1.5 * lat_step))
    lon_ids = np.flatnonzero((raw_lon >= points[:, 1].min() - 1.5 * lon_step) & (raw_lon <= points[:, 1].max() + 1.5 * lon_step))
    lat_slice = slice(int(lat_ids.min()), int(lat_ids.max()) + 1)
    lon_slice = slice(int(lon_ids.min()), int(lon_ids.max()) + 1)
    lat = raw_lat[lat_slice]
    lon = raw_lon[lon_slice]
    return lat_slice, lon_slice, lat, lon, np.argsort(lat), np.argsort(lon)


def pmip_annual(model: str, variable: str, points: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    annual_sum = np.zeros((len(YEARS), len(points)), dtype=np.float64)
    annual_count = np.zeros(len(YEARS), dtype=int)
    audit: list[dict[str, Any]] = []
    for item in pmip_urls(model, variable):
        with Dataset(item["url"]) as dataset:
            time = dataset.variables["time"]
            dates = num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"), only_use_cftime_datetimes=True)
            ce_years = np.asarray([int(date.year) + int(item["offset"]) for date in dates], dtype=int)
            selected = np.flatnonzero((ce_years >= YEARS[0]) & (ce_years <= YEARS[-1]))
            if len(selected) == 0:
                continue
            if np.any((ce_years[selected] >= 1901) & (ce_years[selected] <= 1939)):
                raise RuntimeError("locked-pmip-value-request")
            lat_slice, lon_slice, lat, lon, lat_order, lon_order = source_subset(dataset, points)
            data_var = dataset.variables[variable]
            segments = np.split(selected, np.flatnonzero(np.diff(selected) != 1) + 1)
            request_count = 0
            for segment in segments:
                indexers: list[Any] = []
                kept: list[str] = []
                for dimension in data_var.dimensions:
                    if dimension == "time":
                        indexers.append(slice(int(segment[0]), int(segment[-1]) + 1)); kept.append("time")
                    elif dimension == "lat":
                        indexers.append(lat_slice); kept.append("lat")
                    elif dimension == "lon":
                        indexers.append(lon_slice); kept.append("lon")
                    else:
                        indexers.append(0)
                fields = np.asarray(data_var[tuple(indexers)], dtype=float).squeeze()
                if kept != ["time", "lat", "lon"] or fields.ndim != 3:
                    raise RuntimeError(f"pmip-shape:{model}:{variable}:{data_var.dimensions}:{fields.shape}")
                fields = fields[:, lat_order][:, :, lon_order]
                if variable == "pr":
                    fields, conversion = precip_to_mm_day(fields, str(data_var.units))
                else:
                    conversion = "kelvin_centered_anomaly_equals_degC"
                values = interpolate_fields(fields, lat[lat_order], lon[lon_order], points)
                for row, source_index in enumerate(segment):
                    year = int(ce_years[source_index])
                    annual_sum[year - YEARS[0]] += values[row]
                    annual_count[year - YEARS[0]] += 1
                request_count += len(segment)
            audit.append({"model": model, "variable": variable, "url": item["url"], "source_units": str(data_var.units), "conversion": conversion, "requested_months": request_count})
    if not np.all(annual_count == 12):
        bad = YEARS[annual_count != 12]
        raise RuntimeError(f"pmip-annual-coverage:{model}:{variable}:{bad[:20].tolist()}")
    return annual_sum / annual_count[:, None], audit


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        raise FileExistsError(STATE_FILE)
    sources = {"facts": ROOT / "LIA_PROJECT_FACTS.md", "cesm_temperature": CESM_T, "cesm_precipitation": CESM_P, "catalog": CATALOG, "proxy6": PROXY_PATHS[6], "proxy9": PROXY_PATHS[9], "driver": DRIVER}
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}
    proxy_points, locations, point_indices = proxy_inventory()
    points, native_lat, native_lon = target_points(proxy_points)
    state_count = len(native_lat) * len(native_lon)

    arrays: dict[str, np.ndarray] = {"years": YEARS, "native_lat": native_lat, "native_lon": native_lon, "unique_proxy_points": proxy_points}
    for network in (6, 9):
        arrays[f"proxy{network}_locations"] = locations[network]
        arrays[f"proxy{network}_unique_point_indices"] = point_indices[network]

    temperature = np.load(CESM_T)
    with h5py.File(CESM_P, "r") as handle:
        precipitation = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float)
        if precipitation.shape == (848, 336):
            precipitation = precipitation.T
        member_years = np.asarray(handle["member_years"], dtype=int).reshape(-1)
    selected = np.asarray([int(np.flatnonzero(member_years == year)[0]) for year in YEARS])
    arrays["cesm_tas_state"] = temperature[:, selected].T
    arrays["cesm_pr_state"] = precipitation[:, selected].T

    request_log: list[dict[str, Any]] = []
    remote_audit: dict[str, Any] = {}
    for model_key, model_name in (("mode", "ModE-Sim"), ("mpi", "MPI-ESM1-2-LR"), ("miroc", "MIROC-ES2L")):
        if model_key == "mode":
            tas_all, tas_audit = mode_annual("tas", points, request_log)
            pr_all, pr_audit = mode_annual("pr", points, request_log)
            remote_audit[model_name] = [tas_audit, pr_audit]
        else:
            tas_all, tas_audit = pmip_annual(model_name, "tas", points)
            pr_all, pr_audit = pmip_annual(model_name, "pr", points)
            remote_audit[model_name] = tas_audit + pr_audit
        arrays[f"{model_key}_tas_state"] = tas_all[:, :state_count]
        arrays[f"{model_key}_pr_state"] = pr_all[:, :state_count]
        arrays[f"{model_key}_tas_proxy"] = tas_all[:, state_count:]
        arrays[f"{model_key}_pr_proxy"] = pr_all[:, state_count:]

    for model_key in ("cesm", "mode", "mpi", "miroc"):
        if arrays[f"{model_key}_tas_state"].shape != (len(YEARS), 336):
            raise RuntimeError(f"state-shape:{model_key}:{arrays[f'{model_key}_tas_state'].shape}")
    np.savez_compressed(STATE_FILE, **arrays)
    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in sources)
    summary = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": [int(YEARS[0]), int(YEARS[-1])], "state_count_per_model": len(YEARS),
        "top100_fraction": 100.0 / len(YEARS), "mode_sim_set": "1420-3 ensemble mean",
        "locked_1921_1939_values_accessed": 0, "enkf_run": False, "oggm_run": False,
        "production_modified": False, "source_hashes_unchanged": unchanged,
        "state_file": str(STATE_FILE), "remote_sources": remote_audit,
        "mode_http_range_request_count": len(request_log),
    }
    (OUTPUT / "state_pool_acquisition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "state_pool_source_manifest.json").write_text(json.dumps({"before": before, "after": after, "remote": remote_audit, "mode_requests": request_log}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
