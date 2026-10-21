from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

import acquire_round67_mode_ra_allowed_modern_subset_v2 as range_access
import mode_sim_cdf2_parser_v1 as cdf2


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v1"
STAGE_B = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2/stage_b_pseudoproxy_input_v1.npz"
PREREG = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_preregistration_v1/CANDIDATE_B2_STAGE_C_CONTRACT.md"
CATALOG = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/remote_source_catalog.json"
ARCHIVE = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all\python_data")
PROXY = {6: ARCHIVE / "proxy_data_6.mat", 9: ARCHIVE / "proxy_data_9.mat"}
CESM = {
    "tas": Path(r"D:\VM\gcm_7\cesm\cesm_TREFHT_merged.nc"),
    "pr": Path(r"D:\VM\gcm_7\cesm\cesm_PRECT_merged.nc"),
}
MODE_BASE = (
    "https://mode-ra.unibe.ch/data/ModE-Sim/outdata/"
    "lowres_20mem_Set_1420-3_1850-1/abs/ensstat/by_var/year/"
)
MODE = {
    "tas": MODE_BASE + "ModE-Sim_lowres_20mem_Set_1420-3_1850-1_ensmean_temp2_abs_1420-2009_year.nc",
    "pr": MODE_BASE + "ModE-Sim_lowres_20mem_Set_1420-3_1850-1_ensmean_totprec_abs_1420-2009_year.nc",
}
PERIODS = ((1500, 1920), (1940, 2000))
YEARS = np.concatenate([np.arange(a, b + 1) for a, b in PERIODS]).astype(np.int32)
LOCK = (1921, 1939)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate(fields: np.ndarray, lat: np.ndarray, lon: np.ndarray, points: np.ndarray) -> np.ndarray:
    lat_order = np.argsort(lat)
    lon360 = np.mod(lon, 360.0)
    lon_order = np.argsort(lon360)
    lat = lat[lat_order]
    lon360 = lon360[lon_order]
    fields = fields[:, lat_order][:, :, lon_order]
    query = points.copy()
    query[:, 1] = np.mod(query[:, 1], 360.0)
    out = np.empty((len(fields), len(points)), dtype=np.float64)
    for index, field in enumerate(fields):
        out[index] = RegularGridInterpolator(
            (lat, lon360), field, bounds_error=False, fill_value=np.nan
        )(query)
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"nonfinite-interpolation:{np.size(out) - np.isfinite(out).sum()}")
    return out


def bbox(lat: np.ndarray, lon: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon360 = np.mod(lon, 360.0)
    query_lon = np.mod(points[:, 1], 360.0)
    lat_step = float(np.median(np.abs(np.diff(np.sort(lat)))))
    lon_step = float(np.median(np.abs(np.diff(np.sort(lon360)))))
    lat_ids = np.flatnonzero(
        (lat >= points[:, 0].min() - 1.5 * lat_step)
        & (lat <= points[:, 0].max() + 1.5 * lat_step)
    )
    lon_ids = np.flatnonzero(
        (lon360 >= query_lon.min() - 1.5 * lon_step)
        & (lon360 <= query_lon.max() + 1.5 * lon_step)
    )
    if not len(lat_ids) or not len(lon_ids):
        raise RuntimeError("empty-spatial-subset")
    return lat_ids, lon_ids


def selected_points(stage: np.lib.npyio.NpzFile) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    state = np.column_stack(
        [stage["retained_state_latitude"], stage["retained_state_longitude"]]
    ).astype(float)
    network_locations = {
        network: np.asarray(stage[f"proxy{network}_locations"], dtype=float)
        for network in (6, 9)
    }
    unique_proxy = np.unique(np.vstack(list(network_locations.values())), axis=0)
    lookup = {tuple(row): index for index, row in enumerate(unique_proxy)}
    indices = {
        network: np.asarray([lookup[tuple(row)] for row in locations], dtype=np.int32)
        for network, locations in network_locations.items()
    }
    return np.vstack([state, unique_proxy]), network_locations, indices


def proxy_coefficients(stage: np.lib.npyio.NpzFile) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for network in (6, 9):
        wanted = np.asarray(stage[f"proxy{network}_record_indices_0based"], dtype=int)
        rows = []
        with h5py.File(PROXY[network], "r") as handle:
            refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
            for index in wanted:
                proxy = handle[refs[index]]
                b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
                if b.size != 3:
                    raise RuntimeError(f"invalid-b:{network}:{index}")
                rows.append(b)
        output[network] = np.asarray(rows, dtype=float)
    return output


def annual_cesm(variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    name = "TREFHT" if variable == "tas" else "PRECT"
    with Dataset(CESM[variable], "r") as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        lat_ids, lon_ids = bbox(lat, lon, points)
        annual = np.empty((len(YEARS), len(lat_ids), len(lon_ids)), dtype=float)
        cursor = 0
        for start, end in PERIODS:
            first = (start - 850) * 12
            last = (end - 850 + 1) * 12
            raw = np.asarray(
                dataset.variables[name][first:last, lat_ids.min() : lat_ids.max() + 1, lon_ids.min() : lon_ids.max() + 1],
                dtype=float,
            )
            raw = raw.reshape(end - start + 1, 12, len(lat_ids), len(lon_ids)).mean(axis=1)
            annual[cursor : cursor + len(raw)] = raw
            audit.append({
                "model": "CESM", "variable": variable, "source": str(CESM[variable]),
                "requested_year_start": start, "requested_year_end": end,
                "requested_month_start_index": first, "requested_month_end_index": last - 1,
                "sealed_climate_values_requested": 0,
            })
            cursor += len(raw)
    values = interpolate(annual, lat[lat_ids], lon[lon_ids], points)
    if variable == "tas":
        values -= 273.15
    else:
        values *= 86_400_000.0
    return values


def annual_mode(variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    source = range_access.ExactRangeSource(MODE[variable], f"candidate_b2_mode_{variable}", audit)
    try:
        header = cdf2.parse_header(source.read(0, range_access.HEADER_BYTES - 1, "header_only", None))
        lat_name = range_access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_name = range_access.coordinate_variable(header, ("lon", "longitude", "x"))
        data_name = "temp2" if variable == "tas" else "totprec"
        data_var = header.variables[data_name]
        lat = range_access.read_fixed_variable(source, header, lat_name)
        lon = range_access.normalize_longitude(range_access.read_fixed_variable(source, header, lon_name))
        lat_ids, lon_ids = bbox(lat, lon, points)
        spatial_dims = list(data_var.dimensions[1:])
        lat_axis = spatial_dims.index(header.variables[lat_name].dimensions[0])
        lon_axis = spatial_dims.index(header.variables[lon_name].dimensions[0])
        annual = np.empty((len(YEARS), len(lat_ids), len(lon_ids)), dtype=float)
        cursor = 0
        for start, end in PERIODS:
            first = start - 1420
            last = end - 1420
            byte_start = header.record_base + first * header.record_size
            byte_end = header.record_base + (last + 1) * header.record_size - 1
            payload = source.read(byte_start, byte_end, f"allowed_{variable}_records", (start, end))
            for local, year in enumerate(range(start, end + 1)):
                raw = range_access.record_segment(payload, header, data_var, local)
                field = range_access.decode_numeric(raw, data_var, header.shape(data_var)[1:])
                if (lat_axis, lon_axis) != (0, 1):
                    field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                annual[cursor + local] = field[np.ix_(lat_ids, lon_ids)]
            cursor += end - start + 1
        values = interpolate(annual, lat[lat_ids], lon[lon_ids], points)
        units = str(data_var.attributes.get("units", ""))
    finally:
        source.close()
    if variable == "tas":
        if values.mean() > 100:
            values -= 273.15
    elif "kg" in units.lower() and "s-1" in units.lower().replace(" ", ""):
        values *= 86400.0
    elif values.mean() < 0.01:
        values *= 86400.0
    return values


def pmip_items(model: str, variable: str) -> list[dict[str, Any]]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    items = [item for item in catalog if item["model"] == model]
    if model == "MPI-ESM1-2-LR" and not any(
        int(item["start_year_ce"]) <= 1910 <= int(item["end_year_ce"]) for item in items
    ):
        items.append({
            "model": model, "experiment_role": "historical", "start_year_ce": 1910,
            "end_year_ce": 1929, "offset": 0,
            "url": "https://esgf-node.ornl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_191001-192912.nc",
        })
    output = []
    for item in items:
        if not any(int(item["start_year_ce"]) <= end and int(item["end_year_ce"]) >= start for start, end in PERIODS):
            continue
        url = str(item["url"])
        if variable == "pr":
            url = url.replace("/Amon/tas/", "/Amon/pr/").replace("/tas_Amon_", "/pr_Amon_")
            if model == "MIROC-ES2L" and item["experiment_role"] == "past":
                url = url.replace("/v20231115/", "/v20200318/")
        output.append({**item, "url": url})
    return output


def annual_pmip(model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    total = np.zeros((len(YEARS), len(points)), dtype=float)
    count = np.zeros(len(YEARS), dtype=np.int16)
    year_lookup = {int(year): index for index, year in enumerate(YEARS)}
    for item in pmip_items(model, variable):
        source_start = int(item["start_year_ce"])
        source_end = int(item["end_year_ce"])
        with Dataset(item["url"], "r") as dataset:
            lat = np.asarray(dataset.variables["lat"][:], dtype=float)
            lon = np.asarray(dataset.variables["lon"][:], dtype=float)
            lat_ids, lon_ids = bbox(lat, lon, points)
            data = dataset.variables[variable]
            for allowed_start, allowed_end in PERIODS:
                start = max(source_start, allowed_start)
                end = min(source_end, allowed_end)
                if start > end:
                    continue
                first = (start - source_start) * 12
                last = (end - source_start + 1) * 12
                raw = np.asarray(
                    data[first:last, lat_ids.min() : lat_ids.max() + 1, lon_ids.min() : lon_ids.max() + 1],
                    dtype=float,
                )
                raw = raw.reshape(end - start + 1, 12, len(lat_ids), len(lon_ids)).mean(axis=1)
                if variable == "pr":
                    raw *= 86400.0
                values = interpolate(raw, lat[lat_ids], lon[lon_ids], points)
                if variable == "tas":
                    values -= 273.15
                for row, year in enumerate(range(start, end + 1)):
                    target = year_lookup[year]
                    total[target] += values[row]
                    count[target] += 1
                audit.append({
                    "model": model, "variable": variable, "source": str(item["url"]),
                    "requested_year_start": start, "requested_year_end": end,
                    "requested_month_start_index": first, "requested_month_end_index": last - 1,
                    "sealed_climate_values_requested": 0,
                })
    if not np.all(count == 1):
        raise RuntimeError(f"coverage:{model}:{variable}:{YEARS[count != 1].tolist()[:30]}")
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md", "contract": PREREG,
        "stage_b_static": STAGE_B, "catalog": CATALOG,
        "proxy6": PROXY[6], "proxy9": PROXY[9], "driver": Path(__file__).resolve(),
    }
    for path in [*sources.values(), *CESM.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in sources.items()}
    stage = np.load(STAGE_B, allow_pickle=False)
    points, network_locations, proxy_indices = selected_points(stage)
    state_count = len(stage["retained_state_indices"])
    coefficients = proxy_coefficients(stage)
    audit: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "years": YEARS, "state_count": np.asarray(state_count),
        "point_latitude": points[:, 0], "point_longitude": points[:, 1],
        "retained_state_indices_0based": stage["retained_state_indices"],
    }
    for network in (6, 9):
        arrays[f"proxy{network}_locations"] = network_locations[network]
        arrays[f"proxy{network}_unique_point_indices"] = proxy_indices[network]
        arrays[f"proxy{network}_record_indices_0based"] = stage[f"proxy{network}_record_indices_0based"]
        arrays[f"proxy{network}_psm_b0_b1_b2"] = coefficients[network]
        arrays[f"proxy{network}_residual_variance"] = stage[f"proxy{network}_residual_variance"]

    for key, label in (("cesm", "CESM"), ("mode", "ModE-Sim"), ("mpi", "MPI-ESM1-2-LR"), ("miroc", "MIROC-ES2L")):
        if key == "cesm":
            tas = annual_cesm("tas", points, audit); pr = annual_cesm("pr", points, audit)
        elif key == "mode":
            tas = annual_mode("tas", points, audit); pr = annual_mode("pr", points, audit)
        else:
            tas = annual_pmip(label, "tas", points, audit); pr = annual_pmip(label, "pr", points, audit)
        baseline = (YEARS >= 1961) & (YEARS <= 1990)
        arrays[f"{key}_tas_degC"] = tas.astype(np.float32)
        arrays[f"{key}_tas_anomaly_1961_1990_degC"] = (tas - tas[baseline].mean(axis=0)).astype(np.float32)
        arrays[f"{key}_pr_mm_day"] = pr.astype(np.float32)
        if not np.all(np.isfinite(tas)) or not np.all(np.isfinite(pr)):
            raise RuntimeError(f"nonfinite:{label}")
        if tas.min() < -90 or tas.max() > 65 or pr.min() < 0 or pr.max() > 50:
            raise RuntimeError(f"physical-range:{label}:{tas.min()}:{tas.max()}:{pr.min()}:{pr.max()}")

    if np.any((YEARS >= LOCK[0]) & (YEARS <= LOCK[1])):
        raise RuntimeError("sealed-year-in-output")
    if any(int(row.get("sealed_climate_values_requested", 0)) != 0 for row in audit):
        raise RuntimeError("sealed-climate-request")
    np.savez_compressed(OUT / "candidate_b2_allowed_raw_input_v1.npz", **arrays)
    (OUT / "request_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-hash-changed")
    result = {
        "experiment": OUT.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "years": [int(YEARS[0]), 1920, 1940, int(YEARS[-1])],
        "year_count": int(len(YEARS)), "state_count": int(state_count),
        "unique_proxy_point_count": int(len(points) - state_count),
        "models": ["CESM", "ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"],
        "all_values_finite_and_physical": True, "source_hashes_unchanged": unchanged,
        "sealed_1921_1939_values_requested": 0, "real_proxy_observations_read": False,
        "stage_c_enkf_run": False, "production_modified": False, "oggm_run": False,
        "prohibited_source": "20CRv3", "prohibited_source_accessed": False,
        "next_action": "validate_lowpass_and_psm_operator_closure_then_request_stage_c_authorization",
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
