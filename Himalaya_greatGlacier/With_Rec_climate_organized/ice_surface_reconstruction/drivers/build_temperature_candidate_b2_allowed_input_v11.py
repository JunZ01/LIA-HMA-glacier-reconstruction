from __future__ import annotations

import json
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import h5py
import numpy as np
import requests

import build_temperature_candidate_b2_allowed_input_v6 as v6
import build_temperature_candidate_b2_allowed_input_v7 as v7
import build_temperature_candidate_b2_allowed_input_v8 as v8
from test_esgf_retrying_range_file_v1 import RetryingRangeFile


v1 = v7.v1
OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v11"
THREADS_PER_VARIABLE = 8
_THREAD = threading.local()


def session() -> requests.Session:
    current = getattr(_THREAD, "session", None)
    if current is None:
        current = requests.Session()
        _THREAD.session = current
    return current


def fetch_chunk(url: str, offset: int, size: int) -> bytes:
    last: Exception | None = None
    for attempt in range(1, 9):
        try:
            response = session().get(
                url,
                headers={"Range": f"bytes={offset}-{offset + size - 1}"},
                timeout=(30, 120),
            )
            if response.status_code != 206:
                raise IOError(f"range-status:{response.status_code}")
            payload = response.content
            if len(payload) != size:
                raise IOError(f"range-length:{len(payload)}:{size}")
            return payload
        except Exception as error:
            last = error
            current = getattr(_THREAD, "session", None)
            if current is not None:
                current.close()
            _THREAD.session = requests.Session()
            time.sleep(min(2 * attempt, 12))
    raise last if last is not None else RuntimeError("chunk-fetch-failed")


def decode_chunk(
    payload: bytes,
    dtype: np.dtype,
    shape: tuple[int, ...],
    compression: str | None,
    shuffle: bool,
) -> np.ndarray:
    raw = zlib.decompress(payload) if compression == "gzip" else payload
    itemsize = dtype.itemsize
    count = int(np.prod(shape))
    if len(raw) != count * itemsize:
        raise RuntimeError(f"inflated-size:{len(raw)}:{count * itemsize}")
    if shuffle:
        shuffled = np.frombuffer(raw, dtype=np.uint8).reshape(itemsize, count)
        raw = shuffled.T.copy().reshape(-1).tobytes()
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def decode_numeric(
    values: np.ndarray, fill: float | None, scale: float, offset: float
) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if fill is not None:
        out[np.isclose(out, fill)] = np.nan
    return out * scale + offset


def build_pmip(model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    total = np.zeros((len(v1.YEARS), len(points)), float)
    count = np.zeros(len(v1.YEARS), np.int16)
    lookup = {int(year): index for index, year in enumerate(v1.YEARS)}
    for item in v1.pmip_items(model, variable):
        source_start = int(item["start_year_ce"])
        source_end = int(item["end_year_ce"])
        url = v7.v3.globus_url(str(item["url"]), model, variable, str(item["experiment_role"]))
        requested_years = [
            year
            for allowed_start, allowed_end in v1.PERIODS
            for year in range(max(source_start, allowed_start), min(source_end, allowed_end) + 1)
            if max(source_start, allowed_start) <= min(source_end, allowed_end)
        ]
        if not requested_years:
            continue
        print(
            f"OPEN {model} {variable} {source_start}-{source_end} "
            f"allowed={requested_years[0]}-{requested_years[-1]}",
            flush=True,
        )
        remote = RetryingRangeFile(url)
        compressed_bytes = 0
        metadata_requests_before = remote.requests
        try:
            with h5py.File(remote, "r") as handle:
                lat = v7.v3.decode(handle["lat"], handle["lat"][:])
                lon = v7.v3.decode(handle["lon"], handle["lon"][:])
                lat_ids, lon_ids = v1.bbox(lat, lon, points)
                data = handle[variable]
                if data.chunks is None or int(data.chunks[0]) != 1:
                    raise RuntimeError(f"non-monthly-chunk:{model}:{variable}:{data.chunks}")
                if data.compression not in (None, "gzip") or data.fletcher32:
                    raise RuntimeError(
                        f"unsupported-filter:{model}:{variable}:{data.compression}:{data.fletcher32}"
                    )
                dtype = data.dtype
                chunk_shape = tuple(int(value) for value in data.chunks)
                fill_attr = data.attrs.get("_FillValue")
                fill = None if fill_attr is None else float(np.asarray(fill_attr).reshape(-1)[0])
                scale = float(np.asarray(data.attrs.get("scale_factor", 1.0)).reshape(-1)[0])
                add = float(np.asarray(data.attrs.get("add_offset", 0.0)).reshape(-1)[0])
                annual_fields = np.empty(
                    (len(requested_years), len(lat_ids), len(lon_ids)), dtype=float
                )
                with ThreadPoolExecutor(max_workers=THREADS_PER_VARIABLE) as pool:
                    for year_row, year in enumerate(requested_years):
                        first = (year - source_start) * 12
                        infos = [
                            data.id.get_chunk_info_by_coord((first + month, 0, 0))
                            for month in range(12)
                        ]
                        if any(info.byte_offset is None or info.size <= 0 or info.filter_mask != 0 for info in infos):
                            raise RuntimeError(f"invalid-chunk-info:{model}:{variable}:{year}")
                        futures = [
                            pool.submit(fetch_chunk, url, int(info.byte_offset), int(info.size))
                            for info in infos
                        ]
                        annual_sum = np.zeros((len(lat_ids), len(lon_ids)), dtype=float)
                        for info, future in zip(infos, futures):
                            payload = future.result()
                            compressed_bytes += len(payload)
                            chunk = decode_chunk(
                                payload, dtype, chunk_shape, data.compression, bool(data.shuffle)
                            )[0]
                            decoded = decode_numeric(chunk, fill, scale, add)
                            annual_sum += decoded[np.ix_(lat_ids, lon_ids)]
                        annual_fields[year_row] = annual_sum / 12.0
                        if (year_row + 1) % 25 == 0 or year_row + 1 == len(requested_years):
                            print(
                                f"PROGRESS {model} {variable} {year_row + 1}/{len(requested_years)} "
                                f"through {year}",
                                flush=True,
                            )
                values = v1.interpolate(annual_fields, lat[lat_ids], lon[lon_ids], points)
                if variable == "tas":
                    values -= 273.15
                else:
                    values *= 86400.0
                for row, year in enumerate(requested_years):
                    total[lookup[year]] += values[row]
                    count[lookup[year]] += 1
                audit.append(
                    {
                        "model": model,
                        "variable": variable,
                        "source": url,
                        "transport": "concurrent_exact_raw_hdf5_month_chunks",
                        "time_chunk_months": 1,
                        "requested_year_start": requested_years[0],
                        "requested_year_end": requested_years[-1],
                        "requested_month_chunk_count": len(requested_years) * 12,
                        "compressed_data_bytes_requested": compressed_bytes,
                        "metadata_http_range_requests": remote.requests - metadata_requests_before,
                        "raw_chunk_decoder_fixture": str(
                            v1.LOOP / "temperature_candidate_b2_esgf_raw_chunk_decode_fixture_v1/summary.json"
                        ),
                        "sealed_climate_values_requested": 0,
                    }
                )
        finally:
            remote.close()
        print(f"DONE {model} {variable} {requested_years[0]}-{requested_years[-1]}", flush=True)
    if not np.all(count == 1):
        raise RuntimeError(f"coverage:{model}:{variable}:{v1.YEARS[count != 1].tolist()[:20]}")
    return total


def warm_one(model: str, variable: str) -> dict[str, Any]:
    points = v8.load_points()
    key = v8.cache_key(model, variable, len(points))
    audit: list[dict[str, Any]] = []
    values = v6.cached(key, lambda: build_pmip(model, variable, points, audit))
    if not audit:
        audit = v8.deterministic_pmip_audit(model, variable)
    sidecar = v8.audit_sidecar(key)
    temporary = sidecar.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(sidecar)
    return {
        "model": model,
        "variable": variable,
        "shape": list(values.shape),
        "cache": str(v6.CACHE / f"{key}.npy"),
        "audit": str(sidecar),
    }


def main() -> None:
    jobs = [
        ("MPI-ESM1-2-LR", "tas"),
        ("MPI-ESM1-2-LR", "pr"),
        ("MIROC-ES2L", "tas"),
        ("MIROC-ES2L", "pr"),
    ]
    warmed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(warm_one, *job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            warmed.append(result)
            print(f"WARM COMPLETE {result['model']} {result['variable']}", flush=True)
    v1.OUT = OUT
    v1.annual_mode = v8.cached_mode
    v1.annual_pmip = v8.cached_pmip
    v1.main()
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["v11_transport"] = (
        "concurrent exact raw HDF5 month-chunk ranges with bitwise decoder closure"
    )
    summary["parallel_variable_workers"] = 4
    summary["threads_per_variable"] = THREADS_PER_VARIABLE
    summary["warmed_variables"] = sorted(
        warmed, key=lambda row: (row["model"], row["variable"])
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
