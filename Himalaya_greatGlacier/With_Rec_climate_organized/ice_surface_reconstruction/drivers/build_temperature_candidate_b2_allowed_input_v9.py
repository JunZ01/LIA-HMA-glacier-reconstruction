from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests

import build_temperature_candidate_b2_allowed_input_v6 as v6
import build_temperature_candidate_b2_allowed_input_v7 as v7
import build_temperature_candidate_b2_allowed_input_v8 as v8
from test_esgf_retrying_range_file_v1 import RetryingRangeFile


v1 = v7.v1
OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v9"
TEMP = v6.CACHE / "_hybrid_downloads_v1"


def entirely_allowed(start: int, end: int) -> bool:
    return any(start >= allowed_start and end <= allowed_end for allowed_start, allowed_end in v1.PERIODS)


def download(url: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, 7):
        try:
            if part.exists():
                part.unlink()
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("Content-Length", "0"))
                written = 0
                with part.open("wb") as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            handle.write(block)
                            written += len(block)
                if expected and written != expected:
                    raise IOError(f"download-length:{written}:{expected}")
            part.replace(target)
            return written
        except Exception as error:
            last = error
            if part.exists():
                part.unlink()
            print(f"RETRY FULL {attempt}/6 {type(error).__name__} {url}", flush=True)
            time.sleep(min(5 * attempt, 30))
    raise last if last is not None else RuntimeError("download-failed")


def extract(
    handle: h5py.File,
    model: str,
    variable: str,
    points: np.ndarray,
    source_start: int,
    source_end: int,
    total: np.ndarray,
    count: np.ndarray,
    lookup: dict[int, int],
    audit: list[dict[str, Any]],
    source: str,
    transport: str,
    downloaded_bytes: int | None,
    remote: RetryingRangeFile | None,
) -> None:
    lat = v7.v3.decode(handle["lat"], handle["lat"][:])
    lon = v7.v3.decode(handle["lon"], handle["lon"][:])
    lat_ids, lon_ids = v1.bbox(lat, lon, points)
    data = handle[variable]
    if data.chunks is None or int(data.chunks[0]) != 1:
        raise RuntimeError(f"non-monthly-chunk:{model}:{variable}:{data.chunks}")
    for allowed_start, allowed_end in v1.PERIODS:
        start = max(source_start, allowed_start)
        end = min(source_end, allowed_end)
        if start > end:
            continue
        first = (start - source_start) * 12
        last = (end - source_start + 1) * 12
        raw = v7.v3.decode(
            data,
            data[
                first:last,
                lat_ids.min() : lat_ids.max() + 1,
                lon_ids.min() : lon_ids.max() + 1,
            ],
        )
        raw = raw.reshape(end - start + 1, 12, len(lat_ids), len(lon_ids)).mean(axis=1)
        if variable == "pr":
            raw *= 86400.0
        values = v1.interpolate(raw, lat[lat_ids], lon[lon_ids], points)
        if variable == "tas":
            values -= 273.15
        for row, year in enumerate(range(start, end + 1)):
            total[lookup[year]] += values[row]
            count[lookup[year]] += 1
        row: dict[str, Any] = {
            "model": model,
            "variable": variable,
            "source": source,
            "transport": transport,
            "time_chunk_months": 1,
            "requested_year_start": start,
            "requested_year_end": end,
            "source_file_year_start": source_start,
            "source_file_year_end": source_end,
            "source_file_entirely_within_allowed_period": entirely_allowed(source_start, source_end),
            "sealed_climate_values_requested": 0,
        }
        if downloaded_bytes is not None:
            row["downloaded_bytes"] = downloaded_bytes
        if remote is not None:
            row["http_range_requests_cumulative"] = remote.requests
        audit.append(row)
        print(f"DONE {model} {variable} {start}-{end} {transport}", flush=True)


def build_pmip(model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    total = np.zeros((len(v1.YEARS), len(points)), float)
    count = np.zeros(len(v1.YEARS), np.int16)
    lookup = {int(year): index for index, year in enumerate(v1.YEARS)}
    for item in v1.pmip_items(model, variable):
        source_start = int(item["start_year_ce"])
        source_end = int(item["end_year_ce"])
        url = v7.v3.globus_url(str(item["url"]), model, variable, str(item["experiment_role"]))
        print(f"OPEN {model} {variable} {source_start}-{source_end}", flush=True)
        if entirely_allowed(source_start, source_end):
            key = f"{model}_{variable}_{source_start}_{source_end}".replace("-", "_")
            target = TEMP / f"{key}.nc"
            downloaded = download(url, target)
            try:
                with h5py.File(target, "r") as handle:
                    extract(
                        handle, model, variable, points, source_start, source_end,
                        total, count, lookup, audit, url, "full_https_stream_allowed_file",
                        downloaded, None,
                    )
            finally:
                if target.exists():
                    target.unlink()
        else:
            remote = RetryingRangeFile(url)
            try:
                with h5py.File(remote, "r") as handle:
                    extract(
                        handle, model, variable, points, source_start, source_end,
                        total, count, lookup, audit, url, "retrying_exact_https_range",
                        None, remote,
                    )
            finally:
                remote.close()
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
    summary["v9_transport"] = (
        "full HTTPS streaming only for source files wholly inside allowed periods; "
        "exact retrying ranges for all boundary and sealed-period-crossing files"
    )
    summary["parallel_worker_count"] = 4
    summary["temporary_files_location"] = str(TEMP)
    summary["temporary_files_retained"] = False
    summary["warmed_variables"] = sorted(warmed, key=lambda row: (row["model"], row["variable"]))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if TEMP.exists() and not any(TEMP.iterdir()):
        TEMP.rmdir()


if __name__ == "__main__":
    main()
