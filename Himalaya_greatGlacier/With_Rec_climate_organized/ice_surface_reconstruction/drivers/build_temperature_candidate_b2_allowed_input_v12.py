from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import build_temperature_candidate_b2_allowed_input_v6 as v6
import build_temperature_candidate_b2_allowed_input_v8 as v8
import build_temperature_candidate_b2_allowed_input_v11 as v11
from test_esgf_retrying_range_file_v1 import RetryingRangeFile as BaseRangeFile


v1 = v11.v1
OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v12"
SOURCE_CACHE = v6.CACHE / "_pmip_source_checkpoint_v1"


def open_range_retry(url: str) -> BaseRangeFile:
    last: Exception | None = None
    for attempt in range(1, 9):
        try:
            return BaseRangeFile(url)
        except Exception as error:
            last = error
            print(f"RETRY HEAD {attempt}/8 {type(error).__name__} {url}", flush=True)
            time.sleep(min(4 * attempt, 24))
    raise last if last is not None else RuntimeError("range-head-failed")


def source_key(model: str, variable: str, start: int, end: int) -> str:
    return f"{model}_{variable}_{start}_{end}_554points".lower().replace("-", "_")


def allowed_intersections(start: int, end: int) -> list[tuple[int, int]]:
    output = []
    for allowed_start, allowed_end in v1.PERIODS:
        first = max(start, allowed_start)
        last = min(end, allowed_end)
        if first <= last:
            output.append((first, last))
    return output


def build_pmip_checkpointed(
    model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]
) -> np.ndarray:
    original_years = v1.YEARS.copy()
    original_periods = v1.PERIODS
    original_items = v1.pmip_items
    items = original_items(model, variable)
    rows: dict[int, np.ndarray] = {}
    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        for item in items:
            source_start = int(item["start_year_ce"])
            source_end = int(item["end_year_ce"])
            periods = allowed_intersections(source_start, source_end)
            if not periods:
                continue
            years = np.concatenate(
                [np.arange(start, end + 1, dtype=np.int32) for start, end in periods]
            )
            key = source_key(model, variable, source_start, source_end)
            data_path = SOURCE_CACHE / f"{key}.npz"
            audit_path = SOURCE_CACHE / f"{key}.request_audit.json"
            if data_path.exists() and audit_path.exists():
                with np.load(data_path, allow_pickle=False) as cached:
                    cached_years = cached["years"]
                    values = cached["values"]
                if not np.array_equal(cached_years, years) or values.shape != (len(years), len(points)):
                    raise RuntimeError(f"source-cache-shape:{key}")
                source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
                print(f"SOURCE CACHE HIT {key} {values.shape}", flush=True)
            else:
                v1.YEARS = years
                v1.PERIODS = tuple(periods)
                v1.pmip_items = lambda _model, _variable, frozen=item: [frozen]
                source_audit: list[dict[str, Any]] = []
                values = v11.build_pmip(model, variable, points, source_audit)
                temporary = SOURCE_CACHE / f"{key}.tmp.npz"
                np.savez_compressed(temporary, years=years, values=values)
                temporary.replace(data_path)
                temporary_audit = SOURCE_CACHE / f"{key}.tmp.request_audit.json"
                temporary_audit.write_text(
                    json.dumps(source_audit, indent=2) + "\n", encoding="utf-8"
                )
                temporary_audit.replace(audit_path)
                print(f"SOURCE CACHE WRITE {key} {values.shape}", flush=True)
            audit.extend(source_audit)
            for index, year in enumerate(years):
                if int(year) in rows:
                    raise RuntimeError(f"duplicate-source-year:{model}:{variable}:{year}")
                rows[int(year)] = values[index]
    finally:
        v1.YEARS = original_years
        v1.PERIODS = original_periods
        v1.pmip_items = original_items
    missing = [int(year) for year in original_years if int(year) not in rows]
    if missing:
        raise RuntimeError(f"checkpoint-coverage:{model}:{variable}:{missing[:20]}")
    return np.vstack([rows[int(year)] for year in original_years])


def warm_one(model: str, variable: str) -> dict[str, Any]:
    v11.RetryingRangeFile = open_range_retry
    points = v8.load_points()
    key = v8.cache_key(model, variable, len(points))
    audit: list[dict[str, Any]] = []
    values = v6.cached(
        key, lambda: build_pmip_checkpointed(model, variable, points, audit)
    )
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
    jobs = [("MPI-ESM1-2-LR", "tas"), ("MPI-ESM1-2-LR", "pr")]
    warmed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=2) as pool:
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
    summary["v12_transport"] = (
        "v11 concurrent exact month chunks plus bounded HEAD retry and per-source atomic checkpoints"
    )
    summary["mpi_worker_count"] = 2
    summary["threads_per_variable"] = v11.THREADS_PER_VARIABLE
    summary["source_checkpoint_directory"] = str(SOURCE_CACHE)
    summary["warmed_variables"] = sorted(
        warmed, key=lambda row: (row["model"], row["variable"])
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
