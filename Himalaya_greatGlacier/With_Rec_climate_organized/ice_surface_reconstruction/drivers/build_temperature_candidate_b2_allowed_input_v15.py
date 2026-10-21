from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import build_temperature_candidate_b2_allowed_input_v8 as v8
import build_temperature_candidate_b2_allowed_input_v11 as v11
import build_temperature_candidate_b2_allowed_input_v12 as v12
import build_temperature_candidate_b2_allowed_input_v13 as v13


v1 = v13.v1
OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v15"
SOURCE_CACHE = v1.RESULTS / "b2_pmip_source_cache_v2"
ALLOWED_PERIODS = ((1500, 1920), (1940, 2000))


def corrected_intersections(start: int, end: int) -> list[tuple[int, int]]:
    if start == 1831 and end == 1850:
        end = 1849
    output = []
    for allowed_start, allowed_end in ALLOWED_PERIODS:
        first = max(start, allowed_start)
        last = min(end, allowed_end)
        if first <= last:
            output.append((first, last))
    return output


def prepare_corrected_cache() -> None:
    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    for source in v13.SHORT_SOURCE_CACHE.glob("*.npz"):
        target = SOURCE_CACHE / source.name
        audit_source = source.with_suffix(".request_audit.json")
        audit_target = SOURCE_CACHE / audit_source.name
        if target.exists() and audit_target.exists():
            continue
        with np.load(source, allow_pickle=False) as cached:
            years = cached["years"]
            values = cached["values"]
        audit = json.loads(audit_source.read_text(encoding="utf-8"))
        if "_1831_1850_" in source.name:
            keep = years <= 1849
            years = years[keep]
            values = values[keep]
            for row in audit:
                row["technical_cache_requested_year_end"] = row["requested_year_end"]
                row["requested_year_end"] = 1849
                row["selected_month_chunk_count"] = 19 * 12
                row["excluded_overlap_year"] = 1850
                row["overlap_rule"] = "PMIP past requested_last_year=1849; historical starts 1850"
        temporary = SOURCE_CACHE / f"{source.stem}.tmp.npz"
        np.savez_compressed(temporary, years=years, values=values)
        temporary.replace(target)
        temporary_audit = SOURCE_CACHE / f"{audit_source.stem}.tmp.json"
        temporary_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        temporary_audit.replace(audit_target)


def warm_one(model: str, variable: str) -> dict[str, Any]:
    v12.SOURCE_CACHE = SOURCE_CACHE
    v12.allowed_intersections = corrected_intersections
    return v12.warm_one(model, variable)


def main() -> None:
    prepare_corrected_cache()
    jobs = [("MPI-ESM1-2-LR", "tas"), ("MPI-ESM1-2-LR", "pr")]
    warmed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(warm_one, *job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            warmed.append(result)
            print(f"WARM COMPLETE {result['model']} {result['variable']}", flush=True)
    v1.OUT = OUT
    v1.PERIODS = ALLOWED_PERIODS
    v1.YEARS = np.concatenate(
        [np.arange(start, end + 1) for start, end in ALLOWED_PERIODS]
    ).astype(np.int32)
    v1.annual_mode = v8.cached_mode
    v1.annual_pmip = v8.cached_pmip
    v1.main()
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["v15_change"] = (
        "honor frozen MPI past requested_last_year=1849 and historical start_year=1850"
    )
    summary["mpi_1850_overlap_rule"] = "past through 1849; historical from 1850"
    summary["mpi_worker_count"] = 2
    summary["threads_per_variable"] = v11.THREADS_PER_VARIABLE
    summary["source_checkpoint_directory"] = str(SOURCE_CACHE)
    summary["warmed_variables"] = sorted(
        warmed, key=lambda row: (row["model"], row["variable"])
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
