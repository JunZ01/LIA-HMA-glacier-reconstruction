from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import build_temperature_candidate_b2_allowed_input_v8 as v8
import build_temperature_candidate_b2_allowed_input_v11 as v11
import build_temperature_candidate_b2_allowed_input_v12 as v12
import build_temperature_candidate_b2_allowed_input_v13 as v13


v1 = v13.v1
OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v14"
ALLOWED_PERIODS = ((1500, 1920), (1940, 2000))


def fixed_allowed_intersections(start: int, end: int) -> list[tuple[int, int]]:
    output = []
    for allowed_start, allowed_end in ALLOWED_PERIODS:
        first = max(start, allowed_start)
        last = min(end, allowed_end)
        if first <= last:
            output.append((first, last))
    return output


def warm_one(model: str, variable: str) -> dict[str, Any]:
    v12.SOURCE_CACHE = v13.SHORT_SOURCE_CACHE
    v12.allowed_intersections = fixed_allowed_intersections
    return v12.warm_one(model, variable)


def main() -> None:
    v13.SHORT_SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
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
    v1.YEARS = v11.v1.YEARS
    v1.annual_mode = v8.cached_mode
    v1.annual_pmip = v8.cached_pmip
    v1.main()
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["v14_change"] = (
        "allowed periods are immutable across per-source checkpoint iterations"
    )
    summary["mpi_worker_count"] = 2
    summary["threads_per_variable"] = v11.THREADS_PER_VARIABLE
    summary["source_checkpoint_directory"] = str(v13.SHORT_SOURCE_CACHE)
    summary["warmed_variables"] = sorted(
        warmed, key=lambda row: (row["model"], row["variable"])
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
