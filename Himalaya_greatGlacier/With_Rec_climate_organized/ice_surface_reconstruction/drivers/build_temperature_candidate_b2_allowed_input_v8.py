from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import build_temperature_candidate_b2_allowed_input_v6 as v6
import build_temperature_candidate_b2_allowed_input_v7 as v7


v1 = v7.v1
OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v8"


def cache_key(model: str, variable: str, point_count: int) -> str:
    return f"{model.lower().replace('-', '_')}_{variable}_{point_count}points_482years"


def audit_sidecar(key: str) -> Path:
    return v6.CACHE / f"{key}.request_audit.json"


def deterministic_pmip_audit(model: str, variable: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in v1.pmip_items(model, variable):
        source_start = int(item["start_year_ce"])
        source_end = int(item["end_year_ce"])
        url = v7.v3.globus_url(
            str(item["url"]), model, variable, str(item["experiment_role"])
        )
        for allowed_start, allowed_end in v1.PERIODS:
            start = max(source_start, allowed_start)
            end = min(source_end, allowed_end)
            if start > end:
                continue
            rows.append(
                {
                    "model": model,
                    "variable": variable,
                    "source": url,
                    "transport": "retrying_exact_https_range",
                    "time_chunk_months": 1,
                    "requested_year_start": start,
                    "requested_year_end": end,
                    "sealed_climate_values_requested": 0,
                    "audit_origin": "deterministic_from_frozen_catalog_and_allowed_periods",
                }
            )
    return rows


def load_points() -> np.ndarray:
    with np.load(v1.STAGE_B, allow_pickle=False) as stage:
        points, _, _ = v1.selected_points(stage)
    return points


def warm_one(model: str, variable: str) -> dict[str, Any]:
    points = load_points()
    key = cache_key(model, variable, len(points))
    path = v6.CACHE / f"{key}.npy"
    sidecar = audit_sidecar(key)
    audit: list[dict[str, Any]] = []
    values = v7.annual_pmip(model, variable, points, audit)
    if not audit:
        audit = deterministic_pmip_audit(model, variable)
    temporary = sidecar.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(sidecar)
    return {
        "model": model,
        "variable": variable,
        "shape": list(values.shape),
        "cache": str(path),
        "audit": str(sidecar),
    }


def cached_mode(variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    values = v6.mode(variable, points, [])
    for start, end in v1.PERIODS:
        audit.append(
            {
                "model": "ModE-Sim",
                "variable": variable,
                "source": v1.MODE[variable],
                "transport": "persistent_cache_from_exact_https_range",
                "requested_year_start": start,
                "requested_year_end": end,
                "sealed_climate_values_requested": 0,
                "audit_origin": "v6_cache_created_by_allowed-period-only_v4_reader",
            }
        )
    return values


def cached_pmip(
    model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]
) -> np.ndarray:
    key = cache_key(model, variable, len(points))
    path = v6.CACHE / f"{key}.npy"
    sidecar = audit_sidecar(key)
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(f"incomplete-cache:{key}")
    values = np.load(path)
    audit.extend(json.loads(sidecar.read_text(encoding="utf-8")))
    print(f"CACHE HIT {key} {values.shape}", flush=True)
    return values


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
    v1.annual_mode = cached_mode
    v1.annual_pmip = cached_pmip
    v1.main()
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["v8_transport"] = (
        "four independent retrying exact-range PMIP workers followed by atomic-cache assembly"
    )
    summary["parallel_worker_count"] = 4
    summary["warmed_variables"] = sorted(
        warmed, key=lambda row: (row["model"], row["variable"])
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
