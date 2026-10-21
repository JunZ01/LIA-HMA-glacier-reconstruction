from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import build_temperature_candidate_b2_allowed_input_v3 as v3
import build_temperature_candidate_b2_allowed_input_v4 as v4


v1 = v3.v1
v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v6"
CACHE = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_cache_v1"
ORIGINAL_RANGE_READ = access.ExactRangeSource.read


def retry_range(self, *args, **kwargs):
    last = None
    for attempt in range(1, 9):
        try:
            return ORIGINAL_RANGE_READ(self, *args, **kwargs)
        except Exception as error:
            last = error
            print(f"RETRY ModE range {attempt}/8 {type(error).__name__}", flush=True)
            time.sleep(min(attempt * 5, 30))
    raise last


def cached(name: str, builder: Callable[[], np.ndarray]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        values = np.load(path)
        print(f"CACHE HIT {name} {values.shape}", flush=True)
        return values
    values = builder()
    temporary = CACHE / f"{name}.tmp.npy"
    np.save(temporary, values)
    temporary.replace(path)
    print(f"CACHE WRITE {name} {values.shape}", flush=True)
    return values


def mode(variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    return cached(
        f"mode_{variable}_{len(points)}points_482years",
        lambda: v4.annual_mode(variable, points, audit),
    )


def pmip(model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    key = model.lower().replace("-", "_")
    name = f"{key}_{variable}_{len(points)}points_482years"
    def build() -> np.ndarray:
        last = None
        for attempt in range(1, 9):
            try:
                return v3.annual_pmip(model, variable, points, audit)
            except Exception as error:
                last = error
                print(f"RETRY {model} {variable} {attempt}/8 {type(error).__name__}", flush=True)
                time.sleep(min(attempt * 10, 60))
        raise last
    return cached(name, build)


def main() -> None:
    access.ExactRangeSource.read = retry_range
    v1.annual_mode = mode
    v1.annual_pmip = pmip
    v1.main()
    path = v1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["v6_change"] = "persistent atomic variable caches plus bounded ModE and Globus reconnects"
    summary["cache_directory"] = str(CACHE)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
