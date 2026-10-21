"""One-step equivalence gate for V5 and full-array-cache V6."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v5 as old
from . import network_topology_adaptive_split_system2005_zero_mb_v6 as optimized


PROJECT = Path(__file__).resolve().parents[4]
CONFIG = PROJECT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "config/system2005_branch3_coarsen2x_steady50_conditional_reproduction_v2.json"
)
SOURCE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_static_v2/candidate_model_flowlines.pkl.gz"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_cached_step_equivalence_v2/summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load():
    with gzip.open(SOURCE, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    if OUTPUT.parent.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT.parent}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    old_flowlines = load()
    optimized_flowlines = load()
    coefficients = base.Coefficients(
        float(config["stage2"]["glen_a"]), float(config["stage2"]["fs"])
    )
    closure = max(1e-6, base.total_volume(old_flowlines) * 1e-12)
    dt = float(config["maximum_timestep_seconds"])
    old.conservative_boundary_step.config = config
    optimized.conservative_boundary_step.config = config
    source_hash_before = sha256(SOURCE)

    started = time.perf_counter()
    old_result = old.conservative_boundary_step(
        old_flowlines, coefficients, dt, closure, 0.0
    )
    old_runtime = time.perf_counter() - started
    started = time.perf_counter()
    optimized_result = optimized.conservative_boundary_step(
        optimized_flowlines, coefficients, dt, closure, 0.0
    )
    optimized_runtime = time.perf_counter() - started

    array_equal = {name: True for name in ("section", "thick", "surface_h", "widths_m")}
    array_maxima = {name: 0.0 for name in array_equal}
    for old_flowline, optimized_flowline in zip(old_flowlines, optimized_flowlines):
        for name in array_equal:
            old_array = np.asarray(getattr(old_flowline, name))
            optimized_array = np.asarray(getattr(optimized_flowline, name))
            array_equal[name] &= bool(np.array_equal(old_array, optimized_array))
            difference = old_array - optimized_array
            array_maxima[name] = max(
                array_maxima[name], float(np.max(np.abs(difference)))
            )

    ignored_keys = {"mixedbed_property_access"}
    scalar_keys = sorted((set(old_result) | set(optimized_result)) - ignored_keys)
    scalar_equal = {}
    scalar_differences = {}
    for key in scalar_keys:
        a = old_result.get(key)
        b = optimized_result.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            scalar_equal[key] = bool(a == b)
            scalar_differences[key] = float(b) - float(a)
        else:
            scalar_equal[key] = a == b
            scalar_differences[key] = None if a == b else {"old": a, "optimized": b}
    checks = {
        "flowline_count_equal": len(old_flowlines) == len(optimized_flowlines),
        "all_state_arrays_bitwise_equal": all(array_equal.values()),
        "all_scalar_outputs_equal": all(scalar_equal.values()),
        "source_immutable": source_hash_before == sha256(SOURCE),
    }
    summary = {
        "schema": "lia-191-system2005-cached-step-equivalence-v2",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "dt_seconds": dt,
        "closure_limit_m3": closure,
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "scalar_output_equality": scalar_equal,
        "scalar_output_differences": scalar_differences,
        "old_runtime_seconds": old_runtime,
        "optimized_runtime_seconds": optimized_runtime,
        "single_step_speedup": old_runtime / optimized_runtime,
        "old_result": old_result,
        "optimized_result": optimized_result,
        "source": str(SOURCE),
        "source_sha256": source_hash_before,
        "simulation_scope": "one identical zero-MB solver step on two in-memory source copies",
        "historical_run": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "checks",
                    "maximum_absolute_array_differences",
                    "scalar_output_differences",
                    "old_runtime_seconds",
                    "optimized_runtime_seconds",
                    "single_step_speedup",
                )
            }
        )
    )
    if summary["status"] != "equivalence-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
