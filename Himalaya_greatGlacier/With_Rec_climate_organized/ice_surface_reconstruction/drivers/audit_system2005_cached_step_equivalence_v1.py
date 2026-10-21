"""One-step equivalence gate for V4 and cached-property V5 system-2005 solvers."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v4 as old
from . import network_topology_adaptive_split_system2005_zero_mb_v5 as cached


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
    "lia_191_system2005_cached_step_equivalence_v1/summary.json"
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
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    old_flowlines = load()
    cached_flowlines = load()
    coefficients = base.Coefficients(float(config["stage2"]["glen_a"]), float(config["stage2"]["fs"]))
    closure = max(1e-6, base.total_volume(old_flowlines) * 1e-12)
    dt = float(config["maximum_timestep_seconds"])
    old.conservative_boundary_step.config = config
    cached.conservative_boundary_step.config = config
    old_result = old.conservative_boundary_step(old_flowlines, coefficients, dt, closure, 0.0)
    cached_result = cached.conservative_boundary_step(cached_flowlines, coefficients, dt, closure, 0.0)

    array_maxima = {name: 0.0 for name in ("section", "thick", "surface_h", "widths_m")}
    for old_flowline, cached_flowline in zip(old_flowlines, cached_flowlines):
        for name in array_maxima:
            difference = np.asarray(getattr(old_flowline, name)) - np.asarray(getattr(cached_flowline, name))
            array_maxima[name] = max(array_maxima[name], float(np.max(np.abs(difference))))
    scalar_keys = sorted(set(old_result) & set(cached_result) - {"mixedbed_property_access"})
    scalar_differences = {}
    for key in scalar_keys:
        a, b = old_result[key], cached_result[key]
        if a is None or b is None:
            scalar_differences[key] = 0.0 if a is b else None
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            scalar_differences[key] = float(b) - float(a)
    checks = {
        "flowline_count_equal": len(old_flowlines) == len(cached_flowlines),
        "all_state_arrays_bitwise_equal": all(value == 0.0 for value in array_maxima.values()),
        "all_scalar_outputs_equal": all(value == 0.0 for value in scalar_differences.values()),
        "source_immutable": sha256(SOURCE) == sha256(SOURCE),
    }
    summary = {
        "schema": "lia-191-system2005-cached-step-equivalence-v1",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "dt_seconds": dt,
        "closure_limit_m3": closure,
        "checks": checks,
        "maximum_absolute_array_differences": array_maxima,
        "scalar_output_differences": scalar_differences,
        "old_result": old_result,
        "cached_result": cached_result,
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "simulation_scope": "one identical zero-MB solver step on two in-memory source copies",
        "historical_run": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "checks", "maximum_absolute_array_differences", "scalar_output_differences")}))


if __name__ == "__main__":
    main()
