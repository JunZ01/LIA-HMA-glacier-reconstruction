"""One-step bitwise gate for cached V6 and graph-reuse V8."""

from __future__ import annotations

import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v6 as old
from . import network_topology_adaptive_split_system2005_zero_mb_v8 as graph_reuse


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
    "lia_191_system2005_graph_reuse_step_equivalence_v1/summary.json"
)


def load():
    with gzip.open(SOURCE, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    if OUTPUT.parent.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT.parent}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    old_flowlines = load()
    graph_flowlines = load()
    coefficients = base.Coefficients(
        float(config["stage2"]["glen_a"]), float(config["stage2"]["fs"])
    )
    closure = max(1e-6, base.total_volume(old_flowlines) * 1e-12)
    dt = float(config["maximum_timestep_seconds"])
    old.conservative_boundary_step.config = config
    graph_reuse.conservative_boundary_step.config = config
    graph_reuse.reset_structure_cache()
    started = time.perf_counter()
    old_result = old.conservative_boundary_step(old_flowlines, coefficients, dt, closure, 0.0)
    old_runtime = time.perf_counter() - started
    started = time.perf_counter()
    graph_result = graph_reuse.conservative_boundary_step(
        graph_flowlines, coefficients, dt, closure, 0.0
    )
    graph_runtime = time.perf_counter() - started
    names = ("section", "thick", "surface_h", "widths_m")
    array_equal = {name: True for name in names}
    array_maxima = {name: 0.0 for name in names}
    for old_flowline, graph_flowline in zip(old_flowlines, graph_flowlines):
        for name in names:
            old_array = np.asarray(getattr(old_flowline, name))
            graph_array = np.asarray(getattr(graph_flowline, name))
            array_equal[name] &= bool(np.array_equal(old_array, graph_array))
            array_maxima[name] = max(
                array_maxima[name], float(np.max(np.abs(old_array - graph_array)))
            )
    ignored = {"mixedbed_property_access"}
    keys = sorted((set(old_result) | set(graph_result)) - ignored)
    scalar_equal = {key: old_result.get(key) == graph_result.get(key) for key in keys}
    checks = {
        "all_state_arrays_bitwise_equal": all(array_equal.values()),
        "all_scalar_outputs_equal": all(scalar_equal.values()),
    }
    summary = {
        "schema": "lia-191-system2005-graph-reuse-step-equivalence-v1",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "scalar_output_equality": scalar_equal,
        "old_runtime_seconds": old_runtime,
        "graph_reuse_runtime_seconds": graph_runtime,
        "single_step_speedup": old_runtime / graph_runtime,
        "historical_run": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "equivalence-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
