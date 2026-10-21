"""Picard-relinearized version of the system-2321 obstacle steady gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import run_paired_cell_hybrid_graph_obstacle_steady50_2321_v1 as implementation
from .paired_cell_graph_nonlinear_obstacle_fixture_v1 import solve_picard_obstacle
from .paired_cell_hybrid_graph_2321_common_v1 import conductance


FIXTURE = (
    Path(implementation.PROJECT)
    / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_graph_nonlinear_obstacle_fixture_v1/summary.json"
)
OUTPUT = (
    Path(implementation.PROJECT)
    / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_nonlinear_obstacle_steady50_2321_v1"
)


def nonlinear_step(state, surface, thick, mb, dt, glen_a, fs, glen_n, rhog, closure_limit):
    callback = lambda iterate_surface, iterate_thick: conductance(
        state, iterate_surface, iterate_thick, glen_a, fs, glen_n, rhog
    )
    solution, diagnostics = solve_picard_obstacle(
        surface, state["bed"], state["storage"], state["incidence"], mb,
        dt, callback, closure_limit, convergence_m=1e-8,
        maximum_iterations=100, relaxation=0.7,
    )
    next_thick = solution - state["bed"]
    diagnostics = dict(diagnostics)
    diagnostics["active_set_iterations"] = diagnostics["active_set_iterations_total"]
    diagnostics["newly_exhausted_count"] = int(
        np.count_nonzero((thick > 1e-6) & (next_thick <= 1e-8))
    )
    return solution, next_thick, diagnostics


if __name__ == "__main__":
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if fixture["status"] != "fixture-pass-real-system-not-authorized":
        raise RuntimeError("nonlinear obstacle fixture did not pass")
    implementation.obstacle_step = nonlinear_step
    implementation.OUTPUT = OUTPUT
    implementation.main()
    summary_path = OUTPUT / "steady_50y_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema"] = "lia-191-paired-cell-hybrid-graph-nonlinear-obstacle-steady50-2321-v1"
    summary["solver"] = "paired-cell-hybrid-graph-Picard-relinearized-semiimplicit-SIA-lower-bound-obstacle"
    summary["nonlinear_conductance_relinearized_each_step"] = True
    summary["frozen_slope_direction_flip_is_not_a_failure_gate"] = True
    summary["nonlinear_fixture_sha256"] = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "formal_nonlinear_status": summary["status"],
        "final_year": summary["final"]["year"],
        "review_flags": summary["review_flags"],
    }))
