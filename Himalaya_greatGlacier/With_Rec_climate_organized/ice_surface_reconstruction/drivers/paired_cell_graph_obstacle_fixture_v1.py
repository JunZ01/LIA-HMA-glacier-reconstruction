"""Synthetic fixtures for a nonnegative-thickness graph obstacle solve.

The constrained step minimizes the frozen semi-implicit quadratic subject to
surface >= bed.  Its KKT multiplier is the explicitly recorded unrealized
ablation at an exhausted free boundary; no post-solve thickness clipping is
used.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_graph_obstacle_fixture_v1"


def solve_lower_bound_qp(matrix, rhs, lower, initial, tolerance=1e-10, maximum_iterations=100):
    matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    if matrix.shape != (len(rhs), len(rhs)) or lower.shape != rhs.shape or initial.shape != rhs.shape:
        raise ValueError("incompatible obstacle problem shapes")
    active = initial <= lower + tolerance
    solution = np.maximum(initial, lower).copy()
    total_cg_iterations = 0
    for active_iteration in range(1, maximum_iterations + 1):
        free = ~active
        solution[active] = lower[active]
        if np.any(free):
            aff = matrix[free][:, free]
            reduced_rhs = rhs[free]
            if np.any(active):
                reduced_rhs = reduced_rhs - matrix[free][:, active] @ lower[active]
            diagonal = np.asarray(aff.diagonal(), dtype=float)
            preconditioner = LinearOperator(aff.shape, matvec=lambda vector: vector / diagonal)
            count = [0]
            def callback(_): count[0] += 1
            free_solution, info = cg(
                aff, np.asarray(reduced_rhs).ravel(), x0=solution[free],
                tol=1e-13, atol=0.0, maxiter=5000, M=preconditioner, callback=callback,
            )
            total_cg_iterations += count[0]
            if info != 0 or np.any(~np.isfinite(free_solution)):
                raise RuntimeError(f"reduced CG failed: info={info}")
            solution[free] = free_solution
        violating = free & (solution < lower - tolerance)
        if np.any(violating):
            active[violating] = True
            continue
        gradient = np.asarray(matrix @ solution).ravel() - rhs
        releasable = active & (gradient < -tolerance)
        if np.any(releasable):
            active[releasable] = False
            continue
        free_gradient = float(np.max(np.abs(gradient[free]))) if np.any(free) else 0.0
        active_minimum_gradient = float(np.min(gradient[active])) if np.any(active) else 0.0
        return solution, gradient, active, {
            "active_set_iterations": active_iteration,
            "cg_iterations_total": total_cg_iterations,
            "maximum_free_gradient": free_gradient,
            "minimum_active_gradient": active_minimum_gradient,
        }
    raise RuntimeError("active-set obstacle solver did not converge")


def run_ablation_exhaustion_fixture():
    storage = np.array([1.0, 2.0])
    bed = np.array([0.0, 0.0])
    surface0 = np.array([0.1, 0.0])
    mb = np.array([-0.2, 0.0])
    dt = 1.0
    matrix = sparse.diags(storage, format="csc")
    rhs = storage * surface0 + dt * storage * mb
    surface1, gradient, active, diagnostics = solve_lower_bound_qp(matrix, rhs, bed, surface0)
    expected = np.array([0.0, 0.0])
    reaction = float(np.sum(gradient[active]))
    prescribed_change = float(dt * np.dot(storage, mb))
    actual_change = float(np.dot(storage, surface1 - surface0))
    budget_residual = actual_change - prescribed_change - reaction
    if not np.allclose(surface1, expected, atol=1e-14, rtol=0) or reaction < 0 or abs(budget_residual) > 1e-14:
        raise AssertionError("ablation-exhaustion KKT budget failed")
    return {
        "status": "pass", "surface_after": surface1.tolist(),
        "unrealized_ablation_reaction_m3_equivalent": reaction,
        "prescribed_volume_change_m3_equivalent": prescribed_change,
        "actual_volume_change_m3_equivalent": actual_change,
        "budget_residual": budget_residual, **diagnostics,
    }


def run_diffusive_release_fixture():
    storage = np.ones(2); bed = np.zeros(2); surface0 = np.array([1.0, 0.0]); dt = 1.0
    laplacian = sparse.csc_matrix([[1.0, -1.0], [-1.0, 1.0]])
    matrix = sparse.diags(storage, format="csc") + dt * laplacian
    rhs = storage * surface0
    surface1, gradient, active, diagnostics = solve_lower_bound_qp(matrix, rhs, bed, surface0)
    expected = np.array([2.0 / 3.0, 1.0 / 3.0])
    if not np.allclose(surface1, expected, atol=1e-13, rtol=0) or np.any(active) or np.max(np.abs(gradient)) > 1e-12:
        raise AssertionError("dry downstream control was not released by inflow")
    return {"status": "pass", "surface_after": surface1.tolist(), "dry_node_released": True, **diagnostics}


def run_mixed_obstacle_fixture():
    storage = np.array([2.0, 1.0, 3.0]); bed = np.array([5.0, 4.0, 3.0]); thick0 = np.array([1.0, 0.1, 0.0])
    surface0 = bed + thick0; dt = 2.0
    laplacian = sparse.csc_matrix([[1.0,-1.0,0.0],[-1.0,2.0,-1.0],[0.0,-1.0,1.0]]) * 0.2
    mb = np.array([-0.8, -0.4, 0.0])
    matrix = sparse.diags(storage, format="csc") + dt * laplacian
    rhs = storage * surface0 + dt * storage * mb
    surface1, gradient, active, diagnostics = solve_lower_bound_qp(matrix, rhs, bed, surface0)
    reaction = float(np.sum(gradient[active])); actual = float(np.dot(storage, surface1 - surface0)); prescribed = float(dt * np.dot(storage, mb))
    residual = actual - prescribed - reaction
    if np.min(surface1 - bed) < -1e-12 or np.min(gradient[active]) < -1e-10 or abs(residual) > 1e-11:
        raise AssertionError("mixed obstacle KKT conditions failed")
    return {"status": "pass", "active_indices": np.flatnonzero(active).tolist(), "minimum_thickness": float(np.min(surface1-bed)), "reaction": reaction, "budget_residual": residual, **diagnostics}


def main():
    fixtures = {}; errors = {}
    for name, function in [
        ("ablation_exhaustion", run_ablation_exhaustion_fixture),
        ("diffusive_release", run_diffusive_release_fixture),
        ("mixed_obstacle", run_mixed_obstacle_fixture),
    ]:
        try:
            fixtures[name] = function()
        except Exception as exc:
            fixtures[name] = {"status": "fail", "error": repr(exc)}; errors[name] = traceback.format_exc()
    passed = all(value["status"] == "pass" for value in fixtures.values())
    summary = {
        "schema": "lia-191-paired-cell-graph-obstacle-fixture-v1",
        "status": "fixture-pass-real-system-not-authorized" if passed else "fixture-fail-mechanism-closed",
        "fixtures": fixtures, "tracebacks": errors,
        "post_solve_thickness_clipping": False,
        "real_system_authorized": False, "oggm_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
