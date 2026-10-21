"""Generic Picard-relinearized graph obstacle solver and synthetic fixtures."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse

from .paired_cell_graph_obstacle_fixture_v1 import solve_lower_bound_qp


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_graph_nonlinear_obstacle_fixture_v1"


def solve_picard_obstacle(surface_old, bed, storage, incidence, mb, dt, conductance_callback, closure_limit, convergence_m=1e-8, maximum_iterations=100, relaxation=0.7):
    surface_old = np.asarray(surface_old, dtype=float); bed = np.asarray(bed, dtype=float); storage = np.asarray(storage, dtype=float); mb = np.asarray(mb, dtype=float)
    rhs = storage * surface_old + float(dt) * storage * mb
    iterate = surface_old.copy(); total_active_iterations = 0; total_cg_iterations = 0
    converged = False
    for picard_iteration in range(1, maximum_iterations + 1):
        thick_iterate = iterate - bed
        k = np.asarray(conductance_callback(iterate, thick_iterate), dtype=float)
        if np.any(~np.isfinite(k)) or np.any(k < 0): raise RuntimeError("invalid nonlinear conductance")
        lap = incidence @ sparse.diags(k, format="csc") @ incidence.T
        matrix = sparse.diags(storage, format="csc") + float(dt) * lap
        solved, _, _, diagnostics = solve_lower_bound_qp(matrix, rhs, bed, iterate, tolerance=1e-8, maximum_iterations=100)
        total_active_iterations += diagnostics["active_set_iterations"]; total_cg_iterations += diagnostics["cg_iterations_total"]
        difference = float(np.max(np.abs(solved - iterate)))
        if difference <= convergence_m:
            iterate = solved; converged = True; break
        iterate = relaxation * solved + (1.0 - relaxation) * iterate
    if not converged: raise RuntimeError("Picard obstacle iteration did not converge")
    # Reassemble at the final fixed point and solve once to remove relaxation residue.
    k = np.asarray(conductance_callback(iterate, iterate - bed), dtype=float)
    lap = incidence @ sparse.diags(k, format="csc") @ incidence.T
    matrix = sparse.diags(storage, format="csc") + float(dt) * lap
    solution, gradient, active, diagnostics = solve_lower_bound_qp(matrix, rhs, bed, iterate, tolerance=1e-8, maximum_iterations=100)
    total_active_iterations += diagnostics["active_set_iterations"]; total_cg_iterations += diagnostics["cg_iterations_total"]
    final_fixed_point_error = float(np.max(np.abs(solution - iterate)))
    if final_fixed_point_error > max(convergence_m * 5.0, 1e-7): raise RuntimeError("final nonlinear fixed-point error exceeds gate")
    reaction = float(np.sum(gradient[active])); actual = float(np.dot(storage, solution - surface_old)); prescribed = float(dt * np.dot(storage, mb))
    budget_residual = actual - prescribed - reaction
    free = ~active
    passed = (
        float(np.min(solution - bed)) >= -1e-8
        and (float(np.max(np.abs(gradient[free]))) if np.any(free) else 0.0) <= closure_limit
        and (float(np.min(gradient[active])) if np.any(active) else 0.0) >= -closure_limit
        and reaction >= -closure_limit and abs(budget_residual) <= closure_limit
    )
    return solution, {
        "ok": bool(passed), "picard_iterations": picard_iteration,
        "active_set_iterations_total": total_active_iterations, "cg_iterations_total": total_cg_iterations,
        "final_fixed_point_error_m": final_fixed_point_error,
        "reaction_m3": reaction, "actual_volume_change_m3": actual, "prescribed_mb_m3": prescribed,
        "budget_residual_m3": budget_residual, "active_node_count": int(np.count_nonzero(active)),
    }


def chain_incidence(node_count):
    edges = np.column_stack([np.arange(node_count - 1), np.arange(1, node_count)])
    cols = np.repeat(np.arange(len(edges)), 2); rows = edges.ravel()
    return sparse.csc_matrix((np.tile([1.0,-1.0],len(edges)),(rows,cols)),shape=(node_count,len(edges)))


def run_slope_crossing_fixture():
    bed = np.zeros(2); surface0 = np.array([1.000001, 1.0]); storage = np.ones(2); mb = np.array([-0.001, 0.0]); dt = 1.0
    incidence = chain_incidence(2)
    def callback(surface, thick):
        slope = surface[0] - surface[1]; hf = (thick[0] + thick[1]) / 2.0
        return np.array([1e4 * hf**5 * abs(slope)**2])
    solution, diagnostics = solve_picard_obstacle(surface0, bed, storage, incidence, mb, dt, callback, 1e-6)
    crossed = bool((surface0[0]-surface0[1]) * (solution[0]-solution[1]) < 0)
    if not diagnostics["ok"] or not crossed: raise AssertionError("nonlinear slope-crossing fixture failed")
    return {"status":"pass","surface_before":surface0.tolist(),"surface_after":solution.tolist(),"slope_crossed_zero":crossed,**diagnostics}


def run_zero_mb_conservation_fixture():
    bed = np.array([0.0,0.0,0.0]); surface0=np.array([3.0,2.0,1.0]); storage=np.array([2.0,3.0,4.0]); mb=np.zeros(3); dt=2.0; incidence=chain_incidence(3)
    def callback(surface, thick):
        return 0.05 * ((thick[:-1]+thick[1:])/2.0)**2 * np.abs(np.diff(surface))
    solution, diagnostics=solve_picard_obstacle(surface0,bed,storage,incidence,mb,dt,callback,1e-6)
    if not diagnostics["ok"] or abs(np.dot(storage,solution-surface0))>1e-10: raise AssertionError("nonlinear zero-MB conservation failed")
    return {"status":"pass","volume_change":float(np.dot(storage,solution-surface0)),**diagnostics}


def run_obstacle_crossing_fixture():
    bed=np.zeros(3); surface0=np.array([0.05,0.02,0.0]); storage=np.ones(3); mb=np.array([-0.1,-0.01,0.0]); dt=1.0; incidence=chain_incidence(3)
    def callback(surface, thick): return 0.1*((thick[:-1]+thick[1:])/2.0)**3*np.abs(np.diff(surface))**2
    solution,diagnostics=solve_picard_obstacle(surface0,bed,storage,incidence,mb,dt,callback,1e-6)
    if not diagnostics["ok"] or np.min(solution)<-1e-12 or diagnostics["reaction_m3"]<0: raise AssertionError("nonlinear obstacle exhaustion failed")
    return {"status":"pass","surface_after":solution.tolist(),**diagnostics}


def main():
    fixtures={}; errors={}
    for name,function in [("slope_crossing",run_slope_crossing_fixture),("zero_mb_conservation",run_zero_mb_conservation_fixture),("obstacle_crossing",run_obstacle_crossing_fixture)]:
        try: fixtures[name]=function()
        except Exception as exc: fixtures[name]={"status":"fail","error":repr(exc)}; errors[name]=traceback.format_exc()
    passed=all(value["status"]=="pass" for value in fixtures.values())
    summary={"schema":"lia-191-paired-cell-graph-nonlinear-obstacle-fixture-v1","status":"fixture-pass-real-system-not-authorized" if passed else "fixture-fail-mechanism-closed","fixtures":fixtures,"tracebacks":errors,"real_system_authorized":False,"historical_run":False}
    OUTPUT.mkdir(parents=True,exist_ok=False); (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary))


if __name__=="__main__": main()
