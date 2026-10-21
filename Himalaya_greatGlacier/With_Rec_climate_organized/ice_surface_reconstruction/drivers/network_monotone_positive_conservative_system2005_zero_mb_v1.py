"""0.1 year zero-MB gate for the mass-conserving monotone/positive solver."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import pickle
import time
import traceback

import numpy as np
from scipy.sparse import csr_matrix, eye, vstack

from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_variational_system2005_failure_state_step_v1 as step_v1
from . import network_monotone_positive_conservative_system2005_failure_state_step_v1 as positive_v1


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
TOL = 1e-10


def continuity_failure(flowlines):
    for branch, flowline in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(flowline.thick, dtype=float) > base.ACTIVE_TOLERANCE_M)
        if len(active) > 1 and np.any(np.diff(active) > 1):
            return f"internal-active-gap:branch={branch}"
        if flowline.flows_to is not None and len(active) and int(active[-1]) != len(flowline.thick) - 1:
            return f"tributary-disconnected-from-junction:branch={branch}:last_active={int(active[-1])}:terminal={len(flowline.thick)-1}"
    return None


def constrained_step(flowlines, coefficients, dt_seconds, closure_limit, section_floor):
    active_local, edges, surface_before, storage, matrix, rhs, incidence = step_v1.assemble(
        flowlines, coefficients, dt_seconds
    )
    widths = np.asarray([flowlines[b].widths_m[n] for b, n in active_local], dtype=float)
    old_sections = np.asarray([flowlines[b].section[n] for b, n in active_local], dtype=float)
    surface_lower = surface_before + (section_floor - old_sections) / widths
    constraints = vstack([incidence, eye(len(active_local), format="csr")], format="csr")
    lower = np.r_[np.zeros(incidence.shape[0]), surface_lower]
    volume_row = csr_matrix(storage.reshape(1, -1))
    volume_target = float(np.dot(storage, surface_before))
    candidate, unconstrained, active, multipliers, volume_multiplier, history = positive_v1.solve_affine_active_set(
        matrix, rhs, constraints, lower, volume_row, volume_target
    )
    residual = np.asarray(constraints @ candidate, dtype=float) - lower
    edge_slopes = residual[: incidence.shape[0]]
    section_residual = residual[incidence.shape[0] :]
    target_delta = storage * (candidate - surface_before)
    if abs(float(np.sum(target_delta))) > closure_limit:
        raise RuntimeError(f"linearized-volume-closure:{float(np.sum(target_delta))}")
    candidate_sections = [np.asarray(flowline.section, dtype=float).copy() for flowline in flowlines]
    for global_index, (branch, node) in enumerate(active_local):
        candidate_sections[branch][node] += target_delta[global_index] / float(flowlines[branch].dx_meter)
    active_candidate_sections = np.asarray([
        candidate_sections[branch][node] for branch, node in active_local
    ], dtype=float)
    if float(np.min(active_candidate_sections)) < 0.99 * section_floor:
        raise RuntimeError(f"active-section-interior-floor:{float(np.min(active_candidate_sections))}")
    before_volume = base.total_volume(flowlines)
    for flowline, section in zip(flowlines, candidate_sections):
        flowline.section = section
    after_volume = base.total_volume(flowlines)
    actual_closure = after_volume - before_volume
    if abs(actual_closure) > closure_limit:
        raise RuntimeError(f"actual-volume-closure:{actual_closure}")
    realized_surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    realized_slopes = np.asarray(incidence @ realized_surface, dtype=float)
    if not np.all(np.isfinite(realized_surface)) or float(np.min(realized_slopes)) < -TOL:
        raise RuntimeError(f"realized-reverse-or-nonfinite:{float(np.min(realized_slopes))}")
    topology_failure = continuity_failure(flowlines)
    if topology_failure:
        raise RuntimeError(topology_failure)
    active_matrix = constraints[active, :].tocsr()
    stationarity = np.asarray(matrix @ candidate - rhs, dtype=float)
    stationarity -= np.asarray(active_matrix.T @ multipliers, dtype=float).ravel()
    stationarity += volume_row.toarray().ravel() * volume_multiplier
    maximum_stationarity = float(np.max(np.abs(stationarity)))
    if maximum_stationarity > 1e-5:
        raise RuntimeError(f"KKT-stationarity:{maximum_stationarity}")
    minimum_multiplier = float(np.min(multipliers)) if len(multipliers) else None
    if minimum_multiplier is not None and minimum_multiplier < -TOL:
        raise RuntimeError(f"negative-constraint-multiplier:{minimum_multiplier}")
    return {
        "active_node_count": len(active_local),
        "edge_count": len(edges),
        "unconstrained_reverse_edge_count": int(np.count_nonzero(np.asarray(incidence @ unconstrained) < -TOL)),
        "active_constraint_count": len(active),
        "active_edge_constraint_count": sum(index < incidence.shape[0] for index in active),
        "active_section_constraint_count": sum(index >= incidence.shape[0] for index in active),
        "minimum_inequality_multiplier": minimum_multiplier,
        "minimum_target_edge_slope_m": float(np.min(edge_slopes)),
        "minimum_realized_edge_slope_m": float(np.min(realized_slopes)),
        "minimum_active_section_m2": float(np.min(active_candidate_sections)),
        "maximum_absolute_KKT_stationarity": maximum_stationarity,
        "actual_closure_m3": actual_closure,
        "active_set_iterations": len(history),
    }


def run(config):
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    gate_path = base.resolve(config["failure_state_step_gate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "real-failure-state-step-pass":
        raise RuntimeError("failure-state-step-gate-not-passed")
    source_path = base.resolve(config["model_flowlines"])
    flowlines = base.load_flowlines(source_path)
    coefficients = base.Coefficients(config["glen_a"], config["fs"])
    initial_volume = base.total_volume(flowlines)
    initial_area = base.total_area(flowlines)
    closure_limit = max(1e-6, initial_volume * 1e-12)
    target_seconds = float(config["years"]) * SECONDS_PER_YEAR
    maximum_dt = float(config["maximum_timestep_seconds"])
    minimum_dt = float(config["minimum_timestep_seconds"])
    section_floor = float(config["section_solver_interior_floor_m2"])
    elapsed = 0.0
    current_dt = maximum_dt
    step_index = 0
    rejected = []
    rows = []
    minimum_used_dt = float("inf")
    maximum_used_dt = 0.0
    maximum_step_closure = 0.0
    maximum_stationarity = 0.0
    maximum_constraints = 0
    constrained_steps = 0
    started = time.perf_counter()
    while elapsed < target_seconds - 1e-9:
        remaining = target_seconds - elapsed
        trial = min(current_dt, remaining)
        if 0.0 < remaining - trial < minimum_dt:
            trial = remaining
        if trial < minimum_dt - 1e-9:
            raise RuntimeError(f"required-final-step-below-floor:{trial}")
        snapshot = copy.deepcopy(flowlines)
        try:
            result = constrained_step(flowlines, coefficients, trial, closure_limit, section_floor)
        except RuntimeError as exc:
            flowlines[:] = snapshot
            message = str(exc)
            rejected.append({"elapsed_year": elapsed / SECONDS_PER_YEAR, "trial_dt_seconds": trial, "reason": message})
            adaptive = (
                message.startswith("internal-active-gap:")
                or message.startswith("tributary-disconnected-from-junction:")
                or message.startswith("active-section-interior-floor:")
                or message.startswith("realized-reverse-or-nonfinite:")
            )
            if not adaptive or trial <= minimum_dt + 1e-9:
                raise RuntimeError(f"rejected-at-floor-or-nonadaptive:{trial}:{message}") from exc
            current_dt = max(minimum_dt, trial / 2.0)
            continue
        elapsed += trial
        step_index += 1
        minimum_used_dt = min(minimum_used_dt, trial)
        maximum_used_dt = max(maximum_used_dt, trial)
        maximum_step_closure = max(maximum_step_closure, abs(result["actual_closure_m3"]))
        maximum_stationarity = max(maximum_stationarity, result["maximum_absolute_KKT_stationarity"])
        maximum_constraints = max(maximum_constraints, result["active_constraint_count"])
        constrained_steps += int(result["active_constraint_count"] > 0)
        if step_index == 1 or result["active_constraint_count"] > 0 or elapsed >= target_seconds - 1e-9:
            rows.append({
                "step": step_index,
                "year": elapsed / SECONDS_PER_YEAR,
                "dt_seconds": trial,
                "volume_m3": base.total_volume(flowlines),
                "area_m2": base.total_area(flowlines),
                **result,
            })
    final_volume = base.total_volume(flowlines)
    final_area = base.total_area(flowlines)
    drift = (final_volume - initial_volume) / initial_volume
    root = next(flowline for flowline in flowlines if flowline.flows_to is None)
    topology_failure = continuity_failure(flowlines)
    checks = {
        "completed_target": abs(elapsed - target_seconds) <= 1e-6,
        "volume_drift": abs(drift) <= float(config["gates"]["volume_drift_fraction_max"]),
        "per_step_closure": maximum_step_closure <= closure_limit,
        "KKT_stationarity": maximum_stationarity <= 1e-5,
        "network_continuity": topology_failure is None,
        "terminal_thickness": float(root.thick[-1]) <= float(config["gates"]["terminal_thickness_m_max"]),
        "finite_geometry": all(np.isfinite(flowline.thick).all() and np.isfinite(flowline.section).all() for flowline in flowlines),
        "nonnegative_geometry": all(np.all(flowline.thick >= -TOL) and np.all(flowline.section >= -TOL) for flowline in flowlines),
        "minimum_timestep_floor": minimum_used_dt >= minimum_dt - 1e-9,
    }
    passed = all(checks.values())
    output = base.resolve(config["output_root"])
    state_path = output / "model_flowlines_0p1y.pkl.gz"
    if passed:
        with gzip.open(state_path, "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "schema": "lia-191-network-monotone-positive-conservative-system2005-zero-mb-v1",
        "status": "zero-mb-0p1y-pass" if passed else "zero-mb-0p1y-fail-mechanism-closed",
        "system_id": 2005,
        "years": float(config["years"]),
        "step_count": step_index,
        "rejected_attempt_count": len(rejected),
        "rejected_attempts": rejected,
        "minimum_timestep_seconds": minimum_used_dt,
        "maximum_timestep_seconds": maximum_used_dt,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_drift_fraction": drift,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "maximum_absolute_step_closure_m3": maximum_step_closure,
        "maximum_absolute_KKT_stationarity": maximum_stationarity,
        "maximum_active_constraint_count": maximum_constraints,
        "constrained_step_count": constrained_steps,
        "root_terminal_thickness_m": float(root.thick[-1]),
        "closure_limit_m3": closure_limit,
        "runtime_seconds": time.perf_counter() - started,
        "checks": checks,
        "solver": "network-monotone-positive-conservative-exact-active-set-v1",
        "section_solver_interior_floor_m2": section_floor,
        "runtime_clip_used": False,
        "source_modified": False,
        "saved_state": str(state_path.relative_to(base.PROJECT)) if passed else None,
        "saved_state_sha256": base.sha256(state_path) if passed else None,
        "zero_mb_5y_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run 5y zero-MB under identical solver contract" if passed else "close mechanism",
        "provenance": {
            "gate_sha256": base.sha256(gate_path),
            "source_model_flowlines_sha256": base.sha256(source_path),
        },
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = base.resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        summary, rows = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-monotone-positive-conservative-system2005-zero-mb-v1",
            "status": "zero-mb-technical-or-scientific-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "source_modified": False,
            "zero_mb_5y_run": False,
            "steady_50y_run": False,
            "historical_run": False,
        }
        rows = []
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "checkpoints.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary))
    if summary["status"] != "zero-mb-0p1y-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

