"""0.1 year zero-MB gate for system 2005's monotone variational network solver."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_variational_system2005_failure_state_step_v1 as step_v1
from . import network_monotone_variational_system2005_failure_state_step_v2 as step_v2


PROJECT = Path(__file__).resolve().parents[4]
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
TOL = 1e-10


def monotone_step(flowlines, coefficients, dt_seconds, closure_limit):
    active_local, edges, surface_before, storage, matrix, rhs, incidence = step_v1.assemble(
        flowlines, coefficients, dt_seconds
    )
    candidate, unconstrained, active_edges, multipliers, history = step_v2.solve_active_set(
        matrix, rhs, incidence
    )
    unconstrained_slopes = np.asarray(incidence @ unconstrained, dtype=float)
    target_slopes = np.asarray(incidence @ candidate, dtype=float)
    target_volume_delta = storage * (candidate - surface_before)
    global_target_change = float(np.sum(target_volume_delta))
    if abs(global_target_change) > closure_limit:
        raise RuntimeError(f"target-volume-closure:{global_target_change}")
    sections = [np.asarray(flowline.section, dtype=float).copy() for flowline in flowlines]
    for global_index, (branch, node) in enumerate(active_local):
        sections[branch][node] += target_volume_delta[global_index] / float(flowlines[branch].dx_meter)
    minimum_section = min(float(np.min(section)) for section in sections)
    if minimum_section < -1e-10:
        raise RuntimeError(f"negative-section-required:{minimum_section}")
    before_volume = base.total_volume(flowlines)
    for flowline, section in zip(flowlines, sections):
        flowline.section = section
    after_volume = base.total_volume(flowlines)
    actual_closure = after_volume - before_volume
    if abs(actual_closure) > closure_limit:
        raise RuntimeError(f"actual-volume-closure:{actual_closure}")
    realized_surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    realized_slopes = np.asarray(incidence @ realized_surface, dtype=float)
    if float(np.min(realized_slopes)) < -TOL:
        raise RuntimeError(f"realized-reverse-surface:{float(np.min(realized_slopes))}")
    active_matrix = incidence[active_edges, :].tocsr() if active_edges else None
    stationarity = np.asarray(matrix @ candidate - rhs, dtype=float)
    if active_edges:
        stationarity -= np.asarray(active_matrix.T @ multipliers, dtype=float).ravel()
    maximum_stationarity = float(np.max(np.abs(stationarity)))
    if maximum_stationarity > 1e-5:
        raise RuntimeError(f"KKT-stationarity:{maximum_stationarity}")
    minimum_multiplier = None if not len(multipliers) else float(np.min(multipliers))
    if minimum_multiplier is not None and minimum_multiplier < -TOL:
        raise RuntimeError(f"negative-constraint-multiplier:{minimum_multiplier}")
    root = next(flowline for flowline in flowlines if flowline.flows_to is None)
    return {
        "active_node_count": len(active_local),
        "edge_count": len(edges),
        "unconstrained_reverse_edge_count": int(np.count_nonzero(unconstrained_slopes < -TOL)),
        "active_constraint_count": len(active_edges),
        "minimum_constraint_multiplier": minimum_multiplier,
        "minimum_target_edge_slope_m": float(np.min(target_slopes)),
        "minimum_realized_edge_slope_m": float(np.min(realized_slopes)),
        "maximum_absolute_KKT_stationarity": maximum_stationarity,
        "actual_closure_m3": actual_closure,
        "linearized_target_volume_change_m3": global_target_change,
        "minimum_section_m2": minimum_section,
        "minimum_thickness_m": min(float(np.min(flowline.thick)) for flowline in flowlines),
        "root_terminal_thickness_m": float(root.thick[-1]),
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
    if minimum_dt < 30.0 or maximum_dt < minimum_dt:
        raise RuntimeError("invalid-timestep-contract")
    elapsed = 0.0
    current_dt = maximum_dt
    step_index = 0
    rejected = []
    rows = []
    minimum_used_dt = float("inf")
    maximum_used_dt = 0.0
    maximum_step_closure = 0.0
    maximum_kkt_stationarity = 0.0
    maximum_active_constraints = 0
    constrained_step_count = 0
    maximum_unconstrained_reverse = 0
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
            result = monotone_step(flowlines, coefficients, trial, closure_limit)
        except RuntimeError as exc:
            flowlines[:] = snapshot
            message = str(exc)
            allowed = message.startswith("negative-section-required:") or message.startswith("realized-reverse-surface:")
            rejected.append({"trial_dt_seconds": trial, "reason": message})
            if not allowed or trial <= minimum_dt + 1e-9:
                raise RuntimeError(f"rejected-at-floor-or-nonadaptive:{trial}:{message}") from exc
            current_dt = max(minimum_dt, trial / 2.0)
            continue
        elapsed += trial
        step_index += 1
        minimum_used_dt = min(minimum_used_dt, trial)
        maximum_used_dt = max(maximum_used_dt, trial)
        maximum_step_closure = max(maximum_step_closure, abs(result["actual_closure_m3"]))
        maximum_kkt_stationarity = max(maximum_kkt_stationarity, result["maximum_absolute_KKT_stationarity"])
        maximum_active_constraints = max(maximum_active_constraints, result["active_constraint_count"])
        constrained_step_count += int(result["active_constraint_count"] > 0)
        maximum_unconstrained_reverse = max(maximum_unconstrained_reverse, result["unconstrained_reverse_edge_count"])
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
    gates = config["gates"]
    checks = {
        "completed_target": abs(elapsed - target_seconds) <= 1e-6,
        "volume_drift": abs(drift) <= float(gates["volume_drift_fraction_max"]),
        "per_step_closure": maximum_step_closure <= closure_limit,
        "KKT_stationarity": maximum_kkt_stationarity <= 1e-5,
        "terminal_thickness": float(root.thick[-1]) <= float(gates["terminal_thickness_m_max"]),
        "finite_geometry": all(
            np.isfinite(flowline.thick).all()
            and np.isfinite(flowline.section).all()
            and np.isfinite(flowline.surface_h).all()
            for flowline in flowlines
        ),
        "nonnegative_geometry": all(
            np.all(flowline.thick >= -1e-10) and np.all(flowline.section >= -1e-10)
            for flowline in flowlines
        ),
        "minimum_timestep_floor": minimum_used_dt >= minimum_dt - 1e-9,
    }
    passed = all(checks.values())
    output = base.resolve(config["output_root"])
    state_path = output / "model_flowlines_0p1y.pkl.gz"
    if passed:
        with gzip.open(state_path, "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "schema": "lia-191-network-monotone-variational-system2005-zero-mb-v1",
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
        "maximum_absolute_KKT_stationarity": maximum_kkt_stationarity,
        "maximum_active_constraint_count": maximum_active_constraints,
        "constrained_step_count": constrained_step_count,
        "maximum_unconstrained_reverse_edge_count": maximum_unconstrained_reverse,
        "root_terminal_thickness_m": float(root.thick[-1]),
        "closure_limit_m3": closure_limit,
        "runtime_seconds": time.perf_counter() - started,
        "checks": checks,
        "solver": "network-monotone-variational-exact-sparse-active-set-v1",
        "runtime_clip_used": False,
        "source_modified": False,
        "saved_state": str(state_path.relative_to(PROJECT)) if passed else None,
        "saved_state_sha256": base.sha256(state_path) if passed else None,
        "zero_mb_5y_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run 5y zero-MB from source under identical solver contract" if passed else "close mechanism",
        "provenance": {
            "config_path": str(base.resolve(config["config_path"]).relative_to(PROJECT)),
            "config_sha256": base.sha256(base.resolve(config["config_path"])),
            "gate_sha256": base.sha256(gate_path),
            "source_model_flowlines_sha256": base.sha256(source_path),
        },
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = base.resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        summary, rows = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-monotone-variational-system2005-zero-mb-v1",
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

