"""Mass-conserving monotone/positive variational step at the 2005 failure state.

The predecessor enforces network surface monotonicity but reaches a negative
section request at the frozen 30 s floor.  This one-step gate adds the physical
linear inequalities section_new >= 0 and an exact system-volume equality.
Unlike runtime clipping, the equality redistributes the constrained increment
within the coupled solve and forbids mass creation.  No 0.1 year integration is
performed here.
"""

from __future__ import annotations

import copy
import csv
import json
import traceback

import numpy as np
from scipy.sparse import bmat, csr_matrix, eye, vstack
from scipy.sparse.linalg import spsolve

from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_variational_system2005_failure_state_step_v1 as step_v1
from . import network_monotone_variational_system2005_zero_mb_v1 as monotone_zero


PROJECT = step_v1.PROJECT
CONFIG = PROJECT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/network_monotone_variational_system2005_zero_mb_0p1y_v1.json"
PREDECESSOR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_monotone_variational_system2005_zero_mb_0p1y_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_monotone_positive_conservative_system2005_failure_state_step_v1"
DT_SECONDS = 30.0
TOL = 1e-10


def solve_affine_active_set(matrix, rhs, constraint_matrix, lower, volume_row, volume_target, tolerance=1e-10, maximum_iterations=200):
    unconstrained = np.asarray(spsolve(matrix, rhs), dtype=float)
    residual = np.asarray(constraint_matrix @ unconstrained, dtype=float) - lower
    active = sorted(np.flatnonzero(residual < -tolerance).astype(int).tolist())
    history = []
    candidate = unconstrained
    multipliers = np.asarray([], dtype=float)
    volume_multiplier = 0.0
    for iteration in range(maximum_iterations):
        active_matrix = constraint_matrix[active, :].tocsr() if active else csr_matrix((0, matrix.shape[0]))
        zero_aa = csr_matrix((len(active), len(active)))
        zero_av = csr_matrix((len(active), 1))
        zero_va = csr_matrix((1, len(active)))
        zero_vv = csr_matrix((1, 1))
        kkt = bmat(
            [
                [matrix, -active_matrix.T, volume_row.T],
                [active_matrix, zero_aa, zero_av],
                [volume_row, zero_va, zero_vv],
            ],
            format="csr",
        )
        solution = np.asarray(
            spsolve(kkt, np.r_[rhs, lower[active], volume_target]), dtype=float
        )
        candidate = solution[: matrix.shape[0]]
        multipliers = solution[matrix.shape[0] : matrix.shape[0] + len(active)]
        volume_multiplier = float(solution[-1])
        residual = np.asarray(constraint_matrix @ candidate, dtype=float) - lower
        inactive = np.ones(constraint_matrix.shape[0], dtype=bool)
        inactive[active] = False
        violated = np.flatnonzero(inactive & (residual < -tolerance))
        negative_positions = np.flatnonzero(multipliers < -tolerance)
        history.append({
            "iteration": iteration + 1,
            "active_count": len(active),
            "minimum_constraint_residual": float(np.min(residual)),
            "minimum_multiplier": None if not len(multipliers) else float(np.min(multipliers)),
            "violated_inactive_count": int(len(violated)),
            "negative_multiplier_count": int(len(negative_positions)),
            "volume_residual": float(volume_row @ candidate - volume_target),
        })
        if len(negative_positions):
            position = int(negative_positions[np.argmin(multipliers[negative_positions])])
            active.pop(position)
            continue
        if len(violated):
            constraint = int(violated[np.argmin(residual[violated])])
            active.append(constraint)
            active.sort()
            continue
        return candidate, unconstrained, active, multipliers, volume_multiplier, history
    raise RuntimeError("affine-active-set-maximum-iterations")


def replay_to_negative_section(flowlines, coefficients, closure_limit, target_seconds):
    elapsed = 0.0
    current_dt = 3600.0
    accepted = []
    rejected = []
    while elapsed < target_seconds - 1e-9:
        remaining = target_seconds - elapsed
        trial = min(current_dt, remaining)
        if 0.0 < remaining - trial < DT_SECONDS:
            trial = remaining
        snapshot = copy.deepcopy(flowlines)
        try:
            result = monotone_zero.monotone_step(flowlines, coefficients, trial, closure_limit)
        except RuntimeError as exc:
            flowlines[:] = snapshot
            message = str(exc)
            rejected.append({"elapsed_year": elapsed / monotone_zero.SECONDS_PER_YEAR, "trial_dt_seconds": trial, "reason": message})
            if trial <= DT_SECONDS + 1e-9:
                if not message.startswith("negative-section-required:"):
                    raise RuntimeError(f"unexpected-floor-failure:{message}")
                return elapsed, accepted, rejected
            if not message.startswith("negative-section-required:") and not message.startswith("realized-reverse-surface:"):
                raise
            current_dt = max(DT_SECONDS, trial / 2.0)
            continue
        elapsed += trial
        accepted.append({"dt_seconds": trial, "active_constraint_count": result["active_constraint_count"]})
    raise RuntimeError("predecessor-completed-without-negative-section-failure")


def run():
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if "negative-section-required:-20.41062416456739" not in predecessor.get("error", ""):
        raise RuntimeError("predecessor-is-not-the-registered-floor-negative-section")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    source_path = base.resolve(config["model_flowlines"])
    flowlines = base.load_flowlines(source_path)
    coefficients = base.Coefficients(config["glen_a"], config["fs"])
    source_volume = base.total_volume(flowlines)
    closure_limit = max(1e-6, source_volume * 1e-12)
    elapsed, accepted, rejected = replay_to_negative_section(
        flowlines, coefficients, closure_limit, float(config["years"]) * monotone_zero.SECONDS_PER_YEAR
    )
    pre_volume = base.total_volume(flowlines)
    pre_sections = [np.asarray(flowline.section, dtype=float).copy() for flowline in flowlines]
    active_local, edges, surface_before, storage, matrix, rhs, incidence = step_v1.assemble(
        flowlines, coefficients, DT_SECONDS
    )
    widths = np.asarray([flowlines[b].widths_m[n] for b, n in active_local], dtype=float)
    old_sections = np.asarray([flowlines[b].section[n] for b, n in active_local], dtype=float)
    surface_lower = surface_before - old_sections / widths
    identity = eye(len(active_local), format="csr")
    constraints = vstack([incidence, identity], format="csr")
    lower = np.r_[np.zeros(incidence.shape[0]), surface_lower]
    volume_row = csr_matrix(storage.reshape(1, -1))
    volume_target = float(np.dot(storage, surface_before))
    candidate, unconstrained, active, multipliers, volume_multiplier, history = solve_affine_active_set(
        matrix, rhs, constraints, lower, volume_row, volume_target
    )
    residual = np.asarray(constraints @ candidate, dtype=float) - lower
    edge_slopes = residual[: incidence.shape[0]]
    section_residual = residual[incidence.shape[0] :]
    target_delta = storage * (candidate - surface_before)
    candidate_sections = [section.copy() for section in pre_sections]
    for global_index, (branch, node) in enumerate(active_local):
        candidate_sections[branch][node] += target_delta[global_index] / float(flowlines[branch].dx_meter)
    minimum_section = min(float(np.min(section)) for section in candidate_sections)
    for flowline, section in zip(flowlines, candidate_sections):
        flowline.section = section
    post_volume = base.total_volume(flowlines)
    realized_surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    realized_slopes = np.asarray(incidence @ realized_surface, dtype=float)
    active_matrix = constraints[active, :].tocsr()
    stationarity = np.asarray(matrix @ candidate - rhs, dtype=float)
    stationarity -= np.asarray(active_matrix.T @ multipliers, dtype=float).ravel()
    stationarity += np.asarray(volume_row.T * volume_multiplier, dtype=float).ravel()
    active_edges = [index for index in active if index < incidence.shape[0]]
    active_sections = [index - incidence.shape[0] for index in active if index >= incidence.shape[0]]
    actual_volume_change = post_volume - pre_volume
    global_target_change = float(np.sum(target_delta))
    minimum_multiplier = float(np.min(multipliers)) if len(multipliers) else None
    maximum_stationarity = float(np.max(np.abs(stationarity)))
    root = next(flowline for flowline in flowlines if flowline.flows_to is None)
    checks = {
        "predecessor_negative_section_state_reproduced": bool(rejected) and rejected[-1]["reason"].startswith("negative-section-required:"),
        "active_set_converged": bool(history),
        "all_inequality_multipliers_nonnegative": minimum_multiplier is None or minimum_multiplier >= -TOL,
        "all_network_edges_monotonic": float(np.min(edge_slopes)) >= -TOL,
        "all_section_constraints_nonnegative": float(np.min(section_residual)) >= -TOL,
        "realized_mixedbed_network_monotonic": float(np.min(realized_slopes)) >= -TOL,
        "minimum_section_nonnegative": minimum_section >= -TOL,
        "global_target_volume_closed": abs(global_target_change) <= closure_limit,
        "actual_volume_closed": abs(actual_volume_change) <= closure_limit,
        "KKT_stationarity": maximum_stationarity <= 1e-5,
        "root_terminal_thickness_zero": abs(float(root.thick[-1])) <= 1e-12,
        "finite_geometry": all(
            np.isfinite(flowline.section).all() and np.isfinite(flowline.thick).all() and np.isfinite(flowline.surface_h).all()
            for flowline in flowlines
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema": "lia-191-network-monotone-positive-conservative-system2005-failure-state-step-v1",
        "status": "real-failure-state-step-pass" if passed else "real-failure-state-step-fail-mechanism-closed",
        "system_id": 2005,
        "mechanism": "network-monotone-and-section-positive-variational-inequality-with-exact-system-volume-equality",
        "dt_seconds": DT_SECONDS,
        "elapsed_year_at_predecessor_failure": elapsed / monotone_zero.SECONDS_PER_YEAR,
        "accepted_predecessor_step_count": len(accepted),
        "rejected_predecessor_attempts": rejected,
        "active_node_count": len(active_local),
        "edge_count": len(edges),
        "active_constraint_count": len(active),
        "active_edge_constraint_count": len(active_edges),
        "active_section_constraint_count": len(active_sections),
        "active_section_global_indices": active_sections,
        "minimum_inequality_multiplier": minimum_multiplier,
        "volume_equality_multiplier": volume_multiplier,
        "minimum_edge_constraint_residual_m": float(np.min(edge_slopes)),
        "minimum_section_constraint_residual_m": float(np.min(section_residual)),
        "minimum_realized_mixedbed_edge_slope_m": float(np.min(realized_slopes)),
        "minimum_candidate_section_m2": minimum_section,
        "maximum_absolute_KKT_stationarity": maximum_stationarity,
        "pre_step_volume_m3": pre_volume,
        "post_step_volume_m3": post_volume,
        "actual_volume_change_m3": actual_volume_change,
        "linearized_target_volume_change_m3": global_target_change,
        "closure_limit_m3": closure_limit,
        "active_set_history": history,
        "checks": checks,
        "runtime_clip_used": False,
        "source_modified": False,
        "zero_mb_0p1y_run": False,
        "zero_mb_5y_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run one versioned 0.1y zero-MB with both constraint families" if passed else "close monotone-positive conservative solver",
        "provenance": {
            "config_sha256": base.sha256(CONFIG),
            "predecessor_sha256": base.sha256(PREDECESSOR),
            "source_model_flowlines_sha256": base.sha256(source_path),
        },
    }
    constraint_rows = []
    for position, index in enumerate(active):
        constraint_rows.append({
            "active_position": position,
            "constraint_kind": "edge" if index < incidence.shape[0] else "section",
            "constraint_index": index if index < incidence.shape[0] else index - incidence.shape[0],
            "multiplier": float(multipliers[position]),
        })
    return summary, constraint_rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    try:
        summary, rows = run()
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-monotone-positive-conservative-system2005-failure-state-step-v1",
            "status": "technical-or-scientific-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "source_modified": False,
            "zero_mb_0p1y_run": False,
            "zero_mb_5y_run": False,
            "steady_50y_run": False,
            "historical_run": False,
        }
        rows = []
    if rows:
        with (OUTPUT / "active_constraints.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "real-failure-state-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

