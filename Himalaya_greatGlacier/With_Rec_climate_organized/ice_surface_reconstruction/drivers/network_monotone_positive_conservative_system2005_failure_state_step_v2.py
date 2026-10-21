"""Numerically interior v2 of the unchanged 2005 monotone/positive step.

V1 completed its KKT solve but failed while expanding a sparse diagnostic
column, and a boundary-roundoff negative section entered MixedBed's nonlinear
setter.  V2 uses a 1e-8 m2 solver-interior section floor (with the same exact
system-volume equality) and expands sparse diagnostics explicitly.  It does
not clip the runtime state.
"""

from __future__ import annotations

import csv
import json
import traceback

import numpy as np
from scipy.sparse import csr_matrix, eye, vstack

from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_variational_system2005_failure_state_step_v1 as step_v1
from . import network_monotone_positive_conservative_system2005_failure_state_step_v1 as v1


OUTPUT = v1.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_monotone_positive_conservative_system2005_failure_state_step_v2"
V1_SUMMARY = v1.OUTPUT / "summary.json"
SECTION_INTERIOR_FLOOR_M2 = 1e-8


def run():
    v1_summary = json.loads(V1_SUMMARY.read_text(encoding="utf-8"))
    if "setting an array element with a sequence" not in v1_summary.get("error", ""):
        raise RuntimeError("v1-is-not-the-registered-sparse-diagnostic-failure")
    config = json.loads(v1.CONFIG.read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    source_path = base.resolve(config["model_flowlines"])
    flowlines = base.load_flowlines(source_path)
    coefficients = base.Coefficients(config["glen_a"], config["fs"])
    source_volume = base.total_volume(flowlines)
    closure_limit = max(1e-6, source_volume * 1e-12)
    elapsed, accepted, rejected = v1.replay_to_negative_section(
        flowlines,
        coefficients,
        closure_limit,
        float(config["years"]) * 365.0 * 24.0 * 3600.0,
    )
    pre_volume = base.total_volume(flowlines)
    pre_sections = [np.asarray(flowline.section, dtype=float).copy() for flowline in flowlines]
    active_local, edges, surface_before, storage, matrix, rhs, incidence = step_v1.assemble(
        flowlines, coefficients, v1.DT_SECONDS
    )
    widths = np.asarray([flowlines[b].widths_m[n] for b, n in active_local], dtype=float)
    old_sections = np.asarray([flowlines[b].section[n] for b, n in active_local], dtype=float)
    surface_lower = surface_before + (SECTION_INTERIOR_FLOOR_M2 - old_sections) / widths
    constraints = vstack([incidence, eye(len(active_local), format="csr")], format="csr")
    lower = np.r_[np.zeros(incidence.shape[0]), surface_lower]
    volume_row = csr_matrix(storage.reshape(1, -1))
    volume_target = float(np.dot(storage, surface_before))
    candidate, unconstrained, active, multipliers, volume_multiplier, history = v1.solve_affine_active_set(
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
    if minimum_section < 0.0:
        raise RuntimeError(f"interior-section-constraint-roundoff:{minimum_section}")
    for flowline, section in zip(flowlines, candidate_sections):
        flowline.section = section
    post_volume = base.total_volume(flowlines)
    realized_surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    realized_slopes = np.asarray(incidence @ realized_surface, dtype=float)
    if not np.all(np.isfinite(realized_surface)):
        raise RuntimeError("nonfinite-realized-MixedBed-surface")
    active_matrix = constraints[active, :].tocsr()
    stationarity = np.asarray(matrix @ candidate - rhs, dtype=float)
    stationarity -= np.asarray(active_matrix.T @ multipliers, dtype=float).ravel()
    stationarity += volume_row.toarray().ravel() * volume_multiplier
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
        "all_inequality_multipliers_nonnegative": minimum_multiplier is None or minimum_multiplier >= -v1.TOL,
        "all_network_edges_monotonic": float(np.min(edge_slopes)) >= -v1.TOL,
        "all_section_constraints_interior": float(np.min(section_residual)) >= -v1.TOL,
        "realized_mixedbed_network_monotonic": float(np.min(realized_slopes)) >= -v1.TOL,
        "minimum_section_positive_floor": minimum_section >= SECTION_INTERIOR_FLOOR_M2 * 0.99,
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
        "schema": "lia-191-network-monotone-positive-conservative-system2005-failure-state-step-v2",
        "status": "real-failure-state-step-pass" if passed else "real-failure-state-step-fail-mechanism-closed",
        "system_id": 2005,
        "mechanism": "network-monotone-and-section-positive-variational-inequality-with-exact-system-volume-equality",
        "section_solver_interior_floor_m2": SECTION_INTERIOR_FLOOR_M2,
        "runtime_clip_used": False,
        "dt_seconds": v1.DT_SECONDS,
        "elapsed_year_at_predecessor_failure": elapsed / (365.0 * 24.0 * 3600.0),
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
        "source_modified": False,
        "zero_mb_0p1y_run": False,
        "zero_mb_5y_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run one versioned 0.1y zero-MB with both constraint families" if passed else "close monotone-positive conservative solver",
        "provenance": {
            "v1_summary_sha256": base.sha256(V1_SUMMARY),
            "config_sha256": base.sha256(v1.CONFIG),
            "predecessor_sha256": base.sha256(v1.PREDECESSOR),
            "source_model_flowlines_sha256": base.sha256(source_path),
        },
    }
    rows = []
    for position, index in enumerate(active):
        rows.append({
            "active_position": position,
            "constraint_kind": "edge" if index < incidence.shape[0] else "section",
            "constraint_index": index if index < incidence.shape[0] else index - incidence.shape[0],
            "multiplier": float(multipliers[position]),
        })
    return summary, rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    try:
        summary, rows = run()
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-monotone-positive-conservative-system2005-failure-state-step-v2",
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
    marker = v1.OUTPUT / "SUPERSEDED_TECHNICAL_FAILURE.json"
    if v1.OUTPUT.exists() and not marker.exists():
        marker.write_text(
            json.dumps({
                "status": "superseded-technical-failure",
                "reason": "sparse diagnostic column expansion and boundary-roundoff setter",
                "superseded_by": str(OUTPUT.relative_to(v1.PROJECT)),
                "scientific_contract_changed": False,
            }, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary))
    if summary["status"] != "real-failure-state-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
