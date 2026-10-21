"""Exact sparse active-set v2 for system 2005's monotone variational step.

V1's generic trust-constr solve terminated with poor optimality and nonzero
global constraint reaction, so it did not scientifically evaluate the convex
mechanism.  V2 solves the same unchanged QP using exact sparse KKT systems for
the small active edge set, adding violated inequalities and removing negative
multipliers until the KKT conditions close.
"""

from __future__ import annotations

import csv
import json
import traceback

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve

from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_variational_system2005_failure_state_step_v1 as v1


OUTPUT = v1.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_monotone_variational_system2005_failure_state_step_v2"
V1_SUMMARY = v1.OUTPUT / "summary.json"


def solve_active_set(matrix, rhs, incidence, tolerance=1e-10, maximum_iterations=100):
    unconstrained = np.asarray(spsolve(matrix, rhs), dtype=float)
    slopes = np.asarray(incidence @ unconstrained, dtype=float)
    active = sorted(np.flatnonzero(slopes < -tolerance).astype(int).tolist())
    history = []
    candidate = unconstrained
    multipliers = np.asarray([], dtype=float)
    for iteration in range(maximum_iterations):
        if active:
            active_matrix = incidence[active, :].tocsr()
            kkt = bmat(
                [[matrix, -active_matrix.T], [active_matrix, csr_matrix((len(active), len(active)))]],
                format="csr",
            )
            solution = np.asarray(spsolve(kkt, np.r_[rhs, np.zeros(len(active))]), dtype=float)
            candidate = solution[: matrix.shape[0]]
            multipliers = solution[matrix.shape[0] :]
        else:
            candidate = unconstrained
            multipliers = np.asarray([], dtype=float)
        slopes = np.asarray(incidence @ candidate, dtype=float)
        inactive = np.ones(incidence.shape[0], dtype=bool)
        inactive[active] = False
        violated_inactive = np.flatnonzero(inactive & (slopes < -tolerance))
        negative_multiplier_positions = np.flatnonzero(multipliers < -tolerance)
        history.append({
            "iteration": iteration + 1,
            "active_count": len(active),
            "minimum_slope_m": float(np.min(slopes)),
            "minimum_multiplier": None if not len(multipliers) else float(np.min(multipliers)),
            "violated_inactive_count": int(len(violated_inactive)),
            "negative_multiplier_count": int(len(negative_multiplier_positions)),
        })
        if len(negative_multiplier_positions):
            position = int(negative_multiplier_positions[np.argmin(multipliers[negative_multiplier_positions])])
            active.pop(position)
            continue
        if len(violated_inactive):
            edge = int(violated_inactive[np.argmin(slopes[violated_inactive])])
            active.append(edge)
            active.sort()
            continue
        return candidate, unconstrained, active, multipliers, history
    raise RuntimeError("active-set-maximum-iterations")


def run():
    predecessor = json.loads(v1.PREDECESSOR.read_text(encoding="utf-8"))
    v1_summary = json.loads(V1_SUMMARY.read_text(encoding="utf-8"))
    if v1_summary.get("optimizer_optimality", 0.0) <= 1e-8:
        raise RuntimeError("v1-is-not-the-registered-poor-optimality-failure")
    config = json.loads(v1.CONFIG.read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    source_path = base.resolve(config["model_flowlines"])
    flowlines = base.load_flowlines(source_path)
    coefficients = base.Coefficients(config["glen_a"], config["fs"])
    source_volume = base.total_volume(flowlines)
    closure_limit = max(1e-6, source_volume * 1e-12)
    accepted, rejected = v1.replay_to_failure(flowlines, coefficients, closure_limit)
    pre_volume = base.total_volume(flowlines)
    pre_sections = [np.asarray(flowline.section, dtype=float).copy() for flowline in flowlines]
    active_local, edges, surface_before, storage, matrix, rhs, incidence = v1.assemble(
        flowlines, coefficients, v1.DT_SECONDS
    )
    candidate, unconstrained, active_edges, multipliers, active_history = solve_active_set(
        matrix, rhs, incidence
    )
    unconstrained_slopes = np.asarray(incidence @ unconstrained, dtype=float)
    candidate_slopes = np.asarray(incidence @ candidate, dtype=float)
    target_volume_delta = storage * (candidate - surface_before)
    candidate_sections = [section.copy() for section in pre_sections]
    for global_index, (branch, node) in enumerate(active_local):
        candidate_sections[branch][node] += target_volume_delta[global_index] / float(flowlines[branch].dx_meter)
    minimum_section = min(float(np.min(section)) for section in candidate_sections)
    if minimum_section < -1e-10:
        raise RuntimeError(f"constrained-step-requires-negative-section:{minimum_section}")
    for flowline, section in zip(flowlines, candidate_sections):
        flowline.section = section
    post_volume = base.total_volume(flowlines)
    realized_surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    realized_slopes = np.asarray(incidence @ realized_surface, dtype=float)
    active_matrix = incidence[active_edges, :].tocsr() if active_edges else csr_matrix((0, len(active_local)))
    stationarity = np.asarray(matrix @ candidate - rhs, dtype=float)
    if active_edges:
        stationarity -= np.asarray(active_matrix.T @ multipliers, dtype=float).ravel()
    maximum_stationarity = float(np.max(np.abs(stationarity)))
    global_stationarity_sum = float(np.sum(stationarity))
    root = next(flowline for flowline in flowlines if flowline.flows_to is None)
    reverse_within = 0
    reverse_junction = 0
    edge_rows = []
    for edge_index, (upstream, downstream, conductance, kind) in enumerate(edges):
        realized_slope = float(realized_slopes[edge_index])
        if realized_slope < -v1.TOL:
            if kind == "within":
                reverse_within += 1
            else:
                reverse_junction += 1
        edge_rows.append({
            "edge_index": edge_index,
            "edge_kind": kind,
            "upstream_global": upstream,
            "downstream_global": downstream,
            "conductance_m2_s": conductance,
            "unconstrained_slope_m": float(unconstrained_slopes[edge_index]),
            "constrained_target_slope_m": float(candidate_slopes[edge_index]),
            "realized_mixedbed_slope_m": realized_slope,
            "active_constraint": edge_index in active_edges,
            "constraint_multiplier": float(multipliers[active_edges.index(edge_index)]) if edge_index in active_edges else 0.0,
        })
    actual_volume_change = post_volume - pre_volume
    global_target_change = float(np.sum(target_volume_delta))
    minimum_multiplier = None if not len(multipliers) else float(np.min(multipliers))
    checks = {
        "v3_failure_state_reproduced": int(np.count_nonzero(unconstrained_slopes < -v1.TOL)) == 1,
        "active_set_converged": bool(active_history),
        "all_constraint_multipliers_nonnegative": minimum_multiplier is None or minimum_multiplier >= -1e-10,
        "KKT_stationarity": maximum_stationarity <= 1e-6,
        "KKT_global_stationarity_sum": abs(global_stationarity_sum) <= closure_limit,
        "target_network_monotonic": float(np.min(candidate_slopes)) >= -v1.TOL,
        "realized_mixedbed_network_monotonic": float(np.min(realized_slopes)) >= -v1.TOL,
        "reverse_flux_zero": reverse_within == 0 and reverse_junction == 0,
        "minimum_section_nonnegative": minimum_section >= -1e-10,
        "global_target_volume_closed": abs(global_target_change) <= closure_limit,
        "actual_volume_closed": abs(actual_volume_change) <= closure_limit,
        "root_terminal_thickness_zero": abs(float(root.thick[-1])) <= 1e-12,
        "finite_geometry": all(
            np.isfinite(flowline.section).all()
            and np.isfinite(flowline.thick).all()
            and np.isfinite(flowline.surface_h).all()
            for flowline in flowlines
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema": "lia-191-network-monotone-variational-system2005-failure-state-step-v2",
        "status": "real-failure-state-step-pass" if passed else "real-failure-state-step-fail-mechanism-closed",
        "system_id": 2005,
        "mechanism": "network-coupled-SPD-implicit-variational-inequality-exact-sparse-active-set",
        "dt_seconds": v1.DT_SECONDS,
        "v3_accepted_substep_count_before_failure": len(accepted),
        "v3_rejected_attempts": rejected,
        "active_node_count": len(active_local),
        "edge_count": len(edges),
        "unconstrained_reverse_edge_count": int(np.count_nonzero(unconstrained_slopes < -v1.TOL)),
        "active_constraint_count": len(active_edges),
        "active_constraint_edges": active_edges,
        "minimum_constraint_multiplier": minimum_multiplier,
        "minimum_constrained_target_slope_m": float(np.min(candidate_slopes)),
        "minimum_realized_mixedbed_slope_m": float(np.min(realized_slopes)),
        "realized_reverse_within_count": reverse_within,
        "realized_reverse_junction_count": reverse_junction,
        "maximum_absolute_KKT_stationarity": maximum_stationarity,
        "global_KKT_stationarity_sum": global_stationarity_sum,
        "minimum_candidate_section_m2": minimum_section,
        "pre_step_volume_m3": pre_volume,
        "post_step_volume_m3": post_volume,
        "actual_volume_change_m3": actual_volume_change,
        "linearized_target_volume_change_m3": global_target_change,
        "closure_limit_m3": closure_limit,
        "active_set_history": active_history,
        "checks": checks,
        "runtime_clip_used": False,
        "source_modified": False,
        "zero_mb_0p1y_run": False,
        "zero_mb_5y_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run one versioned 0.1y zero-MB representative" if passed else "close monotone variational solver",
        "provenance": {
            "v1_summary_sha256": base.sha256(V1_SUMMARY),
            "predecessor_sha256": base.sha256(v1.PREDECESSOR),
            "source_model_flowlines_sha256": base.sha256(source_path),
        },
    }
    return summary, edge_rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    try:
        summary, edge_rows = run()
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-monotone-variational-system2005-failure-state-step-v2",
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
        edge_rows = []
    if edge_rows:
        with (OUTPUT / "edge_diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(edge_rows[0]))
            writer.writeheader()
            writer.writerows(edge_rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if v1.OUTPUT.exists():
        marker = v1.OUTPUT / "SUPERSEDED_NUMERICAL_SOLVER_FAILURE.json"
        if not marker.exists():
            marker.write_text(
                json.dumps(
                    {
                        "status": "superseded-numerical-optimizer-failure",
                        "reason": "trust-constr terminated with poor optimality and nonzero global reaction",
                        "superseded_by": str(OUTPUT.relative_to(v1.PROJECT)),
                        "scientific_contract_changed": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "real-failure-state-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

