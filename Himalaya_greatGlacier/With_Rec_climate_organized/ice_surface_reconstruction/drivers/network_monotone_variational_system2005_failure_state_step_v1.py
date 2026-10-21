"""Constrained variational 30 s step at system 2005's first v3 failure state.

The driver exactly replays the rollback-safe v3 dyadic schedule until the
registered 30 s reverse-flux failure, retaining only its already accepted
substeps.  At that frozen pre-failure state it solves the same SPD implicit
energy with linear inequalities s_upstream >= s_downstream on every active
within-branch and junction edge.  Constraint reactions are internal and each
row has zero network sum; no flux, section or surface is clipped.  This is a
single real-state gate, not a 0.1 year run.
"""

from __future__ import annotations

import copy
import csv
import json
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base


PROJECT = Path(__file__).resolve().parents[4]
CONFIG = PROJECT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/network_coupled_semiimplicit_system2005_zero_mb_0p1y_v3.json"
PREDECESSOR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_coupled_semiimplicit_system2005_zero_mb_0p1y_v3/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_monotone_variational_system2005_failure_state_step_v1"
DT_SECONDS = 30.0
TOL = 1e-10


def assemble(flowlines, coefficients, dt_seconds):
    active_local, edges = base.build_active_graph(flowlines, coefficients)
    surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    storage = np.asarray([flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in active_local], dtype=float)
    row, col, data = [], [], []
    for upstream, downstream, conductance, _kind in edges:
        value = float(dt_seconds) * float(conductance)
        row.extend([upstream, downstream, upstream, downstream])
        col.extend([upstream, downstream, downstream, upstream])
        data.extend([value, value, -value, -value])
    graph = coo_matrix((data, (row, col)), shape=(len(active_local), len(active_local))).tocsr()
    matrix = diags(storage, format="csr") + graph
    rhs = storage * surface
    incidence_rows, incidence_cols, incidence_data = [], [], []
    for edge_index, (upstream, downstream, _conductance, _kind) in enumerate(edges):
        incidence_rows.extend([edge_index, edge_index])
        incidence_cols.extend([upstream, downstream])
        incidence_data.extend([1.0, -1.0])
    incidence = coo_matrix(
        (incidence_data, (incidence_rows, incidence_cols)),
        shape=(len(edges), len(active_local)),
    ).tocsr()
    return active_local, edges, surface, storage, matrix, rhs, incidence


def replay_to_failure(flowlines, coefficients, closure_limit):
    current_dt = 3600.0
    remaining = 3600.0
    accepted = []
    rejected = []
    while remaining > 1e-9:
        trial = min(current_dt, remaining)
        snapshot = copy.deepcopy(flowlines)
        try:
            result = base.one_step(flowlines, coefficients, trial, closure_limit)
        except RuntimeError as exc:
            message = str(exc)
            allowed = message.startswith("negative-section-required:") or message.startswith("reverse-flux:")
            if not allowed:
                raise
            flowlines[:] = snapshot
            rejected.append({"trial_dt_seconds": trial, "reason": message})
            if trial <= DT_SECONDS + 1e-9:
                if not message.startswith("reverse-flux:within=1,junction=0"):
                    raise RuntimeError(f"unexpected-registered-floor-failure:{message}")
                return accepted, rejected
            current_dt = max(DT_SECONDS, trial / 2.0)
            continue
        remaining -= trial
        accepted.append({"dt_seconds": trial, **result})
    raise RuntimeError("v3-replay-completed-block-without-registered-failure")


def run() -> tuple[dict, list[dict]]:
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if "rejected-at-30s:reverse-flux:within=1,junction=0" not in predecessor.get("error", ""):
        raise RuntimeError("v3-predecessor-is-not-the-registered-reverse-flux-failure")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    source_path = base.resolve(config["model_flowlines"])
    flowlines = base.load_flowlines(source_path)
    coefficients = base.Coefficients(config["glen_a"], config["fs"])
    source_volume = base.total_volume(flowlines)
    closure_limit = max(1e-6, source_volume * 1e-12)
    accepted, rejected = replay_to_failure(flowlines, coefficients, closure_limit)
    pre_volume = base.total_volume(flowlines)
    pre_sections = [np.asarray(flowline.section, dtype=float).copy() for flowline in flowlines]
    active_local, edges, surface_before, storage, matrix, rhs, incidence = assemble(
        flowlines, coefficients, DT_SECONDS
    )
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.diagonal() <= 0):
        raise RuntimeError("invalid-SPD-matrix")
    unconstrained = np.asarray(spsolve(matrix, rhs), dtype=float)
    unconstrained_edge_slopes = np.asarray(incidence @ unconstrained, dtype=float)
    unconstrained_reverse = int(np.count_nonzero(unconstrained_edge_slopes < -TOL))
    if unconstrained_reverse != 1:
        raise RuntimeError(f"failure-state-not-reproduced:{unconstrained_reverse}")

    def objective(surface):
        return float(0.5 * surface @ (matrix @ surface) - rhs @ surface)

    def gradient(surface):
        return np.asarray(matrix @ surface - rhs, dtype=float)

    constraints = LinearConstraint(incidence, lb=np.zeros(len(edges)), ub=np.full(len(edges), np.inf))
    result = minimize(
        objective,
        surface_before,
        jac=gradient,
        hess=lambda _surface: matrix,
        constraints=[constraints],
        method="trust-constr",
        options={
            "gtol": 1e-10,
            "xtol": 1e-12,
            "barrier_tol": 1e-12,
            "maxiter": 1000,
            "sparse_jacobian": True,
            "factorization_method": "AugmentedSystem",
            "verbose": 0,
        },
    )
    candidate = np.asarray(result.x, dtype=float)
    edge_slopes = np.asarray(incidence @ candidate, dtype=float)
    minimum_edge_slope = float(np.min(edge_slopes))
    target_section_delta = storage * (candidate - surface_before)
    global_target_volume_delta = float(np.sum(target_section_delta))
    candidate_sections = [section.copy() for section in pre_sections]
    for global_index, (branch, node) in enumerate(active_local):
        candidate_sections[branch][node] += target_section_delta[global_index] / float(flowlines[branch].dx_meter)
    minimum_candidate_section = min(float(np.min(section)) for section in candidate_sections)
    if minimum_candidate_section < -1e-10:
        raise RuntimeError(f"constrained-step-requires-negative-section:{minimum_candidate_section}")
    for flowline, section in zip(flowlines, candidate_sections):
        flowline.section = section
    post_volume = base.total_volume(flowlines)
    actual_volume_change = post_volume - pre_volume
    realized_surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    realized_edge_slopes = np.asarray(incidence @ realized_surface, dtype=float)
    minimum_realized_edge_slope = float(np.min(realized_edge_slopes))
    edge_rows = []
    reverse_within = 0
    reverse_junction = 0
    active_constraint_count = 0
    for edge_index, (upstream, downstream, conductance, kind) in enumerate(edges):
        target_slope = float(edge_slopes[edge_index])
        realized_slope = float(realized_edge_slopes[edge_index])
        flux = float(conductance * target_slope)
        active_constraint_count += int(target_slope <= 1e-7)
        if realized_slope < -TOL:
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
            "unconstrained_target_slope_m": float(unconstrained_edge_slopes[edge_index]),
            "constrained_target_slope_m": target_slope,
            "realized_mixedbed_slope_m": realized_slope,
            "constrained_flux_m3_s": flux,
        })
    kkt_stationarity = gradient(candidate)
    global_stationarity_sum = float(np.sum(kkt_stationarity))
    root = next(flowline for flowline in flowlines if flowline.flows_to is None)
    checks = {
        "v3_failure_state_reproduced": unconstrained_reverse == 1,
        "optimizer_success": bool(result.success),
        "optimizer_constraint_violation": float(result.constr_violation) <= 1e-8,
        "optimizer_optimality": float(result.optimality) <= 1e-8,
        "target_network_monotonic": minimum_edge_slope >= -TOL,
        "realized_mixedbed_network_monotonic": minimum_realized_edge_slope >= -TOL,
        "reverse_flux_zero": reverse_within == 0 and reverse_junction == 0,
        "minimum_section_nonnegative": minimum_candidate_section >= -1e-10,
        "global_target_volume_closed": abs(global_target_volume_delta) <= closure_limit,
        "actual_volume_closed": abs(actual_volume_change) <= closure_limit,
        "constraint_reaction_global_sum_closed": abs(global_stationarity_sum) <= closure_limit,
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
        "schema": "lia-191-network-monotone-variational-system2005-failure-state-step-v1",
        "status": "real-failure-state-step-pass" if passed else "real-failure-state-step-fail-mechanism-closed",
        "system_id": 2005,
        "mechanism": "network-coupled-SPD-implicit-variational-inequality-with-edge-surface-monotonicity",
        "dt_seconds": DT_SECONDS,
        "v3_accepted_substep_count_before_failure": len(accepted),
        "v3_accepted_substeps": accepted,
        "v3_rejected_attempts": rejected,
        "active_node_count": len(active_local),
        "edge_count": len(edges),
        "unconstrained_reverse_edge_count": unconstrained_reverse,
        "constrained_active_edge_count": active_constraint_count,
        "minimum_constrained_target_edge_slope_m": minimum_edge_slope,
        "minimum_realized_mixedbed_edge_slope_m": minimum_realized_edge_slope,
        "realized_reverse_within_count": reverse_within,
        "realized_reverse_junction_count": reverse_junction,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.niter),
        "optimizer_optimality": float(result.optimality),
        "optimizer_constraint_violation": float(result.constr_violation),
        "minimum_candidate_section_m2": minimum_candidate_section,
        "pre_step_volume_m3": pre_volume,
        "post_step_volume_m3": post_volume,
        "actual_volume_change_m3": actual_volume_change,
        "linearized_target_volume_change_m3": global_target_volume_delta,
        "constraint_reaction_global_sum_m3": global_stationarity_sum,
        "closure_limit_m3": closure_limit,
        "checks": checks,
        "runtime_clip_used": False,
        "source_modified": False,
        "zero_mb_0p1y_run": False,
        "zero_mb_5y_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run one versioned 0.1y zero-MB representative" if passed else "close monotone variational solver",
        "provenance": {
            "config_sha256": base.sha256(CONFIG),
            "predecessor_sha256": base.sha256(PREDECESSOR),
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
            "schema": "lia-191-network-monotone-variational-system2005-failure-state-step-v1",
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
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "real-failure-state-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

