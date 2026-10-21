"""Exact, mass-conserving zero-boundary implementation for topology split."""

from __future__ import annotations

import argparse
import csv
import json
import traceback

import numpy as np
from scipy.sparse import csr_matrix, eye, vstack

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_positive_conservative_system2005_zero_mb_v1 as predecessor
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as v1


TOL = 1e-10


def conservative_boundary_step(flowlines, coefficients, dt_seconds, closure_limit, section_floor):
    config = conservative_boundary_step.config
    active_local, edges, surface_before, storage, matrix, rhs, incidence = predecessor.step_v1.assemble(
        flowlines, coefficients, dt_seconds
    )
    widths = np.asarray([flowlines[b].widths_m[n] for b, n in active_local], dtype=float)
    old_sections = np.asarray([flowlines[b].section[n] for b, n in active_local], dtype=float)
    surface_lower = surface_before - old_sections / widths
    constraints = vstack([incidence, eye(len(active_local), format="csr")], format="csr")
    lower = np.r_[np.zeros(incidence.shape[0]), surface_lower]
    volume_row = csr_matrix(storage.reshape(1, -1))
    volume_target = float(np.dot(storage, surface_before))
    candidate, unconstrained, active, multipliers, volume_multiplier, history = (
        predecessor.positive_v1.solve_affine_active_set(
            matrix, rhs, constraints, lower, volume_row, volume_target,
            tolerance=float(config["active_set_tolerance"]), maximum_iterations=400,
        )
    )
    residual = np.asarray(constraints @ candidate, dtype=float) - lower
    edge_slopes = residual[: incidence.shape[0]]
    target_delta = storage * (candidate - surface_before)
    if abs(float(np.sum(target_delta))) > closure_limit:
        raise RuntimeError(f"linearized-volume-closure:{float(np.sum(target_delta))}")
    candidate_sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    for global_index, (branch, node) in enumerate(active_local):
        candidate_sections[branch][node] += target_delta[global_index] / float(flowlines[branch].dx_meter)

    boundary_tol = float(config["boundary_canonicalization_tolerance_m2"])
    raw_minimum = min(float(np.min(section)) for section in candidate_sections)
    if raw_minimum < -boundary_tol:
        raise RuntimeError(f"physical-negative-section:{raw_minimum}")
    added_volume = 0.0
    for branch, section in enumerate(candidate_sections):
        negative = np.flatnonzero(section < 0.0)
        for node in negative:
            added_volume += -float(section[node]) * float(flowlines[branch].dx_meter)
            section[node] = 0.0
    donor_removed = 0.0
    if added_volume:
        donors = []
        for branch, section in enumerate(candidate_sections):
            for node in np.flatnonzero(section > 0.0):
                donors.append((float(section[node]) * float(flowlines[branch].dx_meter), branch, int(node)))
        donors.sort(reverse=True)
        available, branch, node = donors[0]
        if available <= added_volume:
            raise RuntimeError("no-positive-donor-for-boundary-canonicalization")
        candidate_sections[branch][node] -= added_volume / float(flowlines[branch].dx_meter)
        donor_removed = added_volume

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
    active_candidate_sections = np.asarray([candidate_sections[b][n] for b, n in active_local], dtype=float)
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
        "raw_minimum_section_before_canonicalization_m2": raw_minimum,
        "boundary_canonicalization_added_m3": added_volume,
        "boundary_canonicalization_donor_removed_m3": donor_removed,
    }


def run(config):
    technical = json.loads(base.resolve(config["technical_predecessor"]).read_text(encoding="utf-8"))
    if "active-section-interior-floor:-2.922107000813412e-13" not in technical.get("error", ""):
        raise RuntimeError("v3-is-not-the-registered-roundoff-only-boundary-failure")
    original = predecessor.constrained_step
    conservative_boundary_step.config = config
    predecessor.constrained_step = conservative_boundary_step
    try:
        summary, rows = v1.run(config)
    finally:
        predecessor.constrained_step = original
    summary["schema"] = "lia-191-network-topology-adaptive-split-system2005-zero-mb-v4"
    summary["boundary_canonicalization"] = {
        "tolerance_m2": float(config["boundary_canonicalization_tolerance_m2"]),
        "mass_conserving": True,
        "positive_support_inserted": False,
    }
    summary["technical_predecessor_sha256"] = base.sha256(base.resolve(config["technical_predecessor"]))
    return summary, rows


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
            "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-v4",
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
    if summary.get("status") != "zero-mb-0p1y-pass-topology-adaptive-split":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

