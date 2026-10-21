"""One real 30-second capacity-constrained network step for system 2005."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_constrained_flux_fixture_v1 import project_flux_dag


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92005/model_flowlines.pkl"
FIXTURE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_capacity_constrained_flux_fixture_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_capacity_constrained_system2005_step_v1"
GLEN_A = 4.355770498109968e-23
FS = 0.0
DT = 30.0


def capacity_one_step(flowlines, coefficients, dt_seconds: float, closure_limit_m3: float) -> dict:
    active_local, edges = base.build_active_graph(flowlines, coefficients)
    count = len(active_local)
    surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    storage = np.asarray([flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in active_local], dtype=float)
    u = np.asarray([edge[0] for edge in edges], dtype=int)
    v = np.asarray([edge[1] for edge in edges], dtype=int)
    conductance = np.asarray([edge[2] for edge in edges], dtype=float)
    forward = (surface[u] - surface[v]) > 1e-12
    index = np.flatnonzero(forward)
    values = float(dt_seconds) * conductance[index]
    rows = np.concatenate([u[index], v[index], u[index], v[index]])
    cols = np.concatenate([u[index], v[index], v[index], u[index]])
    data = np.concatenate([values, values, -values, -values])
    matrix = diags(storage, format="csr") + coo_matrix((data, (rows, cols)), shape=(count, count)).tocsr()
    implicit_surface = np.asarray(spsolve(matrix, storage * surface), dtype=float)
    preferred = conductance * np.maximum(implicit_surface[u] - implicit_surface[v], 0.0)
    node_volume = np.asarray([flowlines[b].section[n] * flowlines[b].dx_meter for b, n in active_local], dtype=float)
    accepted, final_node_volume, limited = project_flux_dag(node_volume, u, v, preferred, dt_seconds)
    divergence_rate = np.zeros(count, dtype=float)
    np.add.at(divergence_rate, u, accepted)
    np.add.at(divergence_rate, v, -accepted)
    before_volume = base.total_volume(flowlines)
    sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    for index_node, (branch, node) in enumerate(active_local):
        sections[branch][node] = final_node_volume[index_node] / float(flowlines[branch].dx_meter)
    minimum_section = min(float(np.min(section)) for section in sections)
    if minimum_section < -1e-8:
        raise RuntimeError(f"negative-section-after-capacity:{minimum_section}")
    for fl, section in zip(flowlines, sections):
        fl.section = section
    after_volume = base.total_volume(flowlines)
    closure = after_volume - before_volume
    if abs(closure) > closure_limit_m3:
        raise RuntimeError(f"volume-closure:{closure}:{closure_limit_m3}")
    internal_gaps = 0
    for fl in flowlines:
        active_nodes = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active_nodes.size:
            internal_gaps += int(np.count_nonzero(np.diff(active_nodes) > 1))
    return {
        "active_node_count": count,
        "edge_count": len(edges),
        "preferred_flux_total_m3_s": float(np.sum(preferred)),
        "accepted_flux_total_m3_s": float(np.sum(accepted)),
        "limited_edge_count": int(np.count_nonzero(limited)),
        "maximum_flux_reduction_m3_s": float(np.max(preferred - accepted)) if len(edges) else 0.0,
        "actual_closure_m3": closure,
        "minimum_section_m2": minimum_section,
        "minimum_thickness_m": min(float(np.min(fl.thick)) for fl in flowlines),
        "internal_active_gap_count": internal_gaps,
        "minimum_accepted_flux_m3_s": float(np.min(accepted)) if len(edges) else 0.0,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        if fixture["status"] != "fixture-pass-real-system-not-authorized":
            raise RuntimeError("fixture-not-pass")
        from oggm import cfg
        cfg.initialize(logging_level="WARNING")
        flowlines = base.load_flowlines(SOURCE)
        initial_volume = base.total_volume(flowlines)
        closure_limit = max(1e-6, initial_volume * 1e-12)
        step = capacity_one_step(flowlines, base.Coefficients(GLEN_A, FS), DT, closure_limit)
        root = next(fl for fl in flowlines if fl.flows_to is None)
        checks = {
            "volume_closure": abs(step["actual_closure_m3"]) <= closure_limit,
            "nonnegative_section": step["minimum_section_m2"] >= -1e-8,
            "nonnegative_thickness": step["minimum_thickness_m"] >= -1e-10,
            "nonnegative_flux": step["minimum_accepted_flux_m3_s"] >= 0.0,
            "no_internal_gaps": step["internal_active_gap_count"] == 0,
            "terminal_zero": float(root.thick[-1]) == 0.0,
        }
        summary = {"status": "real-single-step-pass" if all(checks.values()) else "real-single-step-fail-mechanism-closed", "system_id": 2005, "mechanism": "network-dag-capacity-constrained-least-change-flux-v1", "dt_seconds": DT, "initial_volume_m3": initial_volume, "final_volume_m3": base.total_volume(flowlines), "closure_limit_m3": closure_limit, "step": step, "checks": checks, "source_model_flowlines_sha256": base.sha256(SOURCE), "source_modified": False, "posthoc_section_clip_used": False, "zero_mb_run": False, "steady_run": False, "historical_run": False}
    except Exception as exc:
        summary = {"status": "real-single-step-fail-mechanism-closed", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "source_modified": False, "zero_mb_run": False, "steady_run": False, "historical_run": False}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "real-single-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
