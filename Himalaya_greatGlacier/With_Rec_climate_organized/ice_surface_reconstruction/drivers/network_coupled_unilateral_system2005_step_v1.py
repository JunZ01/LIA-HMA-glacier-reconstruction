"""One real 30-second unilateral active-set network step for system 2005."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92005/model_flowlines.pkl"
OLD_FIXTURE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_coupled_semiimplicit_fixture_v1/summary.json"
NEW_FIXTURE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_coupled_unilateral_active_set_fixture_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_coupled_unilateral_system2005_step_v1"
GLEN_A = 4.355770498109968e-23
FS = 0.0
DT = 30.0
TOLERANCE = 1e-12


def unilateral_one_step(flowlines, coefficients, dt_seconds: float, closure_limit_m3: float) -> dict:
    active_local, edges = base.build_active_graph(flowlines, coefficients)
    node_count = len(active_local)
    surface_before = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    storage = np.asarray([flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in active_local], dtype=float)
    if np.any(~np.isfinite(storage)) or np.any(storage <= 0):
        raise RuntimeError("invalid-storage")
    u = np.asarray([edge[0] for edge in edges], dtype=int)
    v = np.asarray([edge[1] for edge in edges], dtype=int)
    conductance = np.asarray([edge[2] for edge in edges], dtype=float)
    kinds = [edge[3] for edge in edges]
    active = (surface_before[u] - surface_before[v]) > TOLERANCE
    seen = set()
    for iteration in range(1, 101):
        key = active.tobytes()
        if key in seen:
            raise RuntimeError("active-set-cycle")
        seen.add(key)
        index = np.flatnonzero(active)
        values = float(dt_seconds) * conductance[index]
        rows = np.concatenate([u[index], v[index], u[index], v[index]])
        cols = np.concatenate([u[index], v[index], v[index], u[index]])
        data = np.concatenate([values, values, -values, -values])
        graph = coo_matrix((data, (rows, cols)), shape=(node_count, node_count)).tocsr()
        matrix = diags(storage, format="csr") + graph
        surface_after = np.asarray(spsolve(matrix, storage * surface_before), dtype=float)
        gaps = surface_after[u] - surface_after[v]
        new_active = gaps > TOLERANCE
        if np.array_equal(new_active, active):
            break
        active = new_active
    else:
        raise RuntimeError("active-set-no-convergence")
    flux = conductance * np.maximum(gaps, 0.0)
    if np.any(flux < -TOLERANCE):
        raise RuntimeError("negative-flux-after-active-set")
    inactive_positive_gap = float(np.max(np.where(~active, np.maximum(gaps, 0.0), 0.0)))
    if inactive_positive_gap > TOLERANCE:
        raise RuntimeError(f"complementarity-gap:{inactive_positive_gap}")
    divergence = np.zeros(node_count, dtype=float)
    np.add.at(divergence, u, flux * float(dt_seconds))
    np.add.at(divergence, v, -flux * float(dt_seconds))
    linear_residual = storage * (surface_after - surface_before) + divergence
    before_volume = base.total_volume(flowlines)
    sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    for global_index, (branch, node) in enumerate(active_local):
        sections[branch][node] -= divergence[global_index] / float(flowlines[branch].dx_meter)
    minimum_section = min(float(np.min(section)) for section in sections)
    if minimum_section < -1e-10:
        raise RuntimeError(f"negative-section-required:{minimum_section}")
    for fl, section in zip(flowlines, sections):
        fl.section = section
    after_volume = base.total_volume(flowlines)
    closure = after_volume - before_volume
    if abs(closure) > closure_limit_m3:
        raise RuntimeError(f"volume-closure:{closure}:{closure_limit_m3}")
    minimum_thickness = min(float(np.min(fl.thick)) for fl in flowlines)
    if minimum_thickness < -1e-10:
        raise RuntimeError(f"negative-thickness:{minimum_thickness}")
    return {
        "active_node_count": node_count,
        "edge_count": len(edges),
        "active_edge_count": int(np.count_nonzero(active)),
        "inactive_edge_count": int(len(edges) - np.count_nonzero(active)),
        "inactive_junction_edge_count": int(sum((not active[i]) and kinds[i] == "junction" for i in range(len(edges)))),
        "active_set_iterations": iteration,
        "actual_closure_m3": closure,
        "maximum_absolute_linear_residual_m3": float(np.max(np.abs(linear_residual))),
        "minimum_section_m2": minimum_section,
        "minimum_thickness_m": minimum_thickness,
        "minimum_flux_m3_s": float(np.min(flux)) if len(flux) else 0.0,
        "maximum_inactive_positive_gap_m": inactive_positive_gap,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        old_fixture = json.loads(OLD_FIXTURE.read_text(encoding="utf-8"))
        new_fixture = json.loads(NEW_FIXTURE.read_text(encoding="utf-8"))
        if old_fixture["status"] != "fixture-pass-real-system-not-authorized" or new_fixture["status"] != "fixture-pass-real-system-not-authorized":
            raise RuntimeError("fixture-gate-not-pass")
        from oggm import cfg
        cfg.initialize(logging_level="WARNING")
        flowlines = base.load_flowlines(SOURCE)
        coefficients = base.Coefficients(GLEN_A, FS)
        initial_volume = base.total_volume(flowlines)
        initial_area = base.total_area(flowlines)
        closure_limit = max(1e-6, initial_volume * 1e-12)
        step = unilateral_one_step(flowlines, coefficients, DT, closure_limit)
        root = next(fl for fl in flowlines if fl.flows_to is None)
        checks = {
            "fixture_gates": True,
            "volume_closure": abs(step["actual_closure_m3"]) <= closure_limit,
            "linear_residual": step["maximum_absolute_linear_residual_m3"] <= closure_limit,
            "nonnegative_flux": step["minimum_flux_m3_s"] >= -TOLERANCE,
            "complementarity": step["maximum_inactive_positive_gap_m"] <= TOLERANCE,
            "nonnegative_section": step["minimum_section_m2"] >= -1e-10,
            "nonnegative_thickness": step["minimum_thickness_m"] >= -1e-10,
            "terminal_zero": float(root.thick[-1]) == 0.0,
        }
        summary = {
            "status": "real-single-step-pass" if all(checks.values()) else "real-single-step-fail-mechanism-closed",
            "system_id": 2005,
            "mechanism": "network-coupled-unilateral-frozen-diffusivity-active-set-v1",
            "dt_seconds": DT,
            "initial_volume_m3": initial_volume,
            "final_volume_m3": base.total_volume(flowlines),
            "initial_area_m2": initial_area,
            "final_area_m2": base.total_area(flowlines),
            "closure_limit_m3": closure_limit,
            "step": step,
            "checks": checks,
            "source_model_flowlines_sha256": base.sha256(SOURCE),
            "source_modified": False,
            "runtime_flux_clip_used": False,
            "zero_mb_run": False,
            "steady_run": False,
            "historical_run": False,
        }
    except Exception as exc:
        summary = {"status": "real-single-step-fail-mechanism-closed", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "source_modified": False, "zero_mb_run": False, "steady_run": False, "historical_run": False}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "real-single-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
