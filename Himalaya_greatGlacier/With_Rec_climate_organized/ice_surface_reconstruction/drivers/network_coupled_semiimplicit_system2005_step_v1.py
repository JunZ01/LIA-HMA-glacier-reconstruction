"""One frozen-diffusivity 30 s network-semiimplicit step for system 2005.

The synthetic fixtures must already have passed.  This is the first real-state
gate only: it reads the frozen repaired model flowlines, advances one zero-MB
step in memory, writes diagnostics, and never alters the GlacierDirectory.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


PROJECT = Path(__file__).resolve().parents[4]
FIXTURE_SUMMARY = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_network_coupled_semiimplicit_fixture_v1/summary.json"
)
MODEL_FLOWLINES = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/"
    "synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92005/model_flowlines.pkl"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_network_coupled_semiimplicit_system2005_step_v1"
)
DT_SECONDS = 30.0
GLEN_A = 4.355770498109968e-23
FS = 0.0
ACTIVE_TOLERANCE_M = 1e-6
REVERSE_FLUX_TOLERANCE_M3_S = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_flowlines():
    with gzip.open(MODEL_FLOWLINES, "rb") as stream:
        return pickle.load(stream)


def face_conductance(model, upstream_fl, upstream_index, downstream_fl, downstream_index):
    hu = float(upstream_fl.thick[upstream_index])
    hv = float(downstream_fl.thick[downstream_index])
    wu = float(upstream_fl.widths_m[upstream_index])
    wv = float(downstream_fl.widths_m[downstream_index])
    w0u = float(upstream_fl._w0_m[upstream_index])
    w0v = float(downstream_fl._w0_m[downstream_index])
    su = float(upstream_fl.surface_h[upstream_index])
    sv = float(downstream_fl.surface_h[downstream_index])
    distance = 0.5 * (float(upstream_fl.dx_meter) + float(downstream_fl.dx_meter))
    hstag = 0.5 * (hu + hv)
    wstag = 0.5 * (wu + wv)
    w0stag = 0.5 * (w0u + w0v)
    slope = (sv - su) / distance
    n = float(model.glen_n)
    diffusivity = (
        (float(model._fd) * hstag ** (n + 2.0) + float(model.fs) * hstag**n)
        * float(model.rhog)
        * (w0stag + wstag)
        / 2.0
        * abs(slope) ** (n - 1.0)
    )
    return float(diffusivity / distance)


def run() -> dict:
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import ScalarMassBalance

    fixture = json.loads(FIXTURE_SUMMARY.read_text(encoding="utf-8"))
    if fixture.get("status") != "fixture-pass-real-system-not-authorized":
        raise RuntimeError("synthetic fixture gate is not passed")

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    source = load_flowlines()
    flowlines = copy.deepcopy(source)
    root = next(fl for fl in flowlines if fl.flows_to is None)
    coefficient_model = SemiImplicitModel(
        [copy.deepcopy(root)],
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=GLEN_A,
        fs=FS,
        fixed_dt=DT_SECONDS,
        min_dt=DT_SECONDS,
    )

    branch_by_object = {id(fl): index for index, fl in enumerate(flowlines)}
    active_local = []
    global_of = {}
    for branch, fl in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(fl.thick, dtype=float) > ACTIVE_TOLERANCE_M)
        for node in active:
            global_of[(branch, int(node))] = len(active_local)
            active_local.append((branch, int(node)))
    if not active_local:
        raise RuntimeError("system has no active nodes")

    surface_before = np.asarray(
        [flowlines[b].surface_h[n] for b, n in active_local], dtype=float
    )
    storage = np.asarray(
        [flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in active_local],
        dtype=float,
    )
    if np.any(~np.isfinite(storage)) or np.any(storage <= 0):
        raise RuntimeError("active storage is nonfinite or nonpositive")

    edges = []
    for branch, fl in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(fl.thick, dtype=float) > ACTIVE_TOLERANCE_M)
        active_set = set(map(int, active))
        for node in active[:-1]:
            node = int(node)
            if node + 1 not in active_set:
                continue
            k = face_conductance(coefficient_model, fl, node, fl, node + 1)
            edges.append(
                {
                    "kind": "within",
                    "upstream_branch": branch,
                    "upstream_node": node,
                    "downstream_branch": branch,
                    "downstream_node": node + 1,
                    "u": global_of[(branch, node)],
                    "v": global_of[(branch, node + 1)],
                    "conductance_m2_s": k,
                }
            )
        if fl.flows_to is not None and len(active) and int(active[-1]) == len(fl.thick) - 1:
            receiver = branch_by_object[id(fl.flows_to)]
            receiver_node = int(fl.flows_to_indice)
            terminal = int(active[-1])
            if (receiver, receiver_node) not in global_of:
                raise RuntimeError(
                    f"active tributary {branch} targets inactive receiver {receiver}/{receiver_node}"
                )
            k = face_conductance(
                coefficient_model, fl, terminal, flowlines[receiver], receiver_node
            )
            edges.append(
                {
                    "kind": "junction",
                    "upstream_branch": branch,
                    "upstream_node": terminal,
                    "downstream_branch": receiver,
                    "downstream_node": receiver_node,
                    "u": global_of[(branch, terminal)],
                    "v": global_of[(receiver, receiver_node)],
                    "conductance_m2_s": k,
                }
            )

    node_count = len(active_local)
    row = []
    col = []
    data = []
    for edge in edges:
        u, v, k = edge["u"], edge["v"], edge["conductance_m2_s"]
        value = DT_SECONDS * k
        row.extend([u, v, u, v])
        col.extend([u, v, v, u])
        data.extend([value, value, -value, -value])
    graph = coo_matrix((data, (row, col)), shape=(node_count, node_count)).tocsr()
    matrix = diags(storage, format="csr") + graph
    diagonal = matrix.diagonal()
    symmetry_max = float(abs(matrix - matrix.T).max())
    if np.any(~np.isfinite(matrix.data)) or np.any(diagonal <= 0) or symmetry_max > 1e-12:
        raise RuntimeError("assembled matrix failed finite/positive/symmetry gate")

    rhs = storage * surface_before
    surface_after = np.asarray(spsolve(matrix, rhs), dtype=float)
    if np.any(~np.isfinite(surface_after)):
        raise RuntimeError("real-state solve returned nonfinite surface")

    node_residual = storage * (surface_after - surface_before)
    junction_reverse = 0
    for edge in edges:
        u, v, k = edge["u"], edge["v"], edge["conductance_m2_s"]
        flux = float(k * (surface_after[u] - surface_after[v]))
        edge["flux_m3_s"] = flux
        edge["volume_transfer_m3"] = flux * DT_SECONDS
        node_residual[u] += flux * DT_SECONDS
        node_residual[v] -= flux * DT_SECONDS
        if edge["kind"] == "junction" and flux < -REVERSE_FLUX_TOLERANCE_M3_S:
            junction_reverse += 1

    initial_volume = float(sum(np.sum(fl.section) * fl.dx_meter for fl in flowlines))
    initial_area = float(sum(np.sum(fl.widths_m) * fl.dx_meter for fl in flowlines))
    initial_sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    minimum_thickness = float("inf")
    for global_index, (branch, node) in enumerate(active_local):
        fl = flowlines[branch]
        new_thickness = float(surface_after[global_index] - fl.bed_h[node])
        minimum_thickness = min(minimum_thickness, new_thickness)
        fl.thick[node] = new_thickness
    final_sections = [np.asarray(fl.section, dtype=float) for fl in flowlines]
    final_volume = float(sum(np.sum(fl.section) * fl.dx_meter for fl in flowlines))
    final_area = float(sum(np.sum(fl.widths_m) * fl.dx_meter for fl in flowlines))
    actual_volume_change = final_volume - initial_volume
    closure_limit = max(1e-6, initial_volume * 1e-12)
    max_node_residual = float(np.max(np.abs(node_residual)))
    max_section_change = float(
        max(np.max(np.abs(after - before)) for before, after in zip(initial_sections, final_sections))
    )
    root_terminal_thickness = float(flowlines[-1].thick[-1])

    checks = {
        "fixture_gate": True,
        "matrix_finite": bool(np.isfinite(matrix.data).all()),
        "matrix_positive_diagonal": bool(np.all(diagonal > 0)),
        "matrix_symmetric": symmetry_max <= 1e-12,
        "surface_finite": bool(np.isfinite(surface_after).all()),
        "minimum_thickness_nonnegative": minimum_thickness >= -1e-10,
        "junction_reverse_flux_zero": junction_reverse == 0,
        "node_linearized_closure": max_node_residual <= closure_limit,
        "actual_volume_closure": abs(actual_volume_change) <= closure_limit,
        "root_terminal_thickness_zero": abs(root_terminal_thickness) <= 1e-12,
    }
    status = "real-single-step-pass" if all(checks.values()) else "real-single-step-fail-mechanism-closed"
    return {
        "schema": "lia-191-network-coupled-semiimplicit-system2005-step-v1",
        "status": status,
        "system_id": 2005,
        "dt_seconds": DT_SECONDS,
        "flowline_count": len(flowlines),
        "active_node_count": node_count,
        "edge_count": len(edges),
        "within_edge_count": sum(e["kind"] == "within" for e in edges),
        "junction_edge_count": sum(e["kind"] == "junction" for e in edges),
        "matrix_minimum_diagonal": float(diagonal.min()),
        "matrix_symmetry_max_abs": symmetry_max,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "actual_volume_change_m3": actual_volume_change,
        "closure_limit_m3": closure_limit,
        "maximum_absolute_linearized_node_residual_m3": max_node_residual,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "maximum_absolute_section_change_m2": max_section_change,
        "minimum_candidate_thickness_m": minimum_thickness,
        "root_terminal_thickness_m": root_terminal_thickness,
        "junction_reverse_flux_count": junction_reverse,
        "checks": checks,
        "glen_a": GLEN_A,
        "fs": FS,
        "mass_balance": "zero",
        "source_model_flowlines_sha256": sha256(MODEL_FLOWLINES),
        "fixture_summary_sha256": sha256(FIXTURE_SUMMARY),
        "source_modified": False,
        "zero_mb_0p1y_run": False,
        "zero_mb_5y_run": False,
        "steady_run": False,
        "historical_run": False,
        "edges": edges,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        summary = run()
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-coupled-semiimplicit-system2005-step-v1",
            "status": "technical-fail-mechanism-not-evaluated",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "source_modified": False,
            "zero_mb_0p1y_run": False,
            "zero_mb_5y_run": False,
            "steady_run": False,
            "historical_run": False,
        }
    edges = summary.pop("edges", [])
    with (OUTPUT / "edge_diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        if edges:
            writer = csv.DictWriter(stream, fieldnames=list(edges[0]))
            writer.writeheader()
            writer.writerows(edges)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "real-single-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

