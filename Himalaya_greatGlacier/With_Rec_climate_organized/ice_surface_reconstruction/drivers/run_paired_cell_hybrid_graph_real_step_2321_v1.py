"""One frozen-diffusivity, 30-second zero-MB step for system 2321.

This is the first and only real-system step authorized by the paired-cell
hybrid-graph static and synthetic-fixture gates.  It performs no clipping and
does not continue to 0.1 years unless a later versioned driver is authorized.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
GRID = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2321_RGI60-15.04415_system_grid_v1.nc"
MODEL_FLOWLINES = PROJECT / "w395inv2321v1/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92321/model_flowlines.pkl"
STATIC = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_static_2321_v1/summary.json"
FIXTURE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_solver_fixture_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_real_step_2321_v1"
DT_SECONDS = 30.0
GLEN_A = 6.967709123431999e-24
FS = 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if static["status"] != "hybrid-graph-static-contract-pass":
        raise RuntimeError("static contract did not pass")
    if fixture["status"] != "fixture-pass-real-system-not-authorized":
        raise RuntimeError("synthetic solver fixture did not pass")
    sources = [GRID, MODEL_FLOWLINES, STATIC, FIXTURE]
    hashes_before = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}

    cfg.initialize(logging_level="WARNING")
    with xr.open_dataset(GRID) as ds:
        h_source = np.asarray(ds.thickness_m.isel(member=0), dtype=np.float64)
        bed_source = np.asarray(ds.bed_elevation_m.isel(member=0), dtype=np.float64)
        valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool) & np.asarray(ds.lia_mask, dtype=bool) & np.isfinite(bed_source)
        x = np.asarray(ds.x, dtype=np.float64)
        y = np.asarray(ds.y, dtype=np.float64)
    positive = valid & np.isfinite(h_source) & (h_source > 1e-6)
    cell_dx = float(abs(np.median(np.diff(x))))
    cell_dy = float(abs(np.median(np.diff(y))))
    cell_area = cell_dx * cell_dy
    ids = np.full(valid.shape, -1, dtype=np.int64)
    ids[valid] = np.arange(int(valid.sum()), dtype=np.int64)
    domain_ij = np.argwhere(valid)
    node_count_2d = int(valid.sum())
    bed_2d = bed_source[valid]
    thick_2d = np.where(positive[valid], h_source[valid], 0.0)
    surface_2d = bed_2d + thick_2d

    with gzip.open(MODEL_FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    roots = [flowline for flowline in flowlines if getattr(flowline, "flows_to", None) is None]
    if len(roots) != 1:
        raise RuntimeError("expected exactly one root")
    root = roots[0]
    last_active = int(np.flatnonzero(np.asarray(root.thick) > 1e-6)[-1])
    tail_start = last_active + 1
    tail_bed = np.asarray(root.bed_h[tail_start:], dtype=np.float64)
    tail_width = np.asarray(root.widths_m[tail_start:], dtype=np.float64)
    tail_thick = np.zeros_like(tail_bed)
    tail_surface = tail_bed.copy()
    tail_dx = float(root.dx_meter)
    tail_count = len(tail_bed)

    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    h_i, h_j = np.where(horizontal)
    v_i, v_j = np.where(vertical)
    edge_u = [ids[h_i, h_j], ids[v_i, v_j]]
    edge_v = [ids[h_i, h_j + 1], ids[v_i + 1, v_j]]
    edge_distance = [np.full(len(h_i), cell_dx), np.full(len(v_i), cell_dy)]
    edge_width = [np.full(len(h_i), cell_dy), np.full(len(v_i), cell_dx)]

    yy, xx = np.meshgrid(y, x, indexing="ij")
    # The frozen static contract provides the outlet positive cell coordinate.
    outlet_xy = np.asarray(static["hybrid_interface"]["outlet_positive_cell_xy_m"], dtype=float)
    outlet_candidates = np.argwhere(positive)
    outlet_dist2 = (xx[positive] - outlet_xy[0]) ** 2 + (yy[positive] - outlet_xy[1]) ** 2
    outlet_ij = outlet_candidates[int(np.argmin(outlet_dist2))]
    outlet_id = int(ids[outlet_ij[0], outlet_ij[1]])
    tail_ids = node_count_2d + np.arange(tail_count, dtype=np.int64)
    edge_u.append(np.array([outlet_id], dtype=np.int64))
    edge_v.append(np.array([tail_ids[0]], dtype=np.int64))
    edge_distance.append(np.array([float(static["hybrid_interface"]["connector_distance_m"])], dtype=float))
    edge_width.append(np.array([cell_dx], dtype=float))
    if tail_count > 1:
        edge_u.append(tail_ids[:-1])
        edge_v.append(tail_ids[1:])
        edge_distance.append(np.full(tail_count - 1, tail_dx))
        edge_width.append((tail_width[:-1] + tail_width[1:]) / 2.0)
    edge_u = np.concatenate(edge_u)
    edge_v = np.concatenate(edge_v)
    distance = np.concatenate(edge_distance).astype(np.float64)
    face_width = np.concatenate(edge_width).astype(np.float64)
    edges = np.column_stack([edge_u, edge_v])

    bed = np.concatenate([bed_2d, tail_bed])
    thick = np.concatenate([thick_2d, tail_thick])
    surface = np.concatenate([surface_2d, tail_surface])
    storage = np.concatenate([np.full(node_count_2d, cell_area), tail_width * tail_dx])
    total_nodes = len(surface)
    columns = np.repeat(np.arange(len(edges), dtype=np.int64), 2)
    rows = edges.ravel()
    data = np.tile(np.array([1.0, -1.0]), len(edges))
    incidence = sparse.csc_matrix((data, (rows, columns)), shape=(total_nodes, len(edges)))

    thick_face = (thick[edge_u] + thick[edge_v]) / 2.0
    slope = (surface[edge_u] - surface[edge_v]) / distance
    glen_n = float(cfg.PARAMS["glen_n"])
    rhog = float(cfg.PARAMS["ice_density"] * cfg.G)
    fd = 2.0 / (glen_n + 2.0) * GLEN_A
    conductance = (
        (fd * thick_face ** (glen_n + 2.0) + FS * thick_face**glen_n)
        * rhog * face_width * np.abs(slope) ** (glen_n - 1.0) / distance
    )
    laplacian = incidence @ sparse.diags(conductance, format="csc") @ incidence.T
    matrix = sparse.diags(storage, format="csc") + DT_SECONDS * laplacian
    rhs = -DT_SECONDS * np.asarray(laplacian @ surface).ravel()
    inverse_diagonal = 1.0 / np.asarray(matrix.diagonal(), dtype=float)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda vector: inverse_diagonal * vector)
    iterations = [0]

    def callback(_):
        iterations[0] += 1

    start = time.perf_counter()
    delta, info = cg(matrix, rhs, x0=np.zeros(total_nodes), rtol=1e-13, atol=0.0, maxiter=5000, M=preconditioner, callback=callback)
    runtime = time.perf_counter() - start
    surface_after = surface + delta
    thick_after = surface_after - bed
    flux_after = conductance * np.asarray(incidence.T @ surface_after).ravel()
    node_residual = storage * delta + DT_SECONDS * np.asarray(incidence @ flux_after).ravel()
    volume_before = float(np.dot(storage, thick))
    volume_after_linear = float(np.dot(storage, thick_after))
    closure_limit = max(1e-6, abs(volume_before) * 1e-12)
    boundary_ids = ids[valid & (
        np.pad(~valid[1:, :], ((0, 1), (0, 0)), constant_values=True)
        | np.pad(~valid[:-1, :], ((1, 0), (0, 0)), constant_values=True)
        | np.pad(~valid[:, 1:], ((0, 0), (0, 1)), constant_values=True)
        | np.pad(~valid[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    )]
    boundary_ids = boundary_ids[boundary_ids >= 0]
    hashes_after = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}
    gates = {
        "static_and_fixture_inputs_pass": True,
        "cg_converged": int(info) == 0,
        "matrix_coefficients_finite": bool(np.all(np.isfinite(matrix.data)) and np.all(matrix.diagonal() > 0)),
        "conductance_nonnegative_finite": bool(np.all(np.isfinite(conductance)) and np.all(conductance >= 0)),
        "no_negative_thickness_without_clipping": float(np.min(thick_after)) >= -1e-8,
        "maximum_node_closure_within_formal_limit": float(np.max(np.abs(node_residual))) <= closure_limit,
        "global_volume_closure_within_formal_limit": abs(volume_after_linear - volume_before) <= closure_limit,
        "terminal_thickness_le_10m": float(thick_after[-1]) <= 10.0,
        "no_ice_on_2d_domain_boundary": int(np.count_nonzero(thick_after[boundary_ids] > 1e-6)) == 0,
        "sources_immutable": hashes_before == hashes_after,
    }
    passed = all(gates.values())
    summary = {
        "schema": "lia-191-paired-cell-hybrid-graph-real-step-2321-v1",
        "status": "real-step-pass-0p1y-not-yet-run" if passed else "real-step-fail-mechanism-closed",
        "system_id": 2321, "dt_seconds": DT_SECONDS, "glen_a": GLEN_A, "fs": FS,
        "node_count": total_nodes, "paired_domain_node_count": node_count_2d,
        "paired_positive_node_count": int(positive.sum()), "tail_node_count": tail_count,
        "edge_count": len(edges), "cg_info": int(info), "cg_iterations": iterations[0], "solve_runtime_seconds": runtime,
        "conductance_minimum": float(np.min(conductance)), "conductance_maximum": float(np.max(conductance)),
        "surface_change_minimum_m": float(np.min(delta)), "surface_change_maximum_m": float(np.max(delta)),
        "minimum_thickness_after_m": float(np.min(thick_after)),
        "maximum_thickness_after_m": float(np.max(thick_after)),
        "maximum_tail_thickness_after_m": float(np.max(thick_after[node_count_2d:])),
        "terminal_thickness_after_m": float(thick_after[-1]),
        "maximum_absolute_node_closure_residual_m3": float(np.max(np.abs(node_residual))),
        "global_linear_volume_change_m3": volume_after_linear - volume_before,
        "closure_limit_m3": closure_limit,
        "positive_2d_boundary_node_count": int(np.count_nonzero(thick_after[boundary_ids] > 1e-6)),
        "clipping_applied": False, "runtime_limiter_applied": False,
        "gates": gates, "source_sha256": hashes_before,
        "zero_mb_duration_years": DT_SECONDS / float(cfg.SEC_IN_YEAR),
        "zero_mb_0p1y_run": False, "zero_mb_5y_run": False,
        "steady_50y_run": False, "historical_run": False,
        "next_action": "versioned 0.1-year zero-MB pilot" if passed else "close paired-cell hybrid graph mechanism",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if passed:
        np.savez_compressed(
            OUTPUT / "real_step_state.npz", domain_ij=domain_ij, bed_m=bed,
            thickness_before_m=thick, thickness_after_m=thick_after,
            surface_after_m=surface_after, storage_m2=storage, edges=edges,
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
