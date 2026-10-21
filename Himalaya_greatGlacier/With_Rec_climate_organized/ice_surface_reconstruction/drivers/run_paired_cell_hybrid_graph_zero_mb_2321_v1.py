"""0.1-year zero-MB pilot for the system-2321 paired-cell hybrid graph."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.ndimage import binary_erosion
from scipy.sparse.linalg import LinearOperator, cg

from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
GRID = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2321_RGI60-15.04415_system_grid_v1.nc"
MODEL_FLOWLINES = PROJECT / "w395inv2321v1/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92321/model_flowlines.pkl"
STATIC = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_static_2321_v1/summary.json"
REAL_STEP = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_real_step_2321_v2/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_zero_mb_2321_v1"
GLEN_A = 6.967709123431999e-24
FS = 0.0
MAX_DT_SECONDS = 86400.0
MIN_DT_SECONDS = 30.0
TARGET_YEARS = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    real_step = json.loads(REAL_STEP.read_text(encoding="utf-8"))
    if static["status"] != "hybrid-graph-static-contract-pass" or real_step["status"] != "real-step-pass-0p1y-not-yet-run":
        raise RuntimeError("upstream gates did not pass")
    sources = [GRID, MODEL_FLOWLINES, STATIC, REAL_STEP]
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

    with gzip.open(MODEL_FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    root = [flowline for flowline in flowlines if getattr(flowline, "flows_to", None) is None][0]
    last_active = int(np.flatnonzero(np.asarray(root.thick) > 1e-6)[-1])
    tail_start = last_active + 1
    tail_bed = np.asarray(root.bed_h[tail_start:], dtype=np.float64)
    tail_width = np.asarray(root.widths_m[tail_start:], dtype=np.float64)
    tail_dx = float(root.dx_meter)
    tail_count = len(tail_bed)

    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    hi, hj = np.where(horizontal)
    vi, vj = np.where(vertical)
    edge_u = [ids[hi, hj], ids[vi, vj]]
    edge_v = [ids[hi, hj + 1], ids[vi + 1, vj]]
    distances = [np.full(len(hi), cell_dx), np.full(len(vi), cell_dy)]
    widths = [np.full(len(hi), cell_dy), np.full(len(vi), cell_dx)]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    outlet_xy = np.asarray(static["hybrid_interface"]["outlet_positive_cell_xy_m"], dtype=float)
    outlet_ij_all = np.argwhere(positive)
    outlet_dist2 = (xx[positive] - outlet_xy[0]) ** 2 + (yy[positive] - outlet_xy[1]) ** 2
    outlet_ij = outlet_ij_all[int(np.argmin(outlet_dist2))]
    outlet_id = int(ids[outlet_ij[0], outlet_ij[1]])
    tail_ids = node_count_2d + np.arange(tail_count, dtype=np.int64)
    edge_u.append(np.array([outlet_id], dtype=np.int64))
    edge_v.append(np.array([tail_ids[0]], dtype=np.int64))
    distances.append(np.array([float(static["hybrid_interface"]["connector_distance_m"])], dtype=float))
    widths.append(np.array([cell_dx], dtype=float))
    edge_u.append(tail_ids[:-1]); edge_v.append(tail_ids[1:])
    distances.append(np.full(tail_count - 1, tail_dx))
    widths.append((tail_width[:-1] + tail_width[1:]) / 2.0)
    edge_u = np.concatenate(edge_u); edge_v = np.concatenate(edge_v)
    distance = np.concatenate(distances).astype(float)
    face_width = np.concatenate(widths).astype(float)
    edges = np.column_stack([edge_u, edge_v])
    bed = np.concatenate([bed_2d, tail_bed])
    thick = np.concatenate([thick_2d, np.zeros(tail_count)])
    surface = bed + thick
    storage = np.concatenate([np.full(node_count_2d, cell_area), tail_width * tail_dx])
    columns = np.repeat(np.arange(len(edges), dtype=np.int64), 2)
    rows = edges.ravel()
    data = np.tile(np.array([1.0, -1.0]), len(edges))
    incidence = sparse.csc_matrix((data, (rows, columns)), shape=(len(surface), len(edges)))
    boundary_mask = valid & ~binary_erosion(valid, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool), border_value=0)
    boundary_ids = ids[boundary_mask]

    glen_n = float(cfg.PARAMS["glen_n"])
    rhog = float(cfg.PARAMS["ice_density"] * cfg.G)
    fd = 2.0 / (glen_n + 2.0) * GLEN_A
    target_seconds = TARGET_YEARS * float(cfg.SEC_IN_YEAR)
    elapsed = 0.0
    rows_out = []
    minimum_dt_used = np.inf
    maximum_step_residual = 0.0
    maximum_global_step_change = 0.0
    maximum_surface_change = 0.0
    maximum_cg_iterations = 0
    total_retries = 0
    direction_flip_count = 0
    initial_volume = float(np.dot(storage, thick))
    closure_limit = max(1e-6, initial_volume * 1e-12)
    start_time = time.perf_counter()
    failure = None

    while elapsed < target_seconds - 1e-9:
        requested = min(MAX_DT_SECONDS, target_seconds - elapsed)
        dt = requested
        accepted = False
        while dt >= MIN_DT_SECONDS - 1e-12:
            su = surface[edge_u]; sv = surface[edge_v]
            hu = thick[edge_u]; hv = thick[edge_v]
            donor_thick = np.where(su >= sv, hu, hv)
            thick_face = (hu + hv) / 2.0
            slope = (su - sv) / distance
            conductance = (
                (fd * thick_face ** (glen_n + 2.0) + FS * thick_face**glen_n)
                * rhog * face_width * np.abs(slope) ** (glen_n - 1.0) / distance
            )
            conductance[donor_thick <= 1e-6] = 0.0
            laplacian = incidence @ sparse.diags(conductance, format="csc") @ incidence.T
            matrix = sparse.diags(storage, format="csc") + dt * laplacian
            rhs = -dt * np.asarray(laplacian @ surface).ravel()
            inverse_diagonal = 1.0 / np.asarray(matrix.diagonal(), dtype=float)
            preconditioner = LinearOperator(matrix.shape, matvec=lambda vector: inverse_diagonal * vector)
            iteration = [0]
            def callback(_):
                iteration[0] += 1
            delta, info = cg(matrix, rhs, x0=np.zeros(len(surface)), tol=1e-12, atol=0.0, maxiter=5000, M=preconditioner, callback=callback)
            trial_surface = surface + delta
            trial_thick = trial_surface - bed
            flux = conductance * np.asarray(incidence.T @ trial_surface).ravel()
            residual = storage * delta + dt * np.asarray(incidence @ flux).ravel()
            step_global = float(np.dot(storage, trial_thick - thick))
            flipped = int(np.count_nonzero((su - sv) * (trial_surface[edge_u] - trial_surface[edge_v]) < -1e-12))
            step_ok = (
                int(info) == 0 and np.all(np.isfinite(trial_thick))
                and float(np.min(trial_thick)) >= -1e-8
                and float(np.max(np.abs(residual))) <= closure_limit
                and abs(step_global) <= closure_limit
                and flipped == 0
            )
            if step_ok:
                surface = trial_surface; thick = trial_thick; accepted = True
                elapsed += dt
                minimum_dt_used = min(minimum_dt_used, dt)
                maximum_step_residual = max(maximum_step_residual, float(np.max(np.abs(residual))))
                maximum_global_step_change = max(maximum_global_step_change, abs(step_global))
                maximum_surface_change = max(maximum_surface_change, float(np.max(np.abs(delta))))
                maximum_cg_iterations = max(maximum_cg_iterations, iteration[0])
                direction_flip_count += flipped
                rows_out.append({
                    "step": len(rows_out) + 1, "elapsed_years": elapsed / float(cfg.SEC_IN_YEAR),
                    "dt_seconds": dt, "cg_iterations": iteration[0],
                    "volume_m3": float(np.dot(storage, thick)),
                    "maximum_node_residual_m3": float(np.max(np.abs(residual))),
                    "global_step_volume_change_m3": step_global,
                    "minimum_thickness_m": float(np.min(thick)),
                    "maximum_tail_thickness_m": float(np.max(thick[node_count_2d:])),
                    "positive_boundary_nodes": int(np.count_nonzero(thick[boundary_ids] > 1e-6)),
                })
                break
            dt /= 2.0
            total_retries += 1
        if not accepted:
            failure = "no-admissible-step-at-or-above-30s"
            break

    runtime = time.perf_counter() - start_time
    final_volume = float(np.dot(storage, thick))
    hashes_after = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}
    gates = {
        "target_0p1_year_completed": failure is None and abs(elapsed - target_seconds) <= 1e-6,
        "minimum_timestep_ge_30s": float(minimum_dt_used) >= MIN_DT_SECONDS if np.isfinite(minimum_dt_used) else False,
        "volume_drift_le_0p1pct": abs(final_volume - initial_volume) / initial_volume <= 0.001,
        "maximum_step_closure_within_limit": maximum_step_residual <= closure_limit,
        "no_negative_or_nonfinite_thickness": bool(np.all(np.isfinite(thick)) and np.min(thick) >= -1e-8),
        "no_direction_reversal_within_frozen_step": direction_flip_count == 0,
        "no_ice_on_2d_domain_boundary": int(np.count_nonzero(thick[boundary_ids] > 1e-6)) == 0,
        "terminal_thickness_le_10m": float(thick[-1]) <= 10.0,
        "sources_immutable": hashes_before == hashes_after,
    }
    passed = all(gates.values())
    summary = {
        "schema": "lia-191-paired-cell-hybrid-graph-zero-mb-2321-v1",
        "status": "zero-mb-0p1y-pass-5y-not-run" if passed else "zero-mb-0p1y-fail-mechanism-closed",
        "system_id": 2321, "target_years": TARGET_YEARS,
        "elapsed_years": elapsed / float(cfg.SEC_IN_YEAR), "step_count": len(rows_out),
        "minimum_timestep_seconds": float(minimum_dt_used) if np.isfinite(minimum_dt_used) else None,
        "maximum_timestep_seconds": MAX_DT_SECONDS, "retry_count": total_retries,
        "maximum_cg_iterations": maximum_cg_iterations, "runtime_seconds": runtime,
        "initial_volume_m3": initial_volume, "final_volume_m3": final_volume,
        "volume_drift_fraction": (final_volume - initial_volume) / initial_volume,
        "maximum_node_step_closure_residual_m3": maximum_step_residual,
        "maximum_global_step_volume_change_m3": maximum_global_step_change,
        "closure_limit_m3": closure_limit, "maximum_surface_change_per_step_m": maximum_surface_change,
        "minimum_final_thickness_m": float(np.min(thick)),
        "maximum_final_tail_thickness_m": float(np.max(thick[node_count_2d:])),
        "terminal_thickness_m": float(thick[-1]),
        "positive_2d_boundary_node_count": int(np.count_nonzero(thick[boundary_ids] > 1e-6)),
        "failure": failure, "clipping_applied": False, "substep_below_30s": False,
        "ice_free_high_bed_donor_flux_blocked": True,
        "gates": gates, "source_sha256": hashes_before,
        "zero_mb_5y_run": False, "steady_50y_run": False, "historical_run": False,
        "next_action": "versioned 5-year zero-MB pilot" if passed else "close paired-cell hybrid graph mechanism",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "steps.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]) if rows_out else ["step"])
        writer.writeheader(); writer.writerows(rows_out)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if passed:
        np.savez_compressed(OUTPUT / "state_0p1y.npz", domain_ij=domain_ij, bed_m=bed, thickness_m=thick, surface_m=surface, storage_m2=storage, edges=edges)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
