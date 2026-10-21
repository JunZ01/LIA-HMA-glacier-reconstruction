"""Registered 50-year B-window steady gate for system 2321's hybrid graph."""

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
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, cg

from oggm import cfg, utils
from oggm.core.massbalance import MonthlyTIModel


PROJECT = Path(__file__).resolve().parents[4]
GRID = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2321_RGI60-15.04415_system_grid_v1.nc"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/2321/component_0/lia_centerline_network.pkl.gz"
GDIR_ROOT = PROJECT / "w395inv2321v1/synthetic_gdir"
MODEL_FLOWLINES = GDIR_ROOT / "RGI60-15/RGI60-15.92/RGI60-15.92321/model_flowlines.pkl"
ZERO5 = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_zero_mb_2321_5y_v2/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_steady50_2321_v1"
REGISTERED_LENGTH_M = 9091.108842538311
GLEN_A = 6.967709123431999e-24
FS = 0.0
MELT_F = 4.235906754203083
PRCP_FAC = 3.882128640552246
TEMP_MELT_C = -1.0
BASE_TEMP_BIAS_C = 4.89569973117434
DELTA_T_C = 1.0
MAX_DT_SECONDS = 10.0 * 86400.0
MIN_DT_SECONDS = 30.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def false_runs(mask: np.ndarray) -> int:
    active = np.flatnonzero(mask)
    if active.size == 0:
        return 0
    core = mask[active[0]:active[-1] + 1]
    false = ~core
    return int(np.count_nonzero(false & np.r_[True, ~false[:-1]]))


def main() -> None:
    zero5 = json.loads(ZERO5.read_text(encoding="utf-8"))
    if zero5["status"] != "zero-mb-5y-pass-steady-not-run":
        raise RuntimeError("formal five-year zero-MB gate did not pass")
    sources = [GRID, NETWORK, MODEL_FLOWLINES, ZERO5]
    hashes_before = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = TEMP_MELT_C

    with xr.open_dataset(GRID) as ds:
        h_source = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
        bed_source = np.asarray(ds.bed_elevation_m.isel(member=0), dtype=float)
        valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool) & np.asarray(ds.lia_mask, dtype=bool) & np.isfinite(bed_source)
        x = np.asarray(ds.x, dtype=float); y = np.asarray(ds.y, dtype=float)
    positive = valid & np.isfinite(h_source) & (h_source > 1e-6)
    dx = float(abs(np.median(np.diff(x)))); dy = float(abs(np.median(np.diff(y))))
    cell_area = dx * dy
    ids = np.full(valid.shape, -1, dtype=np.int64); ids[valid] = np.arange(int(valid.sum()))
    node_count_2d = int(valid.sum())
    domain_ij = np.argwhere(valid)
    bed_2d = bed_source[valid]
    thick_2d = np.where(positive[valid], h_source[valid], 0.0)

    with gzip.open(MODEL_FLOWLINES, "rb") as stream:
        root = [fl for fl in pickle.load(stream) if getattr(fl, "flows_to", None) is None][0]
    last_active = int(np.flatnonzero(np.asarray(root.thick) > 1e-6)[-1]); tail_start = last_active + 1
    tail_bed = np.asarray(root.bed_h[tail_start:], dtype=float)
    tail_width = np.asarray(root.widths_m[tail_start:], dtype=float)
    tail_dx = float(root.dx_meter); tail_count = len(tail_bed)

    horizontal = valid[:, :-1] & valid[:, 1:]; vertical = valid[:-1, :] & valid[1:, :]
    hi, hj = np.where(horizontal); vi, vj = np.where(vertical)
    edge_u = [ids[hi, hj], ids[vi, vj]]; edge_v = [ids[hi, hj + 1], ids[vi + 1, vj]]
    distances = [np.full(len(hi), dx), np.full(len(vi), dy)]
    widths = [np.full(len(hi), dy), np.full(len(vi), dx)]
    static = json.loads((PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_static_2321_v1/summary.json").read_text(encoding="utf-8"))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    outlet_xy = np.asarray(static["hybrid_interface"]["outlet_positive_cell_xy_m"])
    outlet_all = np.argwhere(positive)
    outlet_dist2 = (xx[positive] - outlet_xy[0])**2 + (yy[positive] - outlet_xy[1])**2
    outlet_ij = outlet_all[int(np.argmin(outlet_dist2))]; outlet_id = int(ids[outlet_ij[0], outlet_ij[1]])
    tail_ids = node_count_2d + np.arange(tail_count, dtype=np.int64)
    edge_u += [np.array([outlet_id]), tail_ids[:-1]]; edge_v += [np.array([tail_ids[0]]), tail_ids[1:]]
    distances += [np.array([float(static["hybrid_interface"]["connector_distance_m"])]), np.full(tail_count - 1, tail_dx)]
    widths += [np.array([dx]), (tail_width[:-1] + tail_width[1:]) / 2.0]
    edge_u = np.concatenate(edge_u); edge_v = np.concatenate(edge_v)
    distance = np.concatenate(distances); face_width = np.concatenate(widths)
    edges = np.column_stack([edge_u, edge_v])
    bed = np.concatenate([bed_2d, tail_bed]); thick = np.concatenate([thick_2d, np.zeros(tail_count)])
    surface = bed + thick; storage = np.concatenate([np.full(node_count_2d, cell_area), tail_width * tail_dx])
    cols = np.repeat(np.arange(len(edges), dtype=np.int64), 2); rows = edges.ravel()
    incidence = sparse.csc_matrix((np.tile([1.0, -1.0], len(edges)), (rows, cols)), shape=(len(surface), len(edges)))
    boundary_ids = ids[valid & ~binary_erosion(valid, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool), border_value=0)]

    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    main_xy = np.asarray(network["branches"][int(network["main_branch_index"])] ["coords_projected_m"], dtype=float)
    main_cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(main_xy, axis=0), axis=1))]
    grid_xy = np.column_stack([xx[valid], yy[valid]])
    _, main_ids = cKDTree(grid_xy).query(main_xy, k=1, workers=1)
    main_ids = np.asarray(main_ids, dtype=np.int64)

    gdir = utils.GlacierDirectory("RGI60-15.92321", base_dir=GDIR_ROOT, reset=False)
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix="_batch1_1441_1470_prcp_off",
        melt_f=MELT_F, prcp_fac=PRCP_FAC, temp_bias=BASE_TEMP_BIAS_C + DELTA_T_C,
        ys=1441, ye=1470, repeat=True, check_calib_params=False,
    )
    n = float(cfg.PARAMS["glen_n"]); rhog = float(cfg.PARAMS["ice_density"] * cfg.G)
    fd = 2.0 / (n + 2.0) * GLEN_A; sec_year = float(cfg.SEC_IN_YEAR)
    initial_volume = float(np.dot(storage, thick)); closure_limit = max(1e-6, initial_volume * 1e-12)
    minimum_dt = np.inf; retries = 0; max_residual = 0.0; max_global_budget = 0.0
    max_iterations = 0; failure = None; elapsed = 0.0; started = time.perf_counter()

    def snapshot(year: int) -> dict:
        active_main = thick[main_ids] > 10.0
        if np.any(active_main):
            first = int(np.flatnonzero(active_main)[0]); last = int(np.flatnonzero(active_main)[-1])
            main_length = float(main_cumulative[last] - main_cumulative[first])
        else:
            first = last = -1; main_length = 0.0
        tail_active = np.flatnonzero(thick[node_count_2d:] > 10.0)
        tail_progress = 0.0 if not tail_active.size else float((tail_active[-1] + 1) * tail_dx)
        length_m = main_length + tail_progress
        free_tail = float(max(0.0, static["hybrid_interface"]["tail_length_m"] - tail_progress))
        active2d = thick[:node_count_2d] > 1e-6
        area = float(np.count_nonzero(active2d) * cell_area + np.sum((thick[node_count_2d:] > 1e-6) * tail_width * tail_dx))
        return {
            "year": year, "volume_m3": float(np.dot(storage, thick)), "area_m2": area,
            "length_m": length_m, "length_relative_error_lia": abs(length_m - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M,
            "main_first_active_index": first, "main_last_active_index": last,
            "main_internal_gap_count": false_runs(active_main), "tail_progress_m": tail_progress,
            "remaining_free_downstream_m": free_tail, "minimum_thickness_m": float(np.min(thick)),
            "maximum_thickness_m": float(np.max(thick)), "positive_boundary_nodes": int(np.count_nonzero(thick[boundary_ids] > 1e-6)),
            "terminal_thickness_m": float(thick[-1]), "finite_geometry": bool(np.all(np.isfinite(thick))),
        }

    snapshots = [snapshot(0)]
    for year in range(50):
        raw_mb = np.asarray(mb_model.get_annual_mb(surface, year=float(year)), dtype=float)
        year_end = float(year + 1) * sec_year
        while elapsed < year_end - 1e-7:
            requested = min(MAX_DT_SECONDS, year_end - elapsed); dt = requested; accepted = False
            while dt >= MIN_DT_SECONDS - 1e-12:
                mb = np.where(thick > 1e-6, raw_mb, 0.0)
                su = surface[edge_u]; sv = surface[edge_v]; hu = thick[edge_u]; hv = thick[edge_v]
                donor_h = np.where(su >= sv, hu, hv); hf = (hu + hv) / 2.0; slope = (su - sv) / distance
                conductance = (fd * hf**(n + 2.0) + FS * hf**n) * rhog * face_width * np.abs(slope)**(n - 1.0) / distance
                conductance[donor_h <= 1e-6] = 0.0
                lap = incidence @ sparse.diags(conductance, format="csc") @ incidence.T
                matrix = sparse.diags(storage, format="csc") + dt * lap
                rhs = dt * storage * mb - dt * np.asarray(lap @ surface).ravel()
                invdiag = 1.0 / np.asarray(matrix.diagonal()); pre = LinearOperator(matrix.shape, matvec=lambda vector: invdiag * vector)
                iterations = [0]
                def callback(_): iterations[0] += 1
                delta, info = cg(matrix, rhs, x0=np.zeros(len(surface)), tol=1e-12, atol=0.0, maxiter=5000, M=pre, callback=callback)
                trial_surface = surface + delta; trial_thick = trial_surface - bed
                flux = conductance * np.asarray(incidence.T @ trial_surface).ravel()
                residual = storage * delta + dt * np.asarray(incidence @ flux).ravel() - dt * storage * mb
                global_budget = float(np.dot(storage, trial_thick - thick) - dt * np.dot(storage, mb))
                flipped = int(np.count_nonzero((su - sv) * (trial_surface[edge_u] - trial_surface[edge_v]) < -1e-12))
                ok = int(info) == 0 and np.all(np.isfinite(trial_thick)) and float(np.min(trial_thick)) >= -1e-8 and float(np.max(np.abs(residual))) <= closure_limit and abs(global_budget) <= closure_limit and flipped == 0
                if ok:
                    surface = trial_surface; thick = trial_thick; elapsed += dt; accepted = True
                    minimum_dt = min(minimum_dt, dt); max_residual = max(max_residual, float(np.max(np.abs(residual))))
                    max_global_budget = max(max_global_budget, abs(global_budget)); max_iterations = max(max_iterations, iterations[0])
                    break
                dt /= 2.0; retries += 1
            if not accepted:
                failure = f"no-admissible-step-at-or-above-30s-year-{year}"
                break
        snapshots.append(snapshot(year + 1 if failure is None else year))
        if failure is not None: break

    runtime = time.perf_counter() - started; final = snapshots[-1]; initial = snapshots[0]
    volume_change = (final["volume_m3"] - initial["volume_m3"]) / initial["volume_m3"]
    last10 = snapshots[-11] if len(snapshots) >= 11 else snapshots[0]
    last10_change = (final["volume_m3"] - last10["volume_m3"]) / last10["volume_m3"]
    hashes_after = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}
    flags = []
    if failure: flags.append("numerical-incomplete")
    if final["year"] != 50: flags.append("50y-incomplete")
    if abs(volume_change) > 0.02: flags.append("50y-volume-change-over-2pct")
    if final["length_relative_error_lia"] > 0.10: flags.append("lia-length-error-over-10pct")
    if final["remaining_free_downstream_m"] < 5000.0: flags.append("remaining-free-domain-under-5km")
    if not final["finite_geometry"] or final["minimum_thickness_m"] < -1e-8: flags.append("invalid-geometry")
    if final["positive_boundary_nodes"]: flags.append("observation-domain-overflow")
    if final["main_internal_gap_count"]: flags.append("main-path-internal-gap")
    if hashes_before != hashes_after: flags.append("source-mutated")
    summary = {
        "schema": "lia-191-paired-cell-hybrid-graph-steady50-2321-v1",
        "status": "steady-50y-pass" if not flags else "steady-50y-fail",
        "system_id": 2321, "solver": "paired-cell-hybrid-graph-semiimplicit-sia",
        "climate": "repeated registered B-window MonthlyTIModel", "spinup_window": [1441, 1470],
        "stage2": {"melt_f": MELT_F, "prcp_fac": PRCP_FAC, "glen_a": GLEN_A, "fs": FS, "temp_melt_c": TEMP_MELT_C, "base_temp_bias_c": BASE_TEMP_BIAS_C, "delta_t_c": DELTA_T_C, "total_temp_bias_c": BASE_TEMP_BIAS_C + DELTA_T_C},
        "initial": initial, "final": final, "volume_change_fraction_50y": volume_change,
        "volume_change_fraction_last_10y": last10_change, "minimum_timestep_seconds": None if not np.isfinite(minimum_dt) else float(minimum_dt),
        "maximum_timestep_seconds": MAX_DT_SECONDS, "retry_count": retries,
        "maximum_cg_iterations": max_iterations, "maximum_node_budget_residual_m3": max_residual,
        "maximum_global_step_budget_residual_m3": max_global_budget, "closure_limit_m3": closure_limit,
        "surface_mass_balance_applied_only_on_current_ice": True, "clipping_applied": False,
        "review_flags": flags, "runtime_error": failure, "runtime_seconds": runtime,
        "source_sha256": hashes_before, "historical_run": False,
        "next_action": "two registered historical endpoints" if not flags else "close paired-cell hybrid graph mechanism",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "steady_50y_timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(snapshots[0])); writer.writeheader(); writer.writerows(snapshots)
    (OUTPUT / "steady_50y_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not flags:
        np.savez_compressed(OUTPUT / "lia_equilibrium_state_50y.npz", domain_ij=domain_ij, bed_m=bed, thickness_m=thick, surface_m=surface, storage_m2=storage, edges=edges)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
