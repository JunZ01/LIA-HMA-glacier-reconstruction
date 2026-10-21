"""Shared immutable geometry and algebra for system 2321's hybrid graph."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from .paired_cell_graph_obstacle_fixture_v1 import solve_lower_bound_qp


PROJECT = Path(__file__).resolve().parents[4]
GRID = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2321_RGI60-15.04415_system_grid_v1.nc"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/2321/component_0/lia_centerline_network.pkl.gz"
MODEL_FLOWLINES = PROJECT / "w395inv2321v1/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92321/model_flowlines.pkl"
STATIC = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_static_2321_v1/summary.json"
REGISTERED_LENGTH_M = 9091.108842538311


def false_runs(mask):
    active = np.flatnonzero(mask)
    if active.size == 0:
        return 0
    core = mask[active[0]:active[-1] + 1]
    false = ~core
    return int(np.count_nonzero(false & np.r_[True, ~false[:-1]]))


def build_state():
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    with xr.open_dataset(GRID) as ds:
        h_source = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
        bed_source = np.asarray(ds.bed_elevation_m.isel(member=0), dtype=float)
        valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool) & np.asarray(ds.lia_mask, dtype=bool) & np.isfinite(bed_source)
        x = np.asarray(ds.x, dtype=float); y = np.asarray(ds.y, dtype=float)
    positive = valid & np.isfinite(h_source) & (h_source > 1e-6)
    dx = float(abs(np.median(np.diff(x)))); dy = float(abs(np.median(np.diff(y)))); cell_area = dx * dy
    ids = np.full(valid.shape, -1, dtype=np.int64); ids[valid] = np.arange(int(valid.sum()))
    domain_ij = np.argwhere(valid); node_count_2d = int(valid.sum())
    bed_2d = bed_source[valid]; thick_2d = np.where(positive[valid], h_source[valid], 0.0)
    with gzip.open(MODEL_FLOWLINES, "rb") as stream:
        root = [fl for fl in pickle.load(stream) if getattr(fl, "flows_to", None) is None][0]
    last_active = int(np.flatnonzero(np.asarray(root.thick) > 1e-6)[-1]); tail_start = last_active + 1
    tail_bed = np.asarray(root.bed_h[tail_start:], dtype=float); tail_width = np.asarray(root.widths_m[tail_start:], dtype=float)
    tail_dx = float(root.dx_meter); tail_count = len(tail_bed)
    horizontal = valid[:, :-1] & valid[:, 1:]; vertical = valid[:-1, :] & valid[1:, :]
    hi, hj = np.where(horizontal); vi, vj = np.where(vertical)
    edge_u = [ids[hi, hj], ids[vi, vj]]; edge_v = [ids[hi, hj + 1], ids[vi + 1, vj]]
    distances = [np.full(len(hi), dx), np.full(len(vi), dy)]; widths = [np.full(len(hi), dy), np.full(len(vi), dx)]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    outlet_xy = np.asarray(static["hybrid_interface"]["outlet_positive_cell_xy_m"])
    outlet_all = np.argwhere(positive); outlet_dist2 = (xx[positive] - outlet_xy[0])**2 + (yy[positive] - outlet_xy[1])**2
    outlet_ij = outlet_all[int(np.argmin(outlet_dist2))]; outlet_id = int(ids[outlet_ij[0], outlet_ij[1]])
    tail_ids = node_count_2d + np.arange(tail_count, dtype=np.int64)
    edge_u += [np.array([outlet_id]), tail_ids[:-1]]; edge_v += [np.array([tail_ids[0]]), tail_ids[1:]]
    distances += [np.array([float(static["hybrid_interface"]["connector_distance_m"])]), np.full(tail_count - 1, tail_dx)]
    widths += [np.array([dx]), (tail_width[:-1] + tail_width[1:]) / 2.0]
    edge_u = np.concatenate(edge_u); edge_v = np.concatenate(edge_v)
    distance = np.concatenate(distances); face_width = np.concatenate(widths); edges = np.column_stack([edge_u, edge_v])
    bed = np.concatenate([bed_2d, tail_bed]); thick = np.concatenate([thick_2d, np.zeros(tail_count)]); surface = bed + thick
    storage = np.concatenate([np.full(node_count_2d, cell_area), tail_width * tail_dx])
    columns = np.repeat(np.arange(len(edges), dtype=np.int64), 2); rows = edges.ravel()
    incidence = sparse.csc_matrix((np.tile([1.0, -1.0], len(edges)), (rows, columns)), shape=(len(surface), len(edges)))
    boundary_ids = ids[valid & ~binary_erosion(valid, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool), border_value=0)]
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    main_xy = np.asarray(network["branches"][int(network["main_branch_index"])]["coords_projected_m"], dtype=float)
    main_cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(main_xy, axis=0), axis=1))]
    grid_xy = np.column_stack([xx[valid], yy[valid]]); _, main_ids = cKDTree(grid_xy).query(main_xy, k=1, workers=1)
    return {
        "static": static, "valid": valid, "domain_ij": domain_ij, "node_count_2d": node_count_2d,
        "cell_area": cell_area, "bed": bed, "thick": thick, "surface": surface, "storage": storage,
        "edge_u": edge_u, "edge_v": edge_v, "edges": edges, "distance": distance, "face_width": face_width,
        "incidence": incidence, "boundary_ids": boundary_ids, "tail_width": tail_width, "tail_dx": tail_dx,
        "main_ids": np.asarray(main_ids, dtype=np.int64), "main_cumulative": main_cumulative,
    }


def conductance(state, surface, thick, glen_a, fs, glen_n, rhog):
    u = state["edge_u"]; v = state["edge_v"]; distance = state["distance"]
    su = surface[u]; sv = surface[v]; hu = thick[u]; hv = thick[v]
    donor_h = np.where(su >= sv, hu, hv); hf = (hu + hv) / 2.0; slope = (su - sv) / distance
    fd = 2.0 / (glen_n + 2.0) * glen_a
    out = (fd * hf**(glen_n + 2.0) + fs * hf**glen_n) * rhog * state["face_width"] * np.abs(slope)**(glen_n - 1.0) / distance
    out[donor_h <= 1e-6] = 0.0
    return out


def obstacle_step(state, surface, thick, mb, dt, glen_a, fs, glen_n, rhog, closure_limit):
    incidence = state["incidence"]; storage = state["storage"]; bed = state["bed"]
    k = conductance(state, surface, thick, glen_a, fs, glen_n, rhog)
    lap = incidence @ sparse.diags(k, format="csc") @ incidence.T
    matrix = sparse.diags(storage, format="csc") + dt * lap
    rhs = storage * surface + dt * storage * mb
    next_surface, gradient, active, solver = solve_lower_bound_qp(matrix, rhs, bed, surface, tolerance=1e-8, maximum_iterations=100)
    next_thick = next_surface - bed
    reaction = float(np.sum(gradient[active])); prescribed = float(dt * np.dot(storage, mb)); actual = float(np.dot(storage, next_thick - thick))
    budget_residual = actual - prescribed - reaction
    su = surface[state["edge_u"]]; sv = surface[state["edge_v"]]
    direction_flips = int(np.count_nonzero((su - sv) * (next_surface[state["edge_u"]] - next_surface[state["edge_v"]]) < -1e-12))
    free = ~active
    ok = (
        np.all(np.isfinite(next_thick)) and float(np.min(next_thick)) >= -1e-8
        and (float(np.max(np.abs(gradient[free]))) if np.any(free) else 0.0) <= closure_limit
        and (float(np.min(gradient[active])) if np.any(active) else 0.0) >= -closure_limit
        and reaction >= -closure_limit and abs(budget_residual) <= closure_limit and direction_flips == 0
    )
    return next_surface, next_thick, {
        "ok": bool(ok), "reaction_m3": reaction, "prescribed_mb_m3": prescribed,
        "actual_volume_change_m3": actual, "budget_residual_m3": budget_residual,
        "direction_flip_count": direction_flips, "active_node_count": int(np.count_nonzero(active)),
        "newly_exhausted_count": int(np.count_nonzero(active & (thick > 1e-6))), **solver,
    }


def snapshot(state, thick, year):
    active_main = thick[state["main_ids"]] > 10.0
    if np.any(active_main):
        first = int(np.flatnonzero(active_main)[0]); last = int(np.flatnonzero(active_main)[-1])
        main_length = float(state["main_cumulative"][last] - state["main_cumulative"][first])
    else:
        first = last = -1; main_length = 0.0
    tail = thick[state["node_count_2d"]:]; tail_active = np.flatnonzero(tail > 10.0)
    tail_progress = 0.0 if not tail_active.size else float((tail_active[-1] + 1) * state["tail_dx"])
    length = main_length + tail_progress
    area = float(np.count_nonzero(thick[:state["node_count_2d"]] > 1e-6) * state["cell_area"] + np.sum((tail > 1e-6) * state["tail_width"] * state["tail_dx"]))
    return {
        "year": int(year), "volume_m3": float(np.dot(state["storage"], thick)), "area_m2": area,
        "length_m": length, "length_relative_error_lia": abs(length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M,
        "main_first_active_index": first, "main_last_active_index": last, "main_internal_gap_count": false_runs(active_main),
        "tail_progress_m": tail_progress, "remaining_free_downstream_m": float(max(0.0, state["static"]["hybrid_interface"]["tail_length_m"] - tail_progress)),
        "minimum_thickness_m": float(np.min(thick)), "maximum_thickness_m": float(np.max(thick)),
        "positive_boundary_nodes": int(np.count_nonzero(thick[state["boundary_ids"]] > 1e-6)),
        "terminal_thickness_m": float(thick[-1]), "finite_geometry": bool(np.all(np.isfinite(thick))),
    }
