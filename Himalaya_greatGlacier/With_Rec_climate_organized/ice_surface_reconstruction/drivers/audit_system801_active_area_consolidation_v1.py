"""Read-only active-area consolidation feasibility audit for system 801."""

from __future__ import annotations

import gzip
import heapq
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
ROOT = PROJECT / "w556inv801v1"
GDIR = ROOT / "synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90801"
PAIRED = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_801_RGI60-15.05606_system_grid_v1.nc"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system801_active_area_consolidation_v1"


def load(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    inversion = load(GDIR / "inversion_flowlines.pkl")
    model = load(GDIR / "model_flowlines.pkl")
    if len(inversion) != len(model):
        raise RuntimeError("branch-count-mismatch")
    model_by_id = {id(fl): i for i, fl in enumerate(model)}
    offsets, inv_nx, branches, nodes, widths, sections, surfaces, dxs, active = [], [], [], [], [], [], [], [], []
    cursor = 0
    for branch, (inv_fl, mod_fl) in enumerate(zip(inversion, model)):
        n = int(inv_fl.nx)
        if mod_fl.nx < n:
            raise RuntimeError(f"model-shorter:{branch}")
        offsets.append(cursor)
        inv_nx.append(n)
        branches.extend([branch] * n)
        nodes.extend(range(n))
        widths.extend(np.asarray(mod_fl.widths_m[:n], dtype=float))
        sections.extend(np.asarray(mod_fl.section[:n], dtype=float))
        surfaces.extend(np.asarray(mod_fl.surface_h[:n], dtype=float))
        dxs.extend([float(mod_fl.dx_meter)] * n)
        active.extend(np.asarray(mod_fl.thick[:n], dtype=float) > 1e-6)
        cursor += n
    branches = np.asarray(branches, dtype=int)
    nodes = np.asarray(nodes, dtype=int)
    widths = np.asarray(widths, dtype=float)
    sections = np.asarray(sections, dtype=float)
    surfaces = np.asarray(surfaces, dtype=float)
    dxs = np.asarray(dxs, dtype=float)
    active = np.asarray(active, dtype=bool)
    n_total = len(widths)
    adjacency = [[] for _ in range(n_total)]
    for branch, mod_fl in enumerate(model):
        off = offsets[branch]
        for node in range(inv_nx[branch] - 1):
            u, v = off + node, off + node + 1
            distance = 0.5 * (dxs[u] + dxs[v])
            adjacency[u].append((v, distance))
            adjacency[v].append((u, distance))
        if mod_fl.flows_to is not None:
            receiver = model_by_id[id(mod_fl.flows_to)]
            receiver_node = int(mod_fl.flows_to_indice)
            if receiver_node < inv_nx[receiver]:
                u, v = off + inv_nx[branch] - 1, offsets[receiver] + receiver_node
                distance = 0.5 * (dxs[u] + dxs[v])
                adjacency[u].append((v, distance))
                adjacency[v].append((u, distance))
    distance = np.full(n_total, np.inf)
    owner = np.full(n_total, -1, dtype=int)
    queue = []
    for node in np.flatnonzero(active):
        distance[node] = 0.0
        owner[node] = node
        heapq.heappush(queue, (0.0, int(node), int(node)))
    while queue:
        dist, source, node = heapq.heappop(queue)
        if dist > distance[node] + 1e-12 or (abs(dist - distance[node]) <= 1e-12 and source != owner[node]):
            continue
        for nxt, edge_length in adjacency[node]:
            candidate = dist + edge_length
            if candidate < distance[nxt] - 1e-12 or (abs(candidate - distance[nxt]) <= 1e-12 and source < owner[nxt]):
                distance[nxt] = candidate
                owner[nxt] = source
                heapq.heappush(queue, (candidate, source, nxt))
    footprint = np.isfinite(widths) & (widths > 0)
    if np.any(owner[footprint] < 0):
        raise RuntimeError("unreachable-footprint-node")
    original_control_area = widths * dxs
    candidate_control_area = np.zeros(n_total, dtype=float)
    for node in np.flatnonzero(footprint):
        candidate_control_area[owner[node]] += original_control_area[node]
    candidate_width = np.zeros(n_total, dtype=float)
    candidate_width[active] = candidate_control_area[active] / dxs[active]
    candidate_thickness = np.zeros(n_total, dtype=float)
    candidate_thickness[active] = sections[active] / candidate_width[active]
    candidate_bed = surfaces - candidate_thickness
    width_multiplier = candidate_width[active] / widths[active]
    original_thickness = np.zeros(n_total, dtype=float)
    original_thickness[active] = sections[active] / widths[active]
    full_area = float(np.sum(original_control_area[footprint]))
    candidate_area = float(np.sum(candidate_control_area[active]))
    original_volume = float(np.sum(sections * dxs))
    candidate_volume = float(np.sum(candidate_width * candidate_thickness * dxs))
    with xr.open_dataset(PAIRED) as ds:
        h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
        s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
        positive = np.asarray(ds.valid_mask.isel(member=0), dtype=bool) & np.asarray(ds.lia_mask, dtype=bool) & np.isfinite(h) & np.isfinite(s) & (h > 1e-6)
        source_surface = s[positive]
    cdf = cdf_error_100m(source_surface, surfaces[active], candidate_control_area[active])
    summary = json.loads((ROOT / "inversion_summary_formal.json").read_text(encoding="utf-8"))
    outline_area = float(summary["outline_area_m2"])
    area_error = abs(candidate_area - outline_area) / outline_area
    source_volume_error = abs(candidate_volume - original_volume) / max(original_volume, 1.0)
    passed = bool(
        area_error <= 0.10 and source_volume_error <= 1e-12 and cdf <= 0.02
        and np.all(np.isfinite(candidate_width)) and np.all(candidate_width[active] > 0)
        and np.all(np.isfinite(candidate_thickness)) and np.all(candidate_thickness[active] > 0)
        and float(np.max(width_multiplier)) <= 5.0
        and float(np.max(distance[footprint])) <= 5000.0
    )
    result = {
        "schema": "lia-191-system801-active-area-consolidation-v1",
        "system_id": 801,
        "status": "static-feasibility-pass-not-materialized" if passed else "static-feasibility-fail-mechanism-closed",
        "branch_count": len(model), "inversion_footprint_node_count": int(np.count_nonzero(footprint)),
        "active_node_count": int(np.count_nonzero(active)), "zero_thickness_footprint_node_count": int(np.count_nonzero(footprint & ~active)),
        "fully_inactive_branch_count": int(sum(not np.any(active[offsets[b]:offsets[b] + inv_nx[b]]) for b in range(len(model)))),
        "full_inversion_width_area_m2": full_area, "candidate_active_area_m2": candidate_area,
        "outline_area_m2": outline_area, "candidate_outline_area_error_fraction": area_error, "area_gate": 0.10,
        "original_model_volume_m3": original_volume, "candidate_volume_m3": candidate_volume, "volume_error_fraction": source_volume_error,
        "paired_hypsometry_cdf_error_100m": cdf, "hypsometry_gate": 0.02,
        "maximum_width_multiplier": float(np.max(width_multiplier)), "width_multiplier_gate": 5.0,
        "p95_width_multiplier": float(np.percentile(width_multiplier, 95)),
        "maximum_area_transfer_distance_m": float(np.max(distance[footprint])), "p95_area_transfer_distance_m": float(np.percentile(distance[footprint], 95)),
        "maximum_thickness_change_m": float(np.max(np.abs(candidate_thickness[active] - original_thickness[active]))),
        "maximum_candidate_thickness_m": float(np.max(candidate_thickness)),
        "minimum_candidate_thickness_m_active": float(np.min(candidate_thickness[active])),
        "maximum_candidate_bed_step_m": float(np.max(np.abs(np.diff(candidate_bed)))),
        "surface_unchanged": True, "section_and_volume_unchanged": True,
        "zero_thickness_nodes_remain_zero": bool(np.all(candidate_thickness[~active] == 0)),
        "topology_unchanged": True, "climate_and_apparent_mb_unchanged": True,
        "candidate_is_rectangular_area_equivalent_representation": True,
        "materialized_geometry": False, "zero_mb_run": False, "steady_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(OUTPUT / "candidate_arrays.npz", candidate_width_m=candidate_width, candidate_thickness_m=candidate_thickness, candidate_bed_m=candidate_bed, owner=owner, transfer_distance_m=distance)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
