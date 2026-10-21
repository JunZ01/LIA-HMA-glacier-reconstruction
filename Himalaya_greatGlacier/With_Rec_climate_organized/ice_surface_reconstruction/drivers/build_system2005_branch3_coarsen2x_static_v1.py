"""Static gate for a conservative 2x coarsening of system-2005 branch 3.

This is a representation test, not a parameter adjustment.  The 18 original
50 m rectangular control volumes of the branch are paired from the head into
9 100 m rectangular control volumes.  Width-weighted surface and bed and the
paired section preserve branch area, volume, and their area-weighted vertical
integrals.  All other branches and the receiver relation remain unchanged.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from oggm import cfg
from oggm.core.flowline import MixedBedFlowline
from shapely.geometry import LineString, Point


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = (
    VALIDATION
    / "lia_191_system2005_terminal_retreat_gap_adjudication_v2"
    / "lia_equilibrium_state_50y.pkl.gz"
)
OUTPUT = VALIDATION / "lia_191_system2005_branch3_coarsen2x_static_v1"
BRANCH = 3
ACTIVE_TOLERANCE_M = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def total_area(flowlines) -> float:
    return float(sum(
        np.sum(np.asarray(fl.widths_m)[np.asarray(fl.thick) > ACTIVE_TOLERANCE_M])
        * float(fl.dx_meter)
        for fl in flowlines
    ))


def total_volume(flowlines) -> float:
    return float(sum(np.sum(np.asarray(fl.section)) * float(fl.dx_meter) for fl in flowlines))


def vertical_integrals(flowline) -> tuple[float, float]:
    active = np.asarray(flowline.thick) > ACTIVE_TOLERANCE_M
    weights = np.asarray(flowline.widths_m)[active] * float(flowline.dx_meter)
    return (
        float(np.sum(np.asarray(flowline.surface_h)[active] * weights)),
        float(np.sum(np.asarray(flowline.bed_h)[active] * weights)),
    )


def hypsometry(flowlines) -> tuple[np.ndarray, np.ndarray]:
    surfaces = []
    areas = []
    for fl in flowlines:
        active = np.asarray(fl.thick) > ACTIVE_TOLERANCE_M
        surfaces.extend(np.asarray(fl.surface_h)[active].tolist())
        areas.extend((np.asarray(fl.widths_m)[active] * float(fl.dx_meter)).tolist())
    return np.asarray(surfaces, dtype=float), np.asarray(areas, dtype=float)


def cumulative_hypsometry_error(source, candidate) -> float:
    s0, a0 = hypsometry(source)
    s1, a1 = hypsometry(candidate)
    lower = 100.0 * np.floor(min(float(np.min(s0)), float(np.min(s1))) / 100.0)
    upper = 100.0 * np.ceil(max(float(np.max(s0)), float(np.max(s1))) / 100.0)
    edges = np.arange(lower, upper + 100.0, 100.0)
    h0, _ = np.histogram(s0, bins=edges, weights=a0)
    h1, _ = np.histogram(s1, bins=edges, weights=a1)
    c0 = np.cumsum(h0) / float(np.sum(h0))
    c1 = np.cumsum(h1) / float(np.sum(h1))
    return float(np.max(np.abs(c1 - c0)))


def active_reverse_edges(flowlines) -> list[dict]:
    rows = []
    for branch, fl in enumerate(flowlines):
        active = np.asarray(fl.thick) > ACTIVE_TOLERANCE_M
        surface = np.asarray(fl.surface_h)
        for node in np.flatnonzero(active[:-1] & active[1:] & (np.diff(surface) > 1e-9)):
            rows.append({"branch": branch, "node": int(node), "rise_m": float(surface[node + 1] - surface[node])})
    return rows


def junction_reverse_edges(flowlines) -> list[dict]:
    rows = []
    index = {id(fl): i for i, fl in enumerate(flowlines)}
    for branch, fl in enumerate(flowlines):
        if fl.flows_to is None:
            continue
        active = np.flatnonzero(np.asarray(fl.thick) > ACTIVE_TOLERANCE_M)
        if not len(active):
            continue
        terminal = int(active[-1])
        receiver = index[id(fl.flows_to)]
        receiver_node = int(fl.flows_to_indice)
        rise = float(fl.flows_to.surface_h[receiver_node] - fl.surface_h[terminal])
        if rise > 1e-9:
            rows.append({"branch": branch, "receiver": receiver, "receiver_node": receiver_node, "rise_m": rise})
    return rows


def build_branch(source):
    if int(source.nx) != 18 or abs(float(source.dx_meter) - 50.0) > 1e-12:
        raise RuntimeError(f"unexpected-branch-grid:{source.nx}:{source.dx_meter}")
    if not np.all(np.asarray(source._lambdas) == 0.0):
        raise RuntimeError("branch3-is-not-rectangular")
    a = np.arange(0, source.nx, 2)
    b = a + 1
    width = np.asarray(source.widths_m, dtype=float)
    section = np.asarray(source.section, dtype=float)
    surface = np.asarray(source.surface_h, dtype=float)
    bed = np.asarray(source.bed_h, dtype=float)
    width_sum = width[a] + width[b]
    coarse_width = 0.5 * width_sum
    coarse_section = 0.5 * (section[a] + section[b])
    coarse_surface = (width[a] * surface[a] + width[b] * surface[b]) / width_sum
    coarse_bed = (width[a] * bed[a] + width[b] * bed[b]) / width_sum
    coarse_thick = coarse_surface - coarse_bed
    if np.max(np.abs(coarse_section - coarse_width * coarse_thick)) > 1e-8:
        raise RuntimeError("rectangular-pair-closure-failed")
    coords = np.asarray(source.line.coords, dtype=float)
    coarse_coords = 0.5 * (coords[a] + coords[b])
    candidate = MixedBedFlowline(
        line=LineString(coarse_coords),
        dx=float(source.dx) * 2.0,
        map_dx=float(source.map_dx),
        surface_h=coarse_surface,
        bed_h=coarse_bed,
        section=coarse_section,
        bed_shape=np.ones(len(a), dtype=float),
        is_trapezoid=np.ones(len(a), dtype=bool),
        lambdas=np.zeros(len(a), dtype=float),
        widths_m=coarse_width,
        rgi_id=source.rgi_id,
        water_level=source.water_level,
    )
    return candidate


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    cfg.initialize(logging_level="WARNING")
    source_hash_before = sha256(SOURCE)
    with gzip.open(SOURCE, "rb") as stream:
        original = pickle.load(stream)
    candidate = copy.deepcopy(original)
    old = candidate[BRANCH]
    receiver = old.flows_to
    old_receiver_index = int(old.flows_to_indice)
    if receiver is None:
        raise RuntimeError("branch3-is-root")
    direct_children = [i for i, fl in enumerate(candidate) if fl.flows_to is old]
    if direct_children:
        raise RuntimeError(f"branch3-has-upstream-tributaries:{direct_children}")
    new = build_branch(old)
    old_inflow_position = next(i for i, fl in enumerate(receiver.inflows) if fl is old)
    receiver.inflows.pop(old_inflow_position)
    receiver.inflow_points.pop(old_inflow_position)
    new.flows_to = receiver
    new.flows_to_point = Point(receiver.line.coords[old_receiver_index])
    receiver.inflows.append(new)
    receiver.inflow_points.append(new.flows_to_point)
    candidate[BRANCH] = new
    new_receiver_index = int(new.flows_to_indice)

    area0, area1 = total_area(original), total_area(candidate)
    volume0, volume1 = total_volume(original), total_volume(candidate)
    branch_area0 = total_area([original[BRANCH]])
    branch_area1 = total_area([candidate[BRANCH]])
    branch_volume0 = total_volume([original[BRANCH]])
    branch_volume1 = total_volume([candidate[BRANCH]])
    surface0, bed0 = vertical_integrals(original[BRANCH])
    surface1, bed1 = vertical_integrals(candidate[BRANCH])
    reverse = active_reverse_edges(candidate)
    junction = junction_reverse_edges(candidate)
    hypsometry_error = cumulative_hypsometry_error(original, candidate)
    roundtrip = copy.deepcopy(new)
    section_before = np.asarray(roundtrip.section).copy()
    volume_before = float(roundtrip.volume_m3)
    roundtrip.section = section_before.copy()
    setter_section_error = float(np.max(np.abs(np.asarray(roundtrip.section) - section_before)))
    setter_volume_error = float(roundtrip.volume_m3 - volume_before)
    source_hash_after = sha256(SOURCE)
    gates = {
        "source_immutable": source_hash_after == source_hash_before,
        "branch_has_no_upstream_tributaries": not direct_children,
        "unique_root": sum(fl.flows_to is None for fl in candidate) == 1,
        "topology_references_valid": all(fl.flows_to is None or fl.flows_to in candidate for fl in candidate),
        "receiver_index_unchanged": new_receiver_index == old_receiver_index,
        "branch_control_volume_length_conserved": abs(old.nx * old.dx_meter - new.nx * new.dx_meter) <= 1e-9,
        "total_area_conserved": abs(area1 - area0) <= max(1e-6, area0 * 1e-12),
        "total_volume_conserved": abs(volume1 - volume0) <= max(1e-6, volume0 * 1e-12),
        "branch_area_conserved": abs(branch_area1 - branch_area0) <= max(1e-6, branch_area0 * 1e-12),
        "branch_volume_conserved": abs(branch_volume1 - branch_volume0) <= max(1e-6, branch_volume0 * 1e-12),
        "branch_surface_integral_conserved": abs(surface1 - surface0) <= max(1e-5, abs(surface0) * 1e-12),
        "branch_bed_integral_conserved": abs(bed1 - bed0) <= max(1e-5, abs(bed0) * 1e-12),
        "hypsometry_100m_cdf_within_2pct": hypsometry_error <= 0.02,
        "active_within_reverse_absent": len(reverse) == 0,
        "active_junction_reverse_absent": len(junction) == 0,
        "finite_nonnegative_geometry": all(
            np.isfinite(fl.section).all() and np.isfinite(fl.thick).all()
            and np.all(np.asarray(fl.section) >= 0.0) and np.all(np.asarray(fl.thick) >= 0.0)
            for fl in candidate
        ),
        "mixedbed_setter_roundtrip": abs(setter_volume_error) <= 1e-5 and setter_section_error <= 1e-6,
    }
    static_pass = all(gates.values())
    candidate_path = OUTPUT / "candidate_model_flowlines.pkl.gz"
    if static_pass:
        with gzip.open(candidate_path, "wb") as stream:
            pickle.dump(candidate, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "schema": "lia-191-system2005-branch3-coarsen2x-static-v1",
        "status": "static-pass" if static_pass else "static-fail-mechanism-closed",
        "system_id": 2005,
        "mechanism": "tributary-branch3-uniform-2x-width-weighted-conservative-control-volumes",
        "source": str(SOURCE),
        "source_sha256": source_hash_before,
        "branch": BRANCH,
        "nodes_before": int(old.nx),
        "nodes_after": int(new.nx),
        "dx_before_m": float(old.dx_meter),
        "dx_after_m": float(new.dx_meter),
        "receiver_branch": int(next(i for i, fl in enumerate(candidate) if fl is receiver)),
        "receiver_node_before": old_receiver_index,
        "receiver_node_after": new_receiver_index,
        "metrics": {
            "total_area_difference_m2": area1 - area0,
            "total_volume_difference_m3": volume1 - volume0,
            "branch_area_difference_m2": branch_area1 - branch_area0,
            "branch_volume_difference_m3": branch_volume1 - branch_volume0,
            "branch_surface_integral_difference_m3": surface1 - surface0,
            "branch_bed_integral_difference_m3": bed1 - bed0,
            "hypsometry_100m_cdf_error": hypsometry_error,
            "within_reverse_count": len(reverse),
            "junction_reverse_count": len(junction),
            "setter_volume_error_m3": setter_volume_error,
            "setter_section_error_m2": setter_section_error,
            "maximum_line_center_shift_m": float(np.max(np.linalg.norm(np.asarray(new.line.coords) - 0.5 * (np.asarray(old.line.coords)[::2] + np.asarray(old.line.coords)[1::2]), axis=1))),
        },
        "gates": gates,
        "candidate_model_flowlines": str(candidate_path) if static_pass else None,
        "candidate_sha256": sha256(candidate_path) if static_pass else None,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_action": "one-0p1y-then-5y-zero-mb" if static_pass else "close-mechanism",
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    if not static_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
