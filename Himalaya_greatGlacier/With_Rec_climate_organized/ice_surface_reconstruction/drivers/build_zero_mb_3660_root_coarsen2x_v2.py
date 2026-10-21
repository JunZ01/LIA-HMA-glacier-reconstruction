"""Build and statically gate one conservative 2x-coarsened root for 3660."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from shapely.geometry import LineString


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = (
    PROJECT
    / "w382inv3660v1/synthetic_gdir/RGI60-14/RGI60-14.93"
    / "RGI60-14.93660/model_flowlines.pkl"
)
CONFIG = PROJECT / "w382inv3660v1/config_snapshot.json"
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v2"
)
ROOT_ID = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_reverse_edges(flowlines) -> list[dict]:
    rows = []
    for flowline_id, flowline in enumerate(flowlines):
        surface = np.asarray(flowline.surface_h, dtype=float)
        active = np.asarray(flowline.thick, dtype=float) > 0.0
        for node in np.flatnonzero(active[:-1] & active[1:] & (np.diff(surface) > 1e-9)):
            rows.append({
                "flowline": int(flowline_id),
                "node": int(node),
                "surface_step_m": float(surface[node + 1] - surface[node]),
            })
    return rows


def _active_junction_reverse_edges(flowlines) -> list[dict]:
    rows = []
    for branch_id, branch in enumerate(flowlines):
        if branch.flows_to is None:
            continue
        receiver_id = next(i for i, candidate in enumerate(flowlines) if candidate is branch.flows_to)
        active = np.flatnonzero(np.asarray(branch.thick, dtype=float) > 0.0)
        if not len(active):
            continue
        terminal = int(active[-1])
        receiver_node = int(branch.flows_to_indice)
        step = float(branch.flows_to.surface_h[receiver_node] - branch.surface_h[terminal])
        if step > 1e-9:
            rows.append({
                "branch": int(branch_id),
                "receiver": int(receiver_id),
                "receiver_node": receiver_node,
                "surface_step_m": step,
            })
    return rows


def _build_root(root):
    from oggm.core.flowline import MixedBedFlowline

    surface = np.asarray(root.surface_h, dtype=float)
    bed = np.asarray(root.bed_h, dtype=float)
    section = np.asarray(root.section, dtype=float)
    width = np.asarray(root.widths_m, dtype=float)
    paired = len(surface) - 1
    a = np.arange(0, paired, 2)
    b = a + 1
    pair_width = width[a] + width[b]
    coarse_width = 0.5 * pair_width
    coarse_section = 0.5 * (section[a] + section[b])
    coarse_surface = np.divide(
        width[a] * surface[a] + width[b] * surface[b],
        pair_width,
        out=0.5 * (surface[a] + surface[b]),
        where=pair_width > 0,
    )
    coarse_bed = np.divide(
        width[a] * bed[a] + width[b] * bed[b],
        pair_width,
        out=0.5 * (bed[a] + bed[b]),
        where=pair_width > 0,
    )
    coarse_surface = np.append(coarse_surface, surface[-1])
    coarse_bed = np.append(coarse_bed, bed[-1])
    coarse_section = np.append(coarse_section, section[-1] * 0.5)
    coarse_width = np.append(coarse_width, width[-1])
    thick = coarse_surface - coarse_bed
    active = coarse_section > 0.0
    lambdas = np.zeros(len(coarse_section), dtype=float)
    w0 = coarse_width.copy()
    raw_lambda = np.full(len(coarse_section), np.nan)
    raw_lambda[active] = (
        2.0 * (coarse_width[active] * thick[active] - coarse_section[active])
        / thick[active] ** 2
    )
    if np.any(raw_lambda[active] < -1e-10):
        raise RuntimeError("material-negative-trapezoid-lambda")
    positive = active & (raw_lambda > 0.0)
    rectangular = active & ~positive
    lambdas[positive] = raw_lambda[positive]
    w0[positive] = 2.0 * coarse_section[positive] / thick[positive] - coarse_width[positive]
    w0[rectangular] = coarse_section[rectangular] / thick[rectangular]
    if np.any(w0 <= 0.0):
        raise RuntimeError("nonpositive-coarse-origin-width")

    coords = np.asarray(root.line.coords)
    coarse_line = LineString(coords[::2])
    new_root = MixedBedFlowline(
        line=coarse_line,
        dx=float(root.dx) * 2.0,
        map_dx=float(root.map_dx),
        surface_h=coarse_surface,
        bed_h=coarse_bed,
        section=coarse_section,
        bed_shape=np.ones(len(coarse_section), dtype=float),
        is_trapezoid=np.ones(len(coarse_section), dtype=bool),
        lambdas=lambdas,
        widths_m=w0,
        rgi_id=root.rgi_id,
        water_level=root.water_level,
    )
    return new_root, {
        "raw_minimum_lambda": float(np.nanmin(raw_lambda[active])),
        "clamped_roundoff_lambda_count": int(np.count_nonzero(active & (raw_lambda <= 0.0))),
        "minimum_origin_width_m": float(np.min(new_root._w0_m)),
    }


def main() -> None:
    from oggm import cfg

    OUTPUT.mkdir(parents=True, exist_ok=False)
    cfg.initialize(logging_level="WARNING")
    with gzip.open(SOURCE, "rb") as stream:
        original = pickle.load(stream)
    candidate = copy.deepcopy(original)
    old_root = candidate[ROOT_ID]
    new_root, shape = _build_root(old_root)
    direct_branches = [
        index for index, branch in enumerate(candidate)
        if branch.flows_to is old_root
    ]
    candidate[ROOT_ID] = new_root
    mapping = []
    for branch_id in direct_branches:
        branch = candidate[branch_id]
        old_node = int(branch.flows_to_indice)
        old_distance = old_node * float(old_root.dx_meter)
        branch.__dict__.pop("flows_to_indice", None)
        branch.set_flows_to(new_root, check_tail=True)
        new_node = int(branch.flows_to_indice)
        new_distance = new_node * float(new_root.dx_meter)
        mapping.append({
            "branch": int(branch_id),
            "old_receiver_node": old_node,
            "new_receiver_node": new_node,
            "old_distance_m": old_distance,
            "new_distance_m": new_distance,
            "distance_error_m": new_distance - old_distance,
        })

    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    registered_length = float(config["registered_lia_length_m"])
    area0 = float(sum(flowline.area_m2 for flowline in original))
    area1 = float(sum(flowline.area_m2 for flowline in candidate))
    volume0 = float(sum(flowline.volume_m3 for flowline in original))
    volume1 = float(sum(flowline.volume_m3 for flowline in candidate))
    old_root_source = original[ROOT_ID]
    active0 = np.asarray(old_root_source.thick, dtype=float) > 0.0
    active1 = np.asarray(new_root.thick, dtype=float) > 0.0
    surface_integral0 = float(np.sum(old_root_source.surface_h[active0] * old_root_source.widths_m[active0]) * old_root_source.dx_meter)
    surface_integral1 = float(np.sum(new_root.surface_h[active1] * new_root.widths_m[active1]) * new_root.dx_meter)
    bed_integral0 = float(np.sum(old_root_source.bed_h[active0] * old_root_source.widths_m[active0]) * old_root_source.dx_meter)
    bed_integral1 = float(np.sum(new_root.bed_h[active1] * new_root.widths_m[active1]) * new_root.dx_meter)
    reverse0 = _active_reverse_edges(original)
    reverse1 = _active_reverse_edges(candidate)
    junction0 = _active_junction_reverse_edges(original)
    junction1 = _active_junction_reverse_edges(candidate)
    topology_valid = all(
        branch.flows_to is None or branch.flows_to in candidate for branch in candidate
    )
    root_length = float(new_root.length_m)
    coordinate_span0 = float((old_root_source.nx - 1) * old_root_source.dx_meter)
    coordinate_span1 = float((new_root.nx - 1) * new_root.dx_meter)
    free_tail = float((new_root.nx - np.count_nonzero(active1)) * new_root.dx_meter)
    gates = {
        "unique_root": sum(flowline.flows_to is None for flowline in candidate) == 1,
        "topology_valid": topology_valid,
        "root_coordinate_span_unchanged": abs(coordinate_span1 - coordinate_span0) <= 1e-8,
        "root_length_formal_10pct": abs(root_length - registered_length) / registered_length <= 0.10,
        "total_area_conserved": abs(area1 - area0) <= max(1e-6, area0 * 1e-12),
        "total_volume_conserved": abs(volume1 - volume0) <= max(1e-6, volume0 * 1e-12),
        "active_surface_integral_conserved": abs(surface_integral1 - surface_integral0) <= max(1e-5, abs(surface_integral0) * 1e-12),
        "active_bed_integral_conserved": abs(bed_integral1 - bed_integral0) <= max(1e-5, abs(bed_integral0) * 1e-12),
        "active_within_reverse_absent": len(reverse1) == 0,
        "active_junction_reverse_absent": len(junction1) == 0,
        "root_terminal_zero": float(new_root.thick[-1]) == 0.0,
        "free_tail_at_least_5km": free_tail >= 5000.0,
        "receiver_mapping_within_half_coarse_cell": max(abs(row["distance_error_m"]) for row in mapping) <= 50.0,
    }
    static_pass = all(gates.values())
    candidate_path = OUTPUT / "candidate_model_flowlines.pkl"
    if static_pass:
        with gzip.open(candidate_path, "wb") as stream:
            pickle.dump(candidate, stream, protocol=pickle.HIGHEST_PROTOCOL)
    payload = {
        "experiment": "zero-mb-3660-conservative-root-coarsen2x-static-preflight-v2",
        "system_id": 3660,
        "mechanism": "uniform-root-2x-width-weighted-conservative-control-volumes",
        "source_model_flowlines": str(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "root_flowline": ROOT_ID,
        "root_nodes_before": int(old_root_source.nx),
        "root_nodes_after": int(new_root.nx),
        "root_dx_before_m": float(old_root_source.dx_meter),
        "root_dx_after_m": float(new_root.dx_meter),
        "shape": shape,
        "receiver_mapping": mapping,
        "metrics": {
            "area_difference_m2": area1 - area0,
            "volume_difference_m3": volume1 - volume0,
            "active_surface_integral_difference_m3": surface_integral1 - surface_integral0,
            "active_bed_integral_difference_m3": bed_integral1 - bed_integral0,
            "root_coordinate_span_before_m": coordinate_span0,
            "root_coordinate_span_after_m": coordinate_span1,
            "root_length_before_m": float(old_root_source.length_m),
            "root_length_after_m": root_length,
            "registered_length_m": registered_length,
            "registered_length_relative_error": abs(root_length - registered_length) / registered_length,
            "free_tail_m": free_tail,
            "within_reverse_before": len(reverse0),
            "within_reverse_after": len(reverse1),
            "junction_reverse_before": len(junction0),
            "junction_reverse_after": len(junction1),
            "maximum_absolute_receiver_mapping_error_m": max(abs(row["distance_error_m"]) for row in mapping),
        },
        "junction_reverse_edges": junction1,
        "gates": gates,
        "static_pass": static_pass,
        "candidate_model_flowlines": str(candidate_path) if static_pass else None,
        "candidate_sha256": _sha256(candidate_path) if static_pass else None,
        "next_action": "one-30s-5y-zeroMB" if static_pass else "close-before-dynamics",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
