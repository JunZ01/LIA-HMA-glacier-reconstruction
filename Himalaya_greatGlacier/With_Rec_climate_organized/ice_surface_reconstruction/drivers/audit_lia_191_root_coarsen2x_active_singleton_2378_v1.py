"""Static-only feasibility audit of an active-singleton 2x root for 2378.

Complete active pairs are coarsened conventionally.  The final 50 m active
cell is represented as a half control volume on the uniform 100 m root by
halving both its transverse width and section.  Zero-ice cells are then paired
from the next source cell.  The candidate exists only in memory: this audit
writes metrics, not model flowlines, and starts no dynamics.
"""

from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import pickle
from pathlib import Path

import numpy as np
from oggm.core.flowline import MixedBedFlowline
from shapely.geometry import LineString, Point

from .build_lia_191_root_coarsen2x_representative_static_v1 import (
    _active_junction_reverse_edges,
    _active_reverse_edges,
    _integral,
    _sha256,
)


PROJECT = Path(__file__).resolve().parents[4]
CONFIG = (
    PROJECT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction"
    / "config/lia_191_root_coarsen2x_active_singleton_2378_v1.json"
)


def _weighted_pair(values, widths, a, b):
    weight = widths[a] + widths[b]
    return np.divide(
        widths[a] * values[a] + widths[b] * values[b],
        weight,
        out=0.5 * (values[a] + values[b]),
        where=weight > 0.0,
    )


def _build(root):
    surface = np.asarray(root.surface_h, dtype=float)
    bed = np.asarray(root.bed_h, dtype=float)
    section = np.asarray(root.section, dtype=float)
    width = np.asarray(root.widths_m, dtype=float)
    active_mask = section > 0.0
    active_count = int(np.count_nonzero(active_mask))
    if not np.array_equal(np.flatnonzero(active_mask), np.arange(active_count)):
        raise RuntimeError("source-active-support-not-contiguous-from-root-head")
    if active_count % 2 != 1:
        raise RuntimeError("source-active-count-is-not-odd")
    tail_count = int(root.nx) - active_count
    if tail_count % 2 != 0:
        raise RuntimeError("source-zero-tail-count-is-not-even")

    active_pair_a = np.arange(0, active_count - 1, 2)
    active_pair_b = active_pair_a + 1
    tail_pair_a = np.arange(active_count, int(root.nx), 2)
    tail_pair_b = tail_pair_a + 1
    singleton = active_count - 1

    pair_surface_active = _weighted_pair(
        surface, width, active_pair_a, active_pair_b
    )
    pair_bed_active = _weighted_pair(bed, width, active_pair_a, active_pair_b)
    pair_width_active = 0.5 * (width[active_pair_a] + width[active_pair_b])
    pair_section_active = 0.5 * (
        section[active_pair_a] + section[active_pair_b]
    )
    pair_surface_tail = _weighted_pair(surface, width, tail_pair_a, tail_pair_b)
    pair_bed_tail = _weighted_pair(bed, width, tail_pair_a, tail_pair_b)
    pair_width_tail = 0.5 * (width[tail_pair_a] + width[tail_pair_b])
    pair_section_tail = 0.5 * (section[tail_pair_a] + section[tail_pair_b])

    coarse_surface = np.concatenate(
        (pair_surface_active, [surface[singleton]], pair_surface_tail)
    )
    coarse_bed = np.concatenate((pair_bed_active, [bed[singleton]], pair_bed_tail))
    coarse_width = np.concatenate(
        (pair_width_active, [0.5 * width[singleton]], pair_width_tail)
    )
    coarse_section = np.concatenate(
        (pair_section_active, [0.5 * section[singleton]], pair_section_tail)
    )
    if len(coarse_surface) != (int(root.nx) + 1) // 2:
        raise RuntimeError("unexpected-coarse-node-count")

    thick = coarse_surface - coarse_bed
    active = coarse_section > 0.0
    raw_lambda = np.full(len(coarse_section), np.nan)
    raw_lambda[active] = (
        2.0
        * (coarse_width[active] * thick[active] - coarse_section[active])
        / thick[active] ** 2
    )
    if np.any(raw_lambda[active] < -1e-10):
        raise RuntimeError("material-negative-trapezoid-lambda")
    lambdas = np.zeros(len(coarse_section), dtype=float)
    w0 = coarse_width.copy()
    positive = active & (raw_lambda >= 1e-10)
    rectangular = active & ~positive
    lambdas[positive] = raw_lambda[positive]
    w0[positive] = (
        2.0 * coarse_section[positive] / thick[positive] - coarse_width[positive]
    )
    w0[rectangular] = coarse_section[rectangular] / thick[rectangular]
    if np.any(w0 <= 0.0):
        raise RuntimeError("nonpositive-coarse-origin-width")

    new_root = MixedBedFlowline(
        line=LineString(np.asarray(root.line.coords)[::2]),
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
        "active_pair_count": int(len(active_pair_a)),
        "singleton_source_node": int(singleton),
        "singleton_source_support_m": float(root.dx_meter),
        "singleton_model_node_dx_m": float(new_root.dx_meter),
        "singleton_width_multiplier": 0.5,
        "singleton_section_multiplier": 0.5,
        "zero_tail_pair_count": int(len(tail_pair_a)),
        "source_active_count": active_count,
        "source_zero_tail_count": tail_count,
        "coarse_active_count": int(np.count_nonzero(active)),
        "raw_minimum_lambda": float(np.nanmin(raw_lambda[active])),
        "rectangular_limit_node_count": int(np.count_nonzero(rectangular)),
        "positive_trapezoid_node_count": int(np.count_nonzero(positive)),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    source = PROJECT / config["source_model_flowlines"]
    output = PROJECT / config["output_root"]
    if output.exists():
        raise FileExistsError(output)
    source_hash = _sha256(source)
    if source_hash != config["source_model_flowlines_sha256"]:
        raise RuntimeError("source-sha256-mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(source, "rb") as stream:
        original = pickle.load(stream)
    roots = [index for index, flowline in enumerate(original) if flowline.flows_to is None]
    if roots != [int(config["root_index"])]:
        raise RuntimeError(f"root-identity-mismatch:{roots}")
    root_id = roots[0]
    root_source = original[root_id]
    candidate = copy.deepcopy(original)
    old_root = candidate[root_id]
    new_root, construction = _build(old_root)

    mapping = []
    for branch_id, branch in enumerate(candidate):
        if branch.flows_to is not old_root:
            continue
        old_node = int(branch.flows_to_indice)
        new_node = int(math.ceil(old_node / 2.0))
        branch.__dict__.pop("flows_to_indice", None)
        branch.__dict__.pop("_lazy_flows_to_indice", None)
        branch.flows_to = new_root
        branch.flows_to_point = Point(new_root.line.coords[new_node])
        new_root.inflow_points.append(branch.flows_to_point)
        new_root.inflows.append(branch)
        mapping.append(
            {
                "branch": int(branch_id),
                "old_receiver_node": old_node,
                "new_receiver_node": new_node,
                "distance_error_m": new_node * float(new_root.dx_meter)
                - old_node * float(old_root.dx_meter),
            }
        )
    candidate[root_id] = new_root

    area0 = float(sum(flowline.area_m2 for flowline in original))
    area1 = float(sum(flowline.area_m2 for flowline in candidate))
    volume0 = float(sum(flowline.volume_m3 for flowline in original))
    volume1 = float(sum(flowline.volume_m3 for flowline in candidate))
    reverse0 = _active_reverse_edges(original)
    reverse1 = _active_reverse_edges(candidate)
    junction0 = _active_junction_reverse_edges(original)
    junction1 = _active_junction_reverse_edges(candidate)
    active_new = np.asarray(new_root.thick, dtype=float) > 0.0
    registered_length = float(config["registered_lia_length_m"])
    length_error = abs(float(new_root.length_m) - registered_length) / registered_length
    free_tail = float(
        (int(new_root.nx) - int(np.count_nonzero(active_new)))
        * float(new_root.dx_meter)
    )
    surface_difference = _integral(new_root, "surface_h") - _integral(
        root_source, "surface_h"
    )
    bed_difference = _integral(new_root, "bed_h") - _integral(root_source, "bed_h")
    test = copy.deepcopy(new_root)
    section0 = np.asarray(new_root.section, dtype=float).copy()
    volume_before_setter = float(new_root.volume_m3)
    test.section = section0.copy()
    setter_volume_difference = float(test.volume_m3 - volume_before_setter)
    setter_section_difference = float(
        np.max(np.abs(np.asarray(test.section, dtype=float) - section0))
    )
    maximum_mapping_error = max(
        [abs(float(row["distance_error_m"])) for row in mapping] or [0.0]
    )
    gates = {
        "scalar_dx_unchanged_within_candidate": bool(
            np.isclose(float(new_root.dx_meter), 100.0)
        ),
        "total_area_conserved": abs(area1 - area0) <= max(1e-6, area0 * 1e-12),
        "total_volume_conserved": abs(volume1 - volume0)
        <= max(1e-6, volume0 * 1e-12),
        "active_surface_integral_conserved": abs(surface_difference)
        <= max(1e-5, abs(_integral(root_source, "surface_h")) * 1e-12),
        "active_bed_integral_conserved": abs(bed_difference)
        <= max(1e-5, abs(_integral(root_source, "bed_h")) * 1e-12),
        "formal_length_within_10pct": length_error <= 0.10,
        "active_within_reverse_absent": len(reverse1) == 0,
        "active_junction_reverse_absent": len(junction1) == 0,
        "receiver_mapping_within_half_coarse_cell": maximum_mapping_error <= 50.0,
        "free_tail_at_least_5km": free_tail >= 5000.0,
        "root_terminal_zero": float(new_root.thick[-1]) == 0.0,
        "setter_roundtrip": abs(setter_volume_difference) <= 1e-5
        and setter_section_difference <= 1e-6,
    }
    feasibility_pass = all(gates.values())
    payload = {
        "experiment": config["experiment"],
        "system_id": 2378,
        "stage": "static-read-only-feasibility",
        "mechanism": "complete-active-pairs-plus-half-control-volume-active-singleton-then-zero-tail-pairs",
        "source_model_flowlines": str(source),
        "source_sha256": source_hash,
        "construction": construction,
        "data_model_assessment": {
            "uniform_scalar_dx_m": float(new_root.dx_meter),
            "requires_local_nonuniform_dx": false,
            "requires_oggm_data_model_change": false,
            "representation": "the 50 m singleton mass and area are encoded by half width and half section on one 100 m node",
            "formal_model_length_effect": "OGGM counts the active singleton node as 100 m, so model length is 50 m longer than source support; the declared 10% registered-length gate still applies",
        },
        "metrics": {
            "area_difference_m2": area1 - area0,
            "volume_difference_m3": volume1 - volume0,
            "active_surface_integral_difference_m3": surface_difference,
            "active_bed_integral_difference_m3": bed_difference,
            "source_root_length_m": float(root_source.length_m),
            "candidate_root_length_m": float(new_root.length_m),
            "registered_length_m": registered_length,
            "registered_length_relative_error": length_error,
            "free_tail_m": free_tail,
            "source_within_reverse_count": len(reverse0),
            "candidate_within_reverse_count": len(reverse1),
            "source_junction_reverse_count": len(junction0),
            "candidate_junction_reverse_count": len(junction1),
            "maximum_receiver_mapping_error_m": maximum_mapping_error,
            "setter_volume_difference_m3": setter_volume_difference,
            "setter_max_section_difference_m2": setter_section_difference,
        },
        "receiver_mapping": mapping,
        "gates": gates,
        "static_feasibility_pass": feasibility_pass,
        "candidate_model_flowlines_written": false,
        "zero_mb_run": false,
        "steady_run": false,
        "historical_run": false,
        "promotion_authorized": false,
        "next_action": "report-only; controller must separately decide whether this new boundary-control-volume mechanism merits a registered pilot",
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "receiver_mapping.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
