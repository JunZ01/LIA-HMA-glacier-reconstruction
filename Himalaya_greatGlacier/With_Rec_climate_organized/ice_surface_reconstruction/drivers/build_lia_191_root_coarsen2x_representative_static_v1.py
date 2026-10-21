"""Build one preregistered conservative 2x-root representative.

This is a static geometry operation only.  It never instantiates an OGGM
dynamical model.  The exact conditioned construction is shared with the
system-3660 v5 preflight, while all source identities and gates come from a
versioned JSON contract.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from .build_zero_mb_3660_root_coarsen2x_v5 import _conditioned_build_root


PROJECT = Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT / path


def _active_reverse_edges(flowlines) -> list[dict]:
    rows = []
    for flowline_id, flowline in enumerate(flowlines):
        surface = np.asarray(flowline.surface_h, dtype=float)
        active = np.asarray(flowline.thick, dtype=float) > 0.0
        for node in np.flatnonzero(active[:-1] & active[1:] & (np.diff(surface) > 1e-9)):
            rows.append(
                {
                    "flowline": int(flowline_id),
                    "node": int(node),
                    "surface_step_m": float(surface[node + 1] - surface[node]),
                }
            )
    return rows


def _active_junction_reverse_edges(flowlines) -> list[dict]:
    rows = []
    for branch_id, branch in enumerate(flowlines):
        if branch.flows_to is None:
            continue
        receiver_id = next(
            index for index, candidate in enumerate(flowlines) if candidate is branch.flows_to
        )
        active = np.flatnonzero(np.asarray(branch.thick, dtype=float) > 0.0)
        if not len(active):
            continue
        terminal = int(active[-1])
        receiver_node = int(branch.flows_to_indice)
        step = float(branch.flows_to.surface_h[receiver_node] - branch.surface_h[terminal])
        if step > 1e-9:
            rows.append(
                {
                    "branch": int(branch_id),
                    "receiver": int(receiver_id),
                    "receiver_node": receiver_node,
                    "surface_step_m": step,
                }
            )
    return rows


def _integral(flowline, field: str) -> float:
    active = np.asarray(flowline.thick, dtype=float) > 0.0
    values = np.asarray(getattr(flowline, field), dtype=float)
    widths = np.asarray(flowline.widths_m, dtype=float)
    return float(np.sum(values[active] * widths[active]) * float(flowline.dx_meter))


def _reconnect_direct_branches(candidate, old_root, new_root) -> list[dict]:
    mapping = []
    for branch_id, branch in enumerate(candidate):
        if branch.flows_to is not old_root:
            continue
        old_node = int(branch.flows_to_indice)
        new_node = int(math.ceil(old_node / 2.0))
        if not 0 <= new_node < int(new_root.nx):
            raise RuntimeError(f"receiver-outside-coarse-root:{branch_id}:{new_node}")
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
                "old_distance_m": old_node * float(old_root.dx_meter),
                "new_distance_m": new_node * float(new_root.dx_meter),
                "distance_error_m": new_node * float(new_root.dx_meter)
                - old_node * float(old_root.dx_meter),
            }
        )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    source = _resolve(config["source_model_flowlines"])
    inversion_summary = _resolve(config["source_inversion_summary"])
    output = _resolve(config["static_output_root"])
    if output.exists():
        raise FileExistsError(output)
    source_hash = _sha256(source)
    inversion_hash = _sha256(inversion_summary)
    if source_hash.lower() != config["source_model_flowlines_sha256"].lower():
        raise RuntimeError("source-model-flowlines-sha256-mismatch")
    if inversion_hash.lower() != config["source_inversion_summary_sha256"].lower():
        raise RuntimeError("source-inversion-summary-sha256-mismatch")

    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(source, "rb") as stream:
        original = pickle.load(stream)
    roots = [index for index, flowline in enumerate(original) if flowline.flows_to is None]
    if roots != [int(config["root_index"])]:
        raise RuntimeError(f"root-identity-mismatch:{roots}")
    root_id = roots[0]
    root_source = original[root_id]
    if abs(float(root_source.dx_meter) - float(config["root_dx_before_m"])) > 1e-9:
        raise RuntimeError("root-dx-mismatch")
    if int(root_source.nx) != int(config["root_nodes_before"]):
        raise RuntimeError("root-node-count-mismatch")
    if int(root_source.nx) % 2 != 1:
        raise RuntimeError("registered-mechanism-requires-odd-root-node-count")

    candidate = copy.deepcopy(original)
    old_root = candidate[root_id]
    new_root, shape = _conditioned_build_root(old_root)
    mapping = _reconnect_direct_branches(candidate, old_root, new_root)
    candidate[root_id] = new_root

    area0 = float(sum(flowline.area_m2 for flowline in original))
    area1 = float(sum(flowline.area_m2 for flowline in candidate))
    volume0 = float(sum(flowline.volume_m3 for flowline in original))
    volume1 = float(sum(flowline.volume_m3 for flowline in candidate))
    reverse0 = _active_reverse_edges(original)
    reverse1 = _active_reverse_edges(candidate)
    junction0 = _active_junction_reverse_edges(original)
    junction1 = _active_junction_reverse_edges(candidate)
    active1 = np.asarray(new_root.thick, dtype=float) > 0.0
    registered_length = float(config["registered_lia_length_m"])
    length_error = abs(float(new_root.length_m) - registered_length) / registered_length
    free_tail = float(
        (int(new_root.nx) - int(np.count_nonzero(active1))) * float(new_root.dx_meter)
    )
    span0 = float((int(root_source.nx) - 1) * float(root_source.dx_meter))
    span1 = float((int(new_root.nx) - 1) * float(new_root.dx_meter))
    maximum_mapping_error = max(
        [abs(float(row["distance_error_m"])) for row in mapping] or [0.0]
    )
    test = copy.deepcopy(new_root)
    section0 = np.asarray(new_root.section, dtype=float).copy()
    test.section = section0.copy()
    setter_volume_difference = float(test.volume_m3 - new_root.volume_m3)
    setter_section_difference = float(
        np.max(np.abs(np.asarray(test.section, dtype=float) - section0))
    )
    thresholds = config["static_gates"]
    gates = {
        "unique_root": sum(flowline.flows_to is None for flowline in candidate) == 1,
        "topology_valid": all(
            flowline.flows_to is None or flowline.flows_to in candidate
            for flowline in candidate
        ),
        "source_active_within_reverse_absent": len(reverse0) == 0,
        "source_active_junction_reverse_absent": len(junction0) == 0,
        "candidate_active_within_reverse_absent": len(reverse1) == 0,
        "candidate_active_junction_reverse_absent": len(junction1) == 0,
        "root_coordinate_span_unchanged": abs(span1 - span0) <= 1e-8,
        "root_length_formal_10pct": length_error
        <= float(thresholds["root_length_relative_error_max"]),
        "total_area_conserved": abs(area1 - area0) <= max(1e-6, area0 * 1e-12),
        "total_volume_conserved": abs(volume1 - volume0)
        <= max(1e-6, volume0 * 1e-12),
        "active_surface_integral_conserved": abs(
            _integral(new_root, "surface_h") - _integral(root_source, "surface_h")
        )
        <= max(1e-5, abs(_integral(root_source, "surface_h")) * 1e-12),
        "active_bed_integral_conserved": abs(
            _integral(new_root, "bed_h") - _integral(root_source, "bed_h")
        )
        <= max(1e-5, abs(_integral(root_source, "bed_h")) * 1e-12),
        "root_terminal_zero": float(new_root.thick[-1]) == 0.0,
        "free_tail_at_least_5km": free_tail
        >= float(thresholds["minimum_free_downstream_m"]),
        "receiver_mapping_within_half_coarse_cell": maximum_mapping_error
        <= 0.5 * float(new_root.dx_meter),
        "root_nodes_enough": int(new_root.nx) >= 3
        and int(np.count_nonzero(active1)) >= 2,
        "mixedbed_setter_roundtrip": abs(setter_volume_difference) <= 1e-5
        and setter_section_difference <= 1e-6,
    }
    static_pass = all(gates.values())
    candidate_path = output / "candidate_model_flowlines.pkl"
    if static_pass:
        with gzip.open(candidate_path, "wb") as stream:
            pickle.dump(candidate, stream, protocol=pickle.HIGHEST_PROTOCOL)
    candidate_hash = _sha256(candidate_path) if candidate_path.exists() else None
    if candidate_path.exists():
        with gzip.open(candidate_path, "rb") as stream:
            reloaded = pickle.load(stream)
        reloaded_area = float(sum(flowline.area_m2 for flowline in reloaded))
        reloaded_volume = float(sum(flowline.volume_m3 for flowline in reloaded))
        reload_pass = (
            abs(reloaded_area - area1) <= max(1e-6, area1 * 1e-12)
            and abs(reloaded_volume - volume1) <= max(1e-6, volume1 * 1e-12)
            and _sha256(candidate_path) == candidate_hash
        )
    else:
        reloaded_area = None
        reloaded_volume = None
        reload_pass = False

    summary = {
        "experiment": config["experiment"],
        "stage": "static-only",
        "system_id": int(config["system_id"]),
        "mechanism": config["mechanism"],
        "source_model_flowlines": str(source),
        "source_model_flowlines_sha256": source_hash,
        "source_inversion_summary": str(inversion_summary),
        "source_inversion_summary_sha256": inversion_hash,
        "root_flowline": root_id,
        "root_nodes_before": int(root_source.nx),
        "root_nodes_after": int(new_root.nx),
        "root_dx_before_m": float(root_source.dx_meter),
        "root_dx_after_m": float(new_root.dx_meter),
        "shape": shape,
        "receiver_mapping": mapping,
        "metrics": {
            "area_difference_m2": area1 - area0,
            "volume_difference_m3": volume1 - volume0,
            "active_surface_integral_difference_m3": _integral(new_root, "surface_h")
            - _integral(root_source, "surface_h"),
            "active_bed_integral_difference_m3": _integral(new_root, "bed_h")
            - _integral(root_source, "bed_h"),
            "root_coordinate_span_before_m": span0,
            "root_coordinate_span_after_m": span1,
            "root_length_before_m": float(root_source.length_m),
            "root_length_after_m": float(new_root.length_m),
            "registered_length_m": registered_length,
            "registered_length_relative_error": length_error,
            "free_tail_m": free_tail,
            "within_reverse_before": len(reverse0),
            "within_reverse_after": len(reverse1),
            "junction_reverse_before": len(junction0),
            "junction_reverse_after": len(junction1),
            "maximum_absolute_receiver_mapping_error_m": maximum_mapping_error,
        },
        "gates": gates,
        "static_pass": static_pass,
        "candidate_model_flowlines": str(candidate_path) if candidate_path.exists() else None,
        "candidate_sha256": candidate_hash,
        "hash_verification_pass": reload_pass,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "hold-for-3660-formal-historical-pass-and-independent-reproduction",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verification = {
        "system_id": int(config["system_id"]),
        "source_hash_expected": config["source_model_flowlines_sha256"],
        "source_hash_observed": source_hash,
        "source_hash_match": source_hash.lower()
        == config["source_model_flowlines_sha256"].lower(),
        "inversion_hash_expected": config["source_inversion_summary_sha256"],
        "inversion_hash_observed": inversion_hash,
        "inversion_hash_match": inversion_hash.lower()
        == config["source_inversion_summary_sha256"].lower(),
        "candidate_hash": candidate_hash,
        "candidate_hash_recomputed": _sha256(candidate_path)
        if candidate_path.exists()
        else None,
        "candidate_reload_area_m2": reloaded_area,
        "candidate_reload_volume_m3": reloaded_volume,
        "candidate_reload_pass": reload_pass,
    }
    (output / "hash_verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
