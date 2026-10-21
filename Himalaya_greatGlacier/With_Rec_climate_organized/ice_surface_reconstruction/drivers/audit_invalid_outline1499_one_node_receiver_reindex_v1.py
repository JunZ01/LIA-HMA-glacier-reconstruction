"""Read-only one-node receiver-reindex subclass audit for repaired system 1499."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
GEOMETRY = PROJECT / "wcr10invpoly1499v2/system_1499/geometry/geometry_summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline1499_one_node_receiver_reindex_v1"
TOLERANCE_M = 1e-7
MAX_SHIFT_NODES = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if geometry.get("junction_reverse_slope_count") != 2 or geometry.get("within_branch_reverse_slope_count") != 0:
        raise RuntimeError("system1499 geometry failure signature changed")
    flowline_path = Path(geometry["synthetic_gdir"]) / "inversion_flowlines.pkl"
    before = sha256(flowline_path)
    with gzip.open(flowline_path, "rb") as stream:
        flowlines = pickle.load(stream)
    by_identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    rows = []
    for branch_index, branch in enumerate(flowlines):
        if branch.flows_to is None:
            continue
        parent = branch.flows_to
        parent_index = by_identity[id(parent)]
        old_node = int(branch.flows_to_indice)
        terminal_surface = float(branch.surface_h[-1])
        old_receiver_surface = float(parent.surface_h[old_node])
        gap = old_receiver_surface - terminal_surface
        if gap <= TOLERANCE_M:
            continue
        new_node = None
        for candidate in range(old_node + 1, min(int(parent.nx), old_node + MAX_SHIFT_NODES + 1)):
            if float(parent.surface_h[candidate]) <= terminal_surface + TOLERANCE_M:
                new_node = candidate
                break
        if new_node is None:
            all_downstream = np.flatnonzero(np.asarray(parent.surface_h[old_node + 1:]) <= terminal_surface + TOLERANCE_M)
            nearest_any = int(old_node + 1 + all_downstream[0]) if all_downstream.size else None
        else:
            nearest_any = new_node
        selected = new_node if new_node is not None else nearest_any
        if selected is None:
            shift_nodes = None
            connector_distance = None
            final_gap = None
        else:
            shift_nodes = int(selected - old_node)
            terminal_xy = np.asarray(branch.line.coords[-1], dtype=float) * float(branch.map_dx)
            receiver_xy = np.asarray(parent.line.coords[selected], dtype=float) * float(parent.map_dx)
            connector_distance = float(np.linalg.norm(receiver_xy - terminal_xy))
            final_gap = float(parent.surface_h[selected] - terminal_surface)
        rows.append({
            "branch_index": branch_index,
            "parent_branch_index": parent_index,
            "old_receiver_node": old_node,
            "nearest_resolving_receiver_node": selected,
            "required_shift_nodes": shift_nodes,
            "required_shift_m": None if shift_nodes is None else shift_nodes * float(parent.dx_meter),
            "connector_distance_m": connector_distance,
            "original_reverse_gap_m": gap,
            "final_gap_m": final_gap,
            "one_node_solution": bool(
                new_node is not None
                and shift_nodes == 1
                and connector_distance <= float(parent.dx_meter) + 1e-6
                and final_gap <= TOLERANCE_M
            ),
        })
    after = sha256(flowline_path)
    passed = bool(len(rows) == 2 and all(row["one_node_solution"] for row in rows) and before == after)
    summary = {
        "schema": "lia-191-invalid-outline1499-one-node-receiver-reindex-v1",
        "status": "one-node-subclass-pass-materialization-not-run" if passed else "one-node-subclass-fail",
        "system_id": 1499,
        "branch_count": len(flowlines),
        "reverse_junction_count": len(rows),
        "resolved_with_one_node_count": sum(int(row["one_node_solution"]) for row in rows),
        "maximum_required_shift_nodes": max((row["required_shift_nodes"] or 10**9 for row in rows), default=0),
        "maximum_required_connector_distance_m": max((row["connector_distance_m"] or float("inf") for row in rows), default=0.0),
        "source_immutable": before == after,
        "receiver_indices_modified": False,
        "geometry_run": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": (
            "materialize only system1499 receiver indices and rerun unchanged static gates"
            if passed else
            "close one-node receiver-reindex subclass for system1499"
        ),
        "provenance": {
            "geometry_summary": str(GEOMETRY.relative_to(PROJECT)),
            "geometry_summary_sha256": sha256(GEOMETRY),
            "inversion_flowlines": str(flowline_path.relative_to(PROJECT)),
            "inversion_flowlines_sha256": before,
        },
    }
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "junctions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
