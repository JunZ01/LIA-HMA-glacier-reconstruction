"""Read-only audit of a new subgrid junction receiver-index mechanism.

The mechanism changes neither surface nor bed.  For a junction whose mapped
receiver is one node too high, it moves only the tributary's receiver index to
the nearest downstream node whose surface is not above the tributary terminal.
This is distinct from the closed surface-raising/isotonic regularizers and
from root/main-line reconstruction.  No flowline pickle is written and no
OGGM model is run.
"""

from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_junction_receiver_reindex_class_audit_v1"
)
TOLERANCE_M = 1e-7
MAX_CLASS_SHIFT_NODES = 1

SOURCES = {
    84: "w404geom84v1/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084/inversion_flowlines.pkl",
    1856: "w447geom1856v1/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91856/inversion_flowlines.pkl",
    1931: "w400geom1931v1/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91931/inversion_flowlines.pkl",
    1886: "w401geom1886v1/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91886/inversion_flowlines.pkl",
    2534: "w505geom2534v1/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92534/inversion_flowlines.pkl",
    2559: "w265geom2559v1/synthetic_gdir/RGI60-13/RGI60-13.92/RGI60-13.92559/inversion_flowlines.pkl",
    5770: "w502geom5770v1/synthetic_gdir/RGI60-15/RGI60-15.95/RGI60-15.95770/inversion_flowlines.pkl",
}

REGISTERED_LENGTHS = {
    84: 10177.894615813211,
    1856: 14211.6322914621,
    1931: 17357.28197028982,
    1886: 22156.176121504963,
    2534: 14476.947804517147,
    2559: 10710.50418338994,
    5770: 13630.278886173555,
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    junction_rows = []
    systems = []
    for system_id, relative in SOURCES.items():
        source = PROJECT / relative
        with gzip.open(source, "rb") as stream:
            flowlines = pickle.load(stream)
        by_identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
        roots = [index for index, flowline in enumerate(flowlines) if flowline.flows_to is None]
        within_reverse = sum(
            int(np.count_nonzero(np.diff(np.asarray(flowline.surface_h)) > TOLERANCE_M))
            for flowline in flowlines
        )
        reverse_count = 0
        unresolved_count = 0
        required_shifts = []
        connector_distances = []
        for branch_id, branch in enumerate(flowlines):
            if branch.flows_to is None:
                continue
            receiver = branch.flows_to
            receiver_id = by_identity[id(receiver)]
            old_node = int(branch.flows_to_indice)
            terminal_surface = float(branch.surface_h[-1])
            old_surface = float(receiver.surface_h[old_node])
            gap = old_surface - terminal_surface
            if gap <= TOLERANCE_M:
                continue
            reverse_count += 1
            new_node = None
            for node in range(old_node + 1, int(receiver.nx)):
                if float(receiver.surface_h[node]) <= terminal_surface + TOLERANCE_M:
                    new_node = node
                    break
            if new_node is None:
                unresolved_count += 1
                shift_nodes = None
                shift_m = None
                new_surface = None
                new_gap = None
                connector_distance_m = None
            else:
                shift_nodes = new_node - old_node
                shift_m = shift_nodes * float(receiver.dx_meter)
                new_surface = float(receiver.surface_h[new_node])
                new_gap = new_surface - terminal_surface
                branch_endpoint = np.asarray(branch.line.coords[-1], dtype=float)
                receiver_point = np.asarray(receiver.line.coords[new_node], dtype=float)
                connector_distance_m = float(
                    np.linalg.norm(receiver_point - branch_endpoint) * float(receiver.map_dx)
                )
                required_shifts.append(shift_nodes)
                connector_distances.append(connector_distance_m)
            junction_rows.append(
                {
                    "lia_unique_id": system_id,
                    "branch": branch_id,
                    "receiver": receiver_id,
                    "old_receiver_node": old_node,
                    "new_receiver_node": "" if new_node is None else new_node,
                    "old_reverse_gap_m": gap,
                    "new_reverse_gap_m": "" if new_gap is None else new_gap,
                    "shift_nodes": "" if shift_nodes is None else shift_nodes,
                    "shift_m": "" if shift_m is None else shift_m,
                    "connector_distance_m": ""
                    if connector_distance_m is None
                    else connector_distance_m,
                    "nearest_downstream_solution_exists": new_node is not None,
                }
            )
        root_index = roots[0] if len(roots) == 1 else None
        root_length = (
            float(flowlines[root_index].line.length * flowlines[root_index].map_dx)
            if root_index is not None
            else float("nan")
        )
        registered = REGISTERED_LENGTHS[system_id]
        length_error = abs(root_length - registered) / registered
        max_shift = max(required_shifts, default=0)
        max_connector = max(connector_distances, default=0.0)
        eligible = bool(
            len(roots) == 1
            and within_reverse == 0
            and reverse_count > 0
            and unresolved_count == 0
            and max_shift <= MAX_CLASS_SHIFT_NODES
            and max_connector <= MAX_CLASS_SHIFT_NODES * float(flowlines[0].dx_meter) + 1e-6
            and length_error <= 0.10
        )
        systems.append(
            {
                "lia_unique_id": system_id,
                "source_flowlines": relative,
                "branch_count": len(flowlines),
                "unique_root": len(roots) == 1,
                "root_index": "" if root_index is None else root_index,
                "within_reverse_count": within_reverse,
                "junction_reverse_count": reverse_count,
                "unresolved_junction_count": unresolved_count,
                "maximum_required_downstream_shift_nodes": max_shift,
                "maximum_required_downstream_shift_m": max_shift
                * float(flowlines[0].dx_meter),
                "maximum_connector_distance_m": max_connector,
                "root_length_m": root_length,
                "registered_length_m": registered,
                "root_length_relative_error": length_error,
                "one_node_receiver_reindex_class_eligible": eligible,
            }
        )

    eligible = [row for row in systems if row["one_node_receiver_reindex_class_eligible"]]
    eligible_ordered = sorted(eligible, key=lambda row: float(row["registered_length_m"]))
    selected = []
    if eligible_ordered:
        selected.append({**eligible_ordered[0], "role": "representative"})
    if len(eligible_ordered) >= 2:
        selected.append({**eligible_ordered[-1], "role": "confirmation"})
    _write_csv(OUTPUT / "junction_reindex_details.csv", junction_rows)
    _write_csv(OUTPUT / "system_classification.csv", systems)
    if selected:
        _write_csv(OUTPUT / "selected_representative_confirmation.csv", selected)
    payload = {
        "experiment": "lia-191-junction-receiver-one-node-reindex-class-audit-v1",
        "read_only": True,
        "route_source": "lia_191_current_route_v9_7/remaining_137_current_route.csv",
        "audited_systems": list(SOURCES),
        "mechanism": "subgrid-one-node-downstream-junction-receiver-index-realignment",
        "distinct_from_closed_mechanisms": [
            "does not raise/lower or isotonic-project any surface or bed node",
            "does not rebuild the root/main line or resample its length",
            "does not repair polygon topology",
            "does not change climate, Stage2 physics, or scoring",
            "changes only a tributary's receiver index on its existing parent by at most one native node",
        ],
        "eligible_systems": [int(row["lia_unique_id"]) for row in eligible_ordered],
        "selected": selected,
        "dynamics_run": False,
        "candidate_written": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
