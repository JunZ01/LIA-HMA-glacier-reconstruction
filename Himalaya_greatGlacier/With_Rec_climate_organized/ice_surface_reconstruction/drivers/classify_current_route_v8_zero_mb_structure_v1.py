from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
ROUTE = VALIDATION / "lia_191_current_route_v8/remaining_138_current_route.csv"
OUT = VALIDATION / "lia_191_current_route_v8_zero_mb_structure_classification_v1"
CFL = re.compile(
    r"required time step smaller than the minimum allowed: ([0-9.]+)s vs ([0-9.]+)s.*?"
    r"simulation year ([0-9.]+), fl_id ([0-9]+), bin_id ([0-9]+) and max_u ([0-9.]+)",
    re.DOTALL,
)


def _local(values: np.ndarray, node: int) -> dict:
    start = max(0, node - 2)
    stop = min(len(values), node + 3)
    return {"start_node": start, "values": [float(x) for x in values[start:stop]]}


def _ratio(a: float, b: float) -> float | None:
    if abs(b) < 1e-15:
        return None
    return float(a / b)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    with ROUTE.open("r", encoding="utf-8-sig", newline="") as stream:
        route_rows = list(csv.DictReader(stream))
    selected = [
        row for row in route_rows
        if row["current_route"] == "zero-mb-blocked" and row["latest_evidence_stage"] == "zero_mb"
    ]
    records = []
    for route in selected:
        summary_path = PROJECT / route["latest_evidence"].replace("\\", "/")
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        match = CFL.search(summary.get("runtime_error") or "")
        if match is None:
            records.append({"system_id": int(route["lia_unique_id"]), "class": "non-cfl-zero-mb-failure"})
            continue
        required_dt, registered_dt, year, fail_fl, fail_node, max_u = match.groups()
        zero_root = summary_path.parent
        config = json.loads((zero_root / "config_snapshot.json").read_text(encoding="utf-8-sig"))
        inversion_root = PROJECT / config["inversion_root"]["sub"]
        model_path = next((inversion_root / "synthetic_gdir").rglob("model_flowlines.pkl"))
        with model_path.open("rb") as stream:
            flowlines = pickle.load(stream)
        fail_fl = int(fail_fl)
        fail_node = int(fail_node)
        flowline = flowlines[fail_fl]
        active = np.flatnonzero(np.asarray(flowline.thick, float) > 0)
        active_last = int(active[-1]) if len(active) else -1
        incoming = []
        for source_id, source in enumerate(flowlines):
            if source.flows_to is flowline:
                incoming.append({"source_flowline": source_id, "receiver_node": int(source.flows_to_indice)})
        nearest_incoming = min((abs(item["receiver_node"] - fail_node) for item in incoming), default=None)
        if flowline.flows_to is None:
            structural_class = "root-confluence-near" if nearest_incoming is not None and nearest_incoming <= 3 else "root-internal"
        else:
            distance = active_last - fail_node
            structural_class = "tributary-terminal" if distance <= 3 else "tributary-internal"
        surface = np.asarray(flowline.surface_h, float)
        bed = np.asarray(flowline.bed_h, float)
        thick = np.asarray(flowline.thick, float)
        width = np.asarray(flowline.widths_m, float)
        section = np.asarray(flowline.section, float)
        left = max(fail_node - 1, 0)
        right = min(fail_node + 1, len(surface) - 1)
        records.append(
            {
                "system_id": int(route["lia_unique_id"]),
                "zero_root": zero_root.name,
                "inversion_root": inversion_root.name,
                "required_dt_seconds": float(required_dt),
                "registered_dt_seconds": float(registered_dt),
                "failure_year": float(year),
                "max_u_m_per_year": float(max_u),
                "flowline": fail_fl,
                "node": fail_node,
                "flowline_role": "root" if flowline.flows_to is None else "tributary",
                "active_last_node": active_last,
                "distance_to_active_terminus_nodes": active_last - fail_node,
                "nearest_incoming_junction_distance_nodes": nearest_incoming,
                "incoming_junctions": incoming,
                "class": structural_class,
                "surface_reverse_count": int(np.count_nonzero(np.diff(surface[: active_last + 1]) > 1e-8)),
                "surface_local": _local(surface, fail_node),
                "bed_local": _local(bed, fail_node),
                "thickness_local": _local(thick, fail_node),
                "width_local": _local(width, fail_node),
                "section_local": _local(section, fail_node),
                "left_to_node_width_ratio": _ratio(float(width[fail_node]), float(width[left])),
                "node_to_right_width_ratio": _ratio(float(width[right]), float(width[fail_node])),
                "left_to_node_thickness_ratio": _ratio(float(thick[fail_node]), float(thick[left])),
                "node_to_right_thickness_ratio": _ratio(float(thick[right]), float(thick[fail_node])),
                "left_surface_step_m": float(surface[fail_node] - surface[left]),
                "right_surface_step_m": float(surface[right] - surface[fail_node]),
                "left_bed_step_m": float(bed[fail_node] - bed[left]),
                "right_bed_step_m": float(bed[right] - bed[fail_node]),
            }
        )
    columns = [
        "system_id", "zero_root", "inversion_root", "required_dt_seconds", "registered_dt_seconds",
        "failure_year", "max_u_m_per_year", "flowline", "node", "flowline_role", "active_last_node",
        "distance_to_active_terminus_nodes", "nearest_incoming_junction_distance_nodes", "class",
        "surface_reverse_count", "left_to_node_width_ratio", "node_to_right_width_ratio",
        "left_to_node_thickness_ratio", "node_to_right_thickness_ratio", "left_surface_step_m",
        "right_surface_step_m", "left_bed_step_m", "right_bed_step_m",
    ]
    with (OUT / "classification.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    counts = {}
    for record in records:
        counts[record["class"]] = counts.get(record["class"], 0) + 1
    payload = {
        "experiment": "current-route-v8-zero-mb-structure-classification-v1",
        "current_route_zero_mb_count": sum(row["current_route"] == "zero-mb-blocked" for row in route_rows),
        "cfl_failure_count": len(records),
        "class_counts": counts,
        "excluded_closed_mechanisms": [
            "terminal-connector", "root-internal-linear-smoothing", "four-node-junction-isotonic"
        ],
        "records": records,
    }
    (OUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"class_counts": counts, "records": records}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
