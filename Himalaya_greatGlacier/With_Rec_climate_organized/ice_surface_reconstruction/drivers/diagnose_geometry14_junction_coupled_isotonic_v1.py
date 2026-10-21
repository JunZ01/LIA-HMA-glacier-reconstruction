"""Read-only minimum weighted-isotonic diagnostic for geometry system 14."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from .pilot_junction_coupled_isotonic_5443_v1 import _project_nonincreasing


PROJECT = Path(__file__).resolve().parents[4]
GEOMETRY_ROOT = PROJECT / "wcr8invpoly14_2692v1/system_14/geometry"
OUTPUT = (
    PROJECT / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_geometry14_junction_coupled_isotonic_diagnostic_v1"
)
TOLERANCE = 1e-7


def _load():
    paths = list((GEOMETRY_ROOT / "synthetic_gdir").rglob("model_flowlines.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one model_flowlines.pkl, got {len(paths)}")
    with gzip.open(paths[0], "rb") as stream:
        return pickle.load(stream), paths[0]


def _metrics(flowlines) -> dict:
    roots = [flowline for flowline in flowlines if flowline.flows_to is None]
    return {
        "volume_m3": float(sum(np.sum(np.asarray(fl.section, float)) * float(fl.dx_meter) for fl in flowlines)),
        "area_m2": float(sum(np.sum(np.asarray(fl.widths_m, float)[np.asarray(fl.thick, float) > 0]) * float(fl.dx_meter) for fl in flowlines)),
        "root_length_m": float(np.count_nonzero(np.asarray(roots[0].thick, float) > 0) * float(roots[0].dx_meter)),
        "negative_section_count": int(sum(np.count_nonzero(np.asarray(fl.section, float) < -1e-10) for fl in flowlines)),
        "finite": bool(all(np.isfinite(np.asarray(fl.surface_h, float)).all() and np.isfinite(np.asarray(fl.section, float)).all() for fl in flowlines)),
    }


def _audit(flowlines) -> dict:
    within = []
    junction = []
    for flowline_id, flowline in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(flowline.thick, float) > 0)
        if not len(active):
            continue
        last = int(active[-1])
        surface = np.asarray(flowline.surface_h, float)
        for node in np.flatnonzero(np.diff(surface[: last + 1]) > TOLERANCE):
            within.append({"flowline": flowline_id, "node": int(node), "reverse_step_m": float(surface[node + 1] - surface[node])})
        if flowline.flows_to is not None:
            receiver_id = next(i for i, candidate in enumerate(flowlines) if candidate is flowline.flows_to)
            receiver_node = int(flowline.flows_to_indice)
            reverse = float(flowline.flows_to.surface_h[receiver_node] - surface[last])
            if reverse > TOLERANCE:
                junction.append({
                    "flowline": flowline_id, "terminal_node": last,
                    "receiver_flowline": receiver_id, "receiver_node": receiver_node,
                    "reverse_step_m": reverse,
                })
    return {"within_count": len(within), "junction_count": len(junction), "within": within, "junction": junction}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original, source = _load()
    before = _metrics(original)
    audit_before = _audit(original)
    if audit_before["within_count"] != 0 or audit_before["junction_count"] != 4:
        raise RuntimeError(f"unexpected source audit: {audit_before}")
    repaired = copy.deepcopy(original)
    paths = []
    touched = set()
    maximum_adjustment = 0.0
    all_rectangular = True
    for reversal in audit_before["junction"]:
        source_id = int(reversal["flowline"])
        terminal = int(reversal["terminal_node"])
        receiver_id = int(reversal["receiver_flowline"])
        receiver_node = int(reversal["receiver_node"])
        targets = [(source_id, terminal - 1), (source_id, terminal), (receiver_id, receiver_node)]
        overlap = sorted(touched.intersection(targets))
        touched.update(targets)
        old_surface = np.asarray([repaired[fid].surface_h[node] for fid, node in targets], float)
        widths = np.asarray([repaired[fid].widths_m[node] for fid, node in targets], float)
        dx = np.asarray([repaired[fid].dx_meter for fid, node in targets], float)
        rectangular = [bool(np.asarray(repaired[fid].is_rectangular)[node]) for fid, node in targets]
        all_rectangular = all_rectangular and all(rectangular)
        projected = _project_nonincreasing(old_surface, widths * dx)
        for (fid, node), surface in zip(targets, projected):
            flowline = repaired[fid]
            section = np.asarray(flowline.section, float).copy()
            thickness = float(surface - flowline.bed_h[node])
            if thickness <= 0:
                raise RuntimeError(f"nonpositive projected thickness at fl{fid}/n{node}")
            if not bool(np.asarray(flowline.is_rectangular)[node]):
                continue
            section[node] = float(flowline.widths_m[node]) * thickness
            flowline.section = section
        actual = np.asarray([repaired[fid].surface_h[node] for fid, node in targets], float)
        maximum_adjustment = max(maximum_adjustment, float(np.max(np.abs(actual - old_surface))))
        paths.append({
            "source_reversal": reversal, "targets": [{"flowline": fid, "node": node} for fid, node in targets],
            "target_overlap_with_prior_paths": [{"flowline": fid, "node": node} for fid, node in overlap],
            "rectangular": rectangular, "old_surface_m": old_surface.tolist(),
            "projected_surface_m": projected.tolist(), "actual_surface_m": actual.tolist(),
            "maximum_absolute_adjustment_m": float(np.max(np.abs(projected - old_surface))),
        })
    after = _metrics(repaired)
    audit_after = _audit(repaired)
    static = {
        "volume_relative_difference": float((after["volume_m3"] - before["volume_m3"]) / before["volume_m3"]),
        "area_relative_difference": float((after["area_m2"] - before["area_m2"]) / before["area_m2"]),
        "root_length_difference_m": float(after["root_length_m"] - before["root_length_m"]),
    }
    read_only_gate_pass = bool(
        all_rectangular and not any(path["target_overlap_with_prior_paths"] for path in paths)
        and audit_after["within_count"] == 0 and audit_after["junction_count"] == 0
        and maximum_adjustment <= 1.0
        and abs(static["volume_relative_difference"]) <= 1e-10
        and abs(static["area_relative_difference"]) <= 1e-12
        and abs(static["root_length_difference_m"]) <= 1e-9
        and after["negative_section_count"] == 0 and after["finite"]
    )
    payload = {
        "experiment": "geometry14-junction-coupled-weighted-isotonic-diagnostic-v1",
        "system_id": 14, "read_only": True, "source_model_flowlines": str(source),
        "source_geometry_summary": str(GEOMETRY_ROOT / "geometry_summary.json"),
        "before": before, "audit_before": audit_before, "paths": paths,
        "all_targets_rectangular": all_rectangular,
        "maximum_absolute_surface_adjustment_m": maximum_adjustment,
        "after_in_memory": after, "audit_after_in_memory": audit_after,
        "static_in_memory": static, "read_only_gate_pass": read_only_gate_pass,
        "next_action": "static-reproduction-then-30s-5y-zeroMB" if read_only_gate_pass else "close-system14-mechanism-without-zeroMB",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
