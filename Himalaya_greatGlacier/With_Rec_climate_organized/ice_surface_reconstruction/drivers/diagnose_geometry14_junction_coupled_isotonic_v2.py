"""Independent v2 diagnostic on geometry-stage inversion flowlines for system 14."""

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
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry14_junction_coupled_isotonic_diagnostic_v2"
TOLERANCE = 1e-7


def _load():
    paths = list((GEOMETRY_ROOT / "synthetic_gdir").rglob("inversion_flowlines.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one inversion_flowlines.pkl, got {len(paths)}")
    with gzip.open(paths[0], "rb") as stream:
        return pickle.load(stream), paths[0]


def _audit(flowlines) -> dict:
    within = []
    junction = []
    for flowline_id, flowline in enumerate(flowlines):
        surface = np.asarray(flowline.surface_h, float)
        for node in np.flatnonzero(np.diff(surface) > TOLERANCE):
            within.append({"flowline": flowline_id, "node": int(node), "reverse_step_m": float(surface[node + 1] - surface[node])})
        if flowline.flows_to is not None:
            receiver_id = next(i for i, candidate in enumerate(flowlines) if candidate is flowline.flows_to)
            receiver_node = int(flowline.flows_to_indice)
            reverse = float(flowline.flows_to.surface_h[receiver_node] - surface[-1])
            if reverse > TOLERANCE:
                junction.append({
                    "flowline": flowline_id, "terminal_node": len(surface) - 1,
                    "receiver_flowline": receiver_id, "receiver_node": receiver_node,
                    "reverse_step_m": reverse,
                })
    return {"within_count": len(within), "junction_count": len(junction), "within": within, "junction": junction}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original, source = _load()
    audit_before = _audit(original)
    if audit_before["within_count"] != 0 or audit_before["junction_count"] != 4:
        raise RuntimeError(f"unexpected source audit: {audit_before}")
    repaired = copy.deepcopy(original)
    paths = []
    touched = set()
    max_adjustment = 0.0
    all_rectangular = True
    for reversal in audit_before["junction"]:
        source_id = int(reversal["flowline"])
        terminal = int(reversal["terminal_node"])
        receiver_id = int(reversal["receiver_flowline"])
        receiver_node = int(reversal["receiver_node"])
        targets = [(source_id, terminal - 1), (source_id, terminal), (receiver_id, receiver_node)]
        overlap = sorted(touched.intersection(targets))
        touched.update(targets)
        old = np.asarray([repaired[fid].surface_h[node] for fid, node in targets], float)
        weights = np.asarray([repaired[fid].widths_m[node] * repaired[fid].dx_meter for fid, node in targets], float)
        rectangular = [bool(np.asarray(repaired[fid].is_rectangular)[node]) for fid, node in targets]
        all_rectangular = all_rectangular and all(rectangular)
        projected = _project_nonincreasing(old, weights)
        for (fid, node), value in zip(targets, projected):
            surface = np.asarray(repaired[fid].surface_h, float).copy()
            surface[node] = value
            repaired[fid]._surface_h = surface
        actual = np.asarray([repaired[fid].surface_h[node] for fid, node in targets], float)
        adjustment = float(np.max(np.abs(actual - old)))
        max_adjustment = max(max_adjustment, adjustment)
        paths.append({
            "source_reversal": reversal,
            "targets": [{"flowline": fid, "node": node} for fid, node in targets],
            "target_overlap_with_prior_paths": [{"flowline": fid, "node": node} for fid, node in overlap],
            "rectangular": rectangular, "weights_m2": weights.tolist(),
            "old_surface_m": old.tolist(), "projected_surface_m": projected.tolist(),
            "maximum_absolute_adjustment_m": adjustment,
        })
    audit_after = _audit(repaired)
    model_geometry_available = bool(all(hasattr(flowline, "section") and hasattr(flowline, "thick") for flowline in original))
    root = [flowline for flowline in original if flowline.flows_to is None][0]
    root_spatial_length_before_m = float((len(root.surface_h) - 1) * root.dx_meter)
    root_after = [flowline for flowline in repaired if flowline.flows_to is None][0]
    root_spatial_length_after_m = float((len(root_after.surface_h) - 1) * root_after.dx_meter)
    topology_surface_gate = bool(
        audit_after["within_count"] == 0 and audit_after["junction_count"] == 0
        and max_adjustment <= 1.0 and all_rectangular
        and not any(path["target_overlap_with_prior_paths"] for path in paths)
        and abs(root_spatial_length_after_m - root_spatial_length_before_m) <= 1e-9
    )
    strict_release_gate = bool(topology_surface_gate and model_geometry_available)
    payload = {
        "experiment": "geometry14-junction-coupled-weighted-isotonic-diagnostic-v2",
        "system_id": 14, "read_only": True,
        "source_geometry_summary": str(GEOMETRY_ROOT / "geometry_summary.json"),
        "source_flowlines": str(source), "source_flowline_type": "geometry-stage inversion_flowlines",
        "model_geometry_available": model_geometry_available,
        "volume_area_conservation_testable": model_geometry_available,
        "audit_before": audit_before, "paths": paths,
        "all_targets_rectangular": all_rectangular,
        "maximum_absolute_surface_adjustment_m": max_adjustment,
        "audit_after_in_memory": audit_after,
        "root_spatial_length_before_m": root_spatial_length_before_m,
        "root_spatial_length_after_m": root_spatial_length_after_m,
        "topology_surface_gate_pass": topology_surface_gate,
        "strict_release_gate_pass": strict_release_gate,
        "failure_reason": None if strict_release_gate else "geometry-stage flowlines have no section/thickness, so strict area/volume conservation cannot be demonstrated",
        "next_action": "static-reproduction-then-30s-5y-zeroMB" if strict_release_gate else "close-system14-mechanism-without-inversion-or-zeroMB",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
