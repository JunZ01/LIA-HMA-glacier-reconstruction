"""Strict read-only identity preflight for inversion-pass system 3660."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
ROOT = PROJECT / "w382inv3660v1"
OUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_identity_3660_preflight_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    formal = json.loads((ROOT / "inversion_summary_formal.json").read_text(encoding="utf-8-sig"))
    paths = list((ROOT / "synthetic_gdir").rglob("model_flowlines.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one model_flowlines.pkl, got {len(paths)}")
    source = paths[0]
    with gzip.open(source, "rb") as stream:
        flowlines = pickle.load(stream)
    roots = [flowline for flowline in flowlines if flowline.flows_to is None]
    volume = float(sum(np.sum(np.asarray(fl.section, float)) * float(fl.dx_meter) for fl in flowlines))
    area = float(sum(np.sum(np.asarray(fl.widths_m, float)[np.asarray(fl.thick, float) > 0]) * float(fl.dx_meter) for fl in flowlines))
    length = float(np.count_nonzero(np.asarray(roots[0].thick, float) > 0) * float(roots[0].dx_meter))
    active_within = []
    active_junction = []
    for flowline_id, flowline in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(flowline.thick, float) > 0)
        if not len(active):
            continue
        last = int(active[-1])
        surface = np.asarray(flowline.surface_h, float)
        for node in np.flatnonzero(np.diff(surface[: last + 1]) > 1e-7):
            active_within.append({"flowline": flowline_id, "node": int(node), "reverse_step_m": float(surface[node + 1] - surface[node])})
        if flowline.flows_to is not None and last == len(surface) - 1:
            receiver_id = next(i for i, candidate in enumerate(flowlines) if candidate is flowline.flows_to)
            receiver_node = int(flowline.flows_to_indice)
            reverse = float(flowline.flows_to.surface_h[receiver_node] - surface[last])
            if reverse > 1e-7:
                active_junction.append({"flowline": flowline_id, "receiver_flowline": receiver_id, "receiver_node": receiver_node, "reverse_step_m": reverse})
    metrics = {
        "volume_m3": volume, "area_m2": area, "root_length_m": length,
        "volume_difference_from_formal_m3": volume - float(formal["model_volume_m3"]),
        "area_difference_from_formal_m2": area - float(formal["model_area_m2"]),
        "length_difference_from_formal_m": length - float(formal["model_root_length_m"]),
        "registered_length_relative_error": abs(length - 27508.72902948928) / 27508.72902948928,
        "negative_section_count": int(sum(np.count_nonzero(np.asarray(fl.section, float) < -1e-10) for fl in flowlines)),
        "finite": bool(all(np.isfinite(np.asarray(fl.section, float)).all() and np.isfinite(np.asarray(fl.surface_h, float)).all() for fl in flowlines)),
        "root_terminal_thickness_m": float(roots[0].thick[-1]),
    }
    static_pass = bool(
        formal["status"] == "inversion-pass" and len(roots) == 1
        and abs(metrics["volume_difference_from_formal_m3"]) <= 1e-6
        and abs(metrics["area_difference_from_formal_m2"]) <= 1e-6
        and abs(metrics["length_difference_from_formal_m"]) <= 1e-9
        and metrics["registered_length_relative_error"] <= 0.1
        and not active_within and not active_junction
        and metrics["negative_section_count"] == 0 and metrics["finite"]
        and metrics["root_terminal_thickness_m"] <= 10.0
    )
    payload = {
        "experiment": "zero-mb-identity-static-preflight-3660-v1",
        "system_id": 3660, "mechanism_class": "inversion-pass-zero-mb-unrun-identity",
        "read_only": True, "geometry_mutation": None, "source_model_flowlines": str(source),
        "source_sha256": _sha(source), "metrics": metrics,
        "active_within_reverse_edges": active_within,
        "active_junction_reverse_edges": active_junction,
        "static_pass": static_pass,
        "next_action": "run-one-30s-5y-zeroMB" if static_pass else "close-identity-representative",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
