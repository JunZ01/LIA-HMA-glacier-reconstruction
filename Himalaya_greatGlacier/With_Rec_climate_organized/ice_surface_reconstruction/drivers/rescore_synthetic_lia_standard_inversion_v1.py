"""Rescore a saved synthetic-system inversion with active-ice gate semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def run(root: Path) -> dict:
    from oggm import cfg, utils

    original = json.loads((root / "inversion_summary.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=root / "synthetic_gdir",
        reset=False,
    )
    flowlines = gdir.read_pickle("model_flowlines")
    index_by_identity = {id(fl): i for i, fl in enumerate(flowlines)}
    tolerance = float(config["gates"]["reverse_slope_tolerance_m"])
    active_reverse = []
    active_junction_reverse = []
    root_indices = []
    for branch_index, flowline in enumerate(flowlines):
        thick = np.asarray(flowline.thick, dtype=float)
        surface = np.asarray(flowline.surface_h, dtype=float)
        active = thick > 0
        for node, rise in enumerate(np.diff(surface)):
            if active[node] and active[node + 1] and rise > tolerance:
                active_reverse.append(
                    {"branch": branch_index, "node": node, "rise_m": float(rise)}
                )
        if flowline.flows_to is None:
            root_indices.append(branch_index)
            continue
        active_nodes = np.flatnonzero(active)
        if not active_nodes.size or active_nodes[-1] != flowline.nx - 1:
            continue
        receiver = flowline.flows_to
        receiver_index = int(flowline._flows_to_indice)
        rise = float(receiver.surface_h[receiver_index] - surface[-1])
        if rise > tolerance:
            active_junction_reverse.append(
                {
                    "branch": branch_index,
                    "receiver_branch": index_by_identity[id(receiver)],
                    "receiver_node": receiver_index,
                    "rise_m": rise,
                }
            )
    if len(root_indices) != 1:
        raise RuntimeError(f"root-count:{len(root_indices)}")
    root = flowlines[root_indices[0]]
    root_thick = np.asarray(root.thick, dtype=float)
    root_active = np.flatnonzero(root_thick > 0)
    root_terminal_thickness = float(root_thick[-1])
    root_free_tail_m = float((root.nx - root_active.size) * root.dx_meter)

    inversion_flowlines = gdir.read_pickle("inversion_flowlines")
    negative_flux = []
    for branch_index, flowline in enumerate(inversion_flowlines):
        flux = np.asarray(flowline.flux, dtype=float)
        locations = np.flatnonzero(flux < 0)
        if locations.size:
            negative_flux.append(
                {
                    "branch": branch_index,
                    "negative_node_count": int(locations.size),
                    "minimum_flux_native_units": float(np.min(flux)),
                }
            )

    flags = []
    if active_reverse:
        flags.append("active-within-branch-reverse-slope")
    if active_junction_reverse:
        flags.append("active-junction-reverse-slope")
    if root_terminal_thickness > 0:
        flags.append("root-safety-tail-has-ice")
    if root_free_tail_m < float(config["gates"]["minimum_free_downstream_m"]):
        flags.append("root-free-tail-too-short")
    if original.get("model_outline_area_relative_error", 1) > float(
        config["gates"]["model_area_relative_error_max"]
    ):
        flags.append("model-outline-area-mismatch")
    if abs(
        float(original["model_root_length_m"])
        - float(config["registered_lia_length_m"])
    ) / float(config["registered_lia_length_m"]) > float(
        config["gates"]["root_length_relative_error_max"]
    ):
        flags.append("model-root-length-mismatch")

    formal = {
        **original,
        "status": "inversion-pass" if not flags else "inversion-static-fail",
        "review_flags": flags,
        "active_gate_definition": (
            "surface slopes are evaluated only when both adjacent nodes carry ice; "
            "junction slope is evaluated only when the tributary reaches its junction; "
            "zero-ice safety-tail gate applies to the unique root"
        ),
        "active_within_branch_reverse_slope_count": len(active_reverse),
        "active_junction_reverse_slope_count": len(active_junction_reverse),
        "active_reverse_edges": active_reverse,
        "active_junction_reverse_edges": active_junction_reverse,
        "root_terminal_thickness_m": root_terminal_thickness,
        "model_free_tail_m": root_free_tail_m,
        "negative_apparent_flux_branches": negative_flux,
        "negative_apparent_flux_is_diagnostic_not_gate": True,
    }
    _write_json(root / "inversion_summary_formal.json", formal)
    return formal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
