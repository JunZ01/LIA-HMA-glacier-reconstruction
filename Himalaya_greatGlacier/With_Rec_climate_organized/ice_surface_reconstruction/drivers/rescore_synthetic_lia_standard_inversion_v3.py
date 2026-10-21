"""Final active-domain rescore for the saved system-3850 inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run(result_root: Path) -> dict:
    from oggm import cfg, utils

    original = json.loads(
        (result_root / "inversion_summary.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (result_root / "config_snapshot.json").read_text(encoding="utf-8")
    )
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=result_root / "synthetic_gdir",
        reset=False,
    )
    flowlines = gdir.read_pickle("model_flowlines")
    identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    tolerance = float(config["gates"]["reverse_slope_tolerance_m"])
    active_reverse = []
    active_junction_reverse = []
    root_indices = []
    for branch, flowline in enumerate(flowlines):
        thick = np.asarray(flowline.thick, dtype=float)
        surface = np.asarray(flowline.surface_h, dtype=float)
        active = thick > 0
        for node, rise in enumerate(np.diff(surface)):
            if active[node] and active[node + 1] and rise > tolerance:
                active_reverse.append(
                    {"branch": branch, "node": node, "rise_m": float(rise)}
                )
        if flowline.flows_to is None:
            root_indices.append(branch)
            continue
        active_nodes = np.flatnonzero(active)
        if not active_nodes.size or active_nodes[-1] != flowline.nx - 1:
            continue
        receiver = flowline.flows_to
        receiver_node = int(flowline.flows_to_indice)
        rise = float(receiver.surface_h[receiver_node] - surface[-1])
        if rise > tolerance:
            active_junction_reverse.append(
                {
                    "branch": branch,
                    "receiver_branch": identity[id(receiver)],
                    "receiver_node": receiver_node,
                    "rise_m": rise,
                }
            )
    if len(root_indices) != 1:
        raise RuntimeError(f"root-count:{len(root_indices)}")
    root_flowline = flowlines[root_indices[0]]
    root_thick = np.asarray(root_flowline.thick, dtype=float)
    root_active_count = int(np.count_nonzero(root_thick > 0))
    root_terminal_thickness = float(root_thick[-1])
    root_free_tail_m = float(
        (root_flowline.nx - root_active_count) * root_flowline.dx_meter
    )
    negative_flux = []
    for branch, flowline in enumerate(gdir.read_pickle("inversion_flowlines")):
        flux = np.asarray(flowline.flux, dtype=float)
        nodes = np.flatnonzero(flux < 0)
        if nodes.size:
            negative_flux.append(
                {
                    "branch": branch,
                    "negative_node_count": int(nodes.size),
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
    if original["model_outline_area_relative_error"] > float(
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
            "adjacent surface slope is checked only where both nodes contain ice; "
            "junction slope is checked only where ice reaches the junction; "
            "the separated zero-ice safety tail belongs to the unique root"
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
    output = result_root / "inversion_summary_formal.json"
    output.write_text(
        json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return formal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.root.resolve()),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
