"""Read-only feasibility audit for local/nonuniform and uniform 2x root grids."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = (
    PROJECT
    / "w382inv3660v1/synthetic_gdir/RGI60-14/RGI60-14.93"
    / "RGI60-14.93660/model_flowlines.pkl"
)
FORMAL = PROJECT / "w382inv3660v1/inversion_summary_formal.json"
CONFIG = PROJECT / "w382inv3660v1/config_snapshot.json"
FLUX_BUDGET = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_flux_budget_v2/summary.json"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_root_coarsen2x_feasibility_v1"
)
ROOT_ID = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coarse_arrays(root, surface_mode: str) -> dict[str, np.ndarray]:
    surface = np.asarray(root.surface_h, dtype=float)
    bed = np.asarray(root.bed_h, dtype=float)
    section = np.asarray(root.section, dtype=float)
    width = np.asarray(root.widths_m, dtype=float)
    if len(surface) % 2 != 1:
        raise RuntimeError("root-node-count-not-odd")
    paired = len(surface) - 1
    a = np.arange(0, paired, 2)
    b = a + 1
    coarse_section = 0.5 * (section[a] + section[b])
    coarse_width = 0.5 * (width[a] + width[b])
    if surface_mode == "axial-average":
        coarse_surface = 0.5 * (surface[a] + surface[b])
        coarse_bed = 0.5 * (bed[a] + bed[b])
    elif surface_mode == "width-weighted":
        denom = width[a] + width[b]
        coarse_surface = np.divide(
            width[a] * surface[a] + width[b] * surface[b],
            denom,
            out=0.5 * (surface[a] + surface[b]),
            where=denom > 0,
        )
        coarse_bed = np.divide(
            width[a] * bed[a] + width[b] * bed[b],
            denom,
            out=0.5 * (bed[a] + bed[b]),
            where=denom > 0,
        )
    else:
        raise ValueError(surface_mode)
    # Keep the physical downstream endpoint (fine node 682) explicitly.
    return {
        "surface": np.append(coarse_surface, surface[-1]),
        "bed": np.append(coarse_bed, bed[-1]),
        "section": np.append(coarse_section, section[-1] * 0.5),
        "width": np.append(coarse_width, width[-1] * 0.5),
    }


def _shape_legality(arrays: dict[str, np.ndarray]) -> dict:
    surface = arrays["surface"]
    bed = arrays["bed"]
    section = arrays["section"]
    width = arrays["width"]
    thick = surface - bed
    active = section > 0.0
    lambdas = np.full(len(section), np.nan)
    w0 = np.full(len(section), np.nan)
    lambdas[active] = 2.0 * (width[active] * thick[active] - section[active]) / thick[active] ** 2
    w0[active] = 2.0 * section[active] / thick[active] - width[active]
    bad_lambda = np.flatnonzero(active & (lambdas < -1e-12))
    bad_w0 = np.flatnonzero(active & (w0 <= 0.0))
    return {
        "active_node_count": int(np.count_nonzero(active)),
        "negative_lambda_node_count": int(len(bad_lambda)),
        "nonpositive_w0_node_count": int(len(bad_w0)),
        "negative_lambda_nodes_first20": bad_lambda[:20].tolist(),
        "nonpositive_w0_nodes_first20": bad_w0[:20].tolist(),
        "minimum_lambda": float(np.nanmin(lambdas[active])),
        "minimum_w0_m": float(np.nanmin(w0[active])),
        "all_active_nodes_legal_trapezoids": bool(len(bad_lambda) == 0 and len(bad_w0) == 0),
    }


def _metrics(root, arrays: dict[str, np.ndarray], registered_length: float) -> dict:
    dx0 = float(root.dx_meter)
    dx1 = 2.0 * dx0
    surface0 = np.asarray(root.surface_h, dtype=float)
    bed0 = np.asarray(root.bed_h, dtype=float)
    section0 = np.asarray(root.section, dtype=float)
    width0 = np.asarray(root.widths_m, dtype=float)
    active0 = np.asarray(root.thick, dtype=float) > 0.0
    surface1 = arrays["surface"]
    bed1 = arrays["bed"]
    section1 = arrays["section"]
    width1 = arrays["width"]
    active1 = section1 > 0.0
    length1 = float(np.count_nonzero(active1) * dx1)
    # The final fine node has zero ice. Half its width/section at the coarse
    # endpoint prevents the uniform 100 m control volume from double counting.
    return {
        "fine_node_count": int(len(surface0)),
        "coarse_node_count": int(len(surface1)),
        "fine_dx_m": dx0,
        "coarse_dx_m": dx1,
        "fine_coordinate_span_m": float((len(surface0) - 1) * dx0),
        "coarse_coordinate_span_m": float((len(surface1) - 1) * dx1),
        "fine_active_length_m": float(np.count_nonzero(active0) * dx0),
        "coarse_active_length_m": length1,
        "registered_length_m": registered_length,
        "coarse_registered_length_relative_error": abs(length1 - registered_length) / registered_length,
        "root_area_difference_m2": float(np.sum(width1[active1]) * dx1 - np.sum(width0[active0]) * dx0),
        "root_volume_difference_m3": float(np.sum(section1) * dx1 - np.sum(section0) * dx0),
        "active_axial_surface_integral_difference_m2": float(
            np.sum(surface1[active1]) * dx1 - np.sum(surface0[active0]) * dx0
        ),
        "active_axial_bed_integral_difference_m2": float(
            np.sum(bed1[active1]) * dx1 - np.sum(bed0[active0]) * dx0
        ),
        "active_area_weighted_surface_integral_difference_m3": float(
            np.sum(surface1[active1] * width1[active1]) * dx1
            - np.sum(surface0[active0] * width0[active0]) * dx0
        ),
        "active_area_weighted_bed_integral_difference_m3": float(
            np.sum(bed1[active1] * width1[active1]) * dx1
            - np.sum(bed0[active0] * width0[active0]) * dx0
        ),
        "active_surface_reverse_edge_count": int(
            np.count_nonzero(active1[:-1] & active1[1:] & (np.diff(surface1) > 1e-9))
        ),
        "root_terminal_thickness_m": float(max(surface1[-1] - bed1[-1], 0.0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with gzip.open(SOURCE, "rb") as stream:
        flowlines = pickle.load(stream)
    formal = json.loads(FORMAL.read_text(encoding="utf-8-sig"))
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    flux = json.loads(FLUX_BUDGET.read_text(encoding="utf-8-sig"))
    root = flowlines[ROOT_ID]
    registered_length = float(config["registered_lia_length_m"])

    incoming = sorted(
        [item for item in flux["all_branch_connections"] if int(item["receiver"]) == ROOT_ID],
        key=lambda item: int(item["receiver_node"]),
    )
    mapping = []
    for item in incoming:
        fine_node = int(item["receiver_node"])
        coarse_node = int(round(fine_node / 2.0))
        mapping.append({
            "branch": int(item["branch"]),
            "fine_receiver_node": fine_node,
            "fine_distance_m": fine_node * float(root.dx_meter),
            "coarse_receiver_node": coarse_node,
            "coarse_distance_m": coarse_node * 2.0 * float(root.dx_meter),
            "mapping_error_m": coarse_node * 2.0 * float(root.dx_meter) - fine_node * float(root.dx_meter),
        })

    alternatives = {}
    for mode in ("axial-average", "width-weighted"):
        arrays = _coarse_arrays(root, mode)
        alternatives[mode] = {
            "shape_legality": _shape_legality(arrays),
            "metrics": _metrics(root, arrays, registered_length),
        }

    local_nonuniform = {
        "supported_by_current_oggm_flowline": False,
        "evidence": [
            "Flowline stores one scalar dx_meter = map_dx * dx for the entire line.",
            "FluxBasedModel sets dx = fl.dx_meter once per flowline and uses it for every slope and continuity update.",
            "Area, volume and model length likewise multiply every node by the same scalar dx_meter.",
        ],
        "consequence": (
            "Merging only bins244-250 while leaving the rest at 50 m cannot be represented "
            "by OGGM MixedBedFlowline/FluxBasedModel without changing the solver/data model."
        ),
    }
    exact_mapping = all(abs(row["mapping_error_m"]) <= 1e-12 for row in mapping)
    any_uniform_pass = any(
        item["shape_legality"]["all_active_nodes_legal_trapezoids"]
        and item["metrics"]["coarse_registered_length_relative_error"] <= 0.10
        and abs(item["metrics"]["root_area_difference_m2"]) <= 1e-6
        and abs(item["metrics"]["root_volume_difference_m3"]) <= 1e-6
        and item["metrics"]["active_surface_reverse_edge_count"] == 0
        for item in alternatives.values()
    )
    payload = {
        "experiment": "zero-mb-3660-root-control-volume-coarsen2x-feasibility-v1",
        "system_id": 3660,
        "read_only": True,
        "source_model_flowlines": str(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "root_flowline": ROOT_ID,
        "local_nonuniform_grid": local_nonuniform,
        "uniform_2x_root": {
            "fine_nodes": int(root.nx),
            "fine_dx_m": float(root.dx_meter),
            "coarse_nodes": int((root.nx - 1) / 2 + 1),
            "coarse_dx_m": float(2 * root.dx_meter),
            "branch_receiver_mapping": mapping,
            "exact_branch_receiver_mapping_preserved": exact_mapping,
            "maximum_absolute_receiver_mapping_error_m": max(abs(row["mapping_error_m"]) for row in mapping),
            "odd_receiver_count": int(sum(int(row["fine_receiver_node"]) % 2 for row in mapping)),
            "even_receiver_count": int(sum(int(row["fine_receiver_node"]) % 2 == 0 for row in mapping)),
            "alternatives": alternatives,
        },
        "formal_reference": {
            "registered_lia_length_m": registered_length,
            "model_root_length_m": formal["model_root_length_m"],
            "root_coordinate_span_m": formal["root_spatial_length_m"],
            "model_free_tail_m": formal["model_free_tail_m"],
        },
        "uniform_static_candidate_exists_under_all_declared_constraints": bool(any_uniform_pass and exact_mapping),
        "pilot_launched": False,
        "next_action": (
            "build-one-uniform-2x-static-candidate"
            if any_uniform_pass and exact_mapping
            else "close-before-dynamics"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
