"""Materialize the gated terminal-control-volume condensation for system 334.

The source synthetic GlacierDirectory is copied byte-for-byte to an independent
versioned root.  Only the copied ``inversion_flowlines`` object is changed:
for each reverse tributary, its shortest conflicting terminal suffix is removed
and its width-area is added to the last retained control volume.  Receiver
indices, surfaces, climate, DEM, outline, and all source files remain frozen.

This stage reruns static geometry checks only.  Inversion is a later gate.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from shapely.geometry import LineString


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
SOURCE_SYSTEM = (
    VALIDATION
    / "lia_191_input_geometry_plus1_pilot_v2/representative_334/"
    "topology_repair/system_334/geometry"
)
SOURCE_GDIR_BASE = SOURCE_SYSTEM / "synthetic_gdir"
SOURCE_GDIR = (
    SOURCE_GDIR_BASE / "RGI60-15/RGI60-15.90/RGI60-15.90334"
)
SOURCE_FLOWLINES = SOURCE_GDIR / "inversion_flowlines.pkl"
SOURCE_SUMMARY = SOURCE_SYSTEM / "geometry_summary.json"
PREFLIGHT = (
    VALIDATION
    / "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "summary.json"
)
OUTPUT = (
    VALIDATION
    / "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "materialized_geometry_v1"
)
TARGET_GDIR_BASE = OUTPUT / "synthetic_gdir"
TARGET_GDIR = TARGET_GDIR_BASE / "RGI60-15/RGI60-15.90/RGI60-15.90334"
TARGET_FLOWLINES = TARGET_GDIR / "inversion_flowlines.pkl"

TOLERANCE_M = 1.0e-7
MAXIMUM_CUT_DISTANCE_M = 250.0
MAXIMUM_CONNECTOR_DISTANCE_M = 300.0
MAXIMUM_CDF_ERROR = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def reverse_counts(flowlines) -> tuple[int, int]:
    within = 0
    junction = 0
    for flowline in flowlines:
        surface = np.asarray(flowline.surface_h, dtype=np.float64)
        within += int(np.count_nonzero(np.diff(surface) > TOLERANCE_M))
        receiver = getattr(flowline, "flows_to", None)
        if receiver is not None:
            receiver_surface = float(receiver.surface_h[int(flowline.flows_to_indice)])
            junction += int(receiver_surface > float(surface[-1]) + TOLERANCE_M)
    return within, junction


def hypsometry(flowlines) -> tuple[np.ndarray, np.ndarray]:
    elevations = []
    areas = []
    for flowline in flowlines:
        elevations.append(np.asarray(flowline.surface_h, dtype=np.float64))
        areas.append(
            np.asarray(flowline.widths, dtype=np.float64)
            * float(flowline.map_dx)
            * float(flowline.dx_meter)
        )
    return np.concatenate(elevations), np.concatenate(areas)


def cdf_error_100m(
    source_elevation: np.ndarray,
    source_area: np.ndarray,
    candidate_elevation: np.ndarray,
    candidate_area: np.ndarray,
) -> float:
    low = 100.0 * np.floor(
        min(float(source_elevation.min()), float(candidate_elevation.min())) / 100.0
    )
    high = 100.0 * np.ceil(
        max(float(source_elevation.max()), float(candidate_elevation.max())) / 100.0
    ) + 100.0
    bins = np.arange(low, high + 1.0, 100.0)
    source_histogram = np.histogram(
        source_elevation, bins=bins, weights=source_area
    )[0]
    candidate_histogram = np.histogram(
        candidate_elevation, bins=bins, weights=candidate_area
    )[0]
    return float(
        np.max(
            np.abs(
                np.cumsum(source_histogram) / float(np.sum(source_area))
                - np.cumsum(candidate_histogram) / float(np.sum(candidate_area))
            )
        )
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != (
        "static-mathematical-feasibility-pass-materialization-not-yet-authorized"
    ):
        raise RuntimeError("registered-preflight-not-pass")
    source_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if source_summary.get("status") != "geometry-fail":
        raise RuntimeError("unexpected-source-geometry-status")
    if source_summary.get("review_flags") != ["junction-reverse-slope"]:
        raise RuntimeError("source-has-failures-beyond-junction-reverse")

    source_tree_before = tree_hash(SOURCE_GDIR_BASE)
    source_flowline_hash = sha256(SOURCE_FLOWLINES)
    with gzip.open(SOURCE_FLOWLINES, "rb") as stream:
        source_flowlines = pickle.load(stream)
    candidate_flowlines = copy.deepcopy(source_flowlines)
    identity = {id(flowline): index for index, flowline in enumerate(candidate_flowlines)}
    source_elevation, source_area = hypsometry(source_flowlines)
    rows = []

    for branch, flowline in enumerate(candidate_flowlines):
        surface = np.asarray(flowline.surface_h, dtype=np.float64)
        receiver = getattr(flowline, "flows_to", None)
        if receiver is None:
            continue
        receiver_index = int(flowline.flows_to_indice)
        receiver_surface = float(receiver.surface_h[receiver_index])
        if receiver_surface <= float(surface[-1]) + TOLERANCE_M:
            continue
        feasible = np.where(surface >= receiver_surface - TOLERANCE_M)[0]
        if feasible.size == 0:
            raise RuntimeError(f"branch-{branch}-has-no-feasible-prefix")
        keep_last = int(feasible[-1])
        if keep_last < 1:
            raise RuntimeError(f"branch-{branch}-would-retain-fewer-than-two-nodes")
        removed_count = int(flowline.nx - keep_last - 1)
        cut_distance_m = float(removed_count * flowline.dx_meter)
        if cut_distance_m > MAXIMUM_CUT_DISTANCE_M:
            raise RuntimeError(f"branch-{branch}-cut-distance-over-gate")
        original_widths = np.asarray(flowline.widths, dtype=np.float64).copy()
        new_widths = original_widths[: keep_last + 1].copy()
        new_widths[-1] += float(np.sum(original_widths[keep_last + 1 :]))
        old_terminal = flowline.tail
        old_node_count = int(flowline.nx)
        new_line = LineString(list(flowline.line.coords)[: keep_last + 1])
        flowline.set_line(new_line)
        flowline.surface_h = surface[: keep_last + 1].copy()
        flowline.widths = new_widths
        if flowline.is_rectangular is not None:
            flowline.is_rectangular = np.asarray(flowline.is_rectangular)[
                : keep_last + 1
            ].copy()
        if flowline.is_trapezoid is not None:
            flowline.is_trapezoid = np.asarray(flowline.is_trapezoid)[
                : keep_last + 1
            ].copy()
        if flowline.geometrical_widths is not None:
            flowline.geometrical_widths = list(flowline.geometrical_widths)[
                : keep_last + 1
            ]
        flowline.reset_flux()
        for cache_name in ("_lazy_normals",):
            flowline.__dict__.pop(cache_name, None)
        connector_distance_m = float(
            flowline.tail.distance(flowline.flows_to_point) * flowline.map_dx
        )
        if connector_distance_m > MAXIMUM_CONNECTOR_DISTANCE_M:
            raise RuntimeError(f"branch-{branch}-connector-distance-over-gate")
        rows.append(
            {
                "branch_index": branch,
                "receiver_branch_index": int(identity[id(receiver)]),
                "receiver_node_index": receiver_index,
                "source_node_count": old_node_count,
                "candidate_node_count": int(flowline.nx),
                "removed_node_count": removed_count,
                "cut_distance_m": cut_distance_m,
                "connector_distance_m": connector_distance_m,
                "old_tail_to_receiver_distance_m": float(
                    old_terminal.distance(flowline.flows_to_point) * flowline.map_dx
                ),
                "source_terminal_surface_m": float(surface[-1]),
                "candidate_terminal_surface_m": float(flowline.surface_h[-1]),
                "receiver_surface_m": receiver_surface,
                "candidate_downhill_drop_m": float(
                    flowline.surface_h[-1] - receiver_surface
                ),
            }
        )

    source_within, source_junction = reverse_counts(source_flowlines)
    candidate_within, candidate_junction = reverse_counts(candidate_flowlines)
    candidate_elevation, candidate_area = hypsometry(candidate_flowlines)
    source_area_total = float(np.sum(source_area))
    candidate_area_total = float(np.sum(candidate_area))
    area_relative_error = (candidate_area_total - source_area_total) / source_area_total
    cdf_error = cdf_error_100m(
        source_elevation, source_area, candidate_elevation, candidate_area
    )
    maximum_connector = max(float(row["connector_distance_m"]) for row in rows)
    checks = {
        "source_within_reverse_zero": source_within == 0,
        "source_junction_reverse_six": source_junction == 6,
        "candidate_within_reverse_zero": candidate_within == 0,
        "candidate_junction_reverse_zero": candidate_junction == 0,
        "modified_branch_count_six": len(rows) == 6,
        "system_area_machine_conservative": abs(area_relative_error) <= 1.0e-12,
        "hypsometry_cdf_within_2pct": cdf_error <= MAXIMUM_CDF_ERROR,
        "maximum_connector_within_300m": (
            maximum_connector <= MAXIMUM_CONNECTOR_DISTANCE_M
        ),
        "root_length_gate_inherited": (
            float(source_summary["root_length_relative_error"]) <= 0.10
        ),
        "outline_area_gate_inherited": (
            float(source_summary["outline_area_relative_error"]) <= 0.10
        ),
        "free_domain_gate_inherited": (
            float(source_summary["downstream_free_length_m"]) >= 5000.0
        ),
        "topology_gate_inherited": bool(source_summary["topology_valid"]),
    }
    passed = all(checks.values())

    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SOURCE_GDIR_BASE, TARGET_GDIR_BASE)
    with gzip.open(TARGET_FLOWLINES, "wb") as stream:
        pickle.dump(candidate_flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    target_hash = sha256(TARGET_FLOWLINES)
    source_tree_after = tree_hash(SOURCE_GDIR_BASE)
    checks["source_gdir_tree_unchanged"] = source_tree_before == source_tree_after
    passed = passed and checks["source_gdir_tree_unchanged"]

    summary = {
        **source_summary,
        "status": "geometry-pass" if passed else "geometry-fail",
        "experiment": "invalid-outline334-terminal-control-volume-condensation-v1",
        "synthetic_gdir": str(TARGET_GDIR),
        "source_geometry_summary": str(SOURCE_SUMMARY),
        "source_geometry_summary_sha256": sha256(SOURCE_SUMMARY),
        "source_inversion_flowlines_sha256": source_flowline_hash,
        "candidate_inversion_flowlines_sha256": target_hash,
        "source_within_branch_reverse_slope_count": source_within,
        "source_junction_reverse_slope_count": source_junction,
        "within_branch_reverse_slope_count": candidate_within,
        "junction_reverse_slope_count": candidate_junction,
        "modified_branch_count": len(rows),
        "total_removed_node_count": int(
            sum(int(row["removed_node_count"]) for row in rows)
        ),
        "maximum_cut_distance_m": max(
            float(row["cut_distance_m"]) for row in rows
        ),
        "maximum_connector_distance_m": maximum_connector,
        "source_flowline_control_area_m2": source_area_total,
        "candidate_flowline_control_area_m2": candidate_area_total,
        "flowline_control_area_relative_error": area_relative_error,
        "cumulative_hypsometry_100m_max_error_fraction": cdf_error,
        "review_flags": [] if passed else [
            key for key, value in checks.items() if not value
        ],
        "checks": checks,
        "source_products_unchanged": checks["source_gdir_tree_unchanged"],
        "candidate_materialized": True,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
    }
    with (OUTPUT / "branch_condensation_materialization.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
