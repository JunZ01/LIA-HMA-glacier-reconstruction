"""Read-only feasibility audit for terminal control-volume condensation on 334.

The mechanism removes the shortest terminal suffix needed to make every
tributary enter its unchanged receiver downhill.  The removed catchment area
is assigned to the last retained tributary control volume.  No surface value,
receiver index, source file, climate value, or physical parameter is changed.

This is a mathematical/static preflight only.  It does not materialize a new
GlacierDirectory and does not run inversion or dynamics.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_input_geometry_plus1_pilot_v2/representative_334/"
    "topology_repair/system_334/geometry/synthetic_gdir/RGI60-15/"
    "RGI60-15.90/RGI60-15.90334/inversion_flowlines.pkl"
)
GEOMETRY_SUMMARY = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_input_geometry_plus1_pilot_v2/representative_334/"
    "topology_repair/system_334/geometry/geometry_summary.json"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_invalid_outline334_terminal_control_volume_condensation_v1"
)

TOLERANCE_M = 1.0e-7
MAXIMUM_CUT_DISTANCE_M = 250.0
MAXIMUM_CDF_ERROR = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within_reverse_count(surface: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(surface) > TOLERANCE_M))


def cumulative_hypsometry_error_100m(
    source_surface: np.ndarray,
    source_area: np.ndarray,
    candidate_surface: np.ndarray,
    candidate_area: np.ndarray,
) -> float:
    lower = 100.0 * np.floor(
        min(float(source_surface.min()), float(candidate_surface.min())) / 100.0
    )
    upper = 100.0 * np.ceil(
        max(float(source_surface.max()), float(candidate_surface.max())) / 100.0
    ) + 100.0
    bins = np.arange(lower, upper + 1.0, 100.0)
    source_histogram = np.histogram(
        source_surface, bins=bins, weights=source_area
    )[0]
    candidate_histogram = np.histogram(
        candidate_surface, bins=bins, weights=candidate_area
    )[0]
    source_cdf = np.cumsum(source_histogram) / float(np.sum(source_area))
    candidate_cdf = np.cumsum(candidate_histogram) / float(np.sum(candidate_area))
    return float(np.max(np.abs(source_cdf - candidate_cdf)))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_hash_before = sha256(SOURCE)
    geometry_hash_before = sha256(GEOMETRY_SUMMARY)
    with gzip.open(SOURCE, "rb") as stream:
        flowlines = pickle.load(stream)

    identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    source_surface_parts: list[np.ndarray] = []
    source_area_parts: list[np.ndarray] = []
    candidate_surface_parts: list[np.ndarray] = []
    candidate_area_parts: list[np.ndarray] = []
    rows: list[dict] = []
    source_within = 0
    candidate_within = 0
    source_junction = 0
    candidate_junction = 0

    for branch, flowline in enumerate(flowlines):
        surface = np.asarray(flowline.surface_h, dtype=np.float64)
        widths_m = np.asarray(flowline.widths, dtype=np.float64) * float(
            flowline.map_dx
        )
        dx_m = float(flowline.dx * flowline.map_dx)
        node_area = widths_m * dx_m
        source_surface_parts.append(surface.copy())
        source_area_parts.append(node_area.copy())
        source_within += within_reverse_count(surface)

        keep_last = surface.size - 1
        receiver_branch = None
        receiver_index = None
        receiver_surface_m = None
        source_gap_m = None
        source_reverse = False
        receiver = getattr(flowline, "flows_to", None)
        if receiver is not None:
            receiver_branch = int(identity[id(receiver)])
            receiver_index = int(flowline.flows_to_indice)
            receiver_surface_m = float(receiver.surface_h[receiver_index])
            source_gap_m = receiver_surface_m - float(surface[-1])
            source_reverse = bool(source_gap_m > TOLERANCE_M)
            source_junction += int(source_reverse)
            if source_reverse:
                feasible = np.where(surface >= receiver_surface_m - TOLERANCE_M)[0]
                if feasible.size == 0:
                    keep_last = -1
                else:
                    keep_last = int(feasible[-1])

        if keep_last < 1:
            candidate_surface = surface[:0]
            candidate_area = node_area[:0]
            candidate_gap_m = None
            candidate_reverse = True
        else:
            candidate_surface = surface[: keep_last + 1].copy()
            candidate_area = node_area[: keep_last + 1].copy()
            candidate_area[-1] += float(np.sum(node_area[keep_last + 1 :]))
            candidate_within += within_reverse_count(candidate_surface)
            candidate_gap_m = (
                None
                if receiver_surface_m is None
                else receiver_surface_m - float(candidate_surface[-1])
            )
            candidate_reverse = bool(
                candidate_gap_m is not None and candidate_gap_m > TOLERANCE_M
            )
        candidate_junction += int(candidate_reverse)
        candidate_surface_parts.append(candidate_surface)
        candidate_area_parts.append(candidate_area)

        removed_node_count = int(surface.size - 1 - keep_last)
        removed_distance_m = float(removed_node_count * dx_m)
        rows.append(
            {
                "branch_index": branch,
                "receiver_branch_index": receiver_branch,
                "receiver_node_index": receiver_index,
                "source_node_count": int(surface.size),
                "candidate_node_count": int(candidate_surface.size),
                "removed_node_count": removed_node_count,
                "removed_distance_m": removed_distance_m,
                "source_terminal_surface_m": float(surface[-1]),
                "candidate_terminal_surface_m": (
                    None if candidate_surface.size == 0 else float(candidate_surface[-1])
                ),
                "receiver_surface_m": receiver_surface_m,
                "source_junction_gap_m": source_gap_m,
                "candidate_junction_gap_m": candidate_gap_m,
                "source_junction_reverse": source_reverse,
                "candidate_junction_reverse": candidate_reverse,
                "source_branch_area_m2": float(np.sum(node_area)),
                "candidate_branch_area_m2": float(np.sum(candidate_area)),
            }
        )

    source_surface = np.concatenate(source_surface_parts)
    source_area = np.concatenate(source_area_parts)
    candidate_surface = np.concatenate(candidate_surface_parts)
    candidate_area = np.concatenate(candidate_area_parts)
    source_area_total = float(np.sum(source_area))
    candidate_area_total = float(np.sum(candidate_area))
    area_relative_error = (candidate_area_total - source_area_total) / source_area_total
    cdf_error = cumulative_hypsometry_error_100m(
        source_surface, source_area, candidate_surface, candidate_area
    )
    cut_rows = [row for row in rows if row["removed_node_count"] > 0]
    maximum_cut_distance = max(
        (float(row["removed_distance_m"]) for row in cut_rows), default=0.0
    )
    source_hash_after = sha256(SOURCE)
    geometry_hash_after = sha256(GEOMETRY_SUMMARY)
    checks = {
        "source_hash_unchanged": source_hash_before == source_hash_after,
        "geometry_summary_hash_unchanged": geometry_hash_before == geometry_hash_after,
        "source_within_branch_reverse_zero": source_within == 0,
        "candidate_within_branch_reverse_zero": candidate_within == 0,
        "candidate_junction_reverse_zero": candidate_junction == 0,
        "area_machine_conservative": abs(area_relative_error) <= 1.0e-12,
        "hypsometry_cdf_within_2pct": cdf_error <= MAXIMUM_CDF_ERROR,
        "maximum_cut_distance_within_250m": (
            maximum_cut_distance <= MAXIMUM_CUT_DISTANCE_M
        ),
        "all_branches_retain_at_least_two_nodes": all(
            int(row["candidate_node_count"]) >= 2 for row in rows
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema": "lia-191-invalid-outline334-terminal-control-volume-condensation-v1",
        "status": (
            "static-mathematical-feasibility-pass-materialization-not-yet-authorized"
            if passed
            else "static-mathematical-feasibility-fail-mechanism-closed"
        ),
        "system_id": 334,
        "mechanism": "minimal-tributary-terminal-control-volume-condensation-v1",
        "flowline_count": len(flowlines),
        "source_junction_reverse_count": source_junction,
        "candidate_junction_reverse_count": candidate_junction,
        "source_within_branch_reverse_count": source_within,
        "candidate_within_branch_reverse_count": candidate_within,
        "modified_branch_count": len(cut_rows),
        "total_removed_node_count": int(
            sum(int(row["removed_node_count"]) for row in cut_rows)
        ),
        "maximum_removed_node_count_per_branch": max(
            (int(row["removed_node_count"]) for row in cut_rows), default=0
        ),
        "maximum_cut_distance_m": maximum_cut_distance,
        "maximum_cut_distance_limit_m": MAXIMUM_CUT_DISTANCE_M,
        "source_area_m2": source_area_total,
        "candidate_area_m2": candidate_area_total,
        "area_relative_error": area_relative_error,
        "cumulative_hypsometry_100m_max_error_fraction": cdf_error,
        "cumulative_hypsometry_100m_max_error_limit": MAXIMUM_CDF_ERROR,
        "checks": checks,
        "source_inversion_flowlines": str(SOURCE),
        "source_inversion_flowlines_sha256": source_hash_before,
        "source_geometry_summary": str(GEOMETRY_SUMMARY),
        "source_geometry_summary_sha256": geometry_hash_before,
        "source_modified": False,
        "candidate_materialized": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_gate": (
            "materialize-independent-gdir-and-rerun-unchanged-geometry-static"
            if passed
            else "none-mechanism-closed"
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "branch_condensation_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
