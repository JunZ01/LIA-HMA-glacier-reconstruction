"""Build a conservative post-inversion zero-ice connector state for 334.

The source standard inversion is left unchanged.  Each tributary receives a
zero-thickness connector tail ending at its unchanged receiver node.  Where the
source model still enters uphill, the shortest active terminal suffix is
condensed into the last downhill-compatible active node.  That node is encoded
as a rectangular control volume with exactly the condensed active area and
volume.  This stage is static only; no dynamics are run here.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
SOURCE_ROOT = PROJECT / "w334tccinvv2"
SOURCE_SUMMARY = SOURCE_ROOT / "inversion_summary.json"
SOURCE_FLOWLINES = (
    SOURCE_ROOT
    / "synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90334/model_flowlines.pkl"
)
OUTPUT = (
    VALIDATION
    / "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "postinversion_zero_ice_connector_v1"
)
CANDIDATE = OUTPUT / "candidate_model_flowlines.pkl.gz"

TOLERANCE_M = 1.0e-7
MAXIMUM_CDF_ERROR = 0.02
MAXIMUM_EXTRA_ACTIVE_CUT_M = 250.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology_counts(flowlines) -> tuple[int, int, int]:
    within = 0
    junction = 0
    terminal_positive = 0
    for flowline in flowlines:
        surface = np.asarray(flowline.surface_h, dtype=np.float64)
        within += int(np.count_nonzero(np.diff(surface) > TOLERANCE_M))
        terminal_positive += int(float(flowline.thick[-1]) > TOLERANCE_M)
        receiver = getattr(flowline, "flows_to", None)
        if receiver is not None:
            receiver_surface = float(receiver.surface_h[int(flowline.flows_to_indice)])
            junction += int(receiver_surface > float(surface[-1]) + TOLERANCE_M)
    return within, junction, terminal_positive


def active_metrics(flowlines) -> tuple[float, float, np.ndarray, np.ndarray]:
    area = 0.0
    volume = 0.0
    elevations = []
    weights = []
    for flowline in flowlines:
        thick = np.asarray(flowline.thick, dtype=np.float64)
        active = thick > TOLERANCE_M
        widths = np.asarray(flowline.widths_m, dtype=np.float64)
        section = np.asarray(flowline.section, dtype=np.float64)
        dx = float(flowline.dx_meter)
        area += float(np.sum(widths[active]) * dx)
        volume += float(np.sum(section) * dx)
        if np.any(active):
            elevations.append(np.asarray(flowline.surface_h)[active])
            weights.append(widths[active] * dx)
    return area, volume, np.concatenate(elevations), np.concatenate(weights)


def cdf_error_100m(z0, w0, z1, w1) -> float:
    low = 100.0 * np.floor(min(float(z0.min()), float(z1.min())) / 100.0)
    high = 100.0 * np.ceil(max(float(z0.max()), float(z1.max())) / 100.0) + 100.0
    bins = np.arange(low, high + 1.0, 100.0)
    h0 = np.histogram(z0, bins=bins, weights=w0)[0]
    h1 = np.histogram(z1, bins=bins, weights=w1)[0]
    return float(np.max(np.abs(np.cumsum(h0) / np.sum(w0) - np.cumsum(h1) / np.sum(w1))))


def rebuild_flowline(source, receiver_surface: float | None, receiver_point: Point | None):
    from oggm.core.flowline import MixedBedFlowline

    surface = np.asarray(source.surface_h, dtype=np.float64)
    bed = np.asarray(source.bed_h, dtype=np.float64)
    section = np.asarray(source.section, dtype=np.float64)
    widths = np.asarray(source.widths_m, dtype=np.float64)
    bed_shape = np.asarray(source.bed_shape, dtype=np.float64)
    is_trapezoid = np.asarray(source.is_trapezoid, dtype=bool)
    lambdas = np.asarray(source._lambdas, dtype=np.float64)
    dx_m = float(source.dx_meter)
    keep_last = source.nx - 1
    source_reverse = False
    if receiver_surface is not None:
        source_reverse = bool(receiver_surface > float(surface[-1]) + TOLERANCE_M)
        if source_reverse:
            feasible = np.where(surface >= receiver_surface - TOLERANCE_M)[0]
            if feasible.size == 0:
                raise RuntimeError("no-downhill-compatible-active-prefix")
            keep_last = int(feasible[-1])
    removed_active = np.flatnonzero(
        (np.arange(source.nx) >= keep_last) & (np.asarray(source.thick) > TOLERANCE_M)
    )
    extra_cut_m = float((source.nx - 1 - keep_last) * dx_m)
    if extra_cut_m > MAXIMUM_EXTRA_ACTIVE_CUT_M:
        raise RuntimeError("postinversion-active-cut-over-250m")

    new_surface = surface[: keep_last + 1].copy()
    new_bed = bed[: keep_last + 1].copy()
    new_section = section[: keep_last + 1].copy()
    new_widths = widths[: keep_last + 1].copy()
    new_bed_shape = bed_shape[: keep_last + 1].copy()
    new_is_trapezoid = is_trapezoid[: keep_last + 1].copy()
    new_lambdas = lambdas[: keep_last + 1].copy()

    if source_reverse:
        block_area = float(np.sum(widths[removed_active]) * dx_m)
        block_volume = float(np.sum(section[removed_active]) * dx_m)
        if block_area <= 0.0 or block_volume < 0.0:
            raise RuntimeError("invalid-condensed-active-budget")
        block_thickness = block_volume / block_area
        new_widths[-1] = block_area / dx_m
        new_section[-1] = block_volume / dx_m
        new_bed[-1] = new_surface[-1] - block_thickness
        new_is_trapezoid[-1] = True
        new_lambdas[-1] = 0.0
        new_bed_shape[-1] = np.nan

    coordinates = list(source.line.coords)[: keep_last + 1]
    connector_count = 0
    connector_distance_m = 0.0
    if receiver_point is not None:
        terminal_point = Point(coordinates[-1])
        connector_distance_m = float(
            terminal_point.distance(receiver_point) * float(source.map_dx)
        )
        connector_count = max(1, int(math.ceil(connector_distance_m / dx_m)))
        start_xy = np.asarray(coordinates[-1], dtype=np.float64)
        end_xy = np.asarray(receiver_point.coords[0], dtype=np.float64)
        connector_surface = np.linspace(
            float(new_surface[-1]), float(receiver_surface), connector_count + 1
        )[1:]
        if np.any(np.diff(np.concatenate(([new_surface[-1]], connector_surface))) > TOLERANCE_M):
            raise RuntimeError("connector-would-rise-downstream")
        for fraction in np.linspace(0.0, 1.0, connector_count + 1)[1:]:
            xy = start_xy + fraction * (end_xy - start_xy)
            coordinates.append((float(xy[0]), float(xy[1])))
        connector_width = max(float(new_widths[-1]), 1.0)
        new_surface = np.concatenate((new_surface, connector_surface))
        new_bed = np.concatenate((new_bed, connector_surface.copy()))
        new_section = np.concatenate((new_section, np.zeros(connector_count)))
        new_widths = np.concatenate(
            (new_widths, np.full(connector_count, connector_width))
        )
        new_bed_shape = np.concatenate(
            (new_bed_shape, np.full(connector_count, np.nan))
        )
        new_is_trapezoid = np.concatenate(
            (new_is_trapezoid, np.ones(connector_count, dtype=bool))
        )
        new_lambdas = np.concatenate(
            (new_lambdas, np.zeros(connector_count))
        )

    candidate = MixedBedFlowline(
        line=LineString(coordinates),
        dx=float(source.dx),
        map_dx=float(source.map_dx),
        surface_h=new_surface,
        bed_h=new_bed,
        section=new_section,
        bed_shape=new_bed_shape,
        is_trapezoid=new_is_trapezoid,
        lambdas=new_lambdas,
        widths_m=new_widths,
        rgi_id=source.rgi_id,
        water_level=getattr(source, "water_level", 0.0),
    )
    return candidate, {
        "source_reverse": source_reverse,
        "source_node_count": int(source.nx),
        "candidate_node_count": int(candidate.nx),
        "retained_source_last_index": int(keep_last),
        "removed_source_node_count": int(source.nx - 1 - keep_last),
        "removed_active_node_count": int(removed_active.size),
        "extra_active_cut_m": extra_cut_m,
        "connector_node_count": connector_count,
        "connector_distance_m": connector_distance_m,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if source_summary.get("status") != "inversion-static-fail":
        raise RuntimeError("unexpected-source-inversion-status")
    if set(source_summary.get("review_flags", [])) != {
        "junction-reverse-slope",
        "terminal-ice-present",
    }:
        raise RuntimeError("source-failures-not-limited-to-registered-two")
    source_hash_before = sha256(SOURCE_FLOWLINES)
    with gzip.open(SOURCE_FLOWLINES, "rb") as stream:
        source_flowlines = pickle.load(stream)
    source_identity = {id(flowline): index for index, flowline in enumerate(source_flowlines)}
    candidates = []
    rows = []
    for branch, source in enumerate(source_flowlines):
        receiver = getattr(source, "flows_to", None)
        receiver_surface = None
        receiver_point = None
        receiver_branch = None
        receiver_index = None
        if receiver is not None:
            receiver_branch = int(source_identity[id(receiver)])
            receiver_index = int(source.flows_to_indice)
            receiver_surface = float(receiver.surface_h[receiver_index])
            receiver_point = Point(receiver.line.coords[receiver_index])
        candidate, diagnostics = rebuild_flowline(
            source, receiver_surface, receiver_point
        )
        candidates.append(candidate)
        rows.append(
            {
                "branch_index": branch,
                "receiver_branch_index": receiver_branch,
                "receiver_node_index": receiver_index,
                **diagnostics,
            }
        )

    for flowline in candidates:
        flowline.flows_to = None
        flowline.flows_to_point = None
        flowline.inflows = []
        flowline.inflow_points = []
    for branch, source in enumerate(source_flowlines):
        receiver = getattr(source, "flows_to", None)
        if receiver is None:
            continue
        receiver_branch = source_identity[id(receiver)]
        receiver_index = int(source.flows_to_indice)
        receiver_candidate = candidates[receiver_branch]
        candidate = candidates[branch]
        candidate.flows_to = receiver_candidate
        candidate.flows_to_point = Point(receiver_candidate.line.coords[receiver_index])
        receiver_candidate.inflows.append(candidate)
        receiver_candidate.inflow_points.append(candidate.flows_to_point)

    source_area, source_volume, source_z, source_w = active_metrics(source_flowlines)
    candidate_area, candidate_volume, candidate_z, candidate_w = active_metrics(candidates)
    area_relative_error = (candidate_area - source_area) / source_area
    volume_relative_error = (candidate_volume - source_volume) / source_volume
    cdf_error = cdf_error_100m(source_z, source_w, candidate_z, candidate_w)
    source_within, source_junction, source_terminal = topology_counts(source_flowlines)
    candidate_within, candidate_junction, candidate_terminal = topology_counts(candidates)
    sections = [np.asarray(flowline.section) for flowline in candidates]
    thickness = [np.asarray(flowline.thick) for flowline in candidates]
    checks = {
        "source_hash_unchanged": source_hash_before == sha256(SOURCE_FLOWLINES),
        "area_machine_conservative": abs(area_relative_error) <= 1.0e-12,
        "volume_machine_conservative": abs(volume_relative_error) <= 1.0e-12,
        "hypsometry_cdf_within_2pct": cdf_error <= MAXIMUM_CDF_ERROR,
        "candidate_within_reverse_zero": candidate_within == 0,
        "candidate_junction_reverse_zero": candidate_junction == 0,
        "candidate_terminal_positive_zero": candidate_terminal == 0,
        "candidate_finite": all(np.isfinite(value).all() for value in sections + thickness),
        "candidate_nonnegative": all(np.all(value >= 0) for value in sections + thickness),
        "root_unchanged": (
            np.array_equal(candidates[-1].section, source_flowlines[-1].section)
            and np.array_equal(candidates[-1].surface_h, source_flowlines[-1].surface_h)
            and np.array_equal(candidates[-1].bed_h, source_flowlines[-1].bed_h)
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema": "lia-191-system334-postinversion-zero-ice-connector-v1",
        "status": "static-pass" if passed else "static-fail",
        "system_id": 334,
        "mechanism": "postinversion-active-terminal-condensation-plus-zero-ice-connectors-v1",
        "source_inversion_summary": str(SOURCE_SUMMARY),
        "source_model_flowlines": str(SOURCE_FLOWLINES),
        "source_model_flowlines_sha256": source_hash_before,
        "source_area_m2": source_area,
        "candidate_area_m2": candidate_area,
        "area_relative_error": area_relative_error,
        "source_volume_m3": source_volume,
        "candidate_volume_m3": candidate_volume,
        "volume_relative_error": volume_relative_error,
        "hypsometry_cdf_error_100m": cdf_error,
        "source_within_reverse_count": source_within,
        "source_junction_reverse_count": source_junction,
        "source_terminal_positive_count": source_terminal,
        "candidate_within_reverse_count": candidate_within,
        "candidate_junction_reverse_count": candidate_junction,
        "candidate_terminal_positive_count": candidate_terminal,
        "modified_active_branch_count": int(sum(bool(row["source_reverse"]) for row in rows)),
        "total_connector_node_count": int(sum(int(row["connector_node_count"]) for row in rows)),
        "maximum_extra_active_cut_m": max(float(row["extra_active_cut_m"]) for row in rows),
        "maximum_connector_distance_m": max(float(row["connector_distance_m"]) for row in rows),
        "checks": checks,
        "candidate_written": passed,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    if passed:
        with gzip.open(CANDIDATE, "wb") as stream:
            pickle.dump(candidates, stream, protocol=pickle.HIGHEST_PROTOCOL)
        summary["candidate_model_flowlines"] = str(CANDIDATE)
        summary["candidate_model_flowlines_sha256"] = sha256(CANDIDATE)
    with (OUTPUT / "branch_connector_inventory.csv").open(
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
