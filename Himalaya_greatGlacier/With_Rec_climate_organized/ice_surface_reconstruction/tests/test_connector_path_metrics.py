"""Tests for direction-free connector path-quality metrics (task #7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ice_surface_reconstruction.drivers.gridded_bed import GriddedBed  # noqa: E402
from ice_surface_reconstruction.drivers.junction_connector import (  # noqa: E402
    ConnectorCostWeights,
    connector_least_cost,
    connector_straight,
)
from ice_surface_reconstruction.drivers.connector_path_metrics import (  # noqa: E402
    VELOCITY_NOT_EVALUATED,
    attach_velocity_evidence,
    path_quality,
    two_bed_stability,
)


def _bed(mask, bed=None, slope=None):
    if bed is None:
        bed = np.zeros_like(mask, dtype=float)
    if slope is None:
        slope = np.zeros_like(mask, dtype=float)
    return GriddedBed(np.asarray(bed, float), np.asarray(mask, bool),
                      np.asarray(slope, float), 66.0, "t")


def test_straight_path_has_length_ratio_one() -> None:
    g = _bed(np.ones((5, 10), bool))
    path = connector_straight((2, 1), (2, 8), g)
    m = path_quality(path, g)
    assert m.status == "ok"
    assert abs(m.length_ratio - 1.0) < 1e-6
    assert m.velocity_status == VELOCITY_NOT_EVALUATED


def test_detour_path_has_length_ratio_above_one() -> None:
    mask = np.ones((5, 10), bool)
    mask[2, 4] = False  # forces the least-cost path off the straight row
    g = _bed(mask)
    path = connector_least_cost((2, 1), (2, 8), g, ConnectorCostWeights())
    m = path_quality(path, g)
    assert m.length_ratio > 1.0


def test_review_path_reports_review_status() -> None:
    mask = np.ones((5, 10), bool)
    mask[2, 4] = False
    g = _bed(mask)
    path = connector_straight((2, 1), (2, 8), g)  # crosses the hole -> review
    m = path_quality(path, g)
    assert m.status == "junction-connector-review"
    assert np.isnan(m.length_ratio)


def test_two_bed_stability_identical_paths() -> None:
    g = _bed(np.ones((5, 10), bool))
    p = connector_straight((2, 1), (2, 8), g)
    s = two_bed_stability(p, p, g)
    assert s["two_bed_status"] == "ok"
    assert s["pixel_jaccard"] == 1.0
    assert s["length_difference_m"] == 0.0
    assert s["mean_symmetric_path_distance_m"] == 0.0
    assert s["hausdorff_distance_m"] == 0.0


def test_two_bed_stability_parallel_paths_small_distance() -> None:
    # Two parallel rows share NO pixels (Jaccard 0) but are one cell apart:
    # the distance metrics must reflect proximity, not the misleading Jaccard.
    g = _bed(np.ones((5, 10), bool))
    p_a = connector_straight((2, 1), (2, 8), g)
    p_b = connector_straight((3, 1), (3, 8), g)
    s = two_bed_stability(p_a, p_b, g)
    assert s["pixel_jaccard"] == 0.0
    assert 0.0 < s["mean_symmetric_path_distance_m"] <= g.map_dx_m * 1.001
    assert s["hausdorff_distance_m"] <= g.map_dx_m * 1.001


def _fake_samples(path, vx_val, vy_val, *, map_dx_m=66.0):
    """Build a VelocitySamples at path SEGMENT MIDPOINTS with a uniform flow."""
    from ice_surface_reconstruction.drivers.velocity_field import VelocitySamples

    n_seg = len(path.pixels) - 1
    # Each segment midpoint sits in its own distinct 120 m velocity cell here.
    prow = np.arange(n_seg)
    pcol = np.zeros(n_seg, dtype=int)
    ones = np.ones(n_seg)
    return VelocitySamples(
        vx=ones * vx_val, vy=ones * vy_val, v_error=ones * 0.5,
        count=ones * 500, outlier_percent=ones * 1.0, valid=ones.astype(bool),
        xy_velocity_crs=np.zeros((n_seg, 2)), pixel_row=prow, pixel_col=pcol,
    )


def test_velocity_evidence_directed_and_undirected() -> None:
    # Connector stored P..T (east); downstream sense T->P (west). Flow west =>
    # directed & undirected both ~0.
    g = _bed(np.ones((5, 10), bool))
    path = connector_straight((2, 1), (2, 8), g)
    verts = np.array([[c, r] for r, c in path.pixels], dtype=float)  # (x=col,y=row)
    samples = _fake_samples(path, -1.0, 0.0)
    m = attach_velocity_evidence(path_quality(path, g), path, samples, verts)
    assert m.velocity_status == "ok"
    assert m.mean_flow_direction_error_rad < 1e-6
    assert m.mean_directed_flow_error_rad < 1e-6
    assert m.n_unique_velocity_cells >= 1
