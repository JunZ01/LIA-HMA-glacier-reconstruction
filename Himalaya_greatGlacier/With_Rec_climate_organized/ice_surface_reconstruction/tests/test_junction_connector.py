"""Unit tests for junction connector paths (synthetic bed, no OGGM)."""

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
    build_connector,
    connector_least_cost,
    connector_straight,
)


def _bed(mask: np.ndarray, bed_values: np.ndarray | None = None) -> GriddedBed:
    if bed_values is None:
        bed_values = np.zeros_like(mask, dtype=float)
    return GriddedBed(
        bed_m=np.asarray(bed_values, dtype=float),
        glacier_mask=np.asarray(mask, dtype=bool),
        slope=np.zeros_like(mask, dtype=float),
        map_dx_m=66.0,
        bed_scenario="test",
    )


def test_straight_connector_inside_mask_ok() -> None:
    mask = np.ones((5, 10), dtype=bool)
    path = connector_straight((2, 1), (2, 8), _bed(mask))
    assert path.ok
    assert path.pixels[0] == (2, 1) and path.pixels[-1] == (2, 8)
    # 7 pixel steps of 66 m along a row.
    assert abs(path.total_length_m - 7 * 66.0) < 1e-9


def test_straight_connector_leaving_mask_flagged() -> None:
    mask = np.ones((5, 10), dtype=bool)
    mask[2, 4] = False  # straight line must cross this cell
    path = connector_straight((2, 1), (2, 8), _bed(mask))
    assert not path.ok
    assert path.status == "junction-connector-review"


def test_least_cost_detours_around_hole_but_stays_in_mask() -> None:
    mask = np.ones((5, 10), dtype=bool)
    mask[2, 4] = False
    path = connector_least_cost((2, 1), (2, 8), _bed(mask), ConnectorCostWeights())
    assert path.ok
    assert all(mask[r, c] for r, c in path.pixels)
    # Detour is at least as long as the straight distance.
    assert path.total_length_m >= 7 * 66.0


def test_least_cost_disconnected_flagged() -> None:
    mask = np.ones((5, 10), dtype=bool)
    mask[:, 5] = False  # full wall splits the domain
    path = connector_least_cost((2, 1), (2, 8), _bed(mask), ConnectorCostWeights())
    assert not path.ok
    assert "no continuous" in path.review_reason


def test_least_cost_avoids_explicit_blocked_network_cells() -> None:
    mask = np.ones((7, 9), dtype=bool)
    blocked = {(3, col) for col in range(2, 7)}
    path = connector_least_cost(
        (3, 1), (3, 7), _bed(mask), ConnectorCostWeights(), blocked
    )
    assert path.ok
    assert all(cell not in blocked for cell in path.pixels[1:-1])


def test_least_cost_prefers_valley_floor() -> None:
    # Two corridors: a high-bed straight row and a low-bed detour. With a strong
    # bed-above-floor weight the path should avoid the high row.
    mask = np.ones((5, 10), dtype=bool)
    bed = np.zeros((5, 10), dtype=float)
    bed[2, :] = 500.0  # the direct row is a high ridge
    weights = ConnectorCostWeights(alpha=50.0, beta=0.0)
    path = connector_least_cost((2, 1), (2, 8), _bed(mask, bed), weights)
    assert path.ok
    # It should leave the ridge row for at least some interior pixels.
    ridge_pixels = sum(1 for r, c in path.pixels if r == 2 and 1 < c < 8)
    assert ridge_pixels < 6


def test_build_connector_dispatch_and_bad_mode() -> None:
    mask = np.ones((3, 3), dtype=bool)
    assert build_connector((1, 0), (1, 2), _bed(mask), mode="straight").ok
    try:
        build_connector((1, 0), (1, 2), _bed(mask), mode="nope")
    except ValueError as e:
        assert "Unknown connector mode" in str(e)
    else:
        raise AssertionError("expected ValueError")
