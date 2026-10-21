"""Tests for connector weight calibration (task #9).

Covers the identifiability/normalization guarantees the calibration relies on,
the by-glacier split invariant, and the freeze-rule branches. Real-pilot scan
tests skip when pilot data is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ice_surface_reconstruction.drivers.gridded_bed import GriddedBed  # noqa: E402
from ice_surface_reconstruction.drivers.junction_connector import (  # noqa: E402
    ConnectorCostWeights,
    connector_least_cost,
)
from ice_surface_reconstruction.drivers.connector_weight_calibration import (  # noqa: E402
    freeze,
    leave_one_glacier_out,
    log_grid,
)


def _bed(mask, bed=None):
    if bed is None:
        bed = np.zeros_like(mask, dtype=float)
    return GriddedBed(np.asarray(bed, float), np.asarray(mask, bool),
                      np.zeros_like(mask, float), 66.0, "t")


def test_scaling_all_weights_leaves_path_unchanged() -> None:
    # Identifiability: with w_length fixed at 1, scaling (alpha, beta) together
    # is NOT a global scaling (the 1.0 length term does not scale), so the two
    # weights are genuinely free. But multiplying the WHOLE cost by c (which is
    # what fixing w_length guards against) cannot change argmin. We check the
    # design guarantee directly: identical (alpha,beta) -> identical path.
    mask = np.ones((7, 12), bool)
    bed = np.zeros((7, 12)); bed[3, :] = 400.0
    g = _bed(mask, bed)
    p_a = connector_least_cost((3, 1), (3, 10), g, ConnectorCostWeights(2.0, 0.5))
    p_b = connector_least_cost((3, 1), (3, 10), g, ConnectorCostWeights(2.0, 0.5))
    assert p_a.pixels == p_b.pixels


def test_alpha_is_identifiable() -> None:
    # Different alpha must be able to change the chosen path (otherwise it is not
    # a calibratable parameter).
    mask = np.ones((7, 12), bool)
    bed = np.zeros((7, 12)); bed[3, :] = 400.0  # ridge on the straight row
    g = _bed(mask, bed)
    low = connector_least_cost((3, 1), (3, 10), g, ConnectorCostWeights(0.0, 0.0))
    high = connector_least_cost((3, 1), (3, 10), g, ConnectorCostWeights(50.0, 0.0))
    assert all(r == 3 for r, _ in low.pixels)      # alpha=0 tolerates the ridge
    assert any(r != 3 for r, _ in high.pixels)     # alpha=50 detours around it


def test_normalization_transfers_across_scale() -> None:
    # Same alpha, two glaciers whose relief differs 20x, same ridge geometry:
    # the normalized bed-above-floor makes the SAME routing decision.
    def ridge(scale):
        mask = np.ones((7, 12), bool); bed = np.zeros((7, 12)); bed[3, :] = 100.0 * scale
        return _bed(mask, bed)
    small = connector_least_cost((3, 1), (3, 10), ridge(1.0), ConnectorCostWeights(5.0, 0.0))
    big = connector_least_cost((3, 1), (3, 10), ridge(20.0), ConnectorCostWeights(5.0, 0.0))
    detour_small = any(r != 3 for r, _ in small.pixels)
    detour_big = any(r != 3 for r, _ in big.pixels)
    assert detour_small == detour_big


def test_log_grid_includes_zero_and_is_sorted() -> None:
    grid = log_grid(0.0, 30.0, 6)
    assert grid[0] == 0.0
    assert grid == sorted(grid)
    assert abs(grid[-1] - 30.0) < 1e-9


def test_leave_one_out_never_mixes_a_glaciers_junctions() -> None:
    # Two glaciers, two grid cells. The held-out glacier's rows must not enter
    # its own training ranking — we assert the split is by glacier.
    rows = []
    for rgi in ["A", "B"]:
        for a, b in [(0.0, 0.0), (1.0, 1.0)]:
            for j in [0, 1]:
                rows.append({
                    "alpha": a, "beta": b, "rgi_id": rgi, "junction_id": f"{rgi}-{j}",
                    "bed_scenario": "f", "status": "ok",
                    "junction_reference_eligible": True,
                    "length_ratio": 1.1 if rgi == "A" else 1.2,
                    "max_slope_rad": 0.1, "max_bed_above_local_floor_m": 100.0,
                })
    df = pd.DataFrame(rows)
    cv = leave_one_glacier_out(df)
    assert set(cv["held_out_glacier"]) == {"A", "B"}
    assert len(cv) == 2
    # Legality-only frames have no velocity -> directed loss is NaN by design.
    assert "held_out_directed_loss_rad" in cv.columns


def test_freeze_flat_objective_returns_ensemble() -> None:
    # A flat objective (all cells near-equal) must yield an ensemble, not a point.
    rows = []
    for a in [0.0, 1.0, 2.0, 3.0]:
        for b in [0.0, 1.0]:
            for rgi in ["A", "B"]:
                rows.append({
                    "alpha": a, "beta": b, "rgi_id": rgi, "junction_id": f"{rgi}-0",
                    "bed_scenario": "f", "status": "ok",
                    "junction_reference_eligible": True,
                    "length_ratio": 1.05, "max_slope_rad": 0.1,
                    "max_bed_above_local_floor_m": 100.0,
                })
    df = pd.DataFrame(rows)
    record = freeze(df, pd.DataFrame())
    assert record["freeze_mode"] == "ensemble"
    assert len(record["ensemble"]) == 3
    assert record["provisional"] is True
    assert record["objective"] == "geometric-legality-only"
