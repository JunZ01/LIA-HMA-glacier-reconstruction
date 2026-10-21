from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "stage3_initial_state"))

from common import (  # noqa: E402
    geometry_score,
    select_regional_pilot,
    timing_weight,
    weighted_quantiles,
)


def test_timing_weight_trapezoid() -> None:
    assert timing_weight(1199, 1200, 1300, 1600, 1900) == 0
    assert timing_weight(1250, 1200, 1300, 1600, 1900) == 0.5
    assert timing_weight(1500, 1200, 1300, 1600, 1900) == 1
    assert timing_weight(1750, 1200, 1300, 1600, 1900) == 0.5
    assert timing_weight(1901, 1200, 1300, 1600, 1900) == 0


def test_weighted_quantiles_respect_weights() -> None:
    result = weighted_quantiles(
        np.array([0.0, 10.0]), np.array([1.0, 9.0]), np.array([0.5])
    )
    assert result[0] > 8


def test_geometry_score_requires_area_and_reweights_missing_components() -> None:
    result = geometry_score(
        area_m2=100,
        target_area_m2=100,
        length_m=None,
        target_length_m=None,
        hypsometry_quantiles_m=None,
        target_hypsometry_quantiles_m=None,
        weights={"area": 0.5, "length": 0.3, "hypsometry": 0.2},
        tolerances={
            "area_relative": 0.15,
            "length_relative": 0.15,
            "hypsometry_rmse_m": 200,
        },
    )
    assert result["geometry_score"] == 1
    assert result["geometry_tolerance_pass"]
    assert result["active_components"] == "area"


def test_regional_selection_keeps_required_systems_and_counts() -> None:
    ids = np.arange(1, 21)
    routes = pd.DataFrame(
        {
            "lia_unique_id": ids,
            "region_code": np.repeat(np.arange(1, 5), 5),
            "farinotti_core_191": True,
        }
    )
    manifest = pd.DataFrame(
        {
            "lia_unique_id": ids,
            "lia_area_km2": ids.astype(float),
            "centroid_lon": 80.0,
            "centroid_lat": 30.0,
        }
    )
    selected = select_regional_pilot(
        routes,
        manifest,
        per_region=3,
        area_quantiles=[0.25, 0.5, 0.75],
        required_ids=[1, 10, 20],
    )
    assert len(selected) == 12
    assert selected.groupby("region_code").size().eq(3).all()
    assert {1, 10, 20}.issubset(set(selected["lia_unique_id"]))
