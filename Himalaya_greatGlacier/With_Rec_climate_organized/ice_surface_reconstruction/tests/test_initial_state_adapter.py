from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ice_surface_reconstruction.drivers.initial_state_adapter import (
    NearestGridSampler,
    grid_xy_from_flowline,
    scale_flowlines_to_area_and_volume,
    scale_flowlines_to_volume,
    total_flowline_area_m2,
    total_flowline_volume_m3,
)
from ice_surface_reconstruction.drivers.initial_state_dynamic_smoke import (
    aggregate_member_results,
    find_member_pickles,
)


class DummyLine:
    def __init__(self, coords):
        self.coords = coords


class DummyFlowline:
    def __init__(self, thick, widths_m=None, dx_meter=10.0):
        self.line = DummyLine([(0.0, 0.0), (2.0, 1.0)])
        self.bed_h = np.zeros(len(thick), dtype=float)
        self.thick = np.asarray(thick, dtype=float)
        self.widths_m = (
            np.ones(len(thick), dtype=float) * 20.0
            if widths_m is None
            else np.asarray(widths_m, dtype=float)
        )
        self.dx_meter = float(dx_meter)

    @property
    def section(self):
        return self.thick * self.widths_m

    @property
    def volume_m3(self):
        return float(np.sum(self.section) * self.dx_meter)


class DummyParabolicFlowline:
    def __init__(self, thick, bed_shape=0.02, dx_meter=10.0):
        self.thick = np.asarray(thick, dtype=float)
        self.bed_shape = np.ones(len(thick), dtype=float) * float(bed_shape)
        self._sqrt_bed = np.sqrt(self.bed_shape)
        self.is_trapezoid = np.zeros(len(thick), dtype=bool)
        self.dx_meter = float(dx_meter)

    @property
    def widths_m(self):
        return np.sqrt(4.0 * self.thick / self.bed_shape)

    @property
    def section(self):
        return 2.0 / 3.0 * self.widths_m * self.thick

    @property
    def bin_area_m2(self):
        return np.where(self.thick > 0, self.widths_m, 0.0) * self.dx_meter

    @property
    def volume_m3(self):
        return float(np.sum(self.section) * self.dx_meter)


def test_grid_xy_from_flowline_uses_col_row_order() -> None:
    flowline = DummyFlowline([1.0, 1.0])
    glacier_grid = {"x0y0": [100.0, 200.0], "dxdy": [30.0, -30.0]}
    x, y = grid_xy_from_flowline(flowline, glacier_grid)
    np.testing.assert_allclose(x, [100.0, 160.0])
    np.testing.assert_allclose(y, [200.0, 170.0])


def test_nearest_grid_sampler_handles_decreasing_y() -> None:
    sampler = NearestGridSampler(
        x=np.array([0.0, 10.0, 20.0]),
        y=np.array([100.0, 90.0]),
        arrays={"value": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
    )
    sampled = sampler.sample(np.array([9.0, 21.0]), np.array([91.0, 101.0]), "value")
    np.testing.assert_allclose(sampled, [5.0, 3.0])


def test_scale_flowlines_to_volume_matches_target() -> None:
    flowlines_by_rgi = {
        "RGI60-00.00001": [DummyFlowline([10.0, 20.0])],
        "RGI60-00.00002": [DummyFlowline([5.0])],
    }
    target = 10000.0
    factor = scale_flowlines_to_volume(flowlines_by_rgi, target)
    assert 0 < factor < 2
    assert abs(total_flowline_volume_m3(flowlines_by_rgi) - target) < 1e-6


def test_scale_flowlines_to_area_and_volume_matches_both_targets() -> None:
    flowlines_by_rgi = {"RGI60-00.00001": [DummyParabolicFlowline([10.0, 20.0])]}
    target_area = total_flowline_area_m2(flowlines_by_rgi) * 1.4
    target_volume = total_flowline_volume_m3(flowlines_by_rgi) * 0.8
    thickness_factor, width_factor = scale_flowlines_to_area_and_volume(
        flowlines_by_rgi, target_area, target_volume
    )
    assert 0 < thickness_factor < 1
    assert width_factor > 1
    assert abs(total_flowline_area_m2(flowlines_by_rgi) - target_area) < 1e-6
    assert abs(total_flowline_volume_m3(flowlines_by_rgi) - target_volume) < 1e-6


def test_find_member_pickles_returns_complete_sorted_package(tmp_path: Path) -> None:
    root = tmp_path / "7" / "farinotti2019"
    expected = []
    for rgi_id in ("RGI60-15.01142", "RGI60-15.01141"):
        path = root / rgi_id / "model_flowlines_lia_farinotti2019.pkl"
        path.parent.mkdir(parents=True)
        path.touch()
        expected.append(path)
    assert find_member_pickles(tmp_path, 7, "farinotti2019") == sorted(expected)


def test_aggregate_member_results_uses_all_system_members() -> None:
    rows = [
        {
            "initial_volume_km3": 1.0,
            "final_volume_km3": 1.1,
            "initial_area_km2": 4.0,
            "final_area_km2": 4.2,
            "initial_length_m": 1000.0,
            "final_length_m": 1100.0,
            "finite_nonnegative": True,
            "dynamic_smoke_status": "ok",
            "dynamic_smoke_error": "",
        },
        {
            "initial_volume_km3": 3.0,
            "final_volume_km3": 2.9,
            "initial_area_km2": 6.0,
            "final_area_km2": 5.8,
            "initial_length_m": 2000.0,
            "final_length_m": 1900.0,
            "finite_nonnegative": True,
            "dynamic_smoke_status": "ok",
            "dynamic_smoke_error": "",
        },
    ]
    result = aggregate_member_results(rows)
    assert result["member_run_count"] == 2
    assert result["initial_volume_km3"] == pytest.approx(4.0)
    assert result["final_volume_km3"] == pytest.approx(4.0)
    assert result["volume_change_fraction"] == pytest.approx(0.0)
    assert result["initial_area_km2"] == pytest.approx(10.0)
    assert result["final_area_km2"] == pytest.approx(10.0)
