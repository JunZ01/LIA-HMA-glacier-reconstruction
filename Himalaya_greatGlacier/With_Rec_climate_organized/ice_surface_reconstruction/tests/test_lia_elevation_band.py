from __future__ import annotations

import numpy as np
import pytest

from ice_surface_reconstruction.drivers.lia_elevation_band import (
    build_elevation_band_state,
    build_rectangular_flowline,
)


def _synthetic_source():
    y, x = np.mgrid[0:20, 0:12]
    surface = 4200.0 - 18.0 * y - 2.0 * x
    thickness = 30.0 + 0.5 * y + 0.2 * x
    bed = surface - thickness
    valid = np.ones_like(surface, dtype=bool)
    valid[:, :2] = False
    valid[:, -2:] = False
    surface = np.where(valid, surface, np.nan)
    bed = np.where(valid, bed, np.nan)
    thickness = np.where(valid, thickness, np.nan)
    return surface, bed, thickness, valid


def test_elevation_band_mapping_preserves_area_volume_and_mean_bed() -> None:
    surface, bed, thickness, valid = _synthetic_source()

    state = build_elevation_band_state(
        surface,
        bed,
        thickness,
        valid,
        cell_dx_m=30.0,
        cell_dy_m=30.0,
        elevation_bin_size_m=30.0,
        node_dx_m=75.0,
    )

    assert np.sum(state.node_width_m * state.node_dx_m) == pytest.approx(
        state.source_area_m2
    )
    assert np.sum(
        state.node_width_m * state.node_thickness_m * state.node_dx_m
    ) == pytest.approx(state.source_volume_m3)
    np.testing.assert_allclose(
        state.node_surface_h_m - state.node_bed_h_m,
        state.node_thickness_m,
    )
    assert np.all(np.diff(state.node_surface_h_m) <= 0)


def test_elevation_band_rectangular_flowline_adds_only_ice_free_buffer() -> None:
    surface, bed, thickness, valid = _synthetic_source()
    state = build_elevation_band_state(
        surface,
        bed,
        thickness,
        valid,
        cell_dx_m=30.0,
        cell_dy_m=30.0,
        elevation_bin_size_m=30.0,
        node_dx_m=75.0,
    )

    flowline = build_rectangular_flowline(
        state, rgi_id="LIA-test", downstream_buffer_m=750.0
    )

    assert flowline.nx == len(state.node_thickness_m) + 10
    np.testing.assert_allclose(
        flowline.thick[: len(state.node_thickness_m)], state.node_thickness_m
    )
    assert np.all(flowline.thick[-10:] == 0)
    assert flowline.area_m2 == pytest.approx(state.source_area_m2)
    assert flowline.volume_m3 == pytest.approx(state.source_volume_m3)


def test_elevation_band_target_length_preserves_area_and_volume() -> None:
    surface, bed, thickness, valid = _synthetic_source()
    target_length = 12_345.0
    state = build_elevation_band_state(
        surface,
        bed,
        thickness,
        valid,
        cell_dx_m=30.0,
        cell_dy_m=30.0,
        elevation_bin_size_m=30.0,
        node_dx_m=75.0,
        target_length_m=target_length,
    )

    assert state.target_total_length_m == target_length
    assert state.longitudinal_scale_factor == pytest.approx(
        target_length / state.unscaled_total_length_m
    )
    assert np.sum(state.node_width_m * state.node_dx_m) == pytest.approx(
        state.source_area_m2
    )
    assert np.sum(
        state.node_width_m * state.node_thickness_m * state.node_dx_m
    ) == pytest.approx(state.source_volume_m3)
    assert abs(len(state.node_width_m) * state.node_dx_m - target_length) <= 75.0
