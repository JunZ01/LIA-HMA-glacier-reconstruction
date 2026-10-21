from __future__ import annotations

import numpy as np
import pytest

from ice_surface_reconstruction.drivers.lia_dynamic_flowlines import (
    FlowlineNodeFields,
    build_variable_lambda_flowlines,
    derive_fixed_bed_trapezoid_geometry,
    derive_trapezoid_geometry,
    project_thickness_to_trapezoid_domain,
)


def test_fixed_bed_contract_preserves_thickness_section_and_positive_bottom() -> None:
    thickness = np.array([20.0, 20.0, 20.0, 0.0])
    width = np.array([100.0, 100.0, 100.0, 80.0])
    # F = 0.4, 0.75 and 1.2 exercise all positive-ice regimes.
    section = np.array([800.0, 1500.0, 2400.0, 0.0])

    result = derive_fixed_bed_trapezoid_geometry(thickness, section, width)

    reconstructed = (
        0.5
        * (result.bottom_width_m + result.effective_surface_width_m)
        * thickness
    )
    np.testing.assert_allclose(reconstructed, section, rtol=1e-12, atol=1e-8)
    assert np.all(result.bottom_width_m[result.positive_ice] > 0)
    assert np.all(result.lambdas[result.positive_ice] >= 0)
    np.testing.assert_array_equal(result.regime_code, [1, 2, 3, 0])
    assert result.effective_surface_width_m[0] == pytest.approx(80.0, rel=2e-6)
    assert result.effective_surface_width_m[1] == pytest.approx(100.0)
    assert result.effective_surface_width_m[2] == pytest.approx(120.0)


def test_fixed_bed_contract_is_continuous_at_shape_factor_boundaries() -> None:
    thickness = np.full(4, 20.0)
    width = np.full(4, 100.0)
    factors = np.array([0.5 - 1e-7, 0.5 + 1e-7, 1.0 - 1e-7, 1.0 + 1e-7])
    section = factors * width * thickness

    result = derive_fixed_bed_trapezoid_geometry(thickness, section, width)

    assert abs(result.effective_surface_width_m[0] - result.effective_surface_width_m[1]) < 1e-3
    assert abs(result.effective_surface_width_m[2] - result.effective_surface_width_m[3]) < 1e-3
    assert abs(result.lambdas[0] - result.lambdas[1]) < 1e-3
    assert abs(result.lambdas[2] - result.lambdas[3]) < 1e-6


def test_fixed_bed_contract_rejects_zero_section_with_positive_thickness() -> None:
    with pytest.raises(
        ValueError, match="fixed-bed-thickness-section-zero-state-mismatch"
    ):
        derive_fixed_bed_trapezoid_geometry([10.0], [0.0], [100.0])


def test_derive_trapezoid_geometry_preserves_all_three_targets() -> None:
    thickness = np.array([20.0, 30.0, 0.0])
    surface_width = np.array([100.0, 140.0, 80.0])
    expected_lambda = np.array([2.0, 1.5])
    expected_w0 = surface_width[:2] - expected_lambda * thickness[:2]
    section = np.array(
        [
            0.5 * (surface_width[0] + expected_w0[0]) * thickness[0],
            0.5 * (surface_width[1] + expected_w0[1]) * thickness[1],
            0.0,
        ]
    )

    result = derive_trapezoid_geometry(
        thickness, section, surface_width, default_lambda=2.0
    )

    np.testing.assert_allclose(result.lambdas[:2], expected_lambda)
    np.testing.assert_allclose(result.bottom_width_m[:2], expected_w0)
    reconstructed_width = result.bottom_width_m + result.lambdas * thickness
    reconstructed_section = (
        0.5 * (reconstructed_width + result.bottom_width_m) * thickness
    )
    np.testing.assert_allclose(reconstructed_width, surface_width)
    np.testing.assert_allclose(reconstructed_section, section)
    assert result.bottom_width_m[2] == pytest.approx(surface_width[2])


def test_derive_trapezoid_geometry_accepts_rectangular_upper_endpoint() -> None:
    result = derive_trapezoid_geometry([10.0], [1000.0], [100.0])
    assert result.lambdas[0] == pytest.approx(0.0)
    assert result.bottom_width_m[0] == pytest.approx(100.0)
    assert result.area_shape_factor[0] == pytest.approx(1.0)


def test_derive_trapezoid_geometry_rejects_infeasible_shape_factor() -> None:
    with pytest.raises(ValueError, match="trapezoid-cross-section-infeasible"):
        derive_trapezoid_geometry([10.0], [400.0], [100.0])


def test_derive_trapezoid_geometry_rejects_thickness_area_zero_mismatch() -> None:
    with pytest.raises(ValueError, match="thickness-section-zero-state-mismatch"):
        derive_trapezoid_geometry([10.0], [0.0], [100.0])


def test_project_thickness_preserves_section_and_width_at_domain_bounds() -> None:
    observed = np.array([30.0, 5.0, 0.0])
    section = np.array([1000.0, 1000.0, 0.0])
    width = np.array([100.0, 100.0, 80.0])
    projection = project_thickness_to_trapezoid_domain(
        observed, section, width, minimum_bottom_width_m=10.0
    )
    # Node 0 is too deep/peaked and reaches the w0=10 m boundary.
    assert projection.effective_thickness_m[0] == pytest.approx(2000.0 / 110.0)
    # Node 1 is too shallow and reaches the rectangular boundary.
    assert projection.effective_thickness_m[1] == pytest.approx(10.0)
    geometry = derive_trapezoid_geometry(
        projection.effective_thickness_m, section, width
    )
    reconstructed_width = (
        geometry.bottom_width_m
        + geometry.lambdas * projection.effective_thickness_m
    )
    reconstructed_section = 0.5 * (
        reconstructed_width + geometry.bottom_width_m
    ) * projection.effective_thickness_m
    np.testing.assert_allclose(reconstructed_width, width)
    np.testing.assert_allclose(reconstructed_section, section)


def test_strip_mean_rectangular_contract_builds_zero_lambda_flowline() -> None:
    oggm = pytest.importorskip("oggm")
    from oggm.core.centerlines import Centerline
    from shapely.geometry import LineString

    centerline = Centerline(
        line=LineString([(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]),
        dx=100.0,
        map_dx=1.0,
        surface_h=np.asarray([120.0, 110.0, 100.0]),
        rgi_id="TEST",
    )
    fields = FlowlineNodeFields(
        surface_h_m=np.asarray([120.0, 110.0, 100.0]),
        bed_h_m=np.asarray([120.0, 90.0, 100.0]),
        section_m2=np.asarray([0.0, 2000.0, 0.0]),
        surface_width_m=np.asarray([100.0, 100.0, 100.0]),
    )
    flowlines, audit = build_variable_lambda_flowlines(
        [centerline],
        [fields],
        map_dx_m=1.0,
        downstream_line=None,
        default_lambda=0.0,
        cross_section_contract="strip-mean-rectangular",
    )
    assert len(flowlines) == 1
    np.testing.assert_allclose(flowlines[0]._lambdas, 0.0)
    assert audit["cross_sections"][0]["cross_section_contract"] == (
        "strip-mean-rectangular"
    )
