"""Synthetic-only tests for the network-coupled semi-implicit algebra."""

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.network_coupled_semiimplicit_fixture_v1 import (
    run_single_line_stock_fixture,
    run_two_tributary_manufactured_fixture,
    run_zero_diffusivity_identity_fixture,
)


def test_single_line_matches_stock_semiimplicit_section() -> None:
    result = run_single_line_stock_fixture()
    assert result["status"] == "pass"
    assert result["section_relative_l2_error"] <= 1e-10
    assert abs(result["volume_residual_m3"]) <= result["closure_limit_m3"]
    assert (
        result["maximum_absolute_node_residual_m3"]
        <= result["closure_limit_m3"]
    )


def test_two_tributary_manufactured_junction_closes() -> None:
    result = run_two_tributary_manufactured_fixture()
    assert result["status"] == "pass"
    assert result["maximum_surface_error_m"] <= 1e-14
    assert result["maximum_absolute_node_residual_m3"] <= result["closure_limit_m3"]
    assert abs(result["global_volume_residual_m3"]) <= result["closure_limit_m3"]
    assert result["matrix_cholesky"] == "pass"


def test_zero_diffusivity_is_bitwise_identity() -> None:
    result = run_zero_diffusivity_identity_fixture()
    assert result["status"] == "pass"
    assert result["surface_bitwise_identity"]
    assert result["section_bitwise_identity"]
    assert result["flux_bitwise_zero"]
    assert result["residual_bitwise_zero"]
