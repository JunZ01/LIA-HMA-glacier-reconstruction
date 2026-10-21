from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.network_capacity_constrained_flux_fixture_v1 import (
    _single_edge_capacity,
    _two_tributary_conservation,
    _unlimited_identity,
)


def test_single_edge_capacity():
    assert _single_edge_capacity()["status"] == "pass"


def test_two_tributary_conservation():
    assert _two_tributary_conservation()["status"] == "pass"


def test_unlimited_identity():
    assert _unlimited_identity()["status"] == "pass"
