from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.planform_2d_frozen_diffusivity_fixtures_v1 import (
    manufactured_conservation_fixture,
    oggm_infinitesimal_consistency_fixture,
    zero_diffusivity_fixture,
)


def test_zero_diffusivity_identity():
    assert zero_diffusivity_fixture()["passed"]


def test_manufactured_mass_conservation():
    assert manufactured_conservation_fixture()["passed"]


def test_oggm_infinitesimal_consistency():
    assert oggm_infinitesimal_consistency_fixture()["passed"]
