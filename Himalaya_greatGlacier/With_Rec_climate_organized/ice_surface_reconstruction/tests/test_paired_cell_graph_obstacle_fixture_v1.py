from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.paired_cell_graph_obstacle_fixture_v1 import (
    run_ablation_exhaustion_fixture,
    run_diffusive_release_fixture,
    run_mixed_obstacle_fixture,
)


def test_ablation_exhaustion():
    assert run_ablation_exhaustion_fixture()["status"] == "pass"


def test_diffusive_release():
    assert run_diffusive_release_fixture()["status"] == "pass"


def test_mixed_obstacle():
    assert run_mixed_obstacle_fixture()["status"] == "pass"
