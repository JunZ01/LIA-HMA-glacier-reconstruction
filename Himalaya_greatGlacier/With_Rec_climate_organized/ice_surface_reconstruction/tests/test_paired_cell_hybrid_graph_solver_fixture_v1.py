from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.paired_cell_hybrid_graph_solver_fixture_v1 import (
    run_hybrid_connector,
    run_rectangular_grid,
    run_two_cell_exact,
    run_zero_conductance_identity,
)


def test_two_cell_exact():
    assert run_two_cell_exact()["status"] == "pass"


def test_rectangular_grid():
    assert run_rectangular_grid()["status"] == "pass"


def test_hybrid_connector():
    assert run_hybrid_connector()["status"] == "pass"


def test_zero_conductance_identity():
    assert run_zero_conductance_identity()["status"] == "pass"
