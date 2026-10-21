from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.paired_cell_graph_nonlinear_obstacle_fixture_v1 import run_obstacle_crossing_fixture, run_slope_crossing_fixture, run_zero_mb_conservation_fixture


def test_slope_crossing(): assert run_slope_crossing_fixture()["status"]=="pass"
def test_zero_mb_conservation(): assert run_zero_mb_conservation_fixture()["status"]=="pass"
def test_obstacle_crossing(): assert run_obstacle_crossing_fixture()["status"]=="pass"
