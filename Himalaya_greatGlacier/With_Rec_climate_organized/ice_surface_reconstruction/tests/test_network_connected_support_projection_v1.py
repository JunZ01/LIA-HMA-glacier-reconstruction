import numpy as np

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.network_capacity_constrained_flux_projection_v2 import project_flux_dag_exact
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.network_connected_support_projection_v1 import (
    project_flux_dag_with_reserve,
    project_lower_bounded_simplex,
)


def test_zero_reserve_matches_registered_projection():
    volume = np.array([4.0, 3.0, 1.0])
    u = np.array([0, 1])
    v = np.array([1, 2])
    flux = np.array([0.2, 0.4])
    expected = project_flux_dag_exact(volume, u, v, flux, 10.0)
    actual = project_flux_dag_with_reserve(volume, np.zeros(3), u, v, flux, 10.0)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.array_equal(actual[2], expected[2])
    assert actual[3] == expected[3]


def test_reserve_is_preserved_and_mass_is_conserved():
    volume = np.array([2.0, 1.0, 0.0])
    reserve = np.array([0.5, 0.25, 0.0])
    accepted, final, limited, conservation = project_flux_dag_with_reserve(
        volume, reserve, np.array([0, 1]), np.array([1, 2]), np.array([1.0, 1.0]), 10.0
    )
    assert np.all(final >= reserve)
    assert np.isclose(final.sum(), volume.sum())
    assert conservation == 0.0
    assert limited.any()
    assert np.all(accepted >= 0)


def test_lower_bounded_simplex_is_exact_and_minimal():
    requested = np.array([5.0, -2.0, 4.0])
    lower = np.array([0.1, 0.2, 0.3])
    projected = project_lower_bounded_simplex(requested, lower)
    assert np.all(projected >= lower)
    assert np.isclose(projected.sum(), requested.sum())
    assert np.allclose(projected, np.array([3.9, 0.2, 2.9]))
