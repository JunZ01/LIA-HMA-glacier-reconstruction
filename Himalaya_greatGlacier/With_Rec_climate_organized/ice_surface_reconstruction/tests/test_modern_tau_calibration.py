"""Tests for modern effective-stress calibration helpers."""

from types import SimpleNamespace

import numpy as np

from ice_surface_reconstruction.drivers.modern_tau_calibration import (
    _interior_truth_points,
    _real_inputs,
)
from ice_surface_reconstruction.drivers.oggm_network_adapter import (
    BranchProvenance,
    FlowlineInput,
)


def test_real_inputs_use_product_specific_inversion_thickness() -> None:
    item = FlowlineInput(
        flowline_index=0,
        flows_to=None,
        flows_to_indice=None,
        bed_elevation_m=np.asarray([0.0, 0.0, 0.0]),
        reference_surface_m=np.asarray([120.0, 110.0, 100.0]),
        dx_meter=10.0,
        map_dx_m=5.0,
        coords_xy=np.asarray([[3.0, 3.0], [3.0, 2.0], [3.0, 1.0]]),
    )
    raw = [SimpleNamespace(surface_h=np.asarray([120.0, 110.0, 100.0]))]
    rebuilt, truth = _real_inputs([item], raw, [{"thick": np.asarray([20.0, 15.0, 5.0])}])
    np.testing.assert_allclose(rebuilt[0].bed_elevation_m, [100.0, 95.0, 95.0])
    np.testing.assert_allclose(truth[0][1], [20.0, 15.0, 5.0])


def test_interior_truth_points_apply_metric_boundary_exclusion() -> None:
    build = SimpleNamespace(
        provenance=[BranchProvenance("fl0", 0, 0, 2)]
    )
    solution = SimpleNamespace(branch_solutions={
        "fl0": SimpleNamespace(
            # Network order maps to native indices 2, 1, 0.
            thickness_m=np.asarray([6.0, 14.0, 19.0]),
            surface_elevation_m=np.asarray([106.0, 114.0, 119.0]),
        )
    })
    truth = {0: (
        np.asarray([120.0, 115.0, 105.0]),
        np.asarray([20.0, 15.0, 5.0]),
    )}
    coords = {0: np.asarray([[3.0, 3.0], [3.0, 2.0], [3.0, 1.0]])}
    model, observed, metrics = _interior_truth_points(
        build,
        solution,
        truth,
        np.ones((7, 7), dtype=bool),
        map_dx_m=1.0,
        boundary_exclusion_m=1.5,
        minimum_thickness_m=1.0,
        coords_by_flowline=coords,
    )
    np.testing.assert_allclose(model, [14.0, 19.0])
    np.testing.assert_allclose(observed, [15.0, 20.0])
    assert metrics["usable_point_count"] == 2
    assert metrics["excluded_boundary_point_count"] == 1
