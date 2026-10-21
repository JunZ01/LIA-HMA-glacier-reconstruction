"""Pure diagnostic contracts for the v1 failure dossier."""

from __future__ import annotations

import numpy as np

from ice_surface_reconstruction.core import FlowlineBranch
from ice_surface_reconstruction.drivers.flowline_step_convergence import refine_branch
from ice_surface_reconstruction.drivers.lia_failure_diagnostics_v1 import (
    _nearest_margin_edge,
    _signed_distance_outside_positive,
)


def test_signed_margin_distance_uses_outside_positive_convention() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    assert _signed_distance_outside_positive(mask, 2, 2, 100.0) < 0
    assert _signed_distance_outside_positive(mask, 0, 0, 100.0) > 0


def test_margin_edge_decomposition_closes_without_smoothing() -> None:
    topo = np.full((3, 4), 1000.0)
    thickness = np.zeros((3, 4), dtype=float)
    mask = np.zeros((3, 4), dtype=bool)
    mask[:, :2] = True
    topo[:, 2:] = 900.0
    thickness[:, 1] = 80.0
    result = _nearest_margin_edge(topo, thickness, mask, 1, 2)
    assert result["seam_edge_found"] is True
    assert np.isclose(result["terrain_step_m"], -100.0)
    assert np.isclose(result["edge_thickness_m"], 80.0)
    assert np.isclose(result["spliced_bed_step_m"], -20.0)
    assert np.isclose(result["decomposition_closure_m"], 0.0)


def test_margin_edge_missing_when_no_mask_transition() -> None:
    result = _nearest_margin_edge(
        np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3), dtype=bool), 1, 1
    )
    assert result["seam_edge_found"] is False
    assert np.isnan(result["seam_edge_distance_pixels"])


def test_refine_branch_preserves_original_polyline_nodes() -> None:
    branch = FlowlineBranch(
        "b", "out", "head",
        np.array([0.0, 100.0, 300.0]),
        np.array([1000.0, 1100.0, 1200.0]),
    )
    refined = refine_branch(branch, 2)
    np.testing.assert_allclose(refined.distance_m, [0, 50, 100, 200, 300])
    np.testing.assert_allclose(refined.bed_elevation_m, [1000, 1050, 1100, 1150, 1200])
    assert refined.downstream_node_id == branch.downstream_node_id
    assert refined.upstream_node_id == branch.upstream_node_id


def test_refine_branch_rejects_non_positive_factor() -> None:
    branch = FlowlineBranch("b", "out", "head", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    try:
        refine_branch(branch, 0)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("factor=0 should fail")
