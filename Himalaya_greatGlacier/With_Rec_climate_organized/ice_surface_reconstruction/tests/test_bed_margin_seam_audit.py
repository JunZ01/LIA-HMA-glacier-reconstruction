"""Tests for decomposition and diagnostic-only bed-margin tapers."""

from __future__ import annotations

import numpy as np

from ice_surface_reconstruction.drivers.bed_margin_seam_audit import (
    audit_scenario,
    margin_edge_components,
    taper_thickness_inside_margin,
)


def test_margin_decomposition_is_signed_outward() -> None:
    topo = np.array([[100.0, 90.0, 80.0]])
    mask = np.array([[True, True, False]])
    thickness = np.array([[30.0, 20.0, 0.0]])
    parts = margin_edge_components(topo, thickness, mask, np.ones_like(mask))
    np.testing.assert_allclose(parts["terrain_step_outward_m"], [-10.0])
    np.testing.assert_allclose(parts["thickness_cutoff_m"], [20.0])
    np.testing.assert_allclose(parts["bed_step_outward_m"], [10.0])


def test_margin_decomposition_is_independent_of_array_orientation() -> None:
    topo = np.array([[80.0, 90.0, 100.0]])
    mask = np.array([[False, True, True]])
    thickness = np.array([[0.0, 20.0, 30.0]])
    parts = margin_edge_components(topo, thickness, mask, np.ones_like(mask))
    np.testing.assert_allclose(parts["terrain_step_outward_m"], [-10.0])
    np.testing.assert_allclose(parts["bed_step_outward_m"], [10.0])


def test_taper_zeroes_boundary_and_preserves_deep_interior() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    thickness = np.where(mask, 100.0, 0.0)
    tapered, weight = taper_thickness_inside_margin(
        thickness, mask, map_dx_m=10.0, taper_width_m=10.0
    )
    assert tapered[1, 3] == 0.0
    assert tapered[3, 3] == 100.0
    assert weight[0, 0] == 0.0


def test_audit_does_not_mutate_inputs_and_closes_decomposition() -> None:
    topo = np.tile(np.arange(7, dtype=float), (7, 1)) * 5.0 + 100.0
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    thickness = np.where(mask, 40.0, 0.0)
    bed = topo - thickness
    topo_before = topo.copy()
    bed_before = bed.copy()
    rows = audit_scenario(
        topo, bed, mask, np.ones_like(mask), 10.0, [10.0, 20.0]
    )
    np.testing.assert_array_equal(topo, topo_before)
    np.testing.assert_array_equal(bed, bed_before)
    assert len(rows) == 3
    assert rows[0]["decomposition_closure_max_abs_m"] == 0.0
    assert rows[1]["candidate_boundary_bed_step_abs_p95_m"] == 5.0
    assert rows[1]["retained_lia_modern_volume_fraction"] < 1.0
