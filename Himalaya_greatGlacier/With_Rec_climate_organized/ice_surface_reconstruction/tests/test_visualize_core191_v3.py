from __future__ import annotations

import numpy as np
import pytest

from ice_surface_reconstruction.drivers.visualize_core191_v3 import (
    _pairwise_gradient_values,
    thickness_qc_metrics,
)


def test_pairwise_gradients_require_two_valid_cells() -> None:
    values = np.asarray([[0.0, 10.0], [20.0, 30.0]])
    valid = np.asarray([[True, True], [False, True]])
    gradients = _pairwise_gradient_values(values, valid, 10.0)
    np.testing.assert_allclose(np.sort(gradients), [1.0, 2.0])


def test_thickness_qc_reports_boundary_and_closure() -> None:
    thickness = np.zeros((5, 5), dtype=float)
    thickness[1:4, 1:4] = 10.0
    valid = np.ones((5, 5), dtype=bool)
    bed = np.full((5, 5), 100.0)
    surface = bed + thickness
    metrics = thickness_qc_metrics(thickness, surface, bed, valid, 10.0)
    assert metrics["maximum_thickness_m"] == 10.0
    assert metrics["boundary_maximum_thickness_m"] == 0.0
    assert metrics["maximum_surface_closure_error_m"] == 0.0
    assert metrics["material_negative_cell_count"] == 0


def test_thickness_qc_rejects_empty_mask() -> None:
    values = np.zeros((2, 2), dtype=float)
    with pytest.raises(ValueError, match="at least one valid"):
        thickness_qc_metrics(values, values, values, np.zeros((2, 2), bool), 10.0)
