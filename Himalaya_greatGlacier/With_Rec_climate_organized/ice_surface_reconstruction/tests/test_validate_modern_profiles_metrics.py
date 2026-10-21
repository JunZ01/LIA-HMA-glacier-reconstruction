from __future__ import annotations

import numpy as np

from ice_surface_reconstruction.drivers.validate_modern_profiles import metrics


def test_zero_truth_thickness_ratios_are_explicitly_not_evaluated() -> None:
    result = metrics(
        np.array([100.0, 110.0]),
        np.zeros(2),
        np.array([100.0, 111.0]),
        np.array([0.0, 1.0]),
    )
    assert np.isnan(result["mean_thickness_ratio"])
    assert np.isnan(result["maximum_thickness_ratio"])
    assert result["mean_thickness_ratio_status"] == (
        "not-evaluated-zero-truth-mean"
    )
    assert result["maximum_thickness_ratio_status"] == (
        "not-evaluated-zero-truth-maximum"
    )
