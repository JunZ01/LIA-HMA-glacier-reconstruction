from __future__ import annotations

import numpy as np

from ice_surface_reconstruction.drivers.lia_design_b_target_and_input_audit import (
    weighted_quantiles,
)


def test_weighted_quantiles_follow_area_weights() -> None:
    result = weighted_quantiles(
        [100.0, 200.0, 300.0],
        [1.0, 8.0, 1.0],
        (0.5,),
    )
    assert result[0] == 200.0


def test_weighted_quantiles_drop_nonfinite_values() -> None:
    result = weighted_quantiles(
        [100.0, np.nan, 300.0],
        [1.0, 1.0, 1.0],
        (0.5,),
    )
    assert result[0] == 200.0
