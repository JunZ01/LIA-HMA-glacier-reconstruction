from __future__ import annotations

import pandas as pd
import pytest

from ice_surface_reconstruction.drivers.lia_elevation_band_pilot import (
    load_main_flowline_length_targets,
)


@pytest.mark.parametrize(
    "column",
    ["target_main_flowline_length_m", "main_centerline_inside_length_m"],
)
def test_load_main_flowline_length_targets_accepts_supported_columns(
    tmp_path, column: str
) -> None:
    path = tmp_path / "targets.csv"
    pd.DataFrame(
        {"lia_unique_id": [7, 3584], column: [12000.0, 24146.8]}
    ).to_csv(path, index=False)

    targets, source_column = load_main_flowline_length_targets(path)

    assert source_column == column
    assert targets.to_dict() == {7: 12000.0, 3584: 24146.8}


def test_load_main_flowline_length_targets_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "lia_unique_id": [7, 7],
            "target_main_flowline_length_m": [12000.0, 12100.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="ids-not-unique"):
        load_main_flowline_length_targets(path)
