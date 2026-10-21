"""Tests for the junction-gap census driver and the flowline IO helper (task #15).

The real-glacier tests skip automatically when the pilot OGGM data is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ice_surface_reconstruction.drivers.junction_gap_census import census_rows  # noqa: E402
from ice_surface_reconstruction.drivers.oggm_flowline_io import (  # noqa: E402
    load_flowline_inputs,
)


PILOT_ROOT = (
    ROOT
    / "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06"
    / "stage0_bed_full_farinotti_v0/oggm_workdir/per_glacier"
)


def _gdir(rgi_id: str) -> Path | None:
    d = PILOT_ROOT / rgi_id[:-6] / rgi_id[:-3] / rgi_id
    return d if (d / "inversion_flowlines.pkl").exists() else None


def test_missing_flowlines_recorded_not_raised() -> None:
    rows = census_rows("RGI60-99.99999", ROOT / "does-not-exist", ["farinotti2019"])
    assert rows == [{"rgi_id": "RGI60-99.99999", "status": "missing-inversion-flowlines"}]


def test_load_flowline_inputs_recovers_junction_indices() -> None:
    gdir = _gdir("RGI60-15.01141")
    if gdir is None:
        pytest.skip("pilot OGGM data not present")
    inputs, fls = load_flowline_inputs(gdir)
    assert len(inputs) == 3
    # Known topology (memory): fl0 -> fl2[35], fl1 -> fl2[4], fl2 -> outlet.
    by_index = {fi.flowline_index: fi for fi in inputs}
    assert by_index[2].flows_to is None
    trib_ftis = sorted(
        fi.flows_to_indice for fi in inputs if fi.flows_to_indice is not None
    )
    assert trib_ftis == [4, 35]
    # coords_xy are real grid-cell coordinates, not None (line.coords not .xy).
    assert by_index[0].coords_xy.shape[1] == 2
    assert np.isfinite(by_index[0].coords_xy).all()


def test_single_line_system_has_no_junctions() -> None:
    gdir = _gdir("RGI60-15.10689")
    if gdir is None:
        pytest.skip("pilot OGGM data not present")
    rows = census_rows("RGI60-15.10689", gdir, ["farinotti2019"])
    assert len(rows) == 1
    assert rows[0]["status"] == "no-junctions"


def test_real_01141_census_two_junctions_connectors_ok() -> None:
    gdir = _gdir("RGI60-15.01141")
    if gdir is None:
        pytest.skip("pilot OGGM data not present")
    rows = census_rows("RGI60-15.01141", gdir, ["farinotti2019"])
    # 2 junctions x 1 bed scenario.
    assert len(rows) == 2
    for row in rows:
        assert row["connector_status"] == "ok"
        assert row["p_inside_mask"] and row["t_inside_mask"]
        # Measured gap ~475-500 m (spec 14 / memory); connector length is real.
        assert 400.0 < row["junction_distance_m"] < 600.0
        assert row["connector_length_m"] > 0.0
