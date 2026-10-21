"""Tests for full-network validation helpers and pilot run (task #16)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ice_surface_reconstruction.core import SurfaceScenario, solve_flowline_network  # noqa: E402
from ice_surface_reconstruction.drivers.full_network_validation import (  # noqa: E402
    _collapsed_vs_connector_diff,
    _real_bed_inputs,
    _unique_point_errors,
    run,
    run_policy,
)
from ice_surface_reconstruction.drivers.oggm_network_adapter import (  # noqa: E402
    FlowlineInput,
    build_network,
)


def _line(nx: int) -> np.ndarray:
    x = np.arange(nx, dtype=float) * 2.0
    return np.column_stack([x, np.zeros(nx)])


def _fl(idx, flows_to, fti, bed, surface):
    return FlowlineInput(
        flowline_index=idx, flows_to=flows_to, flows_to_indice=fti,
        bed_elevation_m=np.asarray(bed, float),
        reference_surface_m=np.asarray(surface, float),
        dx_meter=132.0, map_dx_m=66.0, coords_xy=_line(len(bed)),
    )


def test_unique_point_errors_dedups_and_excludes_connectors() -> None:
    # Parent (10 pts) + one tributary at native 4. Collapsed policy: 5 branches,
    # no connector. Truth = bed + fixed thickness so surface is well-defined.
    parent_bed = np.linspace(300.0, 0.0, 10)
    trib_bed = np.linspace(320.0, 120.0, 5)
    parent = _fl(1, None, None, parent_bed, parent_bed + 50.0)
    trib = _fl(0, 1, 4, trib_bed, trib_bed + 50.0)
    build = build_network([trib, parent])
    truth = {
        1: (parent_bed + 50.0, np.full(10, 50.0)),
        0: (trib_bed + 50.0, np.full(5, 50.0)),
    }
    solution = solve_flowline_network(
        build.branches, SurfaceScenario("c", 85_000.0, 0.85),
        outlet_node_ids=build.outlet_node_ids,
    )
    assert solution.success
    surf_err, thick_err, n_pts = _unique_point_errors(build, solution, truth)
    # Unique OGGM points = 10 (parent) + 5 (trib) = 15; the shared split point
    # fl1[4] is counted once, not twice.
    assert n_pts == 15
    assert len(surf_err) == 15


def test_diff_categories_cover_all_outcomes() -> None:
    keys = ["rgi_id", "bed_scenario", "stress_scenario"]
    collapsed = pd.DataFrame([
        {**dict(zip(keys, ["A", "f", "central"])), "status": "success", "surface_rmse_m": 30.0},
        {**dict(zip(keys, ["B", "f", "central"])), "status": "failed:x", "surface_rmse_m": np.nan},
        {**dict(zip(keys, ["C", "f", "central"])), "status": "failed:x", "surface_rmse_m": np.nan},
    ])
    explicit = pd.DataFrame([
        {**dict(zip(keys, ["A", "f", "central"])), "status": "success", "surface_rmse_m": 34.0},
        {**dict(zip(keys, ["B", "f", "central"])), "status": "success", "surface_rmse_m": 40.0},
        {**dict(zip(keys, ["C", "f", "central"])), "status": "failed:y", "surface_rmse_m": np.nan},
    ])
    diff = _collapsed_vs_connector_diff(collapsed, explicit)
    outcomes = dict(zip(diff["rgi_id"], diff["outcome"]))
    assert outcomes["A"] == "both-solved"
    assert outcomes["B"] == "connector-only-solved"
    assert outcomes["C"] == "neither-solved"
    # Delta only meaningful where both solved.
    a_row = diff[diff["rgi_id"] == "A"].iloc[0]
    assert a_row["surface_rmse_delta_m"] == pytest.approx(4.0)


# --- real pilot 3-way run ---------------------------------------------------


def _have_pilot() -> bool:
    root = (
        ROOT / "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06"
        / "stage0_bed_full_farinotti_v0/oggm_workdir/per_glacier"
    )
    return (root / "RGI60-15/RGI60-15.01/RGI60-15.01141/inversion_flowlines.pkl").exists()


def _config() -> dict:
    return json.loads(
        (ROOT / "config/paleo_surface_validation_v0.json").read_text(encoding="utf-8")
    )


def test_real_pilot_connector_never_worse_solvability() -> None:
    if not _have_pilot():
        pytest.skip("pilot OGGM data not present")
    result = run(_config())
    counts = result["diff"]["outcome"].value_counts().to_dict()
    # The core evidence: explicit-connector solves a strict superset of what
    # collapsed solves. collapsed-only-solved MUST be zero.
    assert counts.get("collapsed-only-solved", 0) == 0
    assert counts.get("connector-only-solved", 0) > 0


def test_real_pilot_failure_semantics_flagged() -> None:
    if not _have_pilot():
        pytest.skip("pilot OGGM data not present")
    glacier_rows, _, _ = run_policy(_config(), "collapsed-transfer-node")
    failed = glacier_rows[glacier_rows["status"].astype(str).str.startswith("failed:")]
    if not failed.empty:
        # Failed systems carry no aggregate RMSE and are flagged partial.
        assert failed["surface_rmse_m"].isna().all()
        assert failed["partial_solution"].all()
