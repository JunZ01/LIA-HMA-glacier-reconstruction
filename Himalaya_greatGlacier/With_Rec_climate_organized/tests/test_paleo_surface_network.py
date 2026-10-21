from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paleo_surface import (  # noqa: E402
    FlowlineBranch,
    SurfaceScenario,
    solve_flowline_network,
)


def branch(
    branch_id: str,
    downstream: str,
    upstream: str,
    bed: list[float],
) -> FlowlineBranch:
    return FlowlineBranch(
        branch_id,
        downstream,
        upstream,
        np.array([0.0, 1000.0]),
        np.asarray(bed, dtype=float),
    )


def test_network_solves_trunk_before_two_upstream_branches() -> None:
    branches = [
        branch("tributary", "junction", "tributary-head", [40.0, 180.0]),
        branch("main-upper", "junction", "main-head", [50.0, 160.0]),
        branch("main-lower", "terminus", "junction", [0.0, 50.0]),
    ]
    result = solve_flowline_network(
        branches,
        SurfaceScenario("test", 80_000.0, 0.8),
        outlet_node_ids={"terminus"},
    )
    assert result.success
    assert result.diagnostics["solve_order"][0] == "main-lower"
    assert result.diagnostics["junction_count"] == 1
    junction = result.junction_diagnostics[0]
    assert junction.node_id == "junction"
    assert junction.maximum_surface_mismatch_m == 0.0
    assert junction.status == "surface-continuous-flux-not-evaluated"
    inherited = result.node_surface_m["junction"]
    assert result.branch_solutions["main-upper"].surface_elevation_m[0] == inherited
    assert result.branch_solutions["tributary"].surface_elevation_m[0] == inherited


def test_network_reports_hanging_or_misregistered_branch_failure() -> None:
    branches = [
        branch("main-lower", "terminus", "junction", [0.0, 50.0]),
        branch("high-tributary", "junction", "tributary-head", [1000.0, 1100.0]),
    ]
    result = solve_flowline_network(
        branches,
        SurfaceScenario("test", 80_000.0, 0.8),
        outlet_node_ids={"terminus"},
    )
    assert result.status == "branch-failed:negative-start-thickness"
    assert result.diagnostics["failed_branch_id"] == "high-tributary"


def test_network_rejects_cycle_or_disconnected_graph() -> None:
    branches = [
        branch("a-to-b", "a", "b", [0.0, 10.0]),
        branch("b-to-a", "b", "a", [10.0, 20.0]),
    ]
    result = solve_flowline_network(
        branches,
        SurfaceScenario("test", 80_000.0, 0.8),
        outlet_node_ids={"terminus"},
    )
    assert result.status == "invalid-network-cycle-or-disconnected"
