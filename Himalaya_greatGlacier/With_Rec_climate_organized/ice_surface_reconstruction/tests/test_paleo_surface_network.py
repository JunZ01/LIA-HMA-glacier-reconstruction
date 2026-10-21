from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ice_surface_reconstruction.core import (  # noqa: E402
    FlowlineBranch,
    SurfaceScenario,
    refine_flowline_network,
    sample_network_solution_at_original_nodes,
    solve_flowline,
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


def _single_branch_equivalence(distance, bed, scenario) -> None:
    """A one-branch network must reproduce solve_flowline point-for-point."""
    direct = solve_flowline(distance, bed, scenario)
    network = solve_flowline_network(
        [FlowlineBranch("only", "terminus", "head", distance, bed)],
        scenario,
        outlet_node_ids={"terminus"},
    )
    assert network.success == direct.success
    assert network.diagnostics["junction_count"] == 0
    branch_solution = network.branch_solutions["only"]
    np.testing.assert_array_equal(
        branch_solution.surface_elevation_m, direct.surface_elevation_m
    )
    np.testing.assert_array_equal(
        branch_solution.thickness_m, direct.thickness_m
    )


def test_single_branch_matches_solver_flat_bed() -> None:
    distance = np.linspace(0.0, 10_000.0, 201)
    _single_branch_equivalence(
        distance, np.zeros_like(distance), SurfaceScenario("flat", 100_000.0, 0.8)
    )


def test_single_branch_matches_solver_sloping_undulating_bed() -> None:
    distance = np.linspace(0.0, 8_000.0, 161)
    bed = 2500.0 + 0.03 * distance + 20.0 * np.sin(distance / 700.0)
    _single_branch_equivalence(
        distance, bed, SurfaceScenario("central", 90_000.0, 0.85)
    )


def test_sampled_network_recomputes_consistent_coarse_diagnostics() -> None:
    original = [FlowlineBranch(
        "only",
        "terminus",
        "head",
        np.array([0.0, 200.0, 500.0]),
        np.array([0.0, 15.0, 45.0]),
    )]
    scenario = SurfaceScenario("sample", 85_000.0, 0.82)
    refined = solve_flowline_network(
        refine_flowline_network(original, 4),
        scenario,
        outlet_node_ids={"terminus"},
    )
    sampled = sample_network_solution_at_original_nodes(original, refined, 4, scenario)
    solution = sampled.branch_solutions["only"]
    expected_slope = np.diff(solution.surface_elevation_m) / np.diff(solution.distance_m)
    np.testing.assert_allclose(solution.surface_slope[1:], expected_slope)
    assert solution.diagnostics["sampled_at_original_nodes"] is True
    assert np.isfinite(solution.driving_shear_stress_pa).all()
