"""Numerical core for boundary-constrained paleo-surface reconstruction."""

from .network import (
    FlowlineBranch,
    FlowlineNetworkSolution,
    JunctionDiagnostic,
    refine_flowline_branch,
    refine_flowline_network,
    sample_network_solution_at_original_nodes,
    solve_flowline_network,
)
from .solver import (
    FlowlineSolution,
    SurfaceScenario,
    flowline_stress_diagnostics,
    harmonic_extend_thickness,
    solve_flowline,
    surface_quality_metrics,
)

__all__ = [
    "FlowlineBranch",
    "FlowlineNetworkSolution",
    "JunctionDiagnostic",
    "refine_flowline_branch",
    "refine_flowline_network",
    "sample_network_solution_at_original_nodes",
    "FlowlineSolution",
    "SurfaceScenario",
    "flowline_stress_diagnostics",
    "harmonic_extend_thickness",
    "solve_flowline",
    "solve_flowline_network",
    "surface_quality_metrics",
]
