"""Boundary-constrained paleo-ice surface reconstruction utilities.

Compatibility layer. The single implementation now lives in
``ice_surface_reconstruction``; this package only re-exports it. The bootstrap
below puts the repository root on ``sys.path`` so the new module is importable
even when only ``src/`` is on the path (e.g. legacy ``sys.path.insert`` callers).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .solver import (
    FlowlineSolution,
    SurfaceScenario,
    harmonic_extend_thickness,
    solve_flowline,
    surface_quality_metrics,
)
from .network import (
    FlowlineBranch,
    FlowlineNetworkSolution,
    JunctionDiagnostic,
    solve_flowline_network,
)

__all__ = [
    "FlowlineSolution",
    "SurfaceScenario",
    "harmonic_extend_thickness",
    "solve_flowline",
    "surface_quality_metrics",
    "FlowlineBranch",
    "FlowlineNetworkSolution",
    "JunctionDiagnostic",
    "solve_flowline_network",
]
