"""Compatibility shim: the solver now lives in ``ice_surface_reconstruction.core``.

This module re-exports the single implementation so existing
``from paleo_surface.solver import ...`` call sites keep working. Do not add
logic here; edit ``ice_surface_reconstruction/core/solver.py`` instead.
"""

from ice_surface_reconstruction.core.solver import (  # noqa: F401
    FlowlineSolution,
    SurfaceScenario,
    harmonic_extend_thickness,
    solve_flowline,
    surface_quality_metrics,
)

__all__ = [
    "FlowlineSolution",
    "SurfaceScenario",
    "harmonic_extend_thickness",
    "solve_flowline",
    "surface_quality_metrics",
]
