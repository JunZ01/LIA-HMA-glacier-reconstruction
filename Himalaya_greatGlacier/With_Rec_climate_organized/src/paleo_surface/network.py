"""Compatibility shim: orchestration now lives in ``ice_surface_reconstruction.core``.

Re-exports the single implementation so existing ``from paleo_surface.network
import ...`` call sites keep working. Do not add logic here; edit
``ice_surface_reconstruction/core/network.py`` instead.
"""

from ice_surface_reconstruction.core.network import (  # noqa: F401
    FlowlineBranch,
    FlowlineNetworkSolution,
    JunctionDiagnostic,
    solve_flowline_network,
)

__all__ = [
    "FlowlineBranch",
    "FlowlineNetworkSolution",
    "JunctionDiagnostic",
    "solve_flowline_network",
]
