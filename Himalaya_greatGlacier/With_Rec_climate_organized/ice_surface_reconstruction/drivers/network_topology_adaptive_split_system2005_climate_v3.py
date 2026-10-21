"""Climate V3: V2 formal gates with the one-step-equivalent cached V5 solver."""

from __future__ import annotations

from . import network_topology_adaptive_split_system2005_climate_v1 as climate_v1
from . import network_topology_adaptive_split_system2005_climate_v2 as climate_v2
from . import network_topology_adaptive_split_system2005_zero_mb_v5 as cached_solver


def run(config):
    original = climate_v1.solver.conservative_boundary_step
    cached_solver.conservative_boundary_step.config = config
    climate_v1.solver.conservative_boundary_step = cached_solver.conservative_boundary_step
    try:
        summary, rows, flowlines = climate_v2.run(config)
    finally:
        climate_v1.solver.conservative_boundary_step = original
    summary["schema"] = "lia-191-network-topology-adaptive-split-system2005-climate-v3"
    summary["mixedbed_property_access"] = "cached-once-per-branch-per-step"
    summary["cached_step_equivalence_gate"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_system2005_cached_step_equivalence_v1/summary.json"
    )
    return summary, rows, flowlines
