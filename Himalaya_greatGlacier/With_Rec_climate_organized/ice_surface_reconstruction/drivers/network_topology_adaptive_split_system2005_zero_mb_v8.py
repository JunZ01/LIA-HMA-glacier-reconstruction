"""V8: scalar-order-preserving face coefficients with graph-structure reuse."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags

from . import network_topology_adaptive_split_system2005_zero_mb_v6 as cached
from . import network_topology_adaptive_split_system2005_zero_mb_v7 as structure_v7


def reset_structure_cache() -> None:
    structure_v7.reset_structure_cache()


def _assemble(flowlines, coefficients, dt_seconds):
    branch_cache = cached._branch_cache(flowlines)
    structure = structure_v7._structure(flowlines, branch_cache)
    active_local = structure["active_local"]
    surface = np.asarray(
        [branch_cache[b]["surface_h"][n] for b, n in active_local], dtype=float
    )
    storage = np.asarray(
        [
            branch_cache[b]["widths_m"][n] * branch_cache[b]["dx_meter"]
            for b, n in active_local
        ],
        dtype=float,
    )
    conductance = []
    for _edge_slice, upstream_branch, upstream_nodes, downstream_branch, downstream_nodes in structure[
        "coefficient_groups"
    ]:
        for upstream_node, downstream_node in zip(upstream_nodes, downstream_nodes):
            conductance.append(
                cached._face_conductance(
                    coefficients,
                    branch_cache[upstream_branch],
                    int(upstream_node),
                    branch_cache[downstream_branch],
                    int(downstream_node),
                )
            )
    value = float(dt_seconds) * np.asarray(conductance, dtype=float)
    graph_data = np.column_stack((value, value, -value, -value)).ravel()
    graph = coo_matrix(
        (graph_data, (structure["graph_row"], structure["graph_col"])),
        shape=(len(active_local), len(active_local)),
    ).tocsr()
    matrix = diags(storage, format="csr") + graph
    rhs = storage * surface
    return (
        active_local,
        range(structure["edge_count"]),
        surface,
        storage,
        matrix,
        rhs,
        structure["incidence"],
        branch_cache,
    )


def conservative_boundary_step(flowlines, coefficients, dt_seconds, closure_limit, section_floor):
    original_assemble = cached._assemble
    cached._assemble = _assemble
    cached.conservative_boundary_step.config = conservative_boundary_step.config
    try:
        result = cached.conservative_boundary_step(
            flowlines, coefficients, dt_seconds, closure_limit, section_floor
        )
    finally:
        cached._assemble = original_assemble
    result["mixedbed_property_access"] = (
        "all-dynamic-arrays-cached; scalar-face-order-preserved; support-keyed-graph-reuse"
    )
    return result
