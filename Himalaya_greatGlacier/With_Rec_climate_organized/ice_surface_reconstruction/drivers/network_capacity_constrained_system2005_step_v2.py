"""Capacity step v2 with an explicit node-wise conservation residual."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_constrained_flux_fixture_v1 import project_flux_dag


def capacity_one_step_v2(flowlines, coefficients, dt_seconds: float, closure_limit_m3: float) -> dict:
    active_local, edges = base.build_active_graph(flowlines, coefficients)
    count = len(active_local)
    surface = np.asarray([flowlines[b].surface_h[n] for b, n in active_local], dtype=float)
    storage = np.asarray([flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in active_local], dtype=float)
    u = np.asarray([edge[0] for edge in edges], dtype=int)
    v = np.asarray([edge[1] for edge in edges], dtype=int)
    conductance = np.asarray([edge[2] for edge in edges], dtype=float)
    forward = (surface[u] - surface[v]) > 1e-12
    selected = np.flatnonzero(forward)
    values = float(dt_seconds) * conductance[selected]
    rows = np.concatenate([u[selected], v[selected], u[selected], v[selected]])
    cols = np.concatenate([u[selected], v[selected], v[selected], u[selected]])
    data = np.concatenate([values, values, -values, -values])
    matrix = diags(storage, format="csr") + coo_matrix((data, (rows, cols)), shape=(count, count)).tocsr()
    implicit_surface = np.asarray(spsolve(matrix, storage * surface), dtype=float)
    preferred = conductance * np.maximum(implicit_surface[u] - implicit_surface[v], 0.0)
    node_volume = np.asarray([flowlines[b].section[n] * flowlines[b].dx_meter for b, n in active_local], dtype=float)
    accepted, final_node_volume, limited = project_flux_dag(node_volume, u, v, preferred, dt_seconds)
    divergence_rate = np.zeros(count, dtype=float)
    np.add.at(divergence_rate, u, accepted)
    np.add.at(divergence_rate, v, -accepted)
    node_residual = final_node_volume - node_volume + float(dt_seconds) * divergence_rate
    maximum_node_residual = float(np.max(np.abs(node_residual)))
    if maximum_node_residual > closure_limit_m3:
        raise RuntimeError(f"node-volume-closure:{maximum_node_residual}:{closure_limit_m3}")
    before_volume = base.total_volume(flowlines)
    sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    for global_index, (branch, node) in enumerate(active_local):
        sections[branch][node] = final_node_volume[global_index] / float(flowlines[branch].dx_meter)
    minimum_section = min(float(np.min(section)) for section in sections)
    if minimum_section < -1e-8:
        raise RuntimeError(f"negative-section-after-capacity:{minimum_section}")
    for fl, section in zip(flowlines, sections):
        fl.section = section
    after_volume = base.total_volume(flowlines)
    closure = after_volume - before_volume
    if abs(closure) > closure_limit_m3:
        raise RuntimeError(f"volume-closure:{closure}:{closure_limit_m3}")
    internal_gaps = 0
    for fl in flowlines:
        active_nodes = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active_nodes.size:
            internal_gaps += int(np.count_nonzero(np.diff(active_nodes) > 1))
    return {
        "active_node_count": count,
        "edge_count": len(edges),
        "preferred_flux_total_m3_s": float(np.sum(preferred)),
        "accepted_flux_total_m3_s": float(np.sum(accepted)),
        "limited_edge_count": int(np.count_nonzero(limited)),
        "maximum_flux_reduction_m3_s": float(np.max(preferred - accepted)) if len(edges) else 0.0,
        "actual_closure_m3": closure,
        "maximum_absolute_linear_residual_m3": maximum_node_residual,
        "minimum_section_m2": minimum_section,
        "minimum_thickness_m": min(float(np.min(fl.thick)) for fl in flowlines),
        "internal_active_gap_count": internal_gaps,
        "minimum_accepted_flux_m3_s": float(np.min(accepted)) if len(edges) else 0.0,
    }
