"""Connected-front network step with exact-zero capacity arithmetic."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_connected_front_step_v1 import build_front_graph
from .network_capacity_constrained_flux_projection_v2 import project_flux_dag_exact


def connected_front_step_v2(flowlines, coefficients, dt_seconds: float, closure_limit_m3: float) -> dict:
    local, edges = build_front_graph(flowlines, coefficients)
    count = len(local)
    surface = np.asarray([flowlines[b].surface_h[n] for b, n in local], dtype=float)
    storage = np.asarray([flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in local], dtype=float)
    if np.any(storage <= 0) or np.any(~np.isfinite(storage)):
        raise RuntimeError("front-storage-nonpositive")
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
    if np.any(~np.isfinite(implicit_surface)):
        raise RuntimeError("front-implicit-solution-nonfinite")
    preferred = conductance * np.maximum(implicit_surface[u] - implicit_surface[v], 0.0)
    node_volume = np.asarray([flowlines[b].section[n] * flowlines[b].dx_meter for b, n in local], dtype=float)
    accepted, final_node_volume, limited, projection_conservation = project_flux_dag_exact(
        node_volume, u, v, preferred, dt_seconds
    )
    divergence = np.zeros(count)
    np.add.at(divergence, u, accepted)
    np.add.at(divergence, v, -accepted)
    residual = final_node_volume - node_volume + float(dt_seconds) * divergence
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > closure_limit_m3:
        raise RuntimeError(f"front-node-closure:{maximum_residual}:{closure_limit_m3}")
    before = base.total_volume(flowlines)
    sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    for index, (branch, node) in enumerate(local):
        sections[branch][node] = final_node_volume[index] / float(flowlines[branch].dx_meter)
    minimum_section = min(float(np.min(section)) for section in sections)
    if minimum_section < 0 or not all(np.isfinite(section).all() for section in sections):
        raise RuntimeError(f"invalid-section-after-front-step:{minimum_section}")
    try:
        with np.errstate(invalid="raise", over="raise"):
            for fl, section in zip(flowlines, sections):
                fl.section = section
    except FloatingPointError as exc:
        raise RuntimeError(f"mixedbed-section-setter-floating-error:{exc}") from exc
    closure = base.total_volume(flowlines) - before
    if abs(closure) > closure_limit_m3:
        raise RuntimeError(f"front-global-closure:{closure}:{closure_limit_m3}")
    gaps = 0
    for fl in flowlines:
        active = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active.size:
            gaps += int(np.count_nonzero(np.diff(active) > 1))
    finite = all(np.isfinite(fl.section).all() and np.isfinite(fl.thick).all() for fl in flowlines)
    if not finite:
        raise RuntimeError("nonfinite-after-front-section-setter")
    return {
        "active_node_count": count,
        "edge_count": len(edges),
        "limited_edge_count": int(np.count_nonzero(limited)),
        "actual_closure_m3": closure,
        "maximum_absolute_linear_residual_m3": maximum_residual,
        "projection_global_conservation_m3": projection_conservation,
        "minimum_section_m2": min(float(np.min(fl.section)) for fl in flowlines),
        "minimum_thickness_m": min(float(np.min(fl.thick)) for fl in flowlines),
        "internal_active_gap_count": gaps,
        "minimum_accepted_flux_m3_s": float(np.min(accepted)) if len(accepted) else 0.0,
        "numerical_zero_canonicalization_volume_m3": 0.0,
    }

