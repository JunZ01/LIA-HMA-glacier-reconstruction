"""Connected-front finite-volume step preserving the current ice support."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_connected_front_step_v1 import build_front_graph
from .network_connected_support_projection_v1 import project_flux_dag_with_reserve


MINIMUM_SUPPORT_THICKNESS_M = 2.0 * base.ACTIVE_TOLERANCE_M


def minimum_section_for_support(fl, node: int) -> float:
    h = MINIMUM_SUPPORT_THICKNESS_M
    shape = float(np.asarray(fl.bed_shape)[node])
    if np.isfinite(shape):
        return float((4.0 / 3.0) * np.sqrt(1.0 / shape) * h ** 1.5)
    w0 = float(np.asarray(fl._w0_m)[node])
    lam = float(np.asarray(fl._lambdas)[node])
    return float(w0 * h + 0.5 * lam * h * h)


def connected_support_step(flowlines, coefficients, dt_seconds: float, closure_limit_m3: float) -> dict:
    local, edges = build_front_graph(flowlines, coefficients)
    count = len(local)
    surface = np.asarray([flowlines[b].surface_h[n] for b, n in local], dtype=float)
    storage = np.asarray([flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in local], dtype=float)
    if np.any(storage <= 0) or np.any(~np.isfinite(storage)):
        raise RuntimeError("support-storage-nonpositive")
    u = np.asarray([edge[0] for edge in edges], dtype=int)
    v = np.asarray([edge[1] for edge in edges], dtype=int)
    conductance = np.asarray([edge[2] for edge in edges], dtype=float)
    selected = np.flatnonzero((surface[u] - surface[v]) > 1e-12)
    values = float(dt_seconds) * conductance[selected]
    rows = np.concatenate([u[selected], v[selected], u[selected], v[selected]])
    cols = np.concatenate([u[selected], v[selected], v[selected], u[selected]])
    data = np.concatenate([values, values, -values, -values])
    matrix = diags(storage, format="csr") + coo_matrix((data, (rows, cols)), shape=(count, count)).tocsr()
    implicit_surface = np.asarray(spsolve(matrix, storage * surface), dtype=float)
    if np.any(~np.isfinite(implicit_surface)):
        raise RuntimeError("support-implicit-solution-nonfinite")
    preferred = conductance * np.maximum(implicit_surface[u] - implicit_surface[v], 0.0)
    node_volume = np.asarray([flowlines[b].section[n] * flowlines[b].dx_meter for b, n in local], dtype=float)
    reserve = np.asarray([
        minimum_section_for_support(flowlines[b], n) * flowlines[b].dx_meter
        if flowlines[b].thick[n] > base.ACTIVE_TOLERANCE_M else 0.0
        for b, n in local
    ], dtype=float)
    accepted, final_node_volume, limited, projection_conservation = project_flux_dag_with_reserve(
        node_volume, reserve, u, v, preferred, dt_seconds
    )
    divergence = np.zeros(count)
    np.add.at(divergence, u, accepted)
    np.add.at(divergence, v, -accepted)
    residual = final_node_volume - node_volume + float(dt_seconds) * divergence
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > closure_limit_m3:
        raise RuntimeError(f"support-node-closure:{maximum_residual}:{closure_limit_m3}")
    before = base.total_volume(flowlines)
    sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    for index, (branch, node) in enumerate(local):
        sections[branch][node] = final_node_volume[index] / float(flowlines[branch].dx_meter)
    if min(float(np.min(section)) for section in sections) < 0:
        raise RuntimeError("negative-section-after-support-step")
    with np.errstate(invalid="raise", over="raise"):
        for fl, section in zip(flowlines, sections):
            fl.section = section
    closure = base.total_volume(flowlines) - before
    if abs(closure) > closure_limit_m3:
        raise RuntimeError(f"support-global-closure:{closure}:{closure_limit_m3}")
    gaps = 0
    for fl in flowlines:
        active = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active.size:
            gaps += int(np.count_nonzero(np.diff(active) > 1))
    if gaps:
        raise RuntimeError(f"support-step-created-internal-gap:{gaps}")
    if not all(np.isfinite(fl.section).all() and np.isfinite(fl.thick).all() for fl in flowlines):
        raise RuntimeError("nonfinite-after-support-step")
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
        "minimum_reserved_volume_m3": float(np.min(reserve[reserve > 0])) if np.any(reserve > 0) else 0.0,
        "support_limited_edge_count": int(np.count_nonzero(limited)),
    }
