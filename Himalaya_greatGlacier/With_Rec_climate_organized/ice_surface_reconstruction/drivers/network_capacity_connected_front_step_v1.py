"""Capacity-constrained network step including one dry downstream front node."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_coupled_semiimplicit_system2005_step_v3 import _mixedbed_face_conductance
from .network_capacity_constrained_flux_fixture_v1 import project_flux_dag


def build_front_graph(flowlines, coefficients):
    branch_by_object = {id(fl): index for index, fl in enumerate(flowlines)}
    included_by_branch = []
    local = []
    global_of = {}
    for branch, fl in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active.size and np.any(np.diff(active) > 1):
            raise RuntimeError(f"preexisting-internal-gap:{branch}")
        included = list(map(int, active))
        if active.size and int(active[-1]) + 1 < fl.nx:
            included.append(int(active[-1]) + 1)
        included = np.asarray(sorted(set(included)), dtype=int)
        included_by_branch.append(included)
        for node in included:
            global_of[(branch, int(node))] = len(local)
            local.append((branch, int(node)))
    edges = []
    for branch, fl in enumerate(flowlines):
        included = included_by_branch[branch]
        aset = set(map(int, included))
        for node in included[:-1]:
            node = int(node)
            if node + 1 in aset:
                edges.append((global_of[(branch, node)], global_of[(branch, node + 1)], _mixedbed_face_conductance(coefficients, fl, node, fl, node + 1), "within"))
        active = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if fl.flows_to is not None and active.size and int(active[-1]) == fl.nx - 1:
            receiver = branch_by_object[id(fl.flows_to)]
            receiver_node = int(fl.flows_to_indice)
            if (receiver, receiver_node) not in global_of:
                raise RuntimeError(f"active-tributary-target-inactive:{branch}:{receiver}:{receiver_node}")
            terminal = int(active[-1])
            edges.append((global_of[(branch, terminal)], global_of[(receiver, receiver_node)], _mixedbed_face_conductance(coefficients, fl, terminal, flowlines[receiver], receiver_node), "junction"))
    return local, edges


def connected_front_step(flowlines, coefficients, dt_seconds: float, closure_limit_m3: float) -> dict:
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
    preferred = conductance * np.maximum(implicit_surface[u] - implicit_surface[v], 0.0)
    node_volume = np.asarray([flowlines[b].section[n] * flowlines[b].dx_meter for b, n in local], dtype=float)
    accepted, final_node_volume, limited = project_flux_dag(node_volume, u, v, preferred, dt_seconds)
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
    if min(float(np.min(section)) for section in sections) < -1e-8:
        raise RuntimeError("negative-section-after-front-step")
    for fl, section in zip(flowlines, sections):
        fl.section = section
    closure = base.total_volume(flowlines) - before
    if abs(closure) > closure_limit_m3:
        raise RuntimeError(f"front-global-closure:{closure}:{closure_limit_m3}")
    gaps = 0
    for fl in flowlines:
        active = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active.size:
            gaps += int(np.count_nonzero(np.diff(active) > 1))
    return {"active_node_count": count, "edge_count": len(edges), "limited_edge_count": int(np.count_nonzero(limited)), "actual_closure_m3": closure, "maximum_absolute_linear_residual_m3": maximum_residual, "minimum_section_m2": min(float(np.min(fl.section)) for fl in flowlines), "minimum_thickness_m": min(float(np.min(fl.thick)) for fl in flowlines), "internal_active_gap_count": gaps, "minimum_accepted_flux_m3_s": float(np.min(accepted)) if len(accepted) else 0.0}
