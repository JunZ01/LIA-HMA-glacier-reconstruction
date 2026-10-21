"""V7: vectorized face coefficients with support-keyed graph-structure reuse."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v6 as cached


_LAST_SIGNATURE = None
_LAST_STRUCTURE = None


def reset_structure_cache() -> None:
    global _LAST_SIGNATURE, _LAST_STRUCTURE
    _LAST_SIGNATURE = None
    _LAST_STRUCTURE = None


def _support_signature(branch_cache):
    return tuple(
        np.packbits(values["thick"] > base.ACTIVE_TOLERANCE_M).tobytes()
        for values in branch_cache
    )


def _build_structure(flowlines, branch_cache):
    branch_by_object = {id(flowline): index for index, flowline in enumerate(flowlines)}
    active_by_branch = [
        np.flatnonzero(values["thick"] > base.ACTIVE_TOLERANCE_M)
        for values in branch_cache
    ]
    global_by_branch = []
    active_local = []
    next_global = 0
    for branch, (values, active) in enumerate(zip(branch_cache, active_by_branch)):
        global_index = np.full(len(values["thick"]), -1, dtype=np.int64)
        global_index[active] = np.arange(next_global, next_global + len(active), dtype=np.int64)
        global_by_branch.append(global_index)
        active_local.extend((branch, int(node)) for node in active)
        next_global += len(active)

    upstream_global_parts = []
    downstream_global_parts = []
    coefficient_groups = []
    edge_offset = 0
    for branch, (flowline, active) in enumerate(zip(flowlines, active_by_branch)):
        if len(active) > 1:
            upstream_node = active[:-1][np.diff(active) == 1]
        else:
            upstream_node = np.empty(0, dtype=np.int64)
        if len(upstream_node):
            downstream_node = upstream_node + 1
            count = len(upstream_node)
            upstream_global_parts.append(global_by_branch[branch][upstream_node])
            downstream_global_parts.append(global_by_branch[branch][downstream_node])
            coefficient_groups.append((
                slice(edge_offset, edge_offset + count),
                branch,
                upstream_node,
                branch,
                downstream_node,
            ))
            edge_offset += count
        if (
            flowline.flows_to is not None
            and len(active)
            and int(active[-1]) == len(branch_cache[branch]["thick"]) - 1
        ):
            receiver = branch_by_object[id(flowline.flows_to)]
            receiver_node = int(flowline.flows_to_indice)
            terminal = int(active[-1])
            if global_by_branch[receiver][receiver_node] < 0:
                raise RuntimeError(
                    f"active tributary {branch} targets inactive receiver {receiver}/{receiver_node}"
                )
            upstream_global_parts.append(
                np.asarray([global_by_branch[branch][terminal]], dtype=np.int64)
            )
            downstream_global_parts.append(
                np.asarray([global_by_branch[receiver][receiver_node]], dtype=np.int64)
            )
            coefficient_groups.append((
                slice(edge_offset, edge_offset + 1),
                branch,
                np.asarray([terminal], dtype=np.int64),
                receiver,
                np.asarray([receiver_node], dtype=np.int64),
            ))
            edge_offset += 1

    upstream_global = np.concatenate(upstream_global_parts)
    downstream_global = np.concatenate(downstream_global_parts)
    edge_index = np.arange(edge_offset, dtype=np.int64)
    graph_row = np.column_stack((upstream_global, downstream_global, upstream_global, downstream_global)).ravel()
    graph_col = np.column_stack((upstream_global, downstream_global, downstream_global, upstream_global)).ravel()
    incidence = coo_matrix(
        (
            np.tile(np.asarray([1.0, -1.0]), edge_offset),
            (
                np.repeat(edge_index, 2),
                np.column_stack((upstream_global, downstream_global)).ravel(),
            ),
        ),
        shape=(edge_offset, len(active_local)),
    ).tocsr()
    return {
        "active_by_branch": active_by_branch,
        "global_by_branch": global_by_branch,
        "active_local": active_local,
        "coefficient_groups": coefficient_groups,
        "edge_count": edge_offset,
        "graph_row": graph_row,
        "graph_col": graph_col,
        "incidence": incidence,
    }


def _structure(flowlines, branch_cache):
    global _LAST_SIGNATURE, _LAST_STRUCTURE
    signature = _support_signature(branch_cache)
    if signature != _LAST_SIGNATURE:
        _LAST_STRUCTURE = _build_structure(flowlines, branch_cache)
        _LAST_SIGNATURE = signature
    return _LAST_STRUCTURE


def _face_conductance_array(coefficients, upstream, upstream_index, downstream, downstream_index):
    hu = upstream["thick"][upstream_index]
    hv = downstream["thick"][downstream_index]
    wu = upstream["widths_m"][upstream_index]
    wv = downstream["widths_m"][downstream_index]
    w0u = upstream["w0_m"][upstream_index]
    w0v = downstream["w0_m"][downstream_index]
    su = upstream["surface_h"][upstream_index]
    sv = downstream["surface_h"][downstream_index]
    distance = 0.5 * (float(upstream["dx_meter"]) + float(downstream["dx_meter"]))
    hstag = 0.5 * (hu + hv)
    slope = (sv - su) / distance
    n = float(coefficients.glen_n)
    trapezoid = np.isfinite(w0u) & np.isfinite(w0v)
    diffusivity = np.empty(len(upstream_index), dtype=float)
    if np.any(trapezoid):
        width_stag = 0.5 * (wu[trapezoid] + wv[trapezoid])
        w0_stag = 0.5 * (w0u[trapezoid] + w0v[trapezoid])
        h = hstag[trapezoid]
        diffusivity[trapezoid] = (
            (
                float(coefficients._fd) * h ** (n + 2.0)
                + float(coefficients.fs) * h**n
            )
            * float(coefficients.rhog)
            * (w0_stag + width_stag)
            / 2.0
            * np.abs(slope[trapezoid]) ** (n - 1.0)
        )
    parabolic = ~trapezoid
    if np.any(parabolic):
        h = hstag[parabolic]
        section_stag = 0.5 * (
            upstream["section"][upstream_index[parabolic]]
            + downstream["section"][downstream_index[parabolic]]
        )
        diffusivity[parabolic] = (
            (
                float(coefficients._fd) * h ** (n + 1.0)
                + float(coefficients.fs) * h ** (n - 1.0)
            )
            * float(coefficients.rhog)
            * section_stag
            * np.abs(slope[parabolic]) ** (n - 1.0)
        )
    return diffusivity / distance


def _assemble(flowlines, coefficients, dt_seconds):
    branch_cache = cached._branch_cache(flowlines)
    structure = _structure(flowlines, branch_cache)
    active_local = structure["active_local"]
    surface = np.empty(len(active_local), dtype=float)
    storage = np.empty(len(active_local), dtype=float)
    for branch, active in enumerate(structure["active_by_branch"]):
        global_index = structure["global_by_branch"][branch][active]
        surface[global_index] = branch_cache[branch]["surface_h"][active]
        storage[global_index] = (
            branch_cache[branch]["widths_m"][active] * branch_cache[branch]["dx_meter"]
        )
    conductance = np.empty(structure["edge_count"], dtype=float)
    for edge_slice, upstream_branch, upstream_node, downstream_branch, downstream_node in structure[
        "coefficient_groups"
    ]:
        conductance[edge_slice] = _face_conductance_array(
            coefficients,
            branch_cache[upstream_branch],
            upstream_node,
            branch_cache[downstream_branch],
            downstream_node,
        )
    value = float(dt_seconds) * conductance
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
        "all-dynamic-arrays-cached; vectorized-face-coefficients; support-keyed-graph-reuse"
    )
    return result
