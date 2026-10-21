"""V6: cache all dynamic MixedBed arrays used by each conservative step."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, vstack

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_positive_conservative_system2005_zero_mb_v1 as predecessor
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as v1


TOL = 1e-10


def _branch_cache(flowlines):
    return [
        {
            "thick": np.asarray(flowline.thick, dtype=float),
            "widths_m": np.asarray(flowline.widths_m, dtype=float),
            "section": np.asarray(flowline.section, dtype=float),
            "surface_h": np.asarray(flowline.surface_h, dtype=float),
            "w0_m": np.asarray(flowline._w0_m, dtype=float),
            "dx_meter": float(flowline.dx_meter),
        }
        for flowline in flowlines
    ]


def _face_conductance(coefficients, upstream, upstream_index, downstream, downstream_index):
    hu = float(upstream["thick"][upstream_index])
    hv = float(downstream["thick"][downstream_index])
    wu = float(upstream["widths_m"][upstream_index])
    wv = float(downstream["widths_m"][downstream_index])
    w0u = float(upstream["w0_m"][upstream_index])
    w0v = float(downstream["w0_m"][downstream_index])
    su = float(upstream["surface_h"][upstream_index])
    sv = float(downstream["surface_h"][downstream_index])
    distance = 0.5 * (float(upstream["dx_meter"]) + float(downstream["dx_meter"]))
    hstag = 0.5 * (hu + hv)
    slope = (sv - su) / distance
    n = float(coefficients.glen_n)
    if np.isfinite(w0u) and np.isfinite(w0v):
        width_stag = 0.5 * (wu + wv)
        w0_stag = 0.5 * (w0u + w0v)
        diffusivity = (
            (float(coefficients._fd) * hstag ** (n + 2.0) + float(coefficients.fs) * hstag**n)
            * float(coefficients.rhog)
            * (w0_stag + width_stag)
            / 2.0
            * abs(slope) ** (n - 1.0)
        )
    else:
        section_stag = 0.5 * (
            float(upstream["section"][upstream_index])
            + float(downstream["section"][downstream_index])
        )
        diffusivity = (
            (
                float(coefficients._fd) * hstag ** (n + 1.0)
                + float(coefficients.fs) * hstag ** (n - 1.0)
            )
            * float(coefficients.rhog)
            * section_stag
            * abs(slope) ** (n - 1.0)
        )
    return float(diffusivity / distance)


def _assemble(flowlines, coefficients, dt_seconds):
    cache = _branch_cache(flowlines)
    branch_by_object = {id(flowline): index for index, flowline in enumerate(flowlines)}
    active_local = []
    global_of = {}
    active_by_branch = []
    for branch, values in enumerate(cache):
        active = np.flatnonzero(values["thick"] > base.ACTIVE_TOLERANCE_M)
        active_by_branch.append(active)
        for node_value in active:
            node = int(node_value)
            global_of[(branch, node)] = len(active_local)
            active_local.append((branch, node))

    edges = []
    for branch, flowline in enumerate(flowlines):
        active = active_by_branch[branch]
        active_set = set(map(int, active))
        for node_value in active[:-1]:
            node = int(node_value)
            if node + 1 not in active_set:
                continue
            edges.append((
                global_of[(branch, node)],
                global_of[(branch, node + 1)],
                _face_conductance(coefficients, cache[branch], node, cache[branch], node + 1),
                "within",
            ))
        if (
            flowline.flows_to is not None
            and len(active)
            and int(active[-1]) == len(cache[branch]["thick"]) - 1
        ):
            receiver = branch_by_object[id(flowline.flows_to)]
            receiver_node = int(flowline.flows_to_indice)
            terminal = int(active[-1])
            if (receiver, receiver_node) not in global_of:
                raise RuntimeError(
                    f"active tributary {branch} targets inactive receiver {receiver}/{receiver_node}"
                )
            edges.append((
                global_of[(branch, terminal)],
                global_of[(receiver, receiver_node)],
                _face_conductance(
                    coefficients, cache[branch], terminal, cache[receiver], receiver_node
                ),
                "junction",
            ))

    surface = np.asarray([cache[b]["surface_h"][n] for b, n in active_local], dtype=float)
    storage = np.asarray(
        [cache[b]["widths_m"][n] * cache[b]["dx_meter"] for b, n in active_local],
        dtype=float,
    )
    row, col, data = [], [], []
    for upstream, downstream, conductance, _kind in edges:
        value = float(dt_seconds) * float(conductance)
        row.extend([upstream, downstream, upstream, downstream])
        col.extend([upstream, downstream, downstream, upstream])
        data.extend([value, value, -value, -value])
    graph = coo_matrix(
        (data, (row, col)), shape=(len(active_local), len(active_local))
    ).tocsr()
    matrix = diags(storage, format="csr") + graph
    rhs = storage * surface
    incidence_rows, incidence_cols, incidence_data = [], [], []
    for edge_index, (upstream, downstream, _conductance, _kind) in enumerate(edges):
        incidence_rows.extend([edge_index, edge_index])
        incidence_cols.extend([upstream, downstream])
        incidence_data.extend([1.0, -1.0])
    incidence = coo_matrix(
        (incidence_data, (incidence_rows, incidence_cols)),
        shape=(len(edges), len(active_local)),
    ).tocsr()
    return active_local, edges, surface, storage, matrix, rhs, incidence, cache


def conservative_boundary_step(flowlines, coefficients, dt_seconds, closure_limit, section_floor):
    config = conservative_boundary_step.config
    (
        active_local,
        edges,
        surface_before,
        storage,
        matrix,
        rhs,
        incidence,
        cache,
    ) = _assemble(flowlines, coefficients, dt_seconds)
    widths = np.asarray([cache[b]["widths_m"][n] for b, n in active_local], dtype=float)
    old_sections = np.asarray([cache[b]["section"][n] for b, n in active_local], dtype=float)
    surface_lower = surface_before - old_sections / widths
    constraints = vstack([incidence, eye(len(active_local), format="csr")], format="csr")
    lower = np.r_[np.zeros(incidence.shape[0]), surface_lower]
    volume_row = csr_matrix(storage.reshape(1, -1))
    volume_target = float(np.dot(storage, surface_before))
    candidate, unconstrained, active, multipliers, volume_multiplier, history = (
        predecessor.positive_v1.solve_affine_active_set(
            matrix,
            rhs,
            constraints,
            lower,
            volume_row,
            volume_target,
            tolerance=float(config["active_set_tolerance"]),
            maximum_iterations=400,
        )
    )
    residual = np.asarray(constraints @ candidate, dtype=float) - lower
    edge_slopes = residual[: incidence.shape[0]]
    target_delta = storage * (candidate - surface_before)
    if abs(float(np.sum(target_delta))) > closure_limit:
        raise RuntimeError(f"linearized-volume-closure:{float(np.sum(target_delta))}")
    candidate_sections = [values["section"].copy() for values in cache]
    for global_index, (branch, node) in enumerate(active_local):
        candidate_sections[branch][node] += target_delta[global_index] / cache[branch]["dx_meter"]

    boundary_tol = float(config["boundary_canonicalization_tolerance_m2"])
    raw_minimum = min(float(np.min(section)) for section in candidate_sections)
    if raw_minimum < -boundary_tol:
        raise RuntimeError(f"physical-negative-section:{raw_minimum}")
    added_volume = 0.0
    for branch, section in enumerate(candidate_sections):
        negative = np.flatnonzero(section < 0.0)
        for node in negative:
            added_volume += -float(section[node]) * cache[branch]["dx_meter"]
            section[node] = 0.0
    donor_removed = 0.0
    if added_volume:
        donors = []
        for branch, section in enumerate(candidate_sections):
            for node in np.flatnonzero(section > 0.0):
                donors.append((float(section[node]) * cache[branch]["dx_meter"], branch, int(node)))
        donors.sort(reverse=True)
        available, branch, node = donors[0]
        if available <= added_volume:
            raise RuntimeError("no-positive-donor-for-boundary-canonicalization")
        candidate_sections[branch][node] -= added_volume / cache[branch]["dx_meter"]
        donor_removed = added_volume

    before_volume = base.total_volume(flowlines)
    for flowline, section in zip(flowlines, candidate_sections):
        flowline.section = section
    after_volume = base.total_volume(flowlines)
    actual_closure = after_volume - before_volume
    if abs(actual_closure) > closure_limit:
        raise RuntimeError(f"actual-volume-closure:{actual_closure}")
    realized_by_branch = [np.asarray(flowline.surface_h, dtype=float) for flowline in flowlines]
    realized_surface = np.asarray(
        [realized_by_branch[b][n] for b, n in active_local], dtype=float
    )
    realized_slopes = np.asarray(incidence @ realized_surface, dtype=float)
    if not np.all(np.isfinite(realized_surface)) or float(np.min(realized_slopes)) < -TOL:
        raise RuntimeError(f"realized-reverse-or-nonfinite:{float(np.min(realized_slopes))}")
    active_matrix = constraints[active, :].tocsr()
    stationarity = np.asarray(matrix @ candidate - rhs, dtype=float)
    stationarity -= np.asarray(active_matrix.T @ multipliers, dtype=float).ravel()
    stationarity += volume_row.toarray().ravel() * volume_multiplier
    maximum_stationarity = float(np.max(np.abs(stationarity)))
    if maximum_stationarity > 1e-5:
        raise RuntimeError(f"KKT-stationarity:{maximum_stationarity}")
    minimum_multiplier = float(np.min(multipliers)) if len(multipliers) else None
    if minimum_multiplier is not None and minimum_multiplier < -TOL:
        raise RuntimeError(f"negative-constraint-multiplier:{minimum_multiplier}")
    active_candidate_sections = np.asarray(
        [candidate_sections[b][n] for b, n in active_local], dtype=float
    )
    return {
        "active_node_count": len(active_local),
        "edge_count": len(edges),
        "unconstrained_reverse_edge_count": int(
            np.count_nonzero(np.asarray(incidence @ unconstrained) < -TOL)
        ),
        "active_constraint_count": len(active),
        "active_edge_constraint_count": sum(index < incidence.shape[0] for index in active),
        "active_section_constraint_count": sum(index >= incidence.shape[0] for index in active),
        "minimum_inequality_multiplier": minimum_multiplier,
        "minimum_target_edge_slope_m": float(np.min(edge_slopes)),
        "minimum_realized_edge_slope_m": float(np.min(realized_slopes)),
        "minimum_active_section_m2": float(np.min(active_candidate_sections)),
        "maximum_absolute_KKT_stationarity": maximum_stationarity,
        "actual_closure_m3": actual_closure,
        "active_set_iterations": len(history),
        "raw_minimum_section_before_canonicalization_m2": raw_minimum,
        "boundary_canonicalization_added_m3": added_volume,
        "boundary_canonicalization_donor_removed_m3": donor_removed,
        "mixedbed_property_access": "all-dynamic-arrays-cached-once-per-branch-per-step",
    }


def run(config):
    technical = __import__("json").loads(
        base.resolve(config["technical_predecessor"]).read_text(encoding="utf-8")
    )
    if "active-section-interior-floor:-2.922107000813412e-13" not in technical.get(
        "error", ""
    ):
        raise RuntimeError("v3-is-not-the-registered-roundoff-only-boundary-failure")
    original = predecessor.constrained_step
    conservative_boundary_step.config = config
    predecessor.constrained_step = conservative_boundary_step
    try:
        summary, rows = v1.run(config)
    finally:
        predecessor.constrained_step = original
    summary["schema"] = "lia-191-network-topology-adaptive-split-system2005-zero-mb-v6"
    summary["mixedbed_property_access"] = (
        "all-dynamic-arrays-cached-once-per-branch-per-step"
    )
    return summary, rows
