"""Exact DAG flux projection with immutable positive-support reserves.

The reserve is a finite-volume representation constraint, not a mass source:
it only limits outgoing transport from a node and every accepted transfer is
added exactly to its receiver.  A zero reserve reduces bit-for-bit to the
registered exact-capacity projection.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def project_flux_dag_with_reserve(
    node_volume_m3: np.ndarray,
    reserve_volume_m3: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    preferred_flux_m3_s: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    volume = np.asarray(node_volume_m3, dtype=float)
    reserve = np.asarray(reserve_volume_m3, dtype=float)
    u = np.asarray(edges_u, dtype=int)
    v = np.asarray(edges_v, dtype=int)
    preferred = np.asarray(preferred_flux_m3_s, dtype=float)
    dt = float(dt_seconds)
    if volume.shape != reserve.shape or dt <= 0:
        raise ValueError("invalid-volume-reserve-shape-or-dt")
    if np.any(~np.isfinite(volume)) or np.any(~np.isfinite(reserve)):
        raise ValueError("nonfinite-volume-or-reserve")
    if np.any(volume < 0) or np.any(reserve < 0) or np.any(reserve > volume):
        raise ValueError("invalid-positive-support-reserve")
    if np.any(~np.isfinite(preferred)) or np.any(preferred < 0):
        raise ValueError("invalid-preferred-flux")

    node_count = volume.size
    indegree = np.zeros(node_count, dtype=int)
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
    for edge, (source, target) in enumerate(zip(u, v)):
        indegree[target] += 1
        outgoing[source].append(edge)
    queue = deque(np.flatnonzero(indegree == 0).tolist())
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for edge in outgoing[node]:
            target = int(v[edge])
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != node_count:
        raise RuntimeError("network-not-dag")

    accepted = np.zeros_like(preferred)
    limited = np.zeros_like(preferred, dtype=bool)
    work = volume.copy()
    for node in order:
        node_edges = outgoing[node]
        if not node_edges:
            continue
        requested = np.asarray([preferred[e] * dt for e in node_edges], dtype=float)
        requested_total = float(np.sum(requested, dtype=np.float64))
        available = max(0.0, float(work[node] - reserve[node]))
        is_limited = requested_total > available and requested_total > 0.0
        transfers = requested * (available / requested_total if is_limited else 1.0)
        if is_limited:
            if len(transfers) > 1:
                transfers[-1] = max(0.0, available - float(np.sum(transfers[:-1], dtype=np.float64)))
            else:
                transfers[0] = available
        for edge, transfer in zip(node_edges, transfers):
            transfer = float(transfer)
            if is_limited and edge == node_edges[-1]:
                transfer = max(0.0, float(work[node] - reserve[node]))
            if transfer < 0 or transfer > work[node] - reserve[node] + 1e-8:
                raise RuntimeError("invalid-reserve-capacity-transfer")
            work[node] -= transfer
            work[int(v[edge])] += transfer
            accepted[edge] = transfer / dt
            limited[edge] = is_limited
        if work[node] < reserve[node] - 1e-8:
            raise RuntimeError("positive-support-reserve-violated")

    if np.any(~np.isfinite(work)) or np.any(work < reserve - 1e-8):
        raise RuntimeError("projection-produced-invalid-reserved-volume")
    conservation = float(np.sum(work, dtype=np.float64) - np.sum(volume, dtype=np.float64))
    return accepted, work, limited, conservation


def project_lower_bounded_simplex(
    requested: np.ndarray, lower: np.ndarray, target_sum: float | None = None
) -> np.ndarray:
    """Euclidean projection onto ``x >= lower, sum(x) == target_sum``."""
    request = np.asarray(requested, dtype=float)
    floor = np.asarray(lower, dtype=float)
    if request.shape != floor.shape or np.any(~np.isfinite(request)) or np.any(~np.isfinite(floor)):
        raise ValueError("invalid-simplex-input")
    target = float(np.sum(request, dtype=np.float64) if target_sum is None else target_sum)
    minimum = float(np.sum(floor, dtype=np.float64))
    if target < minimum - 1e-10:
        raise RuntimeError(f"positive-support-mb-infeasible:{target}:{minimum}")
    if request.size == 0:
        return request.copy()
    shifted = request - floor
    budget = max(0.0, target - minimum)
    order = np.sort(shifted)[::-1]
    cssv = np.cumsum(order) - budget
    indices = np.arange(1, order.size + 1)
    valid = order - cssv / indices > 0
    theta = float(cssv[np.flatnonzero(valid)[-1]] / np.flatnonzero(valid)[-1].__add__(1)) if np.any(valid) else float(cssv[-1] / order.size)
    projected = floor + np.maximum(shifted - theta, 0.0)
    residual = target - float(np.sum(projected, dtype=np.float64))
    projected[int(np.argmax(projected - floor))] += residual
    if np.any(projected < floor - 1e-10) or abs(float(np.sum(projected)) - target) > 1e-7:
        raise RuntimeError("lower-simplex-projection-failed")
    return projected
