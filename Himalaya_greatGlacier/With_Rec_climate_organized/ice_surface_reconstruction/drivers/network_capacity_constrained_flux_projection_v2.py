"""Roundoff-safe DAG capacity projection for conservative network transport.

This is a numerical implementation correction to the v1 projection.  It
updates a working node-volume array edge by edge and, at a capacity-limited
node, assigns the last outgoing transfer from the exact remaining working
volume.  Therefore a depleted node is represented by an exact floating-point
zero before MixedBed section setters are called; no negative section is
clipped after the transport step.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def project_flux_dag_exact(
    node_volume_m3: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    preferred_flux_m3_s: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    volume = np.asarray(node_volume_m3, dtype=float)
    u = np.asarray(edges_u, dtype=int)
    v = np.asarray(edges_v, dtype=int)
    preferred = np.asarray(preferred_flux_m3_s, dtype=float)
    dt = float(dt_seconds)
    if dt <= 0 or np.any(~np.isfinite(volume)) or np.any(volume < 0):
        raise ValueError("invalid-node-volume-or-dt")
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
        requested_volume = np.asarray([preferred[e] * dt for e in node_edges], dtype=float)
        requested_total = float(np.sum(requested_volume, dtype=np.float64))
        available = float(work[node])
        is_limited = requested_total > available and requested_total > 0.0
        factor = available / requested_total if is_limited else 1.0
        transfers = requested_volume * factor
        if is_limited:
            # The final edge receives the exact remaining stored volume.  The
            # source update below is then x -= x, yielding an exact zero.
            if len(transfers) > 1:
                used = float(np.sum(transfers[:-1], dtype=np.float64))
                transfers[-1] = max(0.0, available - used)
            else:
                transfers[0] = available
        for edge, transfer in zip(node_edges, transfers):
            transfer = float(transfer)
            if transfer < 0 or transfer > work[node] + 1e-8:
                raise RuntimeError("invalid-capacity-transfer")
            if is_limited and edge == node_edges[-1]:
                transfer = float(work[node])
            work[node] -= transfer
            work[int(v[edge])] += transfer
            accepted[edge] = transfer / dt
            limited[edge] = is_limited
        if is_limited and work[node] != 0.0:
            raise RuntimeError(f"limited-node-not-exact-zero:{node}:{work[node]}")

    if np.any(~np.isfinite(work)) or np.any(work < 0):
        raise RuntimeError("projection-produced-invalid-volume")
    conservation = float(np.sum(work, dtype=np.float64) - np.sum(volume, dtype=np.float64))
    return accepted, work, limited, conservation

