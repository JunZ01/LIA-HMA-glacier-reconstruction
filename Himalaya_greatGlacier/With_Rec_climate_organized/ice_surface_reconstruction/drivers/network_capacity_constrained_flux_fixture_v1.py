"""Synthetic fixtures for a DAG capacity-constrained conservative flux step."""

from __future__ import annotations

import json
import traceback
from collections import deque
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_capacity_constrained_flux_fixture_v1"


def project_flux_dag(
    node_volume_m3: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    preferred_flux_m3_s: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project nonnegative preferred fluxes onto exact node volume capacity."""

    volume = np.asarray(node_volume_m3, dtype=float)
    u = np.asarray(edges_u, dtype=int)
    v = np.asarray(edges_v, dtype=int)
    preferred = np.asarray(preferred_flux_m3_s, dtype=float)
    if np.any(volume < 0) or np.any(preferred < 0):
        raise ValueError("negative-input")
    node_count = volume.size
    indegree = np.zeros(node_count, dtype=int)
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
    incoming: list[list[int]] = [[] for _ in range(node_count)]
    for edge, (source, target) in enumerate(zip(u, v)):
        indegree[target] += 1
        outgoing[source].append(edge)
        incoming[target].append(edge)
    queue = deque(np.flatnonzero(indegree == 0).tolist())
    order = []
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
    for node in order:
        incoming_rate = float(sum(accepted[edge] for edge in incoming[node]))
        available_rate = float(volume[node]) / float(dt_seconds) + incoming_rate
        requested = float(sum(preferred[edge] for edge in outgoing[node]))
        factor = 1.0 if requested <= available_rate or requested == 0.0 else available_rate / requested
        for edge in outgoing[node]:
            accepted[edge] = preferred[edge] * factor
            limited[edge] = factor < 1.0
    divergence_rate = np.zeros(node_count, dtype=float)
    np.add.at(divergence_rate, u, accepted)
    np.add.at(divergence_rate, v, -accepted)
    final_volume = volume - float(dt_seconds) * divergence_rate
    if np.min(final_volume) < -1e-8:
        raise RuntimeError(f"capacity-projection-negative-volume:{np.min(final_volume)}")
    return accepted, final_volume, limited


def _single_edge_capacity() -> dict:
    q, final, limited = project_flux_dag(np.array([3.0, 0.0]), np.array([0]), np.array([1]), np.array([2.0]), 3.0)
    assert np.allclose(q, [1.0])
    assert np.allclose(final, [0.0, 3.0])
    assert limited.tolist() == [True]
    return {"status": "pass", "accepted_flux": q.tolist(), "final_volume": final.tolist()}


def _two_tributary_conservation() -> dict:
    q, final, limited = project_flux_dag(
        np.array([2.0, 4.0, 1.0, 0.0]),
        np.array([0, 1, 2]),
        np.array([2, 2, 3]),
        np.array([3.0, 1.0, 10.0]),
        2.0,
    )
    assert np.min(final) >= 0
    assert abs(float(np.sum(final)) - 7.0) <= 1e-14
    assert np.all(q >= 0)
    return {"status": "pass", "accepted_flux": q.tolist(), "final_volume": final.tolist(), "limited": limited.tolist()}


def _unlimited_identity() -> dict:
    preferred = np.array([0.1, 0.2])
    q, final, limited = project_flux_dag(np.array([10.0, 10.0, 0.0]), np.array([0, 1]), np.array([2, 2]), preferred, 1.0)
    assert np.array_equal(q, preferred)
    assert not np.any(limited)
    assert abs(float(np.sum(final)) - 20.0) <= 1e-14
    return {"status": "pass", "preferred_flux_bitwise_preserved": True, "final_volume": final.tolist()}


def run_all() -> dict:
    results, failed = [], False
    for name, function in (("single_edge_capacity", _single_edge_capacity), ("two_tributary_conservation", _two_tributary_conservation), ("unlimited_identity", _unlimited_identity)):
        try:
            results.append({"name": name, **function()})
        except Exception as exc:
            failed = True
            results.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    return {
        "status": "fixture-fail-mechanism-closed" if failed else "fixture-pass-real-system-not-authorized",
        "mechanism": "network-dag-capacity-constrained-least-change-flux-v1",
        "fixtures": results,
        "all_fixtures_pass": not failed,
        "projection_rule": "preserve preferred edge-flux ratios while enforcing each node outflow <= initial volume/dt + accepted upstream inflow in topological order",
        "mass_conservative": True,
        "posthoc_section_clip": False,
        "real_system_authorized": False,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary = run_all()
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if not summary["all_fixtures_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
