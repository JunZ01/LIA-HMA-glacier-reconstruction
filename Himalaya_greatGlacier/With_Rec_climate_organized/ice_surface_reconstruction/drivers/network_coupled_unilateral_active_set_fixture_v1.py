"""Synthetic fixtures for a conservative one-way active-set network solve."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_coupled_unilateral_active_set_fixture_v1"


def solve_unilateral(
    storage: np.ndarray,
    surface_before: np.ndarray,
    incidence: np.ndarray,
    conductance: np.ndarray,
    dt_seconds: float,
    *,
    tolerance: float = 1e-12,
    maximum_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve q=K*max(B.T*s,0) by exact active-set matrix reassembly."""

    storage = np.asarray(storage, dtype=float)
    surface_before = np.asarray(surface_before, dtype=float)
    incidence = np.asarray(incidence, dtype=float)
    conductance = np.asarray(conductance, dtype=float)
    if storage.ndim != 1 or np.any(storage <= 0) or np.any(~np.isfinite(storage)):
        raise ValueError("invalid-storage")
    if incidence.shape[0] != storage.size or incidence.shape[1] != conductance.size:
        raise ValueError("incidence-shape")
    if np.any(conductance < 0) or np.any(~np.isfinite(conductance)):
        raise ValueError("invalid-conductance")
    active = (incidence.T @ surface_before) > tolerance
    seen: set[bytes] = set()
    for iteration in range(1, maximum_iterations + 1):
        key = active.tobytes()
        if key in seen:
            raise RuntimeError("active-set-cycle")
        seen.add(key)
        effective = conductance * active
        if np.any(effective):
            laplacian = (incidence * effective[np.newaxis, :]) @ incidence.T
            matrix = np.diag(storage) + float(dt_seconds) * laplacian
            surface_after = np.linalg.solve(matrix, storage * surface_before)
        else:
            surface_after = surface_before.copy()
        gaps = incidence.T @ surface_after
        new_active = gaps > tolerance
        if np.array_equal(new_active, active):
            flux = conductance * np.maximum(gaps, 0.0)
            residual = storage * (surface_after - surface_before) + float(dt_seconds) * incidence @ flux
            return surface_after, flux, residual, iteration
        active = new_active
    raise RuntimeError("active-set-no-convergence")


def _forward_two_tributary() -> dict:
    storage = np.ones(3)
    surface = np.array([2.0, 2.0, 0.0])
    incidence = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    conductance = np.array([1.0 / 30.0, 1.0 / 30.0])
    result, flux, residual, iterations = solve_unilateral(storage, surface, incidence, conductance, 30.0)
    expected = np.array([1.5, 1.5, 1.0])
    assert np.max(np.abs(result - expected)) <= 1e-14
    assert np.all(flux >= 0)
    assert np.max(np.abs(residual)) <= 1e-14
    return {"status": "pass", "surface": result.tolist(), "flux": flux.tolist(), "iterations": iterations}


def _reverse_edge_identity() -> dict:
    storage = np.array([2.0, 3.0])
    surface = np.array([0.0, 1.0])
    incidence = np.array([[1.0], [-1.0]])
    conductance = np.array([7.0])
    result, flux, residual, iterations = solve_unilateral(storage, surface, incidence, conductance, 3600.0)
    assert np.array_equal(result, surface)
    assert np.array_equal(flux, np.zeros(1))
    assert np.array_equal(residual, np.zeros(2))
    return {"status": "pass", "bitwise_identity": True, "flux_bitwise_zero": True, "iterations": iterations}


def _mixed_active_inactive_junction() -> dict:
    storage = np.array([2.0, 1.0, 4.0])
    surface = np.array([3.0, 0.0, 1.0])
    incidence = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    conductance = np.array([0.2, 0.4])
    result, flux, residual, iterations = solve_unilateral(storage, surface, incidence, conductance, 30.0)
    gaps = incidence.T @ result
    assert flux[0] > 0 and flux[1] == 0
    assert gaps[0] > 0 and gaps[1] < 0
    assert np.max(np.abs(residual)) <= 1e-12
    assert abs(np.sum(storage * (result - surface))) <= 1e-12
    return {
        "status": "pass",
        "surface": result.tolist(),
        "flux": flux.tolist(),
        "complementarity_products": (flux * np.minimum(gaps, 0.0)).tolist(),
        "maximum_residual": float(np.max(np.abs(residual))),
        "iterations": iterations,
    }


def run_all() -> dict:
    results = []
    failed = False
    for name, function in (
        ("forward_two_tributary", _forward_two_tributary),
        ("reverse_edge_identity", _reverse_edge_identity),
        ("mixed_active_inactive_junction", _mixed_active_inactive_junction),
    ):
        try:
            results.append({"name": name, **function()})
        except Exception as exc:
            failed = True
            results.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    return {
        "status": "fixture-fail-mechanism-closed" if failed else "fixture-pass-real-system-not-authorized",
        "mechanism": "network-coupled-unilateral-frozen-diffusivity-active-set-v1",
        "equation": "q_e=K_e*max((B.T*s_new)_e,0); (C+dt*B_active*K*B_active.T)s_new=C*s_old",
        "fixtures": results,
        "all_fixtures_pass": not failed,
        "real_system_authorized": False,
        "runtime_flux_clip_used": False,
        "synthetic_only": True,
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
