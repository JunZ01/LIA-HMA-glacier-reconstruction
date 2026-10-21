"""Synthetic sparse fixtures for a paired-cell graph plus a 1-D zero-ice tail.

No project glacier is read here.  Passing validates only the closed-system
semi-implicit graph algebra; it does not authorize a real-system run.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_solver_fixture_v1"


def incidence(node_count: int, edges: np.ndarray) -> sparse.csc_matrix:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (edge_count, 2)")
    if np.any(edges < 0) or np.any(edges >= node_count) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("invalid graph edge")
    cols = np.repeat(np.arange(len(edges), dtype=np.int64), 2)
    rows = edges.ravel()
    data = np.tile(np.array([1.0, -1.0]), len(edges))
    return sparse.csc_matrix((data, (rows, cols)), shape=(node_count, len(edges)))


def step_surface(
    surface_before_m: np.ndarray,
    storage_m2: np.ndarray,
    edges: np.ndarray,
    conductance_m2_s: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, sparse.csc_matrix, np.ndarray, np.ndarray]:
    surface = np.asarray(surface_before_m, dtype=np.float64)
    storage = np.asarray(storage_m2, dtype=np.float64)
    conductance = np.asarray(conductance_m2_s, dtype=np.float64)
    if surface.ndim != 1 or storage.shape != surface.shape:
        raise ValueError("surface/storage shape mismatch")
    if np.any(~np.isfinite(surface)) or np.any(~np.isfinite(storage)) or np.any(storage <= 0):
        raise ValueError("invalid surface or storage")
    bmat = incidence(len(surface), edges)
    if conductance.shape != (bmat.shape[1],) or np.any(~np.isfinite(conductance)) or np.any(conductance < 0):
        raise ValueError("invalid conductance")
    if not np.any(conductance):
        return surface.copy(), sparse.diags(storage, format="csc"), np.zeros_like(conductance), np.zeros_like(surface)
    matrix = sparse.diags(storage, format="csc") + float(dt_seconds) * bmat @ sparse.diags(conductance, format="csc") @ bmat.T
    out = np.asarray(spsolve(matrix, storage * surface), dtype=np.float64)
    if np.any(~np.isfinite(out)):
        raise RuntimeError("non-finite sparse graph solution")
    flux = conductance * np.asarray(bmat.T @ out).ravel()
    residual = storage * (out - surface) + float(dt_seconds) * np.asarray(bmat @ flux).ravel()
    return out, matrix, flux, residual


def grid_edges(ny: int, nx: int) -> np.ndarray:
    ids = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    horizontal = np.column_stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()])
    vertical = np.column_stack([ids[:-1, :].ravel(), ids[1:, :].ravel()])
    return np.vstack([horizontal, vertical])


def formal_diagnostics(before, after, storage, matrix, residual) -> dict:
    volume0 = float(np.dot(storage, before))
    volume1 = float(np.dot(storage, after))
    limit = max(1e-6, abs(volume0) * 1e-12)
    asymmetry = matrix - matrix.T
    maximum_asymmetry = float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.0
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    maximum_residual = float(np.max(np.abs(residual)))
    global_residual = float(np.sum(storage * (after - before)))
    if maximum_asymmetry > 1e-12 or np.any(diagonal <= 0):
        raise AssertionError("matrix symmetry/positive-diagonal gate failed")
    if maximum_residual > limit or abs(global_residual) > limit or abs(volume1 - volume0) > limit:
        raise AssertionError("closed graph mass gate failed")
    return {
        "volume_before_m3_equivalent": volume0,
        "volume_after_m3_equivalent": volume1,
        "volume_change_m3_equivalent": volume1 - volume0,
        "maximum_node_residual_m3_equivalent": maximum_residual,
        "global_residual_m3_equivalent": global_residual,
        "closure_limit_m3_equivalent": limit,
        "matrix_maximum_asymmetry": maximum_asymmetry,
        "matrix_minimum_diagonal": float(np.min(diagonal)),
    }


def run_two_cell_exact() -> dict:
    before = np.array([2.0, 0.0])
    storage = np.ones(2)
    edges = np.array([[0, 1]])
    dt = 30.0
    conductance = np.array([1.0 / 30.0])
    expected = np.array([4.0 / 3.0, 2.0 / 3.0])
    after, matrix, flux, residual = step_surface(before, storage, edges, conductance, dt)
    error = float(np.max(np.abs(after - expected)))
    if error > 1e-14 or flux[0] < 0:
        raise AssertionError("two-cell exact solution failed")
    return {"status": "pass", "maximum_solution_error": error, **formal_diagnostics(before, after, storage, matrix, residual)}


def run_rectangular_grid() -> dict:
    ny, nx = 4, 5
    edges = grid_edges(ny, nx)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    before = (20.0 - 0.7 * xx - 1.2 * yy + 0.15 * np.sin(xx + yy)).ravel()
    storage = (380.0 + 7.0 * xx + 3.0 * yy).ravel()
    conductance = np.linspace(0.05, 0.25, len(edges))
    after, matrix, flux, residual = step_surface(before, storage, edges, conductance, 30.0)
    if np.min(after) < np.min(before) - 1e-12 or np.max(after) > np.max(before) + 1e-12:
        raise AssertionError("closed diffusion violated the discrete maximum principle")
    return {
        "status": "pass", "node_count": len(before), "edge_count": len(edges),
        "minimum_flux": float(np.min(flux)), "maximum_flux": float(np.max(flux)),
        **formal_diagnostics(before, after, storage, matrix, residual),
    }


def run_hybrid_connector() -> dict:
    # 2x2 paired-cell block (nodes 0..3), one connector (3->4), and a
    # three-node native zero-ice tail (4..6).  All controls have positive
    # algebraic storage; the tail starts at zero thickness over a lower bed.
    edges = np.vstack([grid_edges(2, 2), np.array([[3, 4], [4, 5], [5, 6]])])
    bed = np.array([100.0, 98.0, 96.0, 94.0, 90.0, 85.0, 80.0])
    thick = np.array([15.0, 14.0, 13.0, 12.0, 0.0, 0.0, 0.0])
    before = bed + thick
    storage = np.array([390.0625] * 4 + [5000.0] * 3)
    conductance = np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
    after, matrix, flux, residual = step_surface(before, storage, edges, conductance, 30.0)
    if after[3] < after[4] - 1e-12:
        raise AssertionError("connector reversed under the manufactured downhill fixture")
    connector_flux = float(flux[4])
    if connector_flux < 0:
        raise AssertionError("connector flux is uphill")
    return {
        "status": "pass", "node_count": len(before), "edge_count": len(edges),
        "connector_flux_m3_s_equivalent": connector_flux,
        "connector_has_no_extra_storage": True,
        **formal_diagnostics(before, after, storage, matrix, residual),
    }


def run_zero_conductance_identity() -> dict:
    before = np.array([7.0, 3.0, 9.0, -2.0, 4.0], dtype=np.float64)
    storage = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]])
    after, matrix, flux, residual = step_surface(before, storage, edges, np.zeros(len(edges)), 30.0)
    if not np.array_equal(after, before) or not np.array_equal(flux, np.zeros_like(flux)) or not np.array_equal(residual, np.zeros_like(residual)):
        raise AssertionError("zero-conductance graph is not a bitwise identity")
    return {"status": "pass", "surface_bitwise_identity": True, "flux_bitwise_zero": True, "residual_bitwise_zero": True, "matrix_diagonal_only": matrix.nnz == len(before)}


def main() -> None:
    fixtures = {}
    errors = {}
    for name, function in [
        ("two_cell_exact", run_two_cell_exact),
        ("rectangular_grid", run_rectangular_grid),
        ("hybrid_connector", run_hybrid_connector),
        ("zero_conductance_identity", run_zero_conductance_identity),
    ]:
        try:
            fixtures[name] = function()
        except Exception as exc:  # retain all fixture diagnostics in one run
            fixtures[name] = {"status": "fail", "error": repr(exc)}
            errors[name] = traceback.format_exc()
    passed = all(item["status"] == "pass" for item in fixtures.values())
    summary = {
        "schema": "lia-191-paired-cell-hybrid-graph-solver-fixture-v1",
        "status": "fixture-pass-real-system-not-authorized" if passed else "fixture-fail-mechanism-closed",
        "fixtures": fixtures, "tracebacks": errors,
        "real_system_authorized": False,
        "real_glacier_read": False, "oggm_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
