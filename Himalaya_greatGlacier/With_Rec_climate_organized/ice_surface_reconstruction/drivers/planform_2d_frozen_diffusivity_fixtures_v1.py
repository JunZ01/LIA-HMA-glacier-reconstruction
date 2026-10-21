"""Synthetic fixtures for the 2-D frozen-diffusivity semi-implicit operator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .planform_2d_frozen_diffusivity_semiimplicit_core_v1 import (
    frozen_diffusivity_step,
    graph_laplacian,
)


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_fixtures_v1"


def zero_diffusivity_fixture():
    ny, nx = 4, 5
    east = np.zeros((ny, nx - 1))
    south = np.zeros((ny - 1, nx))
    laplacian, edges = graph_laplacian(ny, nx, east, south)
    state = np.arange(ny * nx, dtype=float)
    operator = np.eye(ny * nx) + 123.0 * laplacian.toarray()
    result = np.linalg.solve(operator, state)
    passed = edges == 0 and np.array_equal(result, state)
    return {"passed": bool(passed), "edge_count": edges, "bitwise_identity": bool(np.array_equal(result, state))}


def manufactured_conservation_fixture():
    ny, nx = 1, 3
    east = np.array([[1.0 / 30.0, 1.0 / 30.0]])
    south = np.zeros((0, 3))
    laplacian, edges = graph_laplacian(ny, nx, east, south)
    initial = np.array([2.0, 0.0, 2.0])
    dt = 30.0
    operator = np.eye(3) + dt * laplacian.toarray()
    final = np.linalg.solve(operator, initial)
    residual = operator.dot(final) - initial
    expected = np.array([1.5, 1.0, 1.5])
    passed = (
        edges == 2
        and np.max(np.abs(final - expected)) <= 1e-14
        and abs(np.sum(final - initial)) <= 1e-14
        and np.max(np.abs(residual)) <= 1e-14
    )
    return {
        "passed": bool(passed),
        "edge_count": edges,
        "final": final.tolist(),
        "expected": expected.tolist(),
        "global_storage_change": float(np.sum(final - initial)),
        "maximum_residual": float(np.max(np.abs(residual))),
    }


def oggm_infinitesimal_consistency_fixture():
    from oggm import cfg
    from oggm.core.massbalance import ScalarMassBalance
    from oggm.core.sia2d import Upstream2D

    cfg.initialize(logging_level="WARNING")
    yy, xx = np.meshgrid(np.arange(8), np.arange(10), indexing="ij")
    bed = 3000.0 - 3.0 * xx - 2.0 * yy
    thickness = 80.0 + 0.5 * np.sin(xx / 2.0) + 0.25 * np.cos(yy / 2.0)
    model = Upstream2D(
        bed,
        init_ice_thick=thickness,
        dx=50.0,
        mb_model=ScalarMassBalance(0.0),
        glen_a=2.4e-24,
        ice_thick_filter=None,
    )
    div, _ = model.diffusion_upstream_2d()
    dt = 1e-4
    explicit = thickness + div * dt
    implicit, diagnostics = frozen_diffusivity_step(model, dt, rtol=1e-13)
    scale = max(float(np.linalg.norm(explicit)), 1.0)
    relative = float(np.linalg.norm(implicit - explicit) / scale)
    volume_residual = float(abs(np.sum(implicit - thickness) * model.dxdy))
    passed = (
        diagnostics["solver_info"] == 0
        and diagnostics["matrix_symmetric_max_abs"] == 0.0
        and diagnostics["minimum_thickness_m"] > 0
        and relative <= 1e-10
        and volume_residual <= 1e-6
        and diagnostics["face_symmetry"]["east_max_relative_difference"] <= 1e-14
        and diagnostics["face_symmetry"]["south_max_relative_difference"] <= 1e-14
    )
    return {
        "passed": bool(passed),
        "implicit_vs_explicit_l2_relative": relative,
        "volume_residual_m3": volume_residual,
        "diagnostics": diagnostics,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fixtures = {
        "zero_diffusivity_identity": zero_diffusivity_fixture(),
        "manufactured_mass_conservation": manufactured_conservation_fixture(),
        "oggm_infinitesimal_consistency": oggm_infinitesimal_consistency_fixture(),
    }
    passed = all(item["passed"] for item in fixtures.values())
    summary = {
        "schema": "lia-191-planform2d-frozen-diffusivity-fixtures-v1",
        "status": "fixture-pass-real-system-authorized" if passed else "fixture-fail-mechanism-closed",
        "fixtures": fixtures,
        "real_system_run": False,
        "historical_run": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
