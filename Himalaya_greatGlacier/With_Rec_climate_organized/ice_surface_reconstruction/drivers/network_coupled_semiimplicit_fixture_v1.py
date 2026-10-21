"""Synthetic fixtures for a network-coupled semi-implicit flowline step.

This module does not read a GlacierDirectory or any real glacier state.  It
tests only the algebra needed before a real-system implementation can be
considered:

1. the one-line graph matrix reduces to OGGM's stock SemiImplicitModel;
2. a manufactured two-tributary junction closes mass node by node; and
3. zero diffusivity is a bitwise identity map.

Passing these fixtures does *not* authorize a real glacier run.
"""

from __future__ import annotations

import copy
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_network_coupled_semiimplicit_fixture_v1"
)


def assemble_matrix(
    storage_m2: np.ndarray,
    incidence: np.ndarray,
    conductance_m2_s: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A = diag(C) + dt B diag(K) B.T`` and its graph Laplacian."""

    storage = np.asarray(storage_m2, dtype=np.float64)
    bmat = np.asarray(incidence, dtype=np.float64)
    conductance = np.asarray(conductance_m2_s, dtype=np.float64)
    dt = float(dt_seconds)
    if storage.ndim != 1 or np.any(~np.isfinite(storage)) or np.any(storage <= 0):
        raise ValueError("storage coefficients must be finite and positive")
    if bmat.ndim != 2 or bmat.shape[0] != storage.size:
        raise ValueError("incidence matrix has incompatible shape")
    if conductance.shape != (bmat.shape[1],):
        raise ValueError("conductance vector has incompatible shape")
    if np.any(~np.isfinite(conductance)) or np.any(conductance < 0):
        raise ValueError("conductances must be finite and nonnegative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    laplacian = (bmat * conductance[np.newaxis, :]) @ bmat.T
    matrix = np.diag(storage) + dt * laplacian
    return matrix, laplacian


def step_surface(
    surface_m: np.ndarray,
    storage_m2: np.ndarray,
    incidence: np.ndarray,
    conductance_m2_s: np.ndarray,
    dt_seconds: float,
) -> np.ndarray:
    """Advance a closed, zero-MB network by one frozen-diffusivity step."""

    surface = np.asarray(surface_m, dtype=np.float64)
    conductance = np.asarray(conductance_m2_s, dtype=np.float64)
    if surface.ndim != 1 or np.any(~np.isfinite(surface)):
        raise ValueError("surface must be a finite vector")
    if not np.any(conductance):
        # The exact physical identity, kept explicit so this edge case cannot
        # acquire round-off through a generic linear solve.
        return surface.copy()
    matrix, _ = assemble_matrix(storage_m2, incidence, conductance, dt_seconds)
    rhs = np.asarray(storage_m2, dtype=np.float64) * surface
    out = np.linalg.solve(matrix, rhs)
    if np.any(~np.isfinite(out)):
        raise RuntimeError("non-finite semi-implicit solution")
    return out


def edge_flux(
    surface_m: np.ndarray,
    incidence: np.ndarray,
    conductance_m2_s: np.ndarray,
) -> np.ndarray:
    """Return oriented edge fluxes ``q = K B.T s`` in m3 s-1."""

    return np.asarray(conductance_m2_s, dtype=np.float64) * (
        np.asarray(incidence, dtype=np.float64).T
        @ np.asarray(surface_m, dtype=np.float64)
    )


def control_volume_residual_m3(
    surface_before_m: np.ndarray,
    surface_after_m: np.ndarray,
    storage_m2: np.ndarray,
    incidence: np.ndarray,
    conductance_m2_s: np.ndarray,
    dt_seconds: float,
) -> np.ndarray:
    """Return node-wise storage plus signed-flux residual for one step."""

    flux = edge_flux(surface_after_m, incidence, conductance_m2_s)
    return (
        np.asarray(storage_m2, dtype=np.float64)
        * (
            np.asarray(surface_after_m, dtype=np.float64)
            - np.asarray(surface_before_m, dtype=np.float64)
        )
        + float(dt_seconds) * np.asarray(incidence, dtype=np.float64) @ flux
    )


def chain_incidence(node_count: int) -> np.ndarray:
    """Incidence matrix for a no-flux-boundary one-dimensional chain."""

    if node_count < 2:
        raise ValueError("a chain needs at least two nodes")
    out = np.zeros((node_count, node_count - 1), dtype=np.float64)
    edge = np.arange(node_count - 1)
    out[edge, edge] = 1.0
    out[edge + 1, edge] = -1.0
    return out


def stock_internal_conductance(stock_model) -> np.ndarray:
    """Reproduce OGGM SemiImplicitModel's frozen internal face D/dx."""

    flowline = stock_model.fls[0]
    width = np.asarray(flowline.widths_m, dtype=np.float64)
    thick = np.asarray(flowline.thick, dtype=np.float64)
    surface = np.asarray(flowline.surface_h, dtype=np.float64)
    dx = float(flowline.dx_meter)
    width_stag = (width[:-1] + width[1:]) / 2.0
    w0 = np.asarray(flowline._w0_m, dtype=np.float64)
    w0_stag = (w0[:-1] + w0[1:]) / 2.0
    thick_stag = (thick[:-1] + thick[1:]) / 2.0
    slope = (surface[1:] - surface[:-1]) / dx
    n = float(stock_model.glen_n)
    diffusivity = (
        (
            float(stock_model._fd) * thick_stag ** (n + 2.0)
            + float(stock_model.fs) * thick_stag**n
        )
        * float(stock_model.rhog)
        * (w0_stag + width_stag)
        / 2.0
        * np.abs(slope) ** (n - 1.0)
    )
    return diffusivity / dx


def run_single_line_stock_fixture() -> dict:
    """Compare one synthetic graph step with stock SemiImplicitModel."""

    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel, TrapezoidalBedFlowline
    from oggm.core.massbalance import ScalarMassBalance
    from shapely.geometry import LineString

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    node_count = 9
    map_dx = 100.0
    bed = np.array(
        [3200.0, 3152.0, 3100.0, 3044.0, 2985.0, 2922.0, 2855.0, 2784.0, 2709.0]
    )
    thick = np.array([82.0, 86.0, 91.0, 96.0, 101.0, 99.0, 96.0, 92.0, 88.0])
    surface = bed + thick
    widths_m = np.array([140.0, 145.0, 151.0, 158.0, 166.0, 175.0, 185.0, 196.0, 208.0])
    line = LineString([(float(index), 0.0) for index in range(node_count)])
    flowline = TrapezoidalBedFlowline(
        line=line,
        dx=1.0,
        map_dx=map_dx,
        surface_h=surface,
        bed_h=bed,
        widths=widths_m / map_dx,
        lambdas=np.zeros(node_count),
        rgi_id="SYNTHETIC-FIXTURE",
    )
    dt = 3600.0
    glen_a = float(cfg.PARAMS["glen_a"])
    stock = SemiImplicitModel(
        [copy.deepcopy(flowline)],
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=glen_a,
        fs=0.0,
        fixed_dt=dt,
        min_dt=30.0,
    )
    graph_flowline = copy.deepcopy(flowline)
    surface_before = np.asarray(graph_flowline.surface_h, dtype=np.float64).copy()
    initial_section = np.asarray(graph_flowline.section, dtype=np.float64).copy()
    incidence = chain_incidence(node_count)
    conductance = stock_internal_conductance(stock)
    storage = np.asarray(graph_flowline.widths_m, dtype=np.float64) * map_dx
    surface_graph = step_surface(surface_before, storage, incidence, conductance, dt)
    graph_flowline.thick = surface_graph - np.asarray(graph_flowline.bed_h, dtype=np.float64)
    if np.any(graph_flowline.thick <= 0):
        raise AssertionError("single-line fixture produced nonpositive thickness")
    used_dt = float(stock.step(dt))
    if used_dt != dt:
        raise AssertionError(f"stock used unexpected dt: {used_dt} != {dt}")
    section_graph = np.asarray(graph_flowline.section, dtype=np.float64)
    section_stock = np.asarray(stock.fls[0].section, dtype=np.float64)
    relative_section_error = float(
        np.linalg.norm(section_graph - section_stock)
        / max(float(np.linalg.norm(section_stock)), np.finfo(float).tiny)
    )
    initial_volume = float(np.sum(initial_section) * map_dx)
    graph_volume = float(np.sum(section_graph) * map_dx)
    volume_residual = float(graph_volume - initial_volume)
    closure_limit = max(1e-6, initial_volume * 1e-12)
    node_residual = control_volume_residual_m3(
        surface_before, surface_graph, storage, incidence, conductance, dt
    )
    if relative_section_error > 1e-10:
        raise AssertionError(
            f"single-line stock section relative error {relative_section_error:.3e} > 1e-10"
        )
    if abs(volume_residual) > closure_limit:
        raise AssertionError(
            f"single-line volume residual {volume_residual:.3e} > {closure_limit:.3e}"
        )
    if float(np.max(np.abs(node_residual))) > closure_limit:
        raise AssertionError("single-line node closure exceeds formal limit")
    return {
        "status": "pass",
        "node_count": node_count,
        "dt_seconds": dt,
        "section_relative_l2_error": relative_section_error,
        "section_relative_l2_error_max": 1e-10,
        "volume_residual_m3": volume_residual,
        "maximum_absolute_node_residual_m3": float(np.max(np.abs(node_residual))),
        "closure_limit_m3": closure_limit,
        "stock_model": "OGGM SemiImplicitModel",
    }


def run_two_tributary_manufactured_fixture() -> dict:
    """Verify the exact three-control-volume two-tributary solution."""

    storage = np.ones(3, dtype=np.float64)
    surface_before = np.array([2.0, 2.0, 0.0], dtype=np.float64)
    incidence = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float64
    )
    dt = 30.0
    conductance = np.array([1.0 / 30.0, 1.0 / 30.0], dtype=np.float64)
    expected_surface = np.array([1.5, 1.5, 1.0], dtype=np.float64)
    expected_matrix = np.array(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, -1.0, 3.0]],
        dtype=np.float64,
    )
    surface_after = step_surface(
        surface_before, storage, incidence, conductance, dt
    )
    matrix, _ = assemble_matrix(storage, incidence, conductance, dt)
    flux = edge_flux(surface_after, incidence, conductance)
    residual = control_volume_residual_m3(
        surface_before, surface_after, storage, incidence, conductance, dt
    )
    initial_volume = 3.0
    closure_limit = max(1e-6, initial_volume * 1e-12)
    maximum_surface_error = float(np.max(np.abs(surface_after - expected_surface)))
    maximum_matrix_error = float(np.max(np.abs(matrix - expected_matrix)))
    maximum_node_residual = float(np.max(np.abs(residual)))
    global_volume_residual = float(
        np.sum(storage * (surface_after - surface_before))
    )
    if maximum_surface_error > 1e-14:
        raise AssertionError("manufactured surface solution does not close")
    if maximum_matrix_error > 1e-14:
        raise AssertionError("manufactured junction matrix is incorrect")
    if maximum_node_residual > closure_limit:
        raise AssertionError("manufactured junction node closure failed")
    if abs(global_volume_residual) > closure_limit:
        raise AssertionError("manufactured global volume closure failed")
    if np.any(flux < 0):
        raise AssertionError("manufactured junction produced reverse flux")
    if not np.array_equal(np.sum(incidence, axis=0), np.zeros(2)):
        raise AssertionError("junction incidence columns do not conserve mass")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0):
        raise AssertionError("manufactured matrix is not symmetric")
    if np.any(np.diag(matrix) <= 0):
        raise AssertionError("manufactured matrix has nonpositive diagonal")
    np.linalg.cholesky(matrix)
    return {
        "status": "pass",
        "dt_seconds": dt,
        "surface_solution": surface_after.tolist(),
        "expected_surface_solution": expected_surface.tolist(),
        "edge_flux_m3_s": flux.tolist(),
        "maximum_surface_error_m": maximum_surface_error,
        "maximum_matrix_error": maximum_matrix_error,
        "maximum_absolute_node_residual_m3": maximum_node_residual,
        "global_volume_residual_m3": global_volume_residual,
        "closure_limit_m3": closure_limit,
        "matrix_cholesky": "pass",
    }


def run_zero_diffusivity_identity_fixture() -> dict:
    """Verify an exact bitwise identity when every conductance is zero."""

    surface_before = np.array(
        [4123.25, 4077.5, 4001.125, 3950.75, 3899.0], dtype=np.float64
    )
    bed = np.array([4000.0, 3950.0, 3875.0, 3820.0, 3750.0], dtype=np.float64)
    width = np.array([120.0, 135.0, 150.0, 165.0, 180.0], dtype=np.float64)
    storage = width * 50.0
    incidence = chain_incidence(surface_before.size)
    conductance = np.zeros(incidence.shape[1], dtype=np.float64)
    section_before = (surface_before - bed) * width
    surface_after = step_surface(
        surface_before, storage, incidence, conductance, 30.0
    )
    section_after = (surface_after - bed) * width
    flux = edge_flux(surface_after, incidence, conductance)
    residual = control_volume_residual_m3(
        surface_before, surface_after, storage, incidence, conductance, 30.0
    )
    if not np.array_equal(surface_after, surface_before):
        raise AssertionError("zero diffusivity changed surface bits")
    if not np.array_equal(section_after, section_before):
        raise AssertionError("zero diffusivity changed section bits")
    if not np.array_equal(flux, np.zeros_like(flux)):
        raise AssertionError("zero diffusivity produced flux")
    if not np.array_equal(residual, np.zeros_like(residual)):
        raise AssertionError("zero diffusivity produced a control-volume residual")
    return {
        "status": "pass",
        "surface_bitwise_identity": True,
        "section_bitwise_identity": True,
        "flux_bitwise_zero": True,
        "residual_bitwise_zero": True,
    }


def run_all_fixtures() -> dict:
    fixtures = []
    specs = [
        ("single_line_stock", run_single_line_stock_fixture),
        ("two_tributary_manufactured", run_two_tributary_manufactured_fixture),
        ("zero_diffusivity_identity", run_zero_diffusivity_identity_fixture),
    ]
    any_failure = False
    for name, function in specs:
        try:
            result = function()
            fixtures.append({"name": name, **result})
        except Exception as exc:  # record every fixture, but close on any fail
            any_failure = True
            fixtures.append(
                {
                    "name": name,
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    return {
        "status": (
            "fixture-fail-mechanism-closed"
            if any_failure
            else "fixture-pass-real-system-not-authorized"
        ),
        "experiment": "lia-191-network-coupled-semiimplicit-fixture-v1",
        "fixtures": fixtures,
        "fixture_count": len(fixtures),
        "all_fixtures_pass": not any_failure,
        "mathematical_step": "(diag(C)+dt*B*diag(K)*B.T)*s_new=diag(C)*s_old",
        "minimum_real_system_timestep_seconds": 30.0,
        "real_system_authorized": False,
        "system_2005_read": False,
        "glacier_directory_created": False,
        "historical_run": False,
        "synthetic_only": True,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary = run_all_fixtures()
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if not summary["all_fixtures_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
