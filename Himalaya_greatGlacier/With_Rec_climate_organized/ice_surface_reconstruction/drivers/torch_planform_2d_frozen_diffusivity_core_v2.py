"""Chunk-synchronized PyTorch backend for frozen-diffusivity 2-D SIA.

V2 preserves the v1 linear operator and physical equations.  It validates
fixed face conductances once, uses an unchecked paired-face Laplacian inside
PCG, and synchronizes convergence diagnostics only after a short iteration
chunk instead of on every iteration.
"""

from __future__ import annotations

import torch

from .torch_planform_2d_frozen_diffusivity_core_v1 import (
    CGResult,
    jacobi_diagonal,
    upstream_face_diffusivities,
)


Tensor = torch.Tensor


def _apply_graph_laplacian_unchecked(value: Tensor, east: Tensor, south: Tensor) -> Tensor:
    result = torch.zeros_like(value)
    east_exchange = east * (value[:, :-1] - value[:, 1:])
    result[:, :-1] += east_exchange
    result[:, 1:] -= east_exchange
    south_exchange = south * (value[:-1, :] - value[1:, :])
    result[:-1, :] += south_exchange
    result[1:, :] -= south_exchange
    return result


def preconditioned_conjugate_gradient_chunked(
    rhs: Tensor,
    east: Tensor,
    south: Tensor,
    *,
    dt: float,
    rtol: float = 1e-11,
    atol: float = 0.0,
    maxiter: int = 4000,
    check_interval: int = 8,
) -> CGResult:
    """Solve ``(I + dt L)x = rhs`` with chunked convergence checks."""
    if dt < 0 or rtol < 0 or atol < 0 or maxiter < 1 or check_interval < 1:
        raise ValueError("invalid-cg-parameter")
    ny, nx = rhs.shape
    if east.shape != (ny, nx - 1) or south.shape != (ny - 1, nx):
        raise ValueError("face-conductance-shape-mismatch")
    valid_faces = torch.isfinite(east).all() & torch.isfinite(south).all()
    valid_faces = valid_faces & (east >= 0).all() & (south >= 0).all()
    if not bool(valid_faces):
        raise ValueError("invalid-face-conductance")
    diagonal = jacobi_diagonal(east, south, dt, tuple(rhs.shape))
    if not bool(torch.isfinite(diagonal).all() & (diagonal > 0).all()):
        raise ValueError("invalid-operator-diagonal")

    def operator(value: Tensor) -> Tensor:
        return value + dt * _apply_graph_laplacian_unchecked(value, east, south)

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = float(torch.linalg.vector_norm(rhs).item())
    threshold = max(float(atol), float(rtol) * rhs_norm)
    if rhs_norm == 0.0:
        return CGResult(solution, 0, True, 0.0, 0.0)
    z = residual / diagonal
    direction = z.clone()
    rz_old = torch.sum(residual * z)
    converged = False
    iterations = 0
    residual_norm = rhs_norm
    while iterations < maxiter:
        block_size = min(check_interval, maxiter - iterations)
        for _ in range(block_size):
            product = operator(direction)
            denominator = torch.sum(direction * product)
            alpha = rz_old / denominator
            solution = solution + alpha * direction
            residual = residual - alpha * product
            z = residual / diagonal
            rz_new = torch.sum(residual * z)
            beta = rz_new / rz_old
            direction = z + beta * direction
            rz_old = rz_new
            iterations += 1
        finite_checkpoint = torch.isfinite(residual).all() & torch.isfinite(solution).all()
        finite_checkpoint = finite_checkpoint & torch.isfinite(rz_old)
        if not bool(finite_checkpoint):
            break
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if residual_norm <= threshold:
            converged = True
            break
    final_residual = operator(solution) - rhs
    return CGResult(
        solution=solution,
        iterations=iterations,
        converged=converged,
        residual_l2=float(torch.linalg.vector_norm(final_residual).item()),
        residual_linf=float(torch.max(torch.abs(final_residual)).item()),
    )


def frozen_diffusivity_step_chunked(
    ice_thick: Tensor,
    bed_topo: Tensor,
    *,
    dx: float,
    dy: float,
    glen_n: float,
    gamma: float,
    dt: float,
    rtol: float = 1e-11,
    maxiter: int = 4000,
    check_interval: int = 8,
) -> tuple[Tensor, dict[str, object]]:
    """Advance one unclipped, no-flux step with chunk-synchronized PCG."""
    if ice_thick.shape != bed_topo.shape:
        raise ValueError("state-shape-mismatch")
    surface0 = bed_topo + ice_thick
    east_d, south_d, symmetry = upstream_face_diffusivities(
        ice_thick,
        surface0,
        dx=dx,
        dy=dy,
        glen_n=glen_n,
        gamma=gamma,
    )
    east_k = east_d / dx**2
    south_k = south_d / dy**2
    rhs = -dt * _apply_graph_laplacian_unchecked(surface0, east_k, south_k)
    cg = preconditioned_conjugate_gradient_chunked(
        rhs,
        east_k,
        south_k,
        dt=dt,
        rtol=rtol,
        maxiter=maxiter,
        check_interval=check_interval,
    )
    thickness1 = ice_thick + cg.solution
    diagnostics: dict[str, object] = {
        "solver_converged": cg.converged,
        "solver_iterations": cg.iterations,
        "solver_check_interval": check_interval,
        "linear_residual_l2": cg.residual_l2,
        "linear_residual_linf": cg.residual_linf,
        "delta_sum_m": float(torch.sum(cg.solution).item()),
        "minimum_thickness_m": float(torch.min(thickness1).item()),
        "maximum_thickness_m": float(torch.max(thickness1).item()),
        "negative_cell_count": int(torch.count_nonzero(thickness1 < 0).item()),
        "state_finite": bool(torch.isfinite(thickness1).all()),
        "face_symmetry": symmetry,
        "maximum_east_diffusivity_m2_s": float(torch.max(east_d).item()),
        "maximum_south_diffusivity_m2_s": float(torch.max(south_d).item()),
    }
    return thickness1, diagnostics
