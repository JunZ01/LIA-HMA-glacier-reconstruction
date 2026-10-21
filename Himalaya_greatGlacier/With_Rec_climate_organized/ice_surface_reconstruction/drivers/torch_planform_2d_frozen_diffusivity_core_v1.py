"""Matrix-free PyTorch backend for one frozen-diffusivity 2-D SIA step.

The implementation mirrors the face diffusivity used by the project's
validated NumPy/SciPy operator, but keeps all state and Krylov operations on a
single torch device.  It deliberately performs no thickness clipping, runtime
limiting, mass correction, or boundary outflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


def _east(value: Tensor) -> Tensor:
    return torch.cat((value[:, 1:], value[:, -1:]), dim=1)


def _west(value: Tensor) -> Tensor:
    return torch.cat((value[:, :1], value[:, :-1]), dim=1)


def _south(value: Tensor) -> Tensor:
    return torch.cat((value[1:, :], value[-1:, :]), dim=0)


def _north(value: Tensor) -> Tensor:
    return torch.cat((value[:1, :], value[:-1, :]), dim=0)


def upstream_face_diffusivities(
    ice_thick: Tensor,
    surface_h: Tensor,
    *,
    dx: float,
    dy: float,
    glen_n: float,
    gamma: float,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Return east/south face diffusivities with replicated outer neighbors."""
    if ice_thick.ndim != 2 or surface_h.shape != ice_thick.shape:
        raise ValueError("two-dimensional-shape-mismatch")
    if dx <= 0 or dy <= 0 or glen_n <= 1 or gamma < 0:
        raise ValueError("invalid-physical-or-grid-parameter")
    if not bool(torch.isfinite(ice_thick).all() and torch.isfinite(surface_h).all()):
        raise ValueError("nonfinite-input")
    if bool((ice_thick < 0).any()):
        raise ValueError("negative-input-thickness")

    h0 = ice_thick
    s0 = surface_h
    hlp, hlm = _east(h0), _west(h0)
    hkp, hkm = _south(h0), _north(h0)
    slp, slm = _east(s0), _west(s0)
    skp, skm = _south(s0), _north(s0)

    h_l_up = 0.5 * (hlp + h0)
    h_l_dn = 0.5 * (h0 + hlm)
    h_l_us_up = torch.where(slp > s0, hlp, h0)
    h_l_us_dn = torch.where(s0 > slm, h0, hlm)
    sdiff = skp - skm
    slpdiff = slp - s0
    slmdiff = s0 - slm
    sl_grad_up = (
        ((sdiff + _south(slp) - _north(slp)) ** 2 / (4.0 * dx) ** 2)
        + slpdiff**2 / dy**2
    ) ** ((glen_n - 1.0) / 2.0)
    sl_grad_dn = (
        ((sdiff + _south(slm) - _north(slm)) ** 2 / (4.0 * dx) ** 2)
        + slmdiff**2 / dy**2
    ) ** ((glen_n - 1.0) / 2.0)
    d_l_up = gamma * h_l_up ** (glen_n + 1.0) * h_l_us_up * sl_grad_up
    d_l_dn = gamma * h_l_dn ** (glen_n + 1.0) * h_l_us_dn * sl_grad_dn

    h_k_up = 0.5 * (hkp + h0)
    h_k_dn = 0.5 * (h0 + hkm)
    h_k_us_up = torch.where(skp > s0, hkp, h0)
    h_k_us_dn = torch.where(s0 > skm, h0, hkm)
    sdiff = slp - slm
    skpdiff = skp - s0
    skmdiff = s0 - skm
    sk_grad_up = (
        ((sdiff + _east(skp) - _west(skp)) ** 2 / (4.0 * dy) ** 2)
        + skpdiff**2 / dx**2
    ) ** ((glen_n - 1.0) / 2.0)
    sk_grad_dn = (
        ((sdiff + _east(skm) - _west(skm)) ** 2 / (4.0 * dy) ** 2)
        + skmdiff**2 / dx**2
    ) ** ((glen_n - 1.0) / 2.0)
    d_k_up = gamma * h_k_up ** (glen_n + 1.0) * h_k_us_up * sk_grad_up
    d_k_dn = gamma * h_k_dn ** (glen_n + 1.0) * h_k_us_dn * sk_grad_dn

    east = d_l_up[:, :-1]
    south = d_k_up[:-1, :]
    east_opposite = d_l_dn[:, 1:]
    south_opposite = d_k_dn[1:, :]
    scale_e = max(float(torch.max(torch.abs(east)).item()), 1.0)
    scale_s = max(float(torch.max(torch.abs(south)).item()), 1.0)
    symmetry = {
        "east_max_absolute_difference": float(torch.max(torch.abs(east - east_opposite)).item()),
        "east_max_relative_difference": float(torch.max(torch.abs(east - east_opposite)).item()) / scale_e,
        "south_max_absolute_difference": float(torch.max(torch.abs(south - south_opposite)).item()),
        "south_max_relative_difference": float(torch.max(torch.abs(south - south_opposite)).item()) / scale_s,
    }
    return east, south, symmetry


def apply_graph_laplacian(value: Tensor, east: Tensor, south: Tensor) -> Tensor:
    """Apply a symmetric no-flux graph Laplacian using paired face updates."""
    ny, nx = value.shape
    if east.shape != (ny, nx - 1) or south.shape != (ny - 1, nx):
        raise ValueError("face-conductance-shape-mismatch")
    if not bool(torch.isfinite(east).all() and torch.isfinite(south).all()):
        raise ValueError("nonfinite-face-conductance")
    if bool((east < 0).any() or (south < 0).any()):
        raise ValueError("negative-face-conductance")
    result = torch.zeros_like(value)
    east_exchange = east * (value[:, :-1] - value[:, 1:])
    result[:, :-1] += east_exchange
    result[:, 1:] -= east_exchange
    south_exchange = south * (value[:-1, :] - value[1:, :])
    result[:-1, :] += south_exchange
    result[1:, :] -= south_exchange
    return result


def jacobi_diagonal(east: Tensor, south: Tensor, dt: float, shape: tuple[int, int]) -> Tensor:
    diagonal = torch.ones(shape, dtype=east.dtype, device=east.device)
    diagonal[:, :-1] += dt * east
    diagonal[:, 1:] += dt * east
    diagonal[:-1, :] += dt * south
    diagonal[1:, :] += dt * south
    return diagonal


@dataclass(frozen=True)
class CGResult:
    solution: Tensor
    iterations: int
    converged: bool
    residual_l2: float
    residual_linf: float


def preconditioned_conjugate_gradient(
    rhs: Tensor,
    east: Tensor,
    south: Tensor,
    *,
    dt: float,
    rtol: float = 1e-11,
    atol: float = 0.0,
    maxiter: int = 4000,
) -> CGResult:
    """Solve ``(I + dt L)x = rhs`` with a Jacobi-preconditioned CG loop."""
    if dt < 0 or rtol < 0 or atol < 0 or maxiter < 1:
        raise ValueError("invalid-cg-parameter")
    diagonal = jacobi_diagonal(east, south, dt, tuple(rhs.shape))
    if not bool(torch.isfinite(diagonal).all() and (diagonal > 0).all()):
        raise ValueError("invalid-operator-diagonal")

    def operator(value: Tensor) -> Tensor:
        return value + dt * apply_graph_laplacian(value, east, south)

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
    for iteration in range(1, maxiter + 1):
        product = operator(direction)
        denominator = torch.sum(direction * product)
        if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
            break
        alpha = rz_old / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * product
        iterations = iteration
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if residual_norm <= threshold:
            converged = True
            break
        z = residual / diagonal
        rz_new = torch.sum(residual * z)
        if not bool(torch.isfinite(rz_new)):
            break
        beta = rz_new / rz_old
        direction = z + beta * direction
        rz_old = rz_new
    final_residual = operator(solution) - rhs
    return CGResult(
        solution=solution,
        iterations=iterations,
        converged=converged,
        residual_l2=float(torch.linalg.vector_norm(final_residual).item()),
        residual_linf=float(torch.max(torch.abs(final_residual)).item()),
    )


def frozen_diffusivity_step(
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
) -> tuple[Tensor, dict[str, object]]:
    """Advance one unclipped, no-flux, frozen-diffusivity step."""
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
    rhs = -dt * apply_graph_laplacian(surface0, east_k, south_k)
    cg = preconditioned_conjugate_gradient(
        rhs,
        east_k,
        south_k,
        dt=dt,
        rtol=rtol,
        maxiter=maxiter,
    )
    thickness1 = ice_thick + cg.solution
    diagnostics: dict[str, object] = {
        "solver_converged": cg.converged,
        "solver_iterations": cg.iterations,
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
