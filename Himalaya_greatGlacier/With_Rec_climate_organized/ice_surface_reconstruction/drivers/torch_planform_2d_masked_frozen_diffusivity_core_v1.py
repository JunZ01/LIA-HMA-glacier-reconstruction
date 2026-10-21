"""Source-domain masked frozen-diffusivity core for research fixtures."""

from __future__ import annotations

import torch

from .torch_planform_2d_frozen_diffusivity_core_v1 import upstream_face_diffusivities
from .torch_planform_2d_frozen_diffusivity_core_v2 import (
    _apply_graph_laplacian_unchecked,
    preconditioned_conjugate_gradient_chunked,
)


def masked_frozen_diffusivity_step(
    ice_thick: torch.Tensor,
    bed_topo: torch.Tensor,
    domain_mask: torch.Tensor,
    *,
    dx: float,
    dy: float,
    glen_n: float,
    gamma: float,
    dt: float,
    rtol: float = 1e-11,
    maxiter: int = 4000,
    check_interval: int = 8,
) -> tuple[torch.Tensor, dict[str, object]]:
    if ice_thick.shape != bed_topo.shape or domain_mask.shape != ice_thick.shape:
        raise ValueError("masked-state-shape-mismatch")
    if domain_mask.dtype != torch.bool:
        raise ValueError("domain-mask-not-boolean")
    if bool(torch.any(ice_thick[~domain_mask] != 0.0)):
        raise ValueError("nonzero-initial-thickness-outside-domain")
    surface0 = bed_topo + ice_thick
    east_d, south_d, symmetry = upstream_face_diffusivities(
        ice_thick, surface0, dx=dx, dy=dy, glen_n=glen_n, gamma=gamma
    )
    east_faces = domain_mask[:, :-1] & domain_mask[:, 1:]
    south_faces = domain_mask[:-1, :] & domain_mask[1:, :]
    east_d = torch.where(east_faces, east_d, torch.zeros_like(east_d))
    south_d = torch.where(south_faces, south_d, torch.zeros_like(south_d))
    east_k = east_d / dx**2
    south_k = south_d / dy**2
    rhs = -dt * _apply_graph_laplacian_unchecked(surface0, east_k, south_k)
    cg = preconditioned_conjugate_gradient_chunked(
        rhs, east_k, south_k, dt=dt, rtol=rtol, maxiter=maxiter, check_interval=check_interval
    )
    thickness1 = ice_thick + cg.solution
    outside_max = float(torch.max(torch.abs(thickness1[~domain_mask])).item())
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
        "open_east_face_count": int(torch.count_nonzero(east_faces).item()),
        "open_south_face_count": int(torch.count_nonzero(south_faces).item()),
        "outside_domain_maximum_thickness_m": outside_max,
        "cross_domain_leakage_zero": outside_max == 0.0,
    }
    return thickness1, diagnostics

