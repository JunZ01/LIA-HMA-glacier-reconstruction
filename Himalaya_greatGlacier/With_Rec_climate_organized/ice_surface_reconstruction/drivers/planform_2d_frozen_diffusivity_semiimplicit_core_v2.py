"""Reduced-system semi-implicit step with numerical-zero diffusivity evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from .planform_2d_frozen_diffusivity_semiimplicit_core_v1 import (
    graph_laplacian,
    upstream_face_diffusivities,
)


def frozen_diffusivity_step(model, dt, numerical_zero_m=1e-12, tol=1e-12, maxiter=4000):
    original_h = np.asarray(model.ice_thick, dtype=float)
    if float(np.min(original_h)) < -float(numerical_zero_m):
        raise ValueError("state-below-numerical-zero-tolerance")
    evaluation_h = np.where(original_h < 0.0, 0.0, original_h)
    proxy = SimpleNamespace(
        ice_thick=evaluation_h,
        surface_h=np.asarray(model.bed_topo, float) + evaluation_h,
        glen_n=model.glen_n,
        gamma=model.gamma,
        dx=model.dx,
        dy=model.dy,
        ny=model.ny,
        nx=model.nx,
    )
    east_d, south_d, symmetry = upstream_face_diffusivities(proxy)
    laplacian, edge_count = graph_laplacian(
        model.ny, model.nx,
        east_d / float(model.dx) ** 2,
        south_d / float(model.dy) ** 2,
    )
    active = laplacian.diagonal() > 0
    delta = np.zeros(model.ny * model.nx, dtype=float)
    surface0 = np.asarray(model.surface_h, dtype=float).ravel()
    if np.any(active):
        reduced_l = laplacian[active][:, active].tocsr()
        operator = sparse.identity(int(active.sum()), format="csr") + float(dt) * reduced_l
        rhs = -float(dt) * laplacian[active, :].dot(surface0)
        diagonal = operator.diagonal()
        preconditioner = LinearOperator(operator.shape, matvec=lambda value: value / diagonal)
        solution, info = cg(
            operator, rhs, tol=tol, atol=0.0, maxiter=maxiter, M=preconditioner
        )
        delta[active] = solution
        residual = operator.dot(solution) - rhs
        residual_l2 = float(np.linalg.norm(residual))
        residual_linf = float(np.max(np.abs(residual)))
        symmetric = float(abs(operator - operator.T).max())
        positive_diagonal = bool(np.all(diagonal > 0))
        finite_matrix = bool(np.isfinite(operator.data).all())
    else:
        info, residual_l2, residual_linf, symmetric = 0, 0.0, 0.0, 0.0
        positive_diagonal, finite_matrix = True, True
    thickness1 = original_h + delta.reshape(model.ny, model.nx)
    negative_mask = thickness1 < 0
    diagnostics = {
        "solver_info": int(info),
        "edge_count": edge_count,
        "active_linear_unknown_count": int(active.sum()),
        "matrix_finite": finite_matrix,
        "matrix_positive_diagonal": positive_diagonal,
        "matrix_symmetric_max_abs": symmetric,
        "linear_residual_l2": residual_l2,
        "linear_residual_linf": residual_linf,
        "delta_sum_m": float(np.sum(delta)),
        "minimum_thickness_m": float(np.min(thickness1)),
        "negative_cell_count": int(np.sum(negative_mask)),
        "negative_thickness_sum_m": float(-np.sum(thickness1[negative_mask])),
        "diffusivity_evaluation_zeroed_negative_cell_count": int(np.sum(original_h < 0)),
        "face_symmetry": symmetry,
    }
    return thickness1, diagnostics
