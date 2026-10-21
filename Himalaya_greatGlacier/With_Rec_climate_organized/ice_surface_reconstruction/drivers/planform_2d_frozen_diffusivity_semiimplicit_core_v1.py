"""Mass-conservative frozen-diffusivity semi-implicit operator for 2-D SIA."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg


def upstream_face_diffusivities(model):
    """Reproduce OGGM Upstream2D face diffusivities and symmetry diagnostics."""
    H = np.asarray(model.ice_thick, dtype=float)
    S = np.asarray(model.surface_h, dtype=float)
    n = float(model.glen_n)
    ny, nx = H.shape
    kp = np.r_[np.arange(1, ny), ny - 1]
    km = np.r_[0, np.arange(ny - 1)]
    lp = np.r_[np.arange(1, nx), nx - 1]
    lm = np.r_[0, np.arange(nx - 1)]
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    s0 = S[rows, cols]
    slp, slm = S[rows, lp[cols]], S[rows, lm[cols]]
    skp, skm = S[kp[rows], cols], S[km[rows], cols]
    h0 = H[rows, cols]
    hlp, hlm = H[rows, lp[cols]], H[rows, lm[cols]]
    hkp, hkm = H[kp[rows], cols], H[km[rows], cols]
    dx, dy = float(model.dx), float(model.dy)

    h_l_up = 0.5 * (hlp + h0)
    h_l_dn = 0.5 * (h0 + hlm)
    h_l_us_up = np.where(slp > s0, hlp, h0)
    h_l_us_dn = np.where(s0 > slm, h0, hlm)
    sdiff = skp - skm
    slpdiff = slp - s0
    slmdiff = s0 - slm
    sl_grad_up = (
        ((sdiff + S[kp[rows], lp[cols]] - S[km[rows], lp[cols]]) ** 2 / (4 * dx) ** 2)
        + slpdiff**2 / dy**2
    ) ** ((n - 1) / 2)
    sl_grad_dn = (
        ((sdiff + S[kp[rows], lm[cols]] - S[km[rows], lm[cols]]) ** 2 / (4 * dx) ** 2)
        + slmdiff**2 / dy**2
    ) ** ((n - 1) / 2)
    d_l_up = model.gamma * h_l_up ** (n + 1) * h_l_us_up * sl_grad_up
    d_l_dn = model.gamma * h_l_dn ** (n + 1) * h_l_us_dn * sl_grad_dn

    h_k_up = 0.5 * (hkp + h0)
    h_k_dn = 0.5 * (h0 + hkm)
    h_k_us_up = np.where(skp > s0, hkp, h0)
    h_k_us_dn = np.where(s0 > skm, h0, hkm)
    sdiff = slp - slm
    skpdiff = skp - s0
    skmdiff = s0 - skm
    sk_grad_up = (
        ((sdiff + S[kp[rows], lp[cols]] - S[kp[rows], lm[cols]]) ** 2 / (4 * dy) ** 2)
        + skpdiff**2 / dx**2
    ) ** ((n - 1) / 2)
    sk_grad_dn = (
        ((sdiff + S[km[rows], lp[cols]] - S[km[rows], lm[cols]]) ** 2 / (4 * dy) ** 2)
        + skmdiff**2 / dx**2
    ) ** ((n - 1) / 2)
    d_k_up = model.gamma * h_k_up ** (n + 1) * h_k_us_up * sk_grad_up
    d_k_dn = model.gamma * h_k_dn ** (n + 1) * h_k_us_dn * sk_grad_dn

    east = np.asarray(d_l_up[:, :-1], float)
    south = np.asarray(d_k_up[:-1, :], float)
    east_opposite = np.asarray(d_l_dn[:, 1:], float)
    south_opposite = np.asarray(d_k_dn[1:, :], float)
    scale_e = max(float(np.max(np.abs(east))), 1.0)
    scale_s = max(float(np.max(np.abs(south))), 1.0)
    symmetry = {
        "east_max_absolute_difference": float(np.max(np.abs(east - east_opposite))),
        "east_max_relative_difference": float(np.max(np.abs(east - east_opposite)) / scale_e),
        "south_max_absolute_difference": float(np.max(np.abs(south - south_opposite))),
        "south_max_relative_difference": float(np.max(np.abs(south - south_opposite)) / scale_s),
    }
    return east, south, symmetry


def graph_laplacian(ny, nx, east_conductance, south_conductance):
    """Return the symmetric no-flux graph Laplacian in s^-1."""
    east = np.asarray(east_conductance, dtype=float)
    south = np.asarray(south_conductance, dtype=float)
    if east.shape != (ny, nx - 1) or south.shape != (ny - 1, nx):
        raise ValueError("face-conductance-shape-mismatch")
    if np.any(~np.isfinite(east)) or np.any(~np.isfinite(south)):
        raise ValueError("nonfinite-face-conductance")
    if np.any(east < 0) or np.any(south < 0):
        raise ValueError("negative-face-conductance")
    east_rows, east_cols = np.nonzero(east > 0)
    south_rows, south_cols = np.nonzero(south > 0)
    u_e = east_rows * nx + east_cols
    v_e = u_e + 1
    k_e = east[east_rows, east_cols]
    u_s = south_rows * nx + south_cols
    v_s = u_s + nx
    k_s = south[south_rows, south_cols]
    u = np.r_[u_e, u_s]
    v = np.r_[v_e, v_s]
    weight = np.r_[k_e, k_s]
    rows = np.r_[u, v, u, v]
    cols = np.r_[u, v, v, u]
    data = np.r_[weight, weight, -weight, -weight]
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(ny * nx, ny * nx)).tocsr()
    return matrix, int(len(weight))


def frozen_diffusivity_step(model, dt, rtol=1e-11, maxiter=2000):
    """Advance one un-clipped semi-implicit step and return state plus diagnostics."""
    east_d, south_d, symmetry = upstream_face_diffusivities(model)
    east_k = east_d / float(model.dx) ** 2
    south_k = south_d / float(model.dy) ** 2
    laplacian, edge_count = graph_laplacian(model.ny, model.nx, east_k, south_k)
    size = model.ny * model.nx
    identity = sparse.identity(size, format="csr")
    operator = identity + float(dt) * laplacian
    surface0 = np.asarray(model.surface_h, dtype=float).ravel()
    rhs = -float(dt) * laplacian.dot(surface0)
    diagonal = operator.diagonal()
    preconditioner = LinearOperator((size, size), matvec=lambda value: value / diagonal)
    delta, info = cg(operator, rhs, rtol=rtol, atol=0.0, maxiter=maxiter, M=preconditioner)
    residual = operator.dot(delta) - rhs
    thickness1 = np.asarray(model.ice_thick, dtype=float) + delta.reshape(model.ny, model.nx)
    diagnostics = {
        "solver_info": int(info),
        "edge_count": edge_count,
        "matrix_finite": bool(np.isfinite(operator.data).all()),
        "matrix_positive_diagonal": bool(np.all(diagonal > 0)),
        "matrix_symmetric_max_abs": float(abs(operator - operator.T).max()),
        "linear_residual_l2": float(np.linalg.norm(residual)),
        "linear_residual_linf": float(np.max(np.abs(residual))),
        "delta_sum_m": float(np.sum(delta)),
        "minimum_thickness_m": float(np.min(thickness1)),
        "maximum_thickness_m": float(np.max(thickness1)),
        "face_symmetry": symmetry,
        "maximum_east_diffusivity_m2_s": float(np.max(east_d)),
        "maximum_south_diffusivity_m2_s": float(np.max(south_d)),
    }
    return thickness1, diagnostics
