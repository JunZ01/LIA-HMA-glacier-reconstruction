"""Where does operator A still miss on the three modern controls, and why?

Reads only the NumPy exports from ``export_operator_controls_npz_20261021.py``
(modern_control_<uid>.npz). Uses the same raster boundary and 100 m constraint
exclusion as ``diagnose_extension_operators_20261021.py``.

Experiments, all on the frozen modern grids and constraint cells:
  1. Reference thickness profile against distance to the margin (the
     "margin wall" that the zero-boundary harmonic operator propagates inward).
  2. Operator A with the margin surface at B (current, H = 0) versus at the
     observed DEM (H = reference on margin cells): brackets the effect of the
     reference margin wall on A.
  3. Smoother surface interpolants: thin-plate (biharmonic) and tension splines.
  4. Constraint density: 1/2 and 1/4 of the flowline cells, and an oracle that
     adds DEM surface on medial-axis cells.
  5. Where A clips negative thickness (distance to margin, DEM slope).

Offline operator diagnostic only; no PP/tau fit, LIA reconstruction or model run.

Usage (repo root):
    python analyze_surface_operator_residuals_20261021.py NPZ_DIR OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent / "Himalaya_greatGlacier/With_Rec_climate_organized"))
from ice_surface_reconstruction.core.extension_operators import (  # noqa: E402
    dirichlet_poisson_solve,
    raster_boundary,
)

SYSTEMS = (3632, 3131, 2906)
GAP_M = 100.0
MARGIN_BINS_PX = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 10.5),
                  (10.5, 20.5), (20.5, 40.5), (40.5, np.inf)]


def load(npz_dir: Path, uid: int) -> dict:
    z = np.load(npz_dir / f"modern_control_{uid}.npz", allow_pickle=False)
    H = z["reference_H_m"].astype(float)
    T = z["topo_m"].astype(float)
    M = z["modern_mask"].astype(bool)
    dx = float(z["dx_m"])
    bnd = raster_boundary(M)
    dist = distance_transform_edt(~bnd) * dx
    rr, cc = z["constraint_rows"].astype(int), z["constraint_cols"].astype(int)
    keep = ~bnd[rr, cc] & (dist[rr, cc] >= GAP_M)
    return {"uid": uid, "region": str(z["region"]), "H": H, "T": T, "B": T - H, "M": M,
            "dx": dx, "bnd": bnd, "inner": M & ~bnd, "dist": dist,
            "rr": rr[keep], "cc": cc[keep]}


def _laplacian(M):
    idx = np.full(M.shape, -1)
    idx[M] = np.arange(M.sum())
    inner = M & ~raster_boundary(M)
    r, c = np.nonzero(inner)
    n = len(r)
    rows, cols, vals = [np.arange(n)], [idx[r, c]], [np.full(n, -4.0)]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rows.append(np.arange(n))
        cols.append(idx[r + dr, c + dc])
        vals.append(np.ones(n))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, M.sum()))


def _gradient(M):
    idx = np.full(M.shape, -1)
    idx[M] = np.arange(M.sum())
    R, C, V, k = [], [], [], 0
    for dr, dc in ((1, 0), (0, 1)):
        shifted = np.zeros_like(M)
        if dr:
            shifted[:-1, :] = M[1:, :]
        else:
            shifted[:, :-1] = M[:, 1:]
        r, c = np.nonzero(M & shifted)
        n = len(r)
        R += [np.arange(k, k + n)] * 2
        C += [idx[r, c], idx[r + dr, c + dc]]
        V += [-np.ones(n), np.ones(n)]
        k += n
    return sparse.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                             shape=(k, M.sum()))


def spline_surface(M, fixed, values, tension):
    """Minimise ||lap s||^2 + tension*||grad s||^2 with s fixed on ``fixed``."""
    L, G = _laplacian(M), _gradient(M)
    f = fixed[M]
    u = ~f
    A = (L.T @ L + tension * (G.T @ G)).tocsr()
    xf = values[M][f]
    xu = spsolve(A[u][:, u].tocsc(), -(A[u][:, f] @ xf))
    full = np.empty(M.sum())
    full[f] = xf
    full[u] = xu
    s = np.full(M.shape, np.nan)
    s[M] = full
    return s


def surface_fixed(d, rr, cc, margin_surface):
    fixed = d["bnd"].copy()
    fixed[rr, cc] = True
    values = np.where(d["bnd"], margin_surface, np.nan)
    values[rr, cc] = d["T"][rr, cc]
    return fixed, values


def metrics(d, S, label, **extra):
    M, H, B, T, inner = d["M"], d["H"], d["B"], d["T"], d["inner"]
    raw = S - B
    h = np.where(M, np.maximum(raw, 0.0), 0.0)
    e = S[inner] - T[inner]
    near = inner & (d["dist"] < 100)
    return {"lia_unique_id": d["uid"], "variant": label, **extra,
            "surface_rmse_vs_dem_m": float(np.sqrt(np.mean(e ** 2))),
            "surface_mean_error_m": float(e.mean()),
            "surface_mean_error_within_100m_of_margin_m": float(np.mean(S[near] - T[near])),
            "negative_raw_cells_percent": float(100 * (raw[M] < 0).mean()),
            "raw_volume_bias_percent": float(100 * (raw[M].sum() / H[M].sum() - 1)),
            "clipped_volume_bias_percent": float(100 * (h[M].sum() / H[M].sum() - 1)),
            "thickness_rmse_m": float(np.sqrt(np.mean((h[M] - H[M]) ** 2)))}


def harmonic_surface(d, rr, cc, margin_surface):
    fixed, values = surface_fixed(d, rr, cc, margin_surface)
    S, meta = dirichlet_poisson_solve(d["M"], fixed, values)
    assert meta["status"] == "converged", meta
    return S


def run(npz_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile, variants, clipping = [], [], []
    rng = np.random.default_rng(0)
    for uid in SYSTEMS:
        d = load(npz_dir, uid)
        M, H, T, B, dx = d["M"], d["H"], d["T"], d["B"], d["dx"]
        rr, cc = d["rr"], d["cc"]
        dpx = distance_transform_edt(M)
        for lo, hi in MARGIN_BINS_PX:
            s = M & (dpx >= lo) & (dpx < hi)
            profile.append({"lia_unique_id": uid, "margin_distance_m": f"{lo * dx:.0f}-{hi * dx:.0f}",
                            "cells": int(s.sum()), "mean_reference_H_m": float(H[s].mean()),
                            "reference_volume_share_percent": float(100 * H[s].sum() / H[M].sum())})
        # 2. margin treatment
        S_A = harmonic_surface(d, rr, cc, B)
        variants.append(metrics(d, S_A, "A harmonic, margin S=B (H=0)", constraints=len(rr)))
        variants.append(metrics(d, harmonic_surface(d, rr, cc, T),
                                "A harmonic, margin S=DEM", constraints=len(rr)))
        # 3. smoother interpolants
        fixed, values = surface_fixed(d, rr, cc, B)
        for tension in (1e-1, 1e-2, 0.0):
            label = "biharmonic, margin S=B" if tension == 0 else f"tension {tension:g}, margin S=B"
            variants.append(metrics(d, spline_surface(M, fixed, values, tension), label,
                                    constraints=len(rr)))
        # 4. constraint density
        for k in (2, 4):
            variants.append(metrics(d, harmonic_surface(d, rr[::k], cc[::k], B),
                                    f"A harmonic, 1/{k} constraints", constraints=len(rr[::k])))
        ridge = M & (dpx >= maximum_filter(dpx, size=5) - 0.5) & (d["dist"] >= GAP_M)
        r2, c2 = np.nonzero(ridge)
        pick = rng.permutation(len(r2))[:len(rr)]
        r3, c3 = np.concatenate([rr, r2[pick]]), np.concatenate([cc, c2[pick]])
        variants.append(metrics(d, harmonic_surface(d, r3, c3, B),
                                "A harmonic + oracle medial-axis DEM points", constraints=len(r3)))
        # 5. where A clips
        raw = S_A - B
        neg = M & (raw < 0)
        gy, gx = np.gradient(T, dx)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        clipping.append({
            "lia_unique_id": uid,
            "negative_cells_percent": float(100 * neg.sum() / M.sum()),
            "share_within_100m_of_margin_percent": float(100 * (neg & (d["dist"] < 100)).sum() / max(neg.sum(), 1)),
            "median_dem_slope_negative_cells_deg": float(np.median(slope[neg])),
            "median_dem_slope_glacier_deg": float(np.median(slope[M])),
            "median_reference_H_negative_cells_m": float(np.median(H[neg])),
            "median_reference_H_glacier_m": float(np.median(H[M])),
            "reference_volume_share_negative_cells_percent": float(100 * H[neg].sum() / H[M].sum()),
            "mean_reference_H_on_margin_cells_m": float(H[d["bnd"]].mean()),
        })
        print(uid, "done", flush=True)
    pd.DataFrame(profile).to_csv(out_dir / "reference_margin_profile.csv", index=False)
    pd.DataFrame(variants).to_csv(out_dir / "surface_operator_variants.csv", index=False)
    pd.DataFrame(clipping).to_csv(out_dir / "operator_A_clipping_location.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "status": "completed", "systems": list(SYSTEMS), "gap_m": GAP_M,
        "scope": "offline operator diagnostic on frozen modern controls; reference is an OGGM redistribution, not observation",
        "no_PP_tau_LIA_or_dynamic_runs": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
