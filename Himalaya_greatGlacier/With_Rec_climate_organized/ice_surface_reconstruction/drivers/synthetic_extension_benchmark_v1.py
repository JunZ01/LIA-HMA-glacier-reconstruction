"""Known-geometry benchmark of flowline-to-2D extension operators.

Builds synthetic valley glaciers whose true thickness, bed and surface are
known, feeds the exact centerline values to each operator, and reports the
volume bias. No observed data, OGGM state or LIA product is read or written.

Cases differ in lateral section shape (parabolic, elliptical, OGGM-like
distance**0.25), transverse surface curvature, and planform (straight valley,
Y-shaped confluence with a wide basin). The Poisson curvature q is fitted on
case ``straight_parabolic`` only and then transferred unchanged, which is the
honest analogue of calibrating on modern glaciers and applying to the LIA.

Usage (from the directory that contains ``ice_surface_reconstruction``)::

    python -m ice_surface_reconstruction.drivers.synthetic_extension_benchmark_v1 --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from ..core import harmonic_extend_thickness
from ..core.extension_operators import (
    poisson_extend_thickness,
    raster_boundary,
    surface_extend_thickness,
)

DX = 50.0  # m
EXCLUSION_M = 100.0


def _section(u: np.ndarray, shape: str) -> np.ndarray:
    """Lateral profile for |u| = distance from centre / half-width in [0, 1]."""
    u = np.clip(np.abs(u), 0.0, 1.0)
    if shape == "parabolic":
        return 1.0 - u**2
    if shape == "elliptical":
        return np.sqrt(1.0 - u**2)
    if shape == "oggm_d025":  # OGGM distribute_thickness_per_altitude default exponent
        return (1.0 - u) ** 0.25
    raise ValueError(shape)


def _centre_thickness(s: np.ndarray, length: float, h_max: float) -> np.ndarray:
    """Plastic-like long profile: sqrt rise from the terminus, taper at the head."""
    t = np.clip(s / length, 0.0, 1.0)
    return h_max * np.sqrt(t) * (1.0 - t) ** 0.35 / 0.86


def _polyline_coords(points: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.diff(points, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    s = np.arange(0.0, cum[-1] + 1e-9, spacing)
    x = np.interp(s, cum, points[:, 0])
    y = np.interp(s, cum, points[:, 1])
    return np.column_stack([x, y]), s


def build_case(
    name: str,
    *,
    section: str,
    planform: str,
    transverse_surface_m: float = 0.0,
) -> dict:
    """Return mask, bed, true H, true S and centerline constraints for a case."""
    if planform == "straight":
        branches = [(np.array([[0.0, 0.0], [9000.0, 0.0]]), 9000.0, 300.0,
                     lambda s: 450.0 + 350.0 * s / 9000.0)]
        extent = (-300.0, 9300.0, -1400.0, 1400.0)
    elif planform == "confluence":
        trunk = np.array([[0.0, 0.0], [4500.0, 0.0]])
        left = np.array([[4500.0, 0.0], [6500.0, 1500.0], [9500.0, 2600.0]])
        right = np.array([[4500.0, 0.0], [6500.0, -1500.0], [9000.0, -2400.0]])
        branches = [
            (trunk, 4500.0, 260.0, lambda s: 650.0 + 250.0 * s / 4500.0),
            (left, 5600.0, 230.0, lambda s: 900.0 - 400.0 * s / 5600.0),
            (right, 5000.0, 230.0, lambda s: 800.0 - 350.0 * s / 5000.0),
        ]
        extent = (-300.0, 9900.0, -3600.0, 3800.0)
    else:
        raise ValueError(planform)
    x0, x1, y0, y1 = extent
    xs = np.arange(x0, x1, DX) + DX / 2
    ys = np.arange(y0, y1, DX) + DX / 2
    X, Y = np.meshgrid(xs, ys)
    thickness = np.zeros_like(X)
    u_best = np.ones_like(X)
    inside = np.zeros_like(X, dtype=bool)
    surface_along = np.full_like(X, np.nan)
    centre_rows, centre_cols, centre_s_global = [], [], []
    # Trunk thickness continues into tributaries: tributary s starts at the
    # junction with the trunk value there, then decays to the head.
    for b, (pts, length, h_max, half_width) in enumerate(branches):
        coords, s = _polyline_coords(pts, DX / 4)
        if planform == "confluence" and b > 0:
            s_global = 4500.0 + s
            total = 4500.0 + length
        else:
            s_global = s
            total = length if planform == "straight" else 4500.0 + 5600.0
        hc = _centre_thickness(s_global, total, h_max if planform == "straight" else 260.0)
        if planform == "confluence" and b == 0:
            hc = _centre_thickness(s_global, 4500.0 + 5300.0, 260.0)
        # nearest-point projection of every cell onto this branch
        tree_d = np.full(X.shape, np.inf)
        tree_i = np.zeros(X.shape, dtype=int)
        for k in range(0, len(coords), 2):
            d = np.hypot(X - coords[k, 0], Y - coords[k, 1])
            closer = d < tree_d
            tree_d[closer] = d[closer]
            tree_i[closer] = k
        w = half_width(s[tree_i])
        u = tree_d / w
        branch_inside = (u < 1.0) & (s[tree_i] > 0.0) & (s[tree_i] < s[-1])
        branch_h = hc[tree_i] * _section(u, section) * branch_inside
        replace = branch_h > thickness
        thickness[replace] = branch_h[replace]
        u_best[replace] = np.clip(u[replace], 0.0, 1.0)
        inside |= branch_inside
        sa = 3600.0 + 0.11 * s_global[tree_i] + 12.0 * np.sin(s_global[tree_i] / 1400.0)
        surface_along = np.where(branch_inside & np.isnan(surface_along), sa, surface_along)
        r = np.rint((coords[:, 1] - ys[0]) / DX).astype(int)
        c = np.rint((coords[:, 0] - xs[0]) / DX).astype(int)
        centre_rows.append(r)
        centre_cols.append(c)
    surface_along = np.where(inside, surface_along, 0.0)
    # Optional transverse surface curvature: positive = convex (ablation-like),
    # zero on the centerline and -transverse_surface_m at the margin, parabolic
    # in between (so the surface itself is curved across the glacier).
    surface = surface_along - transverse_surface_m * np.where(inside, u_best**2, 0.0)
    # The zero-thickness margin is the raster boundary (same convention as all
    # operators), so the true field is also zero there.
    thickness = np.where(raster_boundary(inside), 0.0, thickness)
    bed = surface - thickness
    rows = np.concatenate(centre_rows)
    cols = np.concatenate(centre_cols)
    ok = (rows >= 0) & (cols >= 0) & (rows < X.shape[0]) & (cols < X.shape[1])
    rows, cols = rows[ok], cols[ok]
    keep = inside[rows, cols]
    rows, cols = rows[keep], cols[keep]
    flat = np.unique(np.ravel_multi_index((rows, cols), X.shape))
    rows, cols = np.unravel_index(flat, X.shape)
    boundary = raster_boundary(inside)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~boundary) * DX
    keep = ~boundary[rows, cols] & (dist[rows, cols] >= EXCLUSION_M)
    rows, cols = rows[keep], cols[keep]
    thickness = np.where(inside, thickness, 0.0)
    return {
        "name": name, "section": section, "planform": planform,
        "transverse_surface_m": transverse_surface_m,
        "mask": inside, "bed": bed, "thickness": thickness, "surface": surface,
        "rows": rows, "cols": cols,
    }


def _volume(h: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(h[mask]) * DX * DX / 1e9)


def evaluate(case: dict, q_per_m: float) -> dict:
    mask, rows, cols = case["mask"], case["rows"], case["cols"]
    h_true, s_true, bed = case["thickness"], case["surface"], case["bed"]
    v_true = _volume(h_true, mask)
    legacy, meta = harmonic_extend_thickness(
        mask, rows, cols, h_true[rows, cols],
        centerline_surface_m=s_true[rows, cols],
        method="conjugate-gradient", max_iterations=20000,
    )
    assert meta["status"] == "converged", meta
    a, ameta = surface_extend_thickness(mask, bed, rows, cols, s_true[rows, cols])
    c, _ = poisson_extend_thickness(
        mask, rows, cols, h_true[rows, cols], curvature_per_m=q_per_m, cell_size_m=DX
    )
    out = {"case": case["name"], "true_volume_km3": v_true,
           "cells": int(mask.sum()), "constraints": int(len(rows))}
    for label, h in (("legacy_harmonic_H", legacy), ("A_harmonic_surface", a),
                     ("C_poisson_H", c)):
        err = h[mask] - h_true[mask]
        out[label] = {
            "volume_bias_percent": 100.0 * (_volume(h, mask) / v_true - 1.0),
            "rmse_m": float(np.sqrt(np.mean(err**2))),
            "mean_bias_m": float(np.mean(err)),
        }
    out["A_harmonic_surface"]["negative_raw_fraction"] = ameta["negative_raw_thickness_fraction"]
    out["A_surface_interpolation_rmse_m"] = float(
        np.sqrt(np.mean((ameta["surface_m"][mask] - s_true[mask]) ** 2))
    )
    return out


def fit_q(case: dict) -> float:
    mask, rows, cols, h = case["mask"], case["rows"], case["cols"], case["thickness"]
    target = _volume(h, mask)

    def gap(q):
        c, _ = poisson_extend_thickness(mask, rows, cols, h[rows, cols],
                                        curvature_per_m=q, cell_size_m=DX)
        return _volume(c, mask) - target

    return float(brentq(gap, 0.0, 1e-2, xtol=1e-9))


CASES = [
    dict(name="straight_parabolic", section="parabolic", planform="straight"),
    dict(name="straight_elliptical", section="elliptical", planform="straight"),
    dict(name="straight_oggm_d025", section="oggm_d025", planform="straight"),
    dict(name="straight_parabolic_convex20m", section="parabolic", planform="straight",
         transverse_surface_m=20.0),
    dict(name="straight_parabolic_concave20m", section="parabolic", planform="straight",
         transverse_surface_m=-20.0),
    dict(name="confluence_parabolic", section="parabolic", planform="confluence"),
    dict(name="confluence_oggm_d025", section="oggm_d025", planform="confluence"),
]


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    built = [build_case(**spec) for spec in CASES]
    q = fit_q(built[0])
    results = [evaluate(case, q) for case in built]
    report = {
        "status": "completed",
        "grid_dx_m": DX,
        "boundary_exclusion_m": EXCLUSION_M,
        "poisson_q_fitted_on": CASES[0]["name"],
        "poisson_q_per_m": q,
        "results": results,
        "scope": "synthetic known-geometry operator benchmark; no observed or LIA data",
    }
    (out_dir / "synthetic_extension_benchmark_v1.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "synthetic_extension_cases_v1.npz", **{
        f"{c['name']}__{k}": c[k] for c in built
        for k in ("mask", "bed", "thickness", "surface", "rows", "cols")})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    report = run(parser.parse_args().out)
    print(f"q fitted on {report['poisson_q_fitted_on']}: {report['poisson_q_per_m']:.3e} 1/m")
    header = f"{'case':34s} {'legacy':>8s} {'A':>8s} {'C':>8s}  {'A_S_rmse':>8s} {'A_neg':>6s}"
    print(header)
    for r in report["results"]:
        print(f"{r['case']:34s} {r['legacy_harmonic_H']['volume_bias_percent']:8.2f} "
              f"{r['A_harmonic_surface']['volume_bias_percent']:8.2f} "
              f"{r['C_poisson_H']['volume_bias_percent']:8.2f}  "
              f"{r['A_surface_interpolation_rmse_m']:8.2f} "
              f"{r['A_harmonic_surface']['negative_raw_fraction']:6.3f}")


if __name__ == "__main__":
    main()
