"""Unified self-consistent static engine (both epochs, all cohorts).

One code path for the modern and the LIA end, following VOLTA / PalaeoIce
(James & Carrivick 2016; Li 2023): the modern thickness is reconstructed from
the modern surface DEM by perfect plasticity, the bed is DEM minus that
thickness, and the LIA surface is reconstructed on that bed. No external ice
thickness product enters the reconstruction; F19 only sets the tau
coefficient in schemes A and B.

Per system, for a given basal shear stress tau (same value at both ends):
  modern: along every modern centerline H = tau / (F rho g sin(alpha)), alpha
          the along-line slope of the DEM smoothed over SLOPE_WINDOW_M with a
          floor of MIN_SLOPE_DEG; constraint cells >= max(100 m, 1.25 dx) from
          the outline; thin-plate interpolation with H = 0 on the outline;
          V_mod = sum H; bed = DEM - H inside the modern mask, DEM elsewhere.
  LIA:    production network solve (explicit connectors, 16x refinement,
          outlet H = 0) upward from the LIA termini on that bed, production
          constraint collection, same exclusion, thin-plate on the LIA mask;
          V_LIA = sum H.
  dV = V_mod - V_LIA (negative = loss).

tau (Driedger & Kennard 1986, as implemented in VOLTA/PalaeoIce):
  tau_DK = 2.7e4 * (sum_i A_i / cos(alpha_i))**0.106 Pa over 300 m elevation
  bands of the modern system surface (A_i in m2, alpha_i mean band slope).
Schemes:
  C: tau = tau_DK
  A: tau = k_region * tau_DK, k_region such that the region's summed modern
     volume equals its summed F19 volume (main scheme)
  B: tau = k_system * tau_DK, k_system such that each system's modern volume
     equals its F19 volume
tau is clamped to TAU_LIMITS_KPA and clamping is reported.

Input: one NPZ per system (schema 'unified-static-v1'), from
export_unified_static_inputs_20261022.py. NumPy/SciPy only.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates, uniform_filter1d

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("replay", ROOT / "replay_modern_fullchain_20261021.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)
from ice_surface_reconstruction.core import (  # noqa: E402  (path set by replay)
    SurfaceScenario,
    refine_flowline_network,
    sample_network_solution_at_original_nodes,
    solve_flowline_network,
    thinplate_extend_thickness,
)

SHAPE_FACTOR = 0.8
RHO, G = 900.0, 9.81
REFINEMENT = 16
DK_BAND_M = 300.0
SLOPE_WINDOW_M = 500.0
MIN_SLOPE_DEG = 1.5
MAX_THICKNESS_M = 2000.0
TAU_LIMITS_KPA = (40.0, 260.0)
DUPLICATE_TOL_M = 1e-3
CROSS_BRANCH_GRID_FRACTION = 0.1
FAILED_TREE_LENGTH_SHARE_MAX = 0.05


def _bilinear(field, xy):
    return map_coordinates(field, np.vstack([xy[:, 1], xy[:, 0]]), order=1, mode="nearest", prefilter=False)


def _boundary(mask):
    return mask & ~binary_erosion(mask, border_value=0)


def exclusion_m(dx):
    return max(100.0, 1.25 * dx)


def load_system(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ("dem_m",):
        d[k] = d[k].astype(float)
    for k in ("modern_mask", "lia_mask"):
        d[k] = d[k].astype(bool)
    d["dx"] = float(d["dx_m"])
    return d


def lines(d: dict, prefix: str) -> list[dict]:
    off = d[f"{prefix}_fl_offsets"]
    out = []
    for i in range(len(off) - 1):
        s = slice(off[i], off[i + 1])
        out.append({"index": int(d[f"{prefix}_fl_index"][i]), "flows_to": int(d[f"{prefix}_fl_flows_to"][i]),
                    "flows_to_indice": int(d[f"{prefix}_fl_flows_to_indice"][i]),
                    "dx_m": float(d[f"{prefix}_fl_dx_m"][i]), "xy": np.asarray(d[f"{prefix}_fl_xy"][s], float)})
    return out


def trees(fls: list[dict]) -> list[list[dict]]:
    by = {f["index"]: f for f in fls}
    root = {}
    for f in fls:
        r, seen = f, set()
        while r["flows_to"] >= 0 and r["flows_to"] in by and r["index"] not in seen:
            seen.add(r["index"])
            r = by[r["flows_to"]]
        root[f["index"]] = r["index"]
    groups = {}
    for f in fls:
        groups.setdefault(root[f["index"]], []).append(f)
    return list(groups.values())


def dk_tau_kpa(d: dict) -> float:
    dem, mask, dx = d["dem_m"], d["modern_mask"], d["dx"]
    gy, gx = np.gradient(dem, dx)
    alpha = np.arctan(np.hypot(gx, gy))
    z = dem[mask]
    edges = np.arange(np.floor(z.min()), z.max() + DK_BAND_M, DK_BAND_M)
    band = np.digitize(z, edges)
    total = 0.0
    a = alpha[mask]
    for b in np.unique(band):
        sel = band == b
        total += sel.sum() * dx * dx / math.cos(float(a[sel].mean()))
    return 2.7e4 * total ** 0.106 / 1000.0


def clamp_tau(tau):
    return float(min(max(tau, TAU_LIMITS_KPA[0]), TAU_LIMITS_KPA[1]))


def _constraints_to_thinplate(mask, dx, rows, cols, values, surface=None):
    inside = (rows >= 0) & (cols >= 0) & (rows < mask.shape[0]) & (cols < mask.shape[1])
    rows, cols, values = rows[inside], cols[inside], values[inside]
    surface = None if surface is None else surface[inside]
    inm = mask[rows, cols]
    rows, cols, values = rows[inm], cols[inm], values[inm]
    surface = None if surface is None else surface[inm]
    bnd = _boundary(mask)
    dist = distance_transform_edt(~bnd) * dx
    keep = ~bnd[rows, cols] & (dist[rows, cols] >= exclusion_m(dx))
    rows, cols, values = rows[keep], cols[keep], values[keep]
    surface = None if surface is None else surface[keep]
    if not len(rows):
        raise ValueError("no constraint cells after margin exclusion")
    h, meta = thinplate_extend_thickness(mask, rows, cols, values, centerline_surface_m=surface,
                                         duplicate_tolerance_m=np.inf if surface is None else DUPLICATE_TOL_M)
    if meta["status"] != "converged":
        raise ValueError(f"thin-plate {meta['status']}")
    return h, {"constraints": int(len(rows)), "negative_cells_percent": meta["negative_raw_cells_percent"]}


def modern_end(d: dict, tau_kpa: float) -> dict:
    dem, mask, dx = d["dem_m"], d["modern_mask"], d["dx"]
    rows, cols, vals = [], [], []
    floor = math.tan(math.radians(MIN_SLOPE_DEG))
    for f in lines(d, "mod"):
        xy = f["xy"]
        if len(xy) < 2:
            continue
        s = _bilinear(dem, xy)
        win = max(1, int(round(SLOPE_WINDOW_M / f["dx_m"])))
        s_smooth = uniform_filter1d(s, size=win, mode="nearest")
        slope = np.maximum(np.abs(np.gradient(s_smooth, f["dx_m"])), floor)
        h = np.minimum(tau_kpa * 1000.0 / (SHAPE_FACTOR * RHO * G * np.sin(np.arctan(slope))), MAX_THICKNESS_M)
        rows.append(np.rint(xy[:, 1]).astype(int))
        cols.append(np.rint(xy[:, 0]).astype(int))
        vals.append(h)
    if not rows:
        raise ValueError("no modern centerlines")
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    flat = r * mask.shape[1] + c
    uniq, inv = np.unique(flat, return_inverse=True)
    v = np.bincount(inv, weights=v) / np.bincount(inv)
    r, c = np.divmod(uniq, mask.shape[1])
    h, meta = _constraints_to_thinplate(mask, dx, r, c, v)
    bed = np.where(mask, dem - h, dem)
    return {"H": h, "bed": bed, "V_km3": float(h[mask].sum() * dx * dx / 1e9), **meta}


def lia_end(d: dict, bed: np.ndarray, tau_kpa: float) -> dict:
    mask, dx = d["lia_mask"], d["dx"]
    gy, gx = np.gradient(bed, dx)
    gridded = replay.GriddedBed(bed_m=bed, glacier_mask=mask, slope=np.arctan(np.hypot(gx, gy)),
                                map_dx_m=dx, bed_scenario="unified-self-consistent")
    scenario = SurfaceScenario("unified", tau_kpa * 1000.0, SHAPE_FACTOR)
    all_rows, all_cols, all_h, all_s = [], [], [], []
    total_len, failed_len, failures = 0.0, 0.0, []
    for tree in sorted(trees(lines(d, "lia")), key=lambda t: -sum(len(f["xy"]) * f["dx_m"] for f in t)):
        length = sum((len(f["xy"]) - 1) * f["dx_m"] for f in tree)
        total_len += length
        inputs = [replay.FlowlineInput(
            flowline_index=f["index"], flows_to=None if f["flows_to"] < 0 else f["flows_to"],
            flows_to_indice=None if f["flows_to_indice"] < 0 else f["flows_to_indice"],
            bed_elevation_m=_bilinear(bed, f["xy"]), reference_surface_m=_bilinear(bed, f["xy"]),
            dx_meter=f["dx_m"], map_dx_m=dx, coords_xy=f["xy"]) for f in tree]
        try:
            build = replay.build_network(inputs, junction_transfer_policy=replay.EXPLICIT_CONNECTOR,
                                         gridded_bed=gridded,
                                         connector_weights=replay.ConnectorCostWeights(alpha=0.0, beta=0.0),
                                         avoid_existing_network_cells=True)
            net = solve_flowline_network(refine_flowline_network(build.branches, REFINEMENT), scenario,
                                         outlet_node_ids=build.outlet_node_ids)
            if not net.success:
                raise ValueError(f"network-{net.status}")
            net = sample_network_solution_at_original_nodes(build.branches, net, REFINEMENT, scenario)
            r, c, h, s, _ = replay.collect_network_constraints(
                inputs, build, net, duplicate_surface_tolerance_m=DUPLICATE_TOL_M,
                cross_branch_surface_tolerance_m=max(DUPLICATE_TOL_M, CROSS_BRANCH_GRID_FRACTION * dx))
        except Exception as error:  # a failing tree is recorded
            failed_len += length
            failures.append(f"{type(error).__name__}: {error}"[:160])
            continue
        all_rows.append(r); all_cols.append(c); all_h.append(h); all_s.append(s)
    share = failed_len / max(total_len, 1e-9)
    if not all_rows or share > FAILED_TREE_LENGTH_SHARE_MAX:
        raise ValueError(f"LIA trees failed ({share:.1%} of flowline length): {failures[:3]}")
    r, c = np.concatenate(all_rows), np.concatenate(all_cols)
    h, s = np.concatenate(all_h), np.concatenate(all_s)
    flat = r * mask.shape[1] + c
    _, first = np.unique(flat, return_index=True)  # cross-tree duplicates: keep the longer tree's value
    first = np.sort(first)
    hh, meta = _constraints_to_thinplate(mask, dx, r[first], c[first], h[first], s[first])
    return {"H": hh, "V_km3": float(hh[mask].sum() * dx * dx / 1e9), "failed_tree_length_share": share,
            "tree_failures": failures, **meta}


def run_system(d: dict, tau_kpa: float) -> dict:
    mod = modern_end(d, tau_kpa)
    lia = lia_end(d, mod["bed"], tau_kpa)
    return {"tau_kpa": tau_kpa, "V_mod_km3": mod["V_km3"], "V_lia_km3": lia["V_km3"],
            "dV_km3": mod["V_km3"] - lia["V_km3"],
            "modern_area_km2": float(d["modern_mask"].sum() * d["dx"] ** 2 / 1e6),
            "lia_area_km2": float(d["lia_mask"].sum() * d["dx"] ** 2 / 1e6),
            "modern_constraints": mod["constraints"], "lia_constraints": lia["constraints"],
            "lia_failed_tree_length_share": lia["failed_tree_length_share"]}
