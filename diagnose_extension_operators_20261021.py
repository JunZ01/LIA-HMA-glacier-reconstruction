"""Compare 2-D extension operators on the three frozen modern controls.

Follow-up to ``diagnose_legacy191_interpolation_20261020.py``. Same three
systems (selection rule unchanged and recomputed here), same saved LIA grids,
same reprojected modern reference thickness and same flowline constraint cells.
Only the extension operator changes:

  legacy  zero-boundary harmonic thickness (``harmonic_extend_thickness``)
  A       harmonic *surface* with S = B on the margin, H = max(S - B, 0)
  C       Poisson thickness -lap(H) = q; q scanned, then leave-one-system-out

Extra diagnostics recorded per system:
  * share of the reference volume that sits on raster-boundary cells (all
    operators force H = 0 there, so this part of the bias is a convention);
  * bias binned by distance to the nearest constraint cell;
  * for A, the surface interpolation error against the modern DEM. In this
    modern control B = topo - H_ref, so A's thickness error equals its surface
    error exactly; it inherits the reference lateral shape through the bed and
    is NOT an independent test of that shape.
  * OGGM 1-D inversion volume (sum of inversion ``volume``) beside the gridded
    reference volume.

Offline operator diagnostic only: no PP solve, tau fit, LIA reconstruction,
dynamic run or source mutation.
"""

from pathlib import Path
from datetime import datetime, timezone
import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import gzip
import hashlib
import json
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.core import (
    harmonic_extend_thickness,
    poisson_extend_thickness,
    surface_extend_thickness,
)

ROOT = Path(__file__).resolve().parent
BASE = Path("H:/ToYangxin/result_yx")
OUT = BASE / "extension_operator_controls_20261021_v1"
AUD = BASE / "hma_harmonized_analysis_20261012_v1/legacy_volume_comparability"
THICK_VAR = "distributed_thickness_f19_fs0_full_v0"
INVERSION = "inversion_output_f19_fs0_full_v0.pkl"
GAPS_M = [0, 100]
Q_SCAN_PER_M = [0.0, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]
DIST_BINS_M = [0, 100, 200, 400, 800, 1600, np.inf]

sources = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path)
    sources[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return path


def dump(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2,
                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
        encoding="utf-8",
    )


def select_cases():
    """Identical rule to the 2026-10-20 control: upper-median single-member system per region."""
    a = pd.read_csv(bind(AUD / "legacy191_source_pairing_diagnostic.csv"))
    r = pd.read_csv(bind(BASE / "mean_thickness829_20261020_v1/mean_thickness_per_system829.csv"))
    members = pd.read_csv(bind(AUD / "modern_member_array_audit.csv"))
    single = a[a.modern_member_count == 1].merge(
        r[["lia_unique_id", "region", "region_zh"]], on="lia_unique_id")
    chosen = []
    for region, g in single.groupby("region"):
        g = g.sort_values(["modern_outline_union_area_manifest_km2", "lia_unique_id"])
        x = g.iloc[len(g) // 2]
        m = members[members.lia_unique_id == x.lia_unique_id].iloc[0]
        chosen.append({
            "region": region, "region_zh": x.region_zh,
            "lia_unique_id": int(x.lia_unique_id), "rgi_id": m.rgi_id,
            "gdir": str(Path(m.source_path).parent), "lia_netcdf": x.lia_netcdf_path,
            "native_modern_reference_volume_km3": float(x.modern_reference_native_sum_km3),
        })
    return chosen


def load_case(case):
    gdir = Path(case["gdir"])
    source_nc = bind(gdir / "gridded_data.nc")
    gridpath = bind(gdir / "glacier_grid.json")
    flpath = bind(gdir / "inversion_flowlines.pkl")
    invpath = bind(gdir / INVERSION)
    saved = bind(case["lia_netcdf"])
    with xr.open_dataset(source_nc) as ds:
        source_h = np.asarray(ds[THICK_VAR], float)
        source_mask = np.asarray(ds.glacier_mask, bool)
        source_topo = np.asarray(ds.topo, float)
    with xr.open_dataset(saved) as ds:
        yy = np.asarray(ds.y)
        xx = np.asarray(ds.x)
        dst_crs = ds.spatial_ref.attrs["crs_wkt"]
        dsttf = Affine.from_gdal(*map(float, ds.spatial_ref.attrs["GeoTransform"].split()))
    shape = (len(yy), len(xx))
    grid = json.loads(gridpath.read_text(encoding="utf-8"))
    sx, sy = grid["dxdy"]
    x0, y0 = grid["x0y0"]
    src_tf = Affine(sx, 0, x0 - 0.5 * sx, 0, sy, y0 - 0.5 * sy)
    dx = abs(float(xx[1] - xx[0]))

    def warp(array, resampling, nodata=np.nan, dtype=float):
        out = np.full(shape, nodata, dtype=dtype)
        reproject(array, out, src_transform=src_tf, src_crs=grid["proj"],
                  dst_transform=dsttf, dst_crs=dst_crs, src_nodata=nodata,
                  dst_nodata=nodata, resampling=resampling, num_threads=1)
        return out

    hw = warp(np.where(source_mask, source_h, np.nan), Resampling.bilinear)
    mw = warp(source_mask.astype(np.uint8), Resampling.nearest, 0, np.uint8)
    topo = warp(source_topo, Resampling.bilinear)
    modern = (mw > 0) & np.isfinite(hw) & np.isfinite(topo)
    native_volume = float(source_h[source_mask].sum() * abs(sx * sy) / 1e9)
    assert abs(native_volume - case["native_modern_reference_volume_km3"]) < 1e-9

    with gzip.open(flpath, "rb") as f:
        flowlines = pickle.load(f)
    with gzip.open(invpath, "rb") as f:
        inversion = pickle.load(f)
    inversion_1d_volume = float(sum(np.nansum(cl["volume"]) for cl in inversion) / 1e9)

    transformer = Transformer.from_crs(grid["proj"], dst_crs, always_xy=True)
    rr, cc = [], []
    for fl in flowlines:
        coords = np.asarray(fl.line.coords, float)
        wx = x0 + coords[:, 0] * sx
        wy = y0 + coords[:, 1] * sy
        tx, ty = transformer.transform(wx, wy)
        cj = (np.asarray(tx) - xx[0]) / (xx[1] - xx[0])
        rj = (np.asarray(ty) - yy[0]) / (yy[1] - yy[0])
        for j in range(len(cj) - 1):
            count = max(1, int(np.ceil(np.hypot(cj[j + 1] - cj[j], rj[j + 1] - rj[j]) / 2)))
            u = np.linspace(0, 1, count + 1)
            cc.extend(cj[j] + u * (cj[j + 1] - cj[j]))
            rr.extend(rj[j] + u * (rj[j + 1] - rj[j]))
    rr = np.rint(rr).astype(int)
    cc = np.rint(cc).astype(int)
    ok = (rr >= 0) & (cc >= 0) & (rr < shape[0]) & (cc < shape[1])
    rr, cc = rr[ok], cc[ok]
    inside = modern[rr, cc]
    rr, cc = rr[inside], cc[inside]
    flat = np.unique(np.ravel_multi_index((rr, cc), shape))
    rr, cc = np.unravel_index(flat, shape)
    return {
        "hw": np.where(modern, hw, 0.0), "topo": topo, "modern": modern, "dx": dx,
        "rr": rr, "cc": cc, "native_volume": native_volume,
        "inversion_1d_volume": inversion_1d_volume,
    }


def volume_km3(h, mask, dx):
    return float(np.sum(h[mask]) * dx * dx / 1e9)


def binned_bias(h, ref, mask, constraint_mask, dx):
    distance = distance_transform_edt(~constraint_mask) * dx
    rows = []
    for lo, hi in zip(DIST_BINS_M[:-1], DIST_BINS_M[1:]):
        sel = mask & (distance >= lo) & (distance < hi)
        if not sel.any():
            continue
        rows.append({
            "distance_bin_m": f"{lo:g}-{hi:g}", "cells": int(sel.sum()),
            "reference_volume_share": float(ref[sel].sum() / max(ref[mask].sum(), 1e-12)),
            "mean_bias_m": float(np.mean(h[sel] - ref[sel])),
            "relative_volume_bias_percent": (
                float(100 * (h[sel].sum() / ref[sel].sum() - 1))
                if ref[sel].sum() > 0 else float("nan")),
        })
    return rows


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    cases = select_cases()
    bind(ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/core/solver.py")
    bind(ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/core/extension_operators.py")
    bind(Path(__file__))
    bind(ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/core/__init__.py")
    dump("protocol.json", {
        "status": "frozen-before-offline-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Which extension operator recovers the modern reference volume from exact flowline constraints, and how much of the legacy bias is boundary convention versus lateral shape?",
        "cases": cases, "boundary_exclusion_variants_m": GAPS_M,
        "poisson_q_scan_per_m": Q_SCAN_PER_M,
        "poisson_transfer": "leave-one-system-out: q interpolated to zero volume bias on the other two systems (same gap), mean applied to the held-out system",
        "surface_operator_note": "B = topo - H_ref inside the modern mask, so operator A error equals its surface interpolation error; not an independent test of lateral shape",
        "new_dynamic_runs": 0, "new_LIA_reconstruction_runs": 0,
    })

    rows, binned, q_curves, loaded = [], [], [], {}
    for case in cases:
        uid = case["lia_unique_id"]
        d = load_case(case)
        loaded[uid] = d
        hw, topo, modern, dx = d["hw"], d["topo"], d["modern"], d["dx"]
        bed = topo - hw
        ref_volume = volume_km3(hw, modern, dx)
        boundary = modern & ~binary_erosion(modern, border_value=0)
        dist_boundary = distance_transform_edt(~boundary) * dx
        fields = {"reference_H_m": hw, "modern_mask": modern, "topo_m": topo}
        for gap in GAPS_M:
            keep = ~boundary[d["rr"], d["cc"]]
            if gap:
                keep &= dist_boundary[d["rr"], d["cc"]] >= gap
            rows_c, cols_c = d["rr"][keep], d["cc"][keep]
            cmask = np.zeros_like(modern)
            cmask[rows_c, cols_c] = True
            base = {
                **case, "boundary_exclusion_m": gap, "target_grid_dx_m": dx,
                "constraints": int(len(rows_c)),
                "reference_volume_on_target_grid_km3": ref_volume,
                "native_modern_volume_km3": d["native_volume"],
                "oggm_1d_inversion_volume_km3": d["inversion_1d_volume"],
                "boundary_cell_reference_volume_share_percent": float(
                    100 * hw[boundary].sum() / max(hw[modern].sum(), 1e-12)),
            }
            outputs = {}
            t0 = time.perf_counter()
            legacy, meta = harmonic_extend_thickness(
                modern, rows_c, cols_c, hw[rows_c, cols_c],
                duplicate_tolerance_m=1e-3, boundary_tolerance_m=1e-3,
                max_iterations=5000, tolerance_m=1e-3,
                method="conjugate-gradient", cg_relative_tolerance=1e-10)
            assert meta["status"] == "converged", meta
            outputs["legacy"] = (legacy, {"seconds": time.perf_counter() - t0})
            t0 = time.perf_counter()
            a_h, a_meta = surface_extend_thickness(
                modern, bed, rows_c, cols_c, topo[rows_c, cols_c])
            assert a_meta["status"] == "converged", a_meta
            interior = modern & ~boundary
            s_err = a_meta["surface_m"][interior] - topo[interior]
            realized_s_err = bed[interior] + a_h[interior] - topo[interior]
            outputs["A_surface"] = (a_h, {
                "seconds": time.perf_counter() - t0,
                "surface_rmse_vs_dem_m": float(np.sqrt(np.mean(s_err ** 2))),
                "surface_mean_error_vs_dem_m": float(np.mean(s_err)),
                "realized_surface_rmse_vs_dem_m": float(np.sqrt(np.mean(realized_s_err ** 2))),
                "realized_surface_mean_error_vs_dem_m": float(np.mean(realized_s_err)),
                "clipped_volume_km3": float(a_meta["clipped_volume_per_cell_area_m"] * dx**2 / 1e9),
                "negative_raw_thickness_fraction": a_meta["negative_raw_thickness_fraction"],
            })
            for q in Q_SCAN_PER_M:
                c_h, _ = poisson_extend_thickness(
                    modern, rows_c, cols_c, hw[rows_c, cols_c],
                    curvature_per_m=q, cell_size_m=dx)
                q_curves.append({"lia_unique_id": uid, "boundary_exclusion_m": gap, "q_per_m": q,
                                 "volume_bias_percent": 100 * (volume_km3(c_h, modern, dx) / ref_volume - 1)})
            for name, (h, extra) in outputs.items():
                err = h[modern] - hw[modern]
                rows.append({**base, "operator": name,
                             "volume_km3": volume_km3(h, modern, dx),
                             "volume_bias_percent": 100 * (volume_km3(h, modern, dx) / ref_volume - 1),
                             "rmse_m": float(np.sqrt(np.mean(err ** 2))),
                             "constraint_max_abs_error_m": float(np.max(np.abs(h[rows_c, cols_c] - hw[rows_c, cols_c]))),
                             **extra})
                binned.extend({"lia_unique_id": uid, "boundary_exclusion_m": gap, "operator": name, **b}
                              for b in binned_bias(h, hw, modern, cmask, dx))
                fields[f"{name}_H_gap{gap}m"] = h
            print(json.dumps({k: v for k, v in rows[-1].items() if k in (
                "lia_unique_id", "boundary_exclusion_m", "operator", "volume_bias_percent")}), flush=True)
        np.savez_compressed(OUT / f"operator_control_{uid}.npz", **fields)

    # Leave-one-system-out transfer of the Poisson curvature q.
    curves = pd.DataFrame(q_curves)
    transfer = []
    for gap in GAPS_M:
        for uid in curves.lia_unique_id.unique():
            fitted = []
            for other in curves.lia_unique_id.unique():
                if other == uid:
                    continue
                c = curves[(curves.lia_unique_id == other) & (curves.boundary_exclusion_m == gap)].sort_values("q_per_m")
                if c.volume_bias_percent.max() >= 0 >= c.volume_bias_percent.min():
                    fitted.append(float(np.interp(0.0, c.volume_bias_percent, c.q_per_m)))
            if not fitted:
                transfer.append({"lia_unique_id": int(uid), "boundary_exclusion_m": gap, "status": "q-outside-scan"})
                continue
            q = float(np.mean(fitted))
            d = loaded[int(uid)]
            keep = ~(d["modern"] & ~binary_erosion(d["modern"], border_value=0))[d["rr"], d["cc"]]
            if gap:
                b = d["modern"] & ~binary_erosion(d["modern"], border_value=0)
                keep &= (distance_transform_edt(~b) * d["dx"])[d["rr"], d["cc"]] >= gap
            rc, cc_ = d["rr"][keep], d["cc"][keep]
            c_h, _ = poisson_extend_thickness(d["modern"], rc, cc_, d["hw"][rc, cc_],
                                              curvature_per_m=q, cell_size_m=d["dx"])
            ref = volume_km3(d["hw"], d["modern"], d["dx"])
            transfer.append({"lia_unique_id": int(uid), "boundary_exclusion_m": gap, "status": "ok",
                             "q_transferred_per_m": q,
                             "volume_bias_percent": 100 * (volume_km3(c_h, d["modern"], d["dx"]) / ref - 1)})

    pd.DataFrame(rows).to_csv(OUT / "operator_volume_controls.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(binned).to_csv(OUT / "operator_bias_by_constraint_distance.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(OUT / "poisson_q_scan.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(transfer).to_csv(OUT / "poisson_q_leave_one_out.csv", index=False, encoding="utf-8-sig")
    unchanged = {p: sha(Path(p)) == r["sha256"] for p, r in sources.items()}
    dump("source_manifest.json", sources)
    dump("diagnostic_summary.json", {
        "status": "completed", "elapsed_seconds": time.perf_counter() - started,
        "all_sources_unchanged": all(unchanged.values()), "source_hash_checks": unchanged,
        "no_LIA_reconstruction_or_dynamic_model_runs": True,
        "limits": [
            "Operator A inherits the reference lateral shape through B = topo - H_ref; its modern error is a surface-interpolation error, not proof that the shape is right.",
            "The reference is OGGM distribute_thickness_per_altitude output rescaled to a Farinotti-2019 system total; it is a model estimate, not observation.",
            "Three systems do not quantify the 191-system bias; Poisson q transfer across three systems is indicative only.",
        ],
    })
    assert all(unchanged.values())
    shutil.copy2(__file__, OUT / "reproduce_extension_operator_controls.py")
    print("DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
