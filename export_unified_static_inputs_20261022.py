"""Export unified static inputs (schema 'unified-static-v1') for 191 or 829 systems.

One procedure for both cohorts. Everything is put on the system's saved LIA
grid (read from its LIA netCDF):
  * dem_m: the regional DEM named in the netCDF attribute
    topography_source_path (or --dem), bilinear;
  * modern_mask: union of the modern member glacier masks (member
    gridded_data.nc + glacier_grid.json), nearest;
  * modern centerlines: generated for EVERY member with the production OGGM
    centerline function (lia_centerline_pilot.compute_centerline_tree) on the
    DEM, with the frozen OGGM centerline parameters of the config -- the same
    generator for all 829 systems;
  * lia_mask: lia_mask of the LIA netCDF;
  * LIA centerlines: the frozen networks -- Himalaya191 audited dict networks
    (projected metres) or Previous654 OGGM Centerline lists (pixel coordinates of
    the LIA system grid JSON) -- converted to pixel coordinates of this grid.
No ice-thickness product is read; F19 enters only as the system volume
native_reference_volume_km3 used by schemes A/B. A member or LIA component
that fails is recorded in the manifest, not silently dropped. Read-only.

Usage (repo root, oggm_env2):
    python export_unified_static_inputs_20261022.py --config CONFIG.json \
        --systems-index systems829_input_index.csv --members-index modern_members829_input_index.csv \
        --out-dir OUT_DIR [--cohort Himalaya191] [--limit N] [--workers N] [--dem PATH]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import label
from shapely.geometry import shape as shp_shape

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_centerline_pilot import (
    _configure_oggm,
    compute_centerline_tree,
)

_CONFIG = None


def _init(config):
    global _CONFIG
    _CONFIG = config
    _configure_oggm(config)


class Grid:
    def __init__(self, nc_path: Path):
        with xr.open_dataset(nc_path) as ds:
            self.x = np.asarray(ds.x, float)
            self.y = np.asarray(ds.y, float)
            self.crs_wkt = ds.spatial_ref.attrs["crs_wkt"]
            self.lia_mask = np.asarray(ds.lia_mask, bool)
            self.dem_source = ds.attrs.get("topography_source_path")
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.shape = (len(self.y), len(self.x))
        self.transform = Affine(self.dx, 0, self.x[0] - self.dx / 2, 0, self.dy, self.y[0] - self.dy / 2)

    def to_pixel(self, xm, ym):
        return np.column_stack([(np.asarray(xm) - self.x[0]) / self.dx, (np.asarray(ym) - self.y[0]) / self.dy])

    def salem_grid(self, work: Path):
        import salem
        spec = {"proj": CRS.from_wkt(self.crs_wkt).to_proj4(), "x0y0": [float(self.x[0]), float(self.y[0])],
                "nxny": [self.shape[1], self.shape[0]], "dxdy": [self.dx, self.dy], "pixel_ref": "center"}
        path = work / "grid.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        return salem.Grid.from_json(path)


def warp_to(grid: Grid, src, src_transform, src_crs, resampling, nodata=np.nan, dtype=float):
    out = np.full(grid.shape, nodata, dtype=dtype)
    reproject(src, out, src_transform=src_transform, src_crs=src_crs, dst_transform=grid.transform,
              dst_crs=grid.crs_wkt, src_nodata=nodata, dst_nodata=nodata, resampling=resampling, num_threads=1)
    return out


def dem_on_grid(grid: Grid, dem_path: str) -> np.ndarray:
    with rasterio.open(dem_path) as src:
        out = np.full(grid.shape, np.nan)
        reproject(rasterio.band(src, 1), out, dst_transform=grid.transform, dst_crs=grid.crs_wkt,
                  dst_nodata=np.nan, resampling=Resampling.bilinear, num_threads=1)
    return out


def member_mask(grid: Grid, gdir: Path) -> np.ndarray:
    g = json.loads((gdir / "glacier_grid.json").read_text(encoding="utf-8"))
    sx, sy = g["dxdy"]
    x0, y0 = g["x0y0"]
    tf = Affine(sx, 0, x0 - 0.5 * sx, 0, sy, y0 - 0.5 * sy)
    with xr.open_dataset(gdir / "gridded_data.nc") as ds:
        m = np.asarray(ds.glacier_mask, np.uint8)
    return warp_to(grid, m, tf, g["proj"], Resampling.nearest, 0, np.uint8) > 0


def centerline_objects_to_arrays(fls, start_index: int, to_pixel=None) -> list[dict]:
    """OGGM Centerline list -> ragged dicts with global indices (production adapter logic)."""
    ids = {id(f): start_index + i for i, f in enumerate(fls)}
    coords = [np.asarray(f.line.coords, float) for f in fls]
    if to_pixel is not None:
        coords = [to_pixel(c) for c in coords]
    out = []
    for i, f in enumerate(fls):
        parent = ids.get(id(f.flows_to)) if f.flows_to is not None else None
        join = -1
        if parent is not None:
            p = np.asarray(f.__dict__["flows_to_point"].coords[0], float)
            if to_pixel is not None:
                p = to_pixel(p[None, :])[0]
            pc = coords[parent - start_index]
            join = int(np.argmin(np.hypot(pc[:, 0] - p[0], pc[:, 1] - p[1])))
        out.append({"index": start_index + i, "flows_to": -1 if parent is None else parent,
                    "flows_to_indice": join, "dx_m": float(f.dx_meter), "xy": coords[i]})
    return out


def dict_network_to_arrays(payload: dict, grid: Grid, start_index: int) -> list[dict]:
    """Audited projected network -> ragged dicts (mirrors load_projected_centerline_inputs)."""
    if payload.get("format") != "lia-projected-centerline-network-v1":
        raise ValueError("unsupported dict network format")
    tr = Transformer.from_crs(CRS.from_user_input(payload["crs"]), CRS.from_wkt(grid.crs_wkt), always_xy=True)
    branches = sorted(payload["branches"], key=lambda b: b["branch_index"])
    coords, junction = {}, {}
    for b in branches:
        p = np.asarray(b["coords_projected_m"], float)
        coords[int(b["branch_index"])] = grid.to_pixel(*tr.transform(p[:, 0], p[:, 1]))
        j = b.get("flows_to_point_projected_m")
        junction[int(b["branch_index"])] = None if j is None else grid.to_pixel(*tr.transform([float(j[0])], [float(j[1])]))[0]
    remap = {int(b["branch_index"]): start_index + k for k, b in enumerate(branches)}
    out = []
    for b in branches:
        i = int(b["branch_index"])
        parent = b.get("parent_index")
        join = -1
        if parent is not None:
            pc, t = coords[int(parent)], coords[i][-1]
            jn = junction[i]
            join = int(np.argmin(np.hypot(pc[:, 0] - jn[0], pc[:, 1] - jn[1])))
            contact = np.hypot(pc[:, 0] - t[0], pc[:, 1] - t[1])
            if float(contact.min()) <= 1.0:
                join = int(np.argmin(contact))
        out.append({"index": remap[i], "flows_to": -1 if parent is None else remap[int(parent)],
                    "flows_to_indice": join, "dx_m": float(b["dx_meter"]), "xy": coords[i]})
    return out


def lia_lines(row, grid: Grid) -> tuple[list[dict], list[str]]:
    paths = [p for p in str(row["lia_network_paths"]).split(";") if p and p != "nan"]
    if not paths:
        raise ValueError("no LIA network file")
    out, notes, nxt = [], [], 0
    json_grid = None
    if isinstance(row.get("lia_grid_json"), str) and row["lia_grid_json"] not in ("", "nan"):
        json_grid = json.loads(Path(row["lia_grid_json"]).read_text(encoding="utf-8"))
    for p in paths:
        with gzip.open(p, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            part = dict_network_to_arrays(payload, grid, nxt)
        else:
            if json_grid is None:
                raise ValueError("Centerline-list network without system grid JSON")
            gx0, gy0 = json_grid["x0y0"]
            gdx, gdy = json_grid["dxdy"]
            src = CRS.from_user_input(json_grid["proj"])
            tr = Transformer.from_crs(src, CRS.from_wkt(grid.crs_wkt), always_xy=True)

            def to_pixel(c):
                xm, ym = gx0 + c[:, 0] * gdx, gy0 + c[:, 1] * gdy
                return grid.to_pixel(*tr.transform(xm, ym))
            part = centerline_objects_to_arrays(list(payload), nxt, to_pixel)
        nxt += len(part)
        out.extend(part)
    pts = np.concatenate([l["xy"] for l in out])
    r, c = np.rint(pts[:, 1]).astype(int), np.rint(pts[:, 0]).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < grid.shape[0]) & (c < grid.shape[1])
    inside = float(grid.lia_mask[r[ok], c[ok]].mean()) if ok.any() else 0.0
    notes.append(f"lia_points_inside_lia_mask={inside:.3f}")
    if inside < 0.8:
        raise ValueError(f"LIA centerline points mostly outside the LIA mask ({inside:.2f}); grid mismatch")
    return out, notes


def modern_lines(grid: Grid, dem: np.ndarray, members: pd.DataFrame, work: Path):
    sg = grid.salem_grid(work)
    union = np.zeros(grid.shape, bool)
    out, failures, nxt = [], [], 0
    for m in members.itertuples():
        try:
            mask = member_mask(grid, Path(m.static_reference_gdir))
            union |= mask
            lab, n = label(mask)
            if n == 0:
                raise ValueError("empty member mask on LIA grid")
            keep = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
            geom = max((shp_shape(g) for g, v in shapes(keep.astype(np.uint8), mask=keep, transform=grid.transform) if v == 1),
                       key=lambda g: g.area)
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                fls, _, _ = compute_centerline_tree(str(m.rgi_id), geom, sg, dem, Path(tmp))
            part = centerline_objects_to_arrays(fls, nxt)
            nxt += len(part)
            out.extend(part)
        except Exception as error:
            failures.append({"rgi_id": m.rgi_id, "is_main": bool(m.is_main), "error": f"{type(error).__name__}: {error}"[:200]})
    return union, out, failures


def ragged(prefix: str, ls: list[dict]) -> dict:
    off = np.concatenate([[0], np.cumsum([len(l["xy"]) for l in ls])]).astype(np.int64)
    return {f"{prefix}_fl_offsets": off,
            f"{prefix}_fl_xy": np.concatenate([l["xy"] for l in ls]) if ls else np.zeros((0, 2)),
            f"{prefix}_fl_index": np.asarray([l["index"] for l in ls], np.int64),
            f"{prefix}_fl_flows_to": np.asarray([l["flows_to"] for l in ls], np.int64),
            f"{prefix}_fl_flows_to_indice": np.asarray([l["flows_to_indice"] for l in ls], np.int64),
            f"{prefix}_fl_dx_m": np.asarray([l["dx_m"] for l in ls], float)}


def export_system(task):
    row, members, out_dir, dem_override = task
    rid = str(row["record_id"])
    rec = {"record_id": rid, "cohort": row["cohort"], "region": row["region"]}
    try:
        grid = Grid(Path(row["lia_netcdf"]))
        dem_path = dem_override or grid.dem_source
        if not dem_path:
            raise ValueError("no DEM path (netCDF attribute topography_source_path missing; pass --dem)")
        dem = dem_on_grid(grid, dem_path)
        work = out_dir / "_work" / rid
        work.mkdir(parents=True, exist_ok=True)
        mod_mask, mod_lines, failures = modern_lines(grid, dem, members, work)
        if any(f["is_main"] for f in failures) or not mod_lines:
            raise ValueError(f"main member centerlines failed: {failures[:2]}")
        lia, notes = lia_lines(row, grid)
        need = mod_mask | grid.lia_mask
        if not np.isfinite(dem[need]).all():
            raise ValueError("DEM has NaN inside the glacier masks")
        arr = {"schema": np.asarray("unified-static-v1"), "record_id": np.asarray(rid),
               "region": np.asarray(str(row["region"])), "cohort": np.asarray(str(row["cohort"])),
               "n_members": np.asarray(int(len(members))),
               "f19_volume_km3": np.asarray(float(row["native_reference_volume_km3"])),
               "dx_m": np.asarray(abs(grid.dx)), "dem_m": dem.astype(np.float32),
               "modern_mask": mod_mask, "lia_mask": grid.lia_mask,
               "crs_wkt": np.asarray(grid.crs_wkt), "grid_x_m": grid.x, "grid_y_m": grid.y,
               **ragged("mod", mod_lines), **ragged("lia", lia)}
        np.savez_compressed(out_dir / f"system_{rid}.npz", **arr)
        rec.update({"status": "exported", "modern_member_failures": len(failures),
                    "modern_member_failure_detail": json.dumps(failures, ensure_ascii=False),
                    "modern_lines": len(mod_lines), "lia_lines": len(lia), "notes": ";".join(notes),
                    "dem_path": dem_path})
    except Exception as error:
        rec.update({"status": "export-failed", "error": f"{type(error).__name__}: {error}"[:300],
                    "traceback": traceback.format_exc()[-1500:]})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--systems-index", required=True, type=Path)
    ap.add_argument("--members-index", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dem", default=None)
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    systems = pd.read_csv(args.systems_index)
    members = pd.read_csv(args.members_index)
    if args.cohort:
        systems = systems[systems.cohort == args.cohort]
    if args.limit:
        systems = systems.head(args.limit)
    tasks = [(r, members[members.record_id == r["record_id"]], args.out_dir, args.dem)
             for r in systems.to_dict("records")]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(config,)) as pool:
            records = list(pool.map(export_system, tasks, chunksize=1))
    else:
        _init(config)
        records = [export_system(t) for t in tasks]
    table = pd.DataFrame(records)
    table.drop(columns=["traceback"], errors="ignore").to_csv(args.out_dir / "export_manifest.csv", index=False)
    summary = {"systems": int(len(table)), "exported": int((table.status == "exported").sum()),
               "failed": table.loc[table.status != "exported", ["record_id", "error"]].to_dict("records"),
               "modern_member_failures_total": int(table.get("modern_member_failures", pd.Series(dtype=float)).fillna(0).sum()),
               "scope": "read-only export; no thickness product read; no reconstruction run"}
    (args.out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"}, indent=2), flush=True)
    for f in summary["failed"][:20]:
        print("FAILED", f["record_id"], f["error"], flush=True)


if __name__ == "__main__":
    main()
