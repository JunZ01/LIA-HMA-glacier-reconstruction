"""Compare scheme-A modern thickness with measured ice-thickness points.

Independent check of the unified static result (F19 is not used here). For
every system NPZ, measured points are projected into the system grid, points
inside the modern mask are kept, the modern thickness is recomputed with the
engine at the system's scheme-A tau (tau_schemes.csv), and sampled at the
points (bilinear). A point that falls inside several systems is assigned to
the first system that contains it.

Measured files: one or more CSVs with latitude, longitude and thickness
columns (GlaThiDa point names by default). Use --set-col for a dataset
identifier column, otherwise the file name is the dataset id.

Reported per dataset, per system and overall: points, mean measured,
mean modelled, mean bias (model - measured), RMSE, median ratio, Pearson r.
Survey dates differ from the DEM epoch; this is stated in the output, not
corrected.

Usage (repo root, oggm_env2):
    python validate_thickness_points_20261022.py --npz-dir NPZ --tau-csv tau_schemes.csv \
        --points a.csv b.csv ... --out OUT_DIR [--lat-col POINT_LAT --lon-col POINT_LON --h-col THICKNESS]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

import static_unified_engine_20261022 as eng


def load_points(paths, lat, lon, h, set_col):
    frames = []
    for p in paths:
        t = pd.read_csv(p)
        missing = [c for c in (lat, lon, h) if c not in t.columns]
        if missing:
            raise ValueError(f"{p}: missing columns {missing}; columns are {list(t.columns)[:20]}")
        t = t.rename(columns={lat: "lat", lon: "lon", h: "measured_m"})
        t["dataset"] = t[set_col].astype(str) if set_col and set_col in t.columns else Path(p).stem
        frames.append(t[["dataset", "lat", "lon", "measured_m"]])
    pts = pd.concat(frames, ignore_index=True)
    pts = pts[np.isfinite(pts.lat) & np.isfinite(pts.lon) & np.isfinite(pts.measured_m) & (pts.measured_m >= 0)]
    pts["point_id"] = np.arange(len(pts))
    return pts.reset_index(drop=True)


def to_grid(d, lon, lat, transformer_factory):
    tr = transformer_factory(str(d["crs_wkt"]))
    xm, ym = tr(np.asarray(lon), np.asarray(lat))
    x, y = d["grid_x_m"], d["grid_y_m"]
    return (np.asarray(xm) - x[0]) / (x[1] - x[0]), (np.asarray(ym) - y[0]) / (y[1] - y[0])


def pyproj_factory(wkt):
    from pyproj import CRS, Transformer
    t = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_wkt(wkt), always_xy=True)
    return t.transform


def stats(g):
    m, o = g.modelled_m.to_numpy(), g.measured_m.to_numpy()
    pos = o > 0
    return pd.Series({"points": len(g), "mean_measured_m": o.mean(), "mean_modelled_m": m.mean(),
                      "mean_bias_m": (m - o).mean(), "rmse_m": float(np.sqrt(np.mean((m - o) ** 2))),
                      "median_ratio": float(np.median(m[pos] / o[pos])) if pos.any() else np.nan,
                      "pearson_r": float(np.corrcoef(m, o)[0, 1]) if len(g) > 2 and np.std(o) > 0 and np.std(m) > 0 else np.nan})


def run(npz_dir, tau_csv, points, out, transformer_factory=pyproj_factory):
    out.mkdir(parents=True, exist_ok=False)
    taus = pd.read_csv(tau_csv).set_index("file")["tau_A"]
    assigned = np.zeros(len(points), bool)
    rows = []
    for path in sorted(npz_dir.glob("system_*.npz")):
        if path.name not in taus.index or not np.isfinite(taus[path.name]):
            continue
        todo = points[~assigned]
        if todo.empty:
            break
        d = eng.load_system(path)
        col, row = to_grid(d, todo.lon, todo.lat, transformer_factory)
        r, c = np.rint(row).astype(int), np.rint(col).astype(int)
        inside = (r >= 0) & (c >= 0) & (r < d["modern_mask"].shape[0]) & (c < d["modern_mask"].shape[1])
        hit = np.zeros(len(todo), bool)
        hit[inside] = d["modern_mask"][r[inside], c[inside]]
        if not hit.any():
            continue
        h = eng.modern_end(d, float(taus[path.name]))["H"]
        sel = todo[hit]
        mod = map_coordinates(h, np.vstack([row[hit], col[hit]]), order=1, mode="nearest", prefilter=False)
        rows.append(pd.DataFrame({"point_id": sel.point_id.to_numpy(), "dataset": sel.dataset.to_numpy(),
                                  "record_id": str(d["record_id"]), "region": str(d["region"]),
                                  "measured_m": sel.measured_m.to_numpy(), "modelled_m": mod}))
        assigned[sel.point_id.to_numpy()] = True
    if not rows:
        raise ValueError("no measured point falls inside any modern mask")
    matched = pd.concat(rows, ignore_index=True)
    matched.to_csv(out / "matched_points.csv", index=False)
    by_set = matched.groupby("dataset").apply(stats, include_groups=False)
    by_sys = matched.groupby(["dataset", "record_id", "region"]).apply(stats, include_groups=False)
    by_set.to_csv(out / "by_dataset.csv")
    by_sys.to_csv(out / "by_system.csv")
    overall = stats(matched).to_dict()
    summary = {"points_supplied": int(len(points)), "points_matched": int(len(matched)),
               "points_outside_all_modern_masks": int((~assigned).sum()),
               "overall": overall, "by_dataset": by_set.round(3).to_dict(orient="index"),
               "note": "independent of F19; survey dates differ from the DEM epoch and are not corrected"}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, type=Path)
    ap.add_argument("--tau-csv", required=True, type=Path)
    ap.add_argument("--points", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lat-col", default="POINT_LAT")
    ap.add_argument("--lon-col", default="POINT_LON")
    ap.add_argument("--h-col", default="THICKNESS")
    ap.add_argument("--set-col", default=None)
    a = ap.parse_args()
    pts = load_points(a.points, a.lat_col, a.lon_col, a.h_col, a.set_col)
    s = run(a.npz_dir, a.tau_csv, pts, a.out)
    print(json.dumps({k: v for k, v in s.items() if k != "by_dataset"}, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
