from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT = Path(__file__).resolve().parents[4]
GRID = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_84_RGI60-15.01996_system_grid_v1.nc"
GDIR = PROJECT / "w404geom84v1/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"


def main():
    with xr.open_dataset(GRID) as ds:
        x = np.asarray(ds.x.values, dtype=float)
        y = np.asarray(ds.y.values, dtype=float)
        valid = np.asarray(ds.valid_mask.isel(member=0).values, dtype=bool)
        lia = np.asarray(ds.lia_mask.values, dtype=bool)
        thick = np.asarray(ds.thickness_m.isel(member=0).values, dtype=float)
        attrs = {k: str(v) for k, v in ds.attrs.items()}
        spatial = {k: str(v) for k, v in ds.spatial_ref.attrs.items()}
    with open(GDIR / "glacier_grid.json", "r", encoding="utf-8") as handle:
        ggrid = json.load(handle)
    with gzip.open(GDIR / "inversion_flowlines.pkl", "rb") as handle:
        fls = pickle.load(handle)
    x0, y0 = map(float, ggrid["x0y0"])
    dx, dy = map(float, ggrid["dxdy"])
    controls = []
    for i, fl in enumerate(fls):
        pix = np.asarray(fl.line.coords, dtype=float)
        xp = x0 + pix[:, 0] * dx
        yp = y0 + pix[:, 1] * dy
        controls.append({
            "index": i,
            "x_min": float(xp.min()), "x_max": float(xp.max()),
            "y_min": float(yp.min()), "y_max": float(yp.max()),
            "start": [float(xp[0]), float(yp[0])],
            "end": [float(xp[-1]), float(yp[-1])],
        })
    mask = valid & lia & np.isfinite(thick) & (thick > 0)
    out = {
        "paired_x": [float(x.min()), float(x.max()), float(np.median(np.diff(x)))],
        "paired_y": [float(y.min()), float(y.max()), float(np.median(np.diff(y)))],
        "positive_cells": int(mask.sum()),
        "paired_attrs": attrs,
        "spatial_ref_attrs": spatial,
        "gdir_grid": ggrid,
        "controls": controls,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
