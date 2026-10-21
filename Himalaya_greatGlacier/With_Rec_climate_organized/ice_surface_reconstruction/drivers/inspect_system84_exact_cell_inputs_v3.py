from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT = Path(__file__).resolve().parents[4]
GRID = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_84_RGI60-15.01996_system_grid_v1.nc"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/84/component_0/lia_centerline_network.pkl.gz"
GDIR = PROJECT / "w404geom84v1/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"


def arr_summary(value):
    a = np.asarray(value, dtype=float)
    finite = a[np.isfinite(a)]
    return {
        "shape": list(a.shape),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def main():
    with xr.open_dataset(GRID) as ds:
        grid = {
            "dims": dict(ds.sizes),
            "variables": {name: {"dims": list(v.dims), "shape": list(v.shape)} for name, v in ds.variables.items()},
        }
    with gzip.open(NETWORK, "rb") as handle:
        network = pickle.load(handle)
    branches = network.get("branches", [])
    network_out = {
        "keys": sorted(network),
        "main_branch_index": network.get("main_branch_index"),
        "branch_count": len(branches),
        "branch_keys": sorted(branches[0]) if branches else [],
        "branches": [],
    }
    for i, branch in enumerate(branches):
        item = {"index": i}
        for key in ("coords", "line_coords", "surface_h", "bed_h", "widths_m", "widths", "thick", "thickness_m"):
            if key in branch:
                try:
                    item[key] = arr_summary(branch[key])
                except Exception:
                    item[key] = {"type": type(branch[key]).__name__}
        for key in ("flows_to", "flows_to_index", "flows_to_indice", "order", "is_main"):
            if key in branch:
                value = branch[key]
                item[key] = value.item() if isinstance(value, np.generic) else value
        network_out["branches"].append(item)
    with gzip.open(GDIR / "inversion_flowlines.pkl", "rb") as handle:
        flowlines = pickle.load(handle)
    index = {id(fl): i for i, fl in enumerate(flowlines)}
    flow_out = []
    for i, fl in enumerate(flowlines):
        item = {
            "index": i,
            "nx": int(fl.nx),
            "dx": float(fl.dx),
            "map_dx": float(fl.map_dx),
            "dx_meter": float(fl.dx_meter),
            "flows_to_index": index.get(id(fl.flows_to)) if fl.flows_to is not None else None,
            "flows_to_indice": int(fl.flows_to_indice) if fl.flows_to is not None else None,
            "line": arr_summary(np.asarray(fl.line.coords)),
            "surface_h": arr_summary(fl.surface_h),
            "widths": arr_summary(fl.widths),
        }
        if hasattr(fl, "bed_h"):
            item["bed_h"] = arr_summary(fl.bed_h)
        flow_out.append(item)
    print(json.dumps({"grid": grid, "network": network_out, "flowlines": flow_out}, indent=2, default=str))


if __name__ == "__main__":
    main()
