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


def describe(value):
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__}


def main():
    with xr.open_dataset(GRID) as ds:
        grid = {
            "dims": dict(ds.sizes),
            "variables": {name: {"dims": list(var.dims), "shape": list(var.shape), "dtype": str(var.dtype)} for name, var in ds.variables.items()},
            "attrs": dict(ds.attrs),
        }
    with gzip.open(NETWORK, "rb") as handle:
        network = pickle.load(handle)
    network_desc = {
        "keys": list(network.keys()),
        "main_branch_index": network.get("main_branch_index"),
        "branch_count": len(network.get("branches", [])),
        "branch_keys": [list(branch.keys()) for branch in network.get("branches", [])],
        "branch_shapes": [{key: describe(value) for key, value in branch.items()} for branch in network.get("branches", [])],
    }
    with (GDIR / "inversion_flowlines.pkl").open("rb") as handle:
        flowlines = pickle.load(handle)
    flow_desc = []
    index = {id(fl): i for i, fl in enumerate(flowlines)}
    for i, fl in enumerate(flowlines):
        attrs = {}
        for name in ("surface_h", "bed_h", "widths", "widths_m", "dx", "map_dx", "dx_meter", "nx", "flows_to_indice", "inflow_indices"):
            try:
                attrs[name] = describe(getattr(fl, name))
            except Exception as exc:
                attrs[name] = {"error": repr(exc)}
        attrs["line_shape"] = list(np.asarray(fl.line.coords).shape)
        attrs["flows_to_index"] = index.get(id(fl.flows_to)) if fl.flows_to is not None else None
        flow_desc.append(attrs)
    out = {"grid": grid, "network": network_desc, "flowlines": flow_desc}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
