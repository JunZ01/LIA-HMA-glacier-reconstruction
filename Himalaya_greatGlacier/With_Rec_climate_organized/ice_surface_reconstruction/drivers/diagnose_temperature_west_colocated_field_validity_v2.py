"""Read-only native-grid field validity diagnostic using the V5 globals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

import run_temperature_west_colocated_independent_validation_v5 as runner


def main() -> int:
    e = runner.experiment
    with netcdf_file(e.REFERENCE, "r", mmap=False) as dataset:
        lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    valid_mask = np.load(e.HMA_MASK).astype(bool)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lineage = pd.read_csv(e.LINEAGE)
    paths = [Path(path) for path in lineage.loc[lineage.variable.eq("tas"), "source_tif"]]
    fields = [runner.read_tif_prefix(path, lat, lon) for path in paths]
    project = np.nanmean(np.stack(fields), axis=0)
    rows = []
    for site, config in e.SITES.items():
        distance = e.distance_km(lat_grid, lon_grid, config["latitude"], config["longitude"])
        valid_distance = np.where(valid_mask, distance, np.inf)
        nearest = np.zeros_like(valid_mask)
        nearest.flat[int(np.argmin(valid_distance))] = True
        radius = valid_mask & (distance <= e.RADIUS_KM)
        for scale, mask in (("nearest", nearest), ("radius100km", radius)):
            values = e.weighted_target(project, mask, lat)
            baseline = e.YEARS <= 1860
            rows.append({
                "target": f"{site}__{scale}",
                "cells": int(mask.sum()),
                "finite_fraction": float(np.isfinite(values).mean()),
                "baseline_sd": float(np.nanstd(values[baseline], ddof=0)),
                "first": float(values[0]),
            })
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
