"""Read-only field-validity preflight for the frozen co-located experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

import run_temperature_west_colocated_independent_validation_v4 as runner


def main() -> int:
    experiment = runner.experiment
    with netcdf_file(experiment.REFERENCE, "r", mmap=False) as dataset:
        lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    hma = np.load(experiment.HMA_MASK).astype(bool)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lineage = pd.read_csv(experiment.LINEAGE)
    paths = [Path(path) for path in lineage.loc[lineage.variable.eq("tas"), "source_tif"]]
    fields = [runner.read_tif_prefix(path, lat, lon) for path in paths]
    project = np.nanmean(np.stack(fields), axis=0)
    rows = []
    for site, config in experiment.SITES.items():
        distance = experiment.distance_km(
            lat_grid, lon_grid, config["latitude"], config["longitude"]
        )
        valid_distance = np.where(hma, distance, np.inf)
        nearest = np.zeros_like(hma)
        nearest.flat[int(np.argmin(valid_distance))] = True
        radius = hma & (distance <= experiment.RADIUS_KM)
        for scale, mask in (("nearest", nearest), ("radius100km", radius)):
            values = experiment.weighted_target(project, mask, lat)
            baseline = experiment.YEARS <= 1860
            rows.append(
                {
                    "target": f"{site}__{scale}",
                    "cell_count": int(mask.sum()),
                    "finite_fraction": float(np.isfinite(values).mean()),
                    "baseline_sd_degC": float(np.nanstd(values[baseline], ddof=0)),
                    "baseline_min_degC": float(np.nanmin(values[baseline])),
                    "baseline_max_degC": float(np.nanmax(values[baseline])),
                }
            )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
