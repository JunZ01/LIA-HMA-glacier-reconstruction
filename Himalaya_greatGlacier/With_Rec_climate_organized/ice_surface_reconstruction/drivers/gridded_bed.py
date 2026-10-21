"""Load 2-D bed, glacier mask and slope from an OGGM gdir onto one pixel grid.

Driver-layer helper (reads rasters/NetCDF); ``core`` never imports this. The
returned arrays share the ``gridded_data.nc`` pixel frame, which is ALSO the
frame OGGM flowline ``line.coords`` live in: a flowline coordinate ``(col, row)``
indexes directly into these arrays. One pixel is ``map_dx_m`` metres (66 m in
the pilot), so a path step of k pixels is ``k * map_dx_m`` metres of real
horizontal distance.

Bed is ``topo - distributed_thickness`` on the glacier, ``topo`` elsewhere; the
distributed thickness variable is bed-scenario specific.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


_THICKNESS_VAR = {
    "farinotti2019": "distributed_thickness_f19_fs0_full_v0",
    "millan2022": "distributed_thickness_m22_fs0_full_v0",
}


@dataclass(frozen=True)
class GriddedBed:
    """2-D bed and masks on the gdir pixel grid (col=x index, row=y index)."""

    bed_m: np.ndarray            # (nrow, ncol); topo - ice where glaciated
    glacier_mask: np.ndarray     # (nrow, ncol) bool
    slope: np.ndarray | None     # (nrow, ncol) radians, or None
    map_dx_m: float              # metres per pixel
    bed_scenario: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.bed_m.shape

    def in_bounds(self, row: int, col: int) -> bool:
        nrow, ncol = self.bed_m.shape
        return 0 <= row < nrow and 0 <= col < ncol

    def inside_mask(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and bool(self.glacier_mask[row, col])


def xy_to_rc(coord_xy: np.ndarray | tuple[float, float]) -> tuple[int, int]:
    """Convert OGGM ``(col, row)`` coordinates to array ``(row, col)``."""
    coordinate = np.asarray(coord_xy, dtype=float)
    if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
        raise ValueError("coord_xy must contain exactly two finite values")
    col, row = coordinate
    return int(round(float(row))), int(round(float(col)))


def load_gridded_bed(gdir_dir: Path, bed_scenario: str) -> GriddedBed:
    """Read topo, distributed thickness, glacier mask and slope from a gdir."""
    thickness_var = _THICKNESS_VAR.get(bed_scenario)
    if thickness_var is None:
        raise ValueError(f"Unknown bed_scenario {bed_scenario!r}")
    grid = json.loads((gdir_dir / "glacier_grid.json").read_text())
    map_dx_m = abs(float(grid["dxdy"][0]))
    with xr.open_dataset(gdir_dir / "gridded_data.nc") as ds:
        topo = np.asarray(ds["topo"].values, dtype=float)
        mask = np.asarray(ds["glacier_mask"].values).astype(bool)
        slope = np.asarray(ds["slope"].values, dtype=float) if "slope" in ds else None
        if thickness_var not in ds:
            raise ValueError(
                f"{thickness_var} not in gridded_data.nc for {gdir_dir.name}"
            )
        thickness = np.asarray(ds[thickness_var].values, dtype=float)
    thickness = np.where(np.isfinite(thickness), thickness, 0.0)
    bed = topo - np.where(mask, thickness, 0.0)
    return GriddedBed(
        bed_m=bed,
        glacier_mask=mask,
        slope=slope,
        map_dx_m=map_dx_m,
        bed_scenario=bed_scenario,
    )
