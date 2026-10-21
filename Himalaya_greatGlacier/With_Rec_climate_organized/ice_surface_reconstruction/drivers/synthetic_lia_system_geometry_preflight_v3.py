"""Section-182 25 m edge-coverage wrapper for the synthetic LIA gdir."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from oggm import cfg
from scipy.ndimage import binary_erosion, distance_transform_edt

from ..paths import resolve
from . import synthetic_lia_system_geometry_preflight_v1 as v1
from .lia_centerline_pilot import _prepare_oggm_lia_geometry


_configure_v1 = v1._configure_oggm
_run_v1 = v1.run


def _configure_oggm(output_root, flowline_dx):
    version = _configure_v1(output_root, flowline_dx)
    cfg.PARAMS["use_tar_shapefiles"] = False
    return version


def _prepare_gridded_data(gdir, grid, lia_geometry, surface_source, regional_dem):
    pixel_geometry, glacier_mask, _ = _prepare_oggm_lia_geometry(
        lia_geometry, grid
    )
    if not np.isfinite(regional_dem).all():
        raise ValueError("regional-dem-has-nonfinite-cells")
    surface = np.asarray(surface_source, dtype=float).copy()
    finite = np.isfinite(surface)
    missing_surface = glacier_mask & ~finite
    maximum_distance_m = 0.0
    if missing_surface.any():
        distances, nearest = distance_transform_edt(
            ~finite,
            sampling=(abs(float(grid.dy)), abs(float(grid.dx))),
            return_indices=True,
        )
        maximum_distance_m = float(np.max(distances[missing_surface]))
        surface[missing_surface] = surface[
            nearest[0][missing_surface], nearest[1][missing_surface]
        ]
    if not np.isfinite(surface[glacier_mask]).all():
        raise ValueError("paired-edge-fill-left-nonfinite-surface")
    topo = np.asarray(regional_dem, dtype=float).copy()
    topo[glacier_mask] = surface[glacier_mask]
    exterior = glacier_mask & ~binary_erosion(
        glacier_mask, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    dataset = xr.Dataset(
        data_vars={
            "glacier_mask": (("y", "x"), glacier_mask.astype(np.int16)),
            "glacier_ext": (("y", "x"), exterior.astype(np.int16)),
            "topo": (("y", "x"), topo.astype(np.float32)),
            "topo_smoothed": (("y", "x"), topo.astype(np.float32)),
        },
        coords={"x": np.arange(grid.nx), "y": np.arange(grid.ny)},
    )
    dataset.to_netcdf(gdir.get_filepath("gridded_data"), engine="netcdf4")
    gdir.write_pickle(
        {
            "polygon_pix": pixel_geometry,
            "polygon_area": float(lia_geometry.area),
        },
        "geometries",
    )
    audit = {
        "method": "nearest-finite-paired-lia-surface",
        "filled_cell_count": int(missing_surface.sum()),
        "glacier_mask_cell_count": int(glacier_mask.sum()),
        "filled_fraction": float(missing_surface.sum() / glacier_mask.sum()),
        "filled_area_m2": float(
            missing_surface.sum() * abs(float(grid.dx * grid.dy))
        ),
        "maximum_fill_distance_m": maximum_distance_m,
        "regional_dem_used_inside_mask": False,
    }
    v1._write_json(Path(gdir.dir) / "surface_fill_audit.json", audit)
    return glacier_mask, topo


def run(config_path: Path):
    summary = _run_v1(config_path)
    audit_path = Path(summary["synthetic_gdir"]) / "surface_fill_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    summary["surface_fill"] = audit
    config = json.loads(config_path.read_text(encoding="utf-8"))
    v1._write_json(resolve(config["output_root"]) / "geometry_summary.json", summary)
    return summary


v1._configure_oggm = _configure_oggm
v1._prepare_gridded_data = _prepare_gridded_data
v1.run = run


if __name__ == "__main__":
    v1.main()
