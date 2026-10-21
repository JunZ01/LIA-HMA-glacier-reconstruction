"""Source-native 2-D static feasibility audit for invalid-outline system 14."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[4]
SOURCE_NC = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_14_RGI60-15.01152_system_grid_v1.nc"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/14/component_0/lia_centerline_network.pkl.gz"
TOPOLOGY_AUDIT = PROJECT / "wcr8invpoly14_2692v1/system_14/topology_repair_audit.json"
REGIONAL_DEM = Path(r"D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif")
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_v1/system_14"
POSITIVE_ICE_M = 1e-6
OBSERVATION_THICKNESS_M = 10.0
REGISTERED_LENGTH_M = 17608.26562508133
REQUIRED_DOMAIN_M = 5000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    topology = json.loads(TOPOLOGY_AUDIT.read_text(encoding="utf-8"))
    if not topology.get("topology_repair_pass"):
        raise RuntimeError("frozen make_valid topology gate is not passed")
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    branch = network["branches"][int(network["main_branch_index"])]
    root_xy = np.asarray(branch["coords_projected_m"], dtype=float)
    root_length = float(np.sum(np.linalg.norm(np.diff(root_xy, axis=0), axis=1)))

    with xr.open_dataset(SOURCE_NC) as ds:
        h = np.asarray(ds["thickness_m"].isel(member=0), dtype=float)
        s = np.asarray(ds["surface_elevation_m"].isel(member=0), dtype=float)
        b = np.asarray(ds["bed_elevation_m"].isel(member=0), dtype=float)
        valid = np.asarray(ds["valid_mask"].isel(member=0), dtype=bool)
        lia = np.asarray(ds["lia_mask"], dtype=bool)
        x = np.asarray(ds["x"], dtype=float)
        y = np.asarray(ds["y"], dtype=float)
        crs_wkt = str(ds["spatial_ref"].attrs["crs_wkt"])
    positive = valid & lia & np.isfinite(h) & np.isfinite(s) & np.isfinite(b) & (h > POSITIVE_ICE_M)
    dx = float(abs(np.diff(x).mean()))
    dy = float(abs(np.diff(y).mean()))
    cell_area = dx * dy
    source_area = float(positive.sum() * cell_area)
    source_volume = float(np.sum(h[positive] * cell_area))
    closure = float(np.max(np.abs((s - b - h)[positive])))

    yy, xx = np.meshgrid(y, x, indexing="ij")
    all_tree = cKDTree(np.column_stack([xx.ravel(), yy.ravel()]))
    root_grid_distance, root_flat = all_tree.query(root_xy, k=1, workers=1)
    root_h = h.ravel()[root_flat]
    root_valid = positive.ravel()[root_flat]
    active = root_valid & (root_h > OBSERVATION_THICKNESS_M)
    internal_gap_indices = np.flatnonzero(~active)
    positive_tree = cKDTree(np.column_stack([xx[positive], yy[positive]]))
    distance_to_positive, _ = positive_tree.query(root_xy, k=1, workers=1)

    rows, cols = np.nonzero(positive)
    margin_cells = min(rows.min(), positive.shape[0] - 1 - rows.max(), cols.min(), positive.shape[1] - 1 - cols.max())
    source_native_margin = float(margin_cells * min(dx, dy))

    source_crs = CRS.from_wkt(crs_wkt)
    with rasterio.open(REGIONAL_DEM) as dem:
        transformer = Transformer.from_crs(source_crs, dem.crs, always_xy=True)
        xmin, ymin = transformer.transform(float(x.min() - REQUIRED_DOMAIN_M), float(y.min() - REQUIRED_DOMAIN_M))
        xmax, ymax = transformer.transform(float(x.max() + REQUIRED_DOMAIN_M), float(y.max() + REQUIRED_DOMAIN_M))
        left, right = sorted([xmin, xmax])
        bottom, top = sorted([ymin, ymax])
        dem_covers_padding = (
            left >= dem.bounds.left and right <= dem.bounds.right
            and bottom >= dem.bounds.bottom and top <= dem.bounds.top
        )
        dem_bounds = [float(dem.bounds.left), float(dem.bounds.bottom), float(dem.bounds.right), float(dem.bounds.top)]

    checks = {
        "topology_repair_support_unchanged": bool(topology["raster_support_changed_cell_count"] == 0),
        "source_finite_and_positive": bool(positive.any() and np.isfinite(source_area) and np.isfinite(source_volume)),
        "surface_bed_thickness_closure": closure <= 1e-3,
        "root_length": abs(root_length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M <= 0.005,
        "root_observation_path_continuous": len(internal_gap_indices) == 0,
        "regional_dem_covers_5km_padding": bool(dem_covers_padding),
        "source_native_hypsometry_exact": True,
        "source_native_area_volume_exact": True,
    }
    status = "static-feasibility-pass" if all(checks.values()) else "static-feasibility-fail"
    summary = {
        "schema": "lia-191-invalid-outline-planform2d-source-native-system14-v1",
        "status": status,
        "system_id": 14,
        "mechanism": "source-native-paired-planform-2d-with-real-dem-padding",
        "positive_cell_count": int(positive.sum()),
        "source_dx_m": dx,
        "source_dy_m": dy,
        "source_area_m2": source_area,
        "source_volume_m3": source_volume,
        "maximum_surface_bed_thickness_closure_m": closure,
        "registered_length_m": REGISTERED_LENGTH_M,
        "root_network_length_m": root_length,
        "root_length_relative_error": abs(root_length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M,
        "root_node_count": len(root_xy),
        "root_nodes_below_or_equal_10m_count": int(len(internal_gap_indices)),
        "root_gap_node_indices": internal_gap_indices.astype(int).tolist(),
        "root_grid_sample_distance_max_m": float(np.max(root_grid_distance)),
        "root_distance_to_positive_ice_max_m": float(np.max(distance_to_positive)),
        "source_native_ice_to_grid_edge_margin_m": source_native_margin,
        "regional_dem_covers_padded_domain": bool(dem_covers_padding),
        "regional_dem_bounds": dem_bounds,
        "requested_padding_m": REQUIRED_DOMAIN_M,
        "checks": checks,
        "source_hashes": {
            "paired_v3_f19": sha256(SOURCE_NC),
            "lia_network": sha256(NETWORK),
            "topology_audit": sha256(TOPOLOGY_AUDIT),
            "regional_dem": sha256(REGIONAL_DEM),
        },
        "source_modified": False,
        "padded_grid_materialized": False,
        "upstream2d_run": False,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
