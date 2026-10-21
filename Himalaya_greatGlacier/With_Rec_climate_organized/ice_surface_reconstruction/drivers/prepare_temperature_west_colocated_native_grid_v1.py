"""Prepare a read-only-derived native project grid for co-located validation."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.io import netcdf_file


sys.modules.setdefault("rasterio", types.ModuleType("rasterio"))
import run_climate_reconstruction_loop_v2_round23_corrected_reconstruction_reference_envelope_v1 as round23


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_colocated_independent_validation_v1"
LINEAGE = RESULTS / "climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
REFERENCE = RESULTS / "climate_reconstruction_improvement_loop_v2/round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
ADDENDUM = OUTPUT / "SPATIAL_GRID_ADDENDUM.md"
OUT_REFERENCE = OUTPUT / "native_source_grid_reference_1812_1900.nc"
OUT_MASK = OUTPUT / "native_source_grid_six_member_valid_mask.npy"
YEARS = np.arange(1812, 1901)
SITES = {
    "hirpora": (33.066, 74.068),
    "kumrat": (35.570, 72.207),
    "upper_indus": (35.34, 74.79),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        scale = tuple(float(value) for value in page.tags[33550].value)
        tie = tuple(float(value) for value in page.tags[33922].value)
        nodata = float(page.tags[42113].value)
        height, width, count = tif.series[0].shape
    if count != 1001:
        raise RuntimeError(f"unexpected-source-sample-count:{count}")
    lon = tie[3] + (np.arange(width) + 0.5) * scale[0]
    lat = tie[4] - (np.arange(height) + 0.5) * scale[1]
    return lat, lon, nodata


def read_prefix(path: Path, nodata: float) -> np.ndarray:
    source = tifffile.memmap(path, series=0, mode="r")
    selected = np.asarray(source[:, :, YEARS - 1000], dtype=float).transpose(2, 0, 1)
    selected[np.isclose(selected, nodata) | (selected < -9000)] = np.nan
    return selected


def main() -> int:
    if not ADDENDUM.exists():
        raise FileNotFoundError(ADDENDUM)
    lineage = pd.read_csv(LINEAGE)
    temperature = lineage[lineage.variable.eq("tas")].copy()
    expected = {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}
    if set(temperature.member) != expected:
        raise RuntimeError("temperature-lineage-contract-failed")
    members = {row.member: Path(row.source_tif) for row in temperature.itertuples(index=False)}
    paths = {"lineage": LINEAGE, "reference": REFERENCE, "addendum": ADDENDUM, **members}
    before = {name: sha256(path) for name, path in paths.items()}
    for row in temperature.itertuples(index=False):
        if before[row.member] != row.source_tif_sha256:
            raise RuntimeError(f"member-hash-mismatch:{row.member}")

    coordinates = [source_coordinates(path) for path in members.values()]
    lat, lon, nodata = coordinates[0]
    if not all(np.array_equal(item[0], lat) and np.array_equal(item[1], lon) and item[2] == nodata for item in coordinates[1:]):
        raise RuntimeError("member-native-grid-mismatch")
    fields = [read_prefix(path, nodata) for path in members.values()]
    valid = np.logical_and.reduce([np.isfinite(field).all(axis=0) for field in fields])
    np.save(OUT_MASK, valid)

    with netcdf_file(REFERENCE, "r", mmap=False) as dataset:
        ref_year = np.asarray(dataset.variables["year"].data, dtype=int).copy()
        indices = np.flatnonzero((ref_year >= YEARS[0]) & (ref_year <= YEARS[-1]))
        if not np.array_equal(ref_year[indices], YEARS):
            raise RuntimeError("reference-prefix-year-closure-failed")
        ref_lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        ref_lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"].data[indices], dtype=float).copy()
    reference[reference < -9000] = np.nan
    projected, orientation = round23.interpolate_to_reference(reference, ref_lat, ref_lon, lat, lon)
    with netcdf_file(OUT_REFERENCE, "w") as dataset:
        dataset.createDimension("year", len(YEARS))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i", ("year",))[:] = YEARS
        dataset.createVariable("lat", "f", ("lat",))[:] = lat
        dataset.createVariable("lon", "f", ("lon",))[:] = lon
        dataset.createVariable("tas_reference_family_balanced", "f", ("year", "lat", "lon"))[:] = projected.astype(np.float32)

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    site_rows = []
    for site, (latitude, longitude) in SITES.items():
        distance = np.sqrt(
            ((lat_grid - latitude) * 111.0) ** 2
            + ((lon_grid - longitude) * 111.0 * np.cos(np.deg2rad(latitude))) ** 2
        )
        valid_distance = np.where(valid, distance, np.inf)
        nearest = int(np.argmin(valid_distance))
        radius_count = int((valid & (distance <= 100.0)).sum())
        site_rows.append({
            "site": site,
            "nearest_latitude": float(lat_grid.flat[nearest]),
            "nearest_longitude": float(lon_grid.flat[nearest]),
            "nearest_distance_km": float(valid_distance.flat[nearest]),
            "radius100km_valid_cell_count": radius_count,
        })
    site_table = pd.DataFrame(site_rows)
    site_table.to_csv(OUTPUT / "native_grid_site_preflight.csv", index=False)
    if site_table.nearest_distance_km.max() > 35.0 or site_table.radius100km_valid_cell_count.min() < 3:
        raise RuntimeError("native-grid-site-gate-failed")
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "years_read": [1812, 1900],
        "native_grid_shape": [len(lat), len(lon)],
        "native_latitude_range": [float(lat.min()), float(lat.max())],
        "native_longitude_range": [float(lon.min()), float(lon.max())],
        "jointly_valid_cell_count": int(valid.sum()),
        "reference_interpolation": orientation,
        "site_preflight": site_rows,
        "source_hashes_unchanged": before == after,
        "post1900_values_read": False,
    }
    (OUTPUT / "native_grid_preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "native_grid_preflight_source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
