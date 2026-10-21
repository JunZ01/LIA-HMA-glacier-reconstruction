"""Build an LIA-system grid and mosaic the available modern-member beds.

The original pilot used the main modern glacier directory as the complete
computational domain.  A LIA system can contain many modern members and can
extend beyond that one directory.  This module creates a target grid covering
the complete LIA geometry, then reprojects every available member gdir onto
it.  Overlapping source values are checked; they are never silently averaged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPolygon, Polygon

from ..paths import resolve
from .gridded_bed import GriddedBed, _THICKNESS_VAR
from .lia_geometry_preflight import _gdir_path


INTERIOR_DISTANCE_POLICY = "maximum-source-interior-distance-wins"
MAIN_MEMBER_FIRST_POLICY = "main-member-first"
_LEGACY_INTERIOR_DISTANCE_POLICY = (
    "maximum-source-interior-distance-wins-no-average"
)


@dataclass(frozen=True)
class SystemGrid:
    """Pixel grid aligned to the main member's resolution and projection."""

    crs: str
    x0: float
    y0: float
    nx: int
    ny: int
    dx: float
    dy: float
    json_path: Path

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def transform(self) -> Affine:
        return Affine(
            self.dx,
            0.0,
            self.x0 - 0.5 * self.dx,
            0.0,
            self.dy,
            self.y0 - 0.5 * self.dy,
        )

    @property
    def center_transform(self) -> Affine:
        return Affine(self.dx, 0.0, self.x0, 0.0, self.dy, self.y0)

    def ij_to_crs(self, cols, rows):
        cols = np.asarray(cols, dtype=float)
        rows = np.asarray(rows, dtype=float)
        return (
            self.x0 + cols * self.dx,
            self.y0 + rows * self.dy,
        )


@dataclass(frozen=True)
class SystemGridContext:
    rgi_id: str
    lia_unique_id: int
    grid: SystemGrid
    lia_geometry: Polygon | MultiPolygon
    component_geometries: tuple[Polygon, ...]
    member_rgi_ids: tuple[str, ...]
    topography_m: np.ndarray
    topography_metadata: dict
    bed_by_scenario: dict[str, GriddedBed]
    modern_mask: np.ndarray
    mosaic_metadata_by_scenario: dict[str, dict]
    bed_errors_by_scenario: dict[str, str]


def normalize_lia_components(geometry) -> tuple[Polygon, ...]:
    """Return valid connected polygon components in descending area order."""
    repaired = geometry.buffer(0)
    if repaired.is_empty:
        raise ValueError("lia-geometry-empty-after-make-valid")
    if isinstance(repaired, Polygon):
        components = [repaired]
    elif isinstance(repaired, MultiPolygon):
        components = list(repaired.geoms)
    else:
        raise ValueError(f"unsupported-lia-geometry-type:{repaired.geom_type}")
    components = [item for item in components if not item.is_empty and item.area > 0]
    if not components:
        raise ValueError("lia-geometry-has-no-positive-area-component")
    return tuple(sorted(components, key=lambda item: item.area, reverse=True))


def _member_ids(
    manifest: pd.DataFrame | None, lia_unique_id: int, main_rgi_id: str
) -> tuple[str, ...]:
    if manifest is None or lia_unique_id not in manifest.index:
        return (main_rgi_id,)
    raw = str(manifest.loc[lia_unique_id].get("download_member_rgi_ids", ""))
    values = [item for item in raw.split(";") if item and item != "nan"]
    values = [main_rgi_id] + [item for item in values if item != main_rgi_id]
    return tuple(dict.fromkeys(values)) or (main_rgi_id,)


def _grid_from_gdir(gdir_dir: Path) -> dict:
    return json.loads((gdir_dir / "glacier_grid.json").read_text(encoding="utf-8"))


def _target_grid(
    main_grid: dict,
    geometry,
    output_path: Path,
    buffer_m: float,
    resolution_factor: int = 1,
) -> SystemGrid:
    crs = CRS.from_user_input(main_grid["proj"])
    projected = gpd.GeoSeries([geometry], crs=None).set_crs(crs).iloc[0]
    minx, miny, maxx, maxy = projected.bounds
    if isinstance(resolution_factor, bool) or not isinstance(
        resolution_factor, (int, np.integer)
    ) or int(resolution_factor) < 1:
        raise ValueError("system-grid-resolution-factor-must-be-positive-integer")
    resolution_factor = int(resolution_factor)
    base_dx = abs(float(main_grid["dxdy"][0]))
    dx = base_dx / resolution_factor
    dy = -dx if float(main_grid["dxdy"][1]) < 0 else dx
    main_x0, main_y0 = map(float, main_grid["x0y0"])
    # Anchor refined pixel edges to the original gdir pixel edges. This makes
    # factor 2/4 true nested grids instead of unrelated shifted rasters.
    anchor_x0 = main_x0 - 0.5 * base_dx + 0.5 * dx
    anchor_y0 = (
        main_y0 + 0.5 * base_dx - 0.5 * dx
        if dy < 0
        else main_y0 - 0.5 * base_dx + 0.5 * dx
    )
    minx -= float(buffer_m)
    maxx += float(buffer_m)
    miny -= float(buffer_m)
    maxy += float(buffer_m)
    x0 = anchor_x0 + np.floor((minx - anchor_x0) / dx) * dx
    if dy < 0:
        steps = np.floor((anchor_y0 - maxy) / abs(dy))
        y0 = anchor_y0 - steps * abs(dy)
    else:
        y0 = anchor_y0 + np.floor((miny - anchor_y0) / dy) * dy
    nx = int(np.ceil((maxx - x0) / dx)) + 1
    ny = int(np.ceil(abs(y0 - miny) / abs(dy))) + 1
    payload = {
        "proj": str(crs),
        "x0y0": [float(x0), float(y0)],
        "nxny": [nx, ny],
        "dxdy": [float(dx), float(dy)],
        "pixel_ref": "center",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return SystemGrid(
        crs=str(crs), x0=float(x0), y0=float(y0), nx=nx, ny=ny,
        dx=float(dx), dy=float(dy), json_path=output_path,
    )


def _source_transform(grid: dict) -> Affine:
    dx, dy = map(float, grid["dxdy"])
    x0, y0 = map(float, grid["x0y0"])
    return Affine(dx, 0.0, x0 - 0.5 * dx, 0.0, dy, y0 - 0.5 * dy)


def _reproject(
    values: np.ndarray,
    source_grid: dict,
    target: SystemGrid,
    source_crs: str,
    *,
    resampling: Resampling,
    source_nodata: float | None = np.nan,
) -> np.ndarray:
    destination = np.full(target.shape, np.nan, dtype=float)
    reproject(
        np.asarray(values, dtype=float),
        destination,
        src_transform=_source_transform(source_grid),
        src_crs=source_crs,
        src_nodata=source_nodata,
        dst_transform=target.transform,
        dst_crs=target.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return destination


def _load_system_topography(
    surface_dem_path: Path,
    target: SystemGrid,
) -> tuple[np.ndarray, dict]:
    """Reproject the authoritative regional DEM onto one system grid.

    The source raster is passed to GDAL as a band so the full regional DEM is
    never materialized in memory.
    """
    surface_dem_path = Path(surface_dem_path)
    if not surface_dem_path.is_file():
        raise FileNotFoundError(f"missing-system-surface-dem:{surface_dem_path}")
    destination = np.full(target.shape, np.nan, dtype=float)
    with rasterio.open(surface_dem_path) as source:
        if source.crs is None:
            raise ValueError(f"system-surface-dem-missing-crs:{surface_dem_path}")
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_nodata=source.nodata,
            dst_transform=target.transform,
            dst_crs=target.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        metadata = {
            "topography_source_policy": "regional-dem-only-no-gdir-fallback",
            "topography_source_path": str(surface_dem_path.resolve()),
            "topography_source_crs": str(source.crs),
            "topography_source_native_pixel_width": abs(float(source.transform.a)),
            "topography_source_native_pixel_height": abs(float(source.transform.e)),
            "topography_source_nodata": (
                float(source.nodata) if source.nodata is not None else np.nan
            ),
            "topography_resampling": "bilinear",
            "topography_target_finite_cell_count": int(np.isfinite(destination).sum()),
            "topography_target_finite_fraction": float(np.isfinite(destination).mean()),
        }
    return destination, metadata


def _merge_checked(
    current: np.ndarray,
    incoming: np.ndarray,
    *,
    tolerance_m: float,
    label: str,
) -> tuple[np.ndarray, int, float, int]:
    current = np.asarray(current, dtype=float)
    incoming = np.asarray(incoming, dtype=float)
    both = np.isfinite(current) & np.isfinite(incoming)
    difference = np.abs(current[both] - incoming[both])
    maximum = float(np.max(difference)) if difference.size else 0.0
    conflicts = int((difference > float(tolerance_m)).sum())
    added = ~np.isfinite(current) & np.isfinite(incoming)
    merged = current.copy()
    merged[added] = incoming[added]
    return merged, int(both.sum()), maximum, conflicts


def _interior_distance(mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Distance to a source-domain edge, with the array exterior fixed false."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    distance = distance_transform_edt(padded)[1:-1, 1:-1]
    return distance * float(pixel_size_m)


def _merge_by_quality(
    current: np.ndarray,
    current_quality: np.ndarray,
    incoming: np.ndarray,
    incoming_quality: np.ndarray,
    *,
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, int, float, int]:
    """Select one source per cell; never average overlapping source values."""
    current = np.asarray(current, dtype=float)
    incoming = np.asarray(incoming, dtype=float)
    current_quality = np.asarray(current_quality, dtype=float)
    incoming_quality = np.asarray(incoming_quality, dtype=float)
    both = np.isfinite(current) & np.isfinite(incoming)
    difference = np.abs(current[both] - incoming[both])
    maximum = float(np.max(difference)) if difference.size else 0.0
    conflicts = int((difference > float(tolerance_m)).sum())
    choose = np.isfinite(incoming) & (
        ~np.isfinite(current) | (incoming_quality > current_quality)
    )
    merged = current.copy()
    quality = current_quality.copy()
    merged[choose] = incoming[choose]
    quality[choose] = incoming_quality[choose]
    return merged, quality, int(both.sum()), maximum, conflicts


def normalize_overlap_resolution_policy(policy: str | None) -> str:
    """Return a supported canonical overlap policy name.

    The longer legacy name remains accepted so existing frozen configs keep
    their behavior. Both policies select exactly one source value per cell.
    """
    value = policy or INTERIOR_DISTANCE_POLICY
    if value == _LEGACY_INTERIOR_DISTANCE_POLICY:
        return INTERIOR_DISTANCE_POLICY
    if value not in {INTERIOR_DISTANCE_POLICY, MAIN_MEMBER_FIRST_POLICY}:
        raise ValueError(f"unsupported-mosaic-overlap-policy:{value}")
    return value


def _load_source(gdir_dir: Path, bed_scenario: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    grid = _grid_from_gdir(gdir_dir)
    thickness_var = _THICKNESS_VAR[bed_scenario]
    with xr.open_dataset(gdir_dir / "gridded_data.nc") as ds:
        topo = np.asarray(ds["topo"].values, dtype=float)
        mask = np.asarray(ds["glacier_mask"].values).astype(bool)
        if thickness_var not in ds:
            raise FileNotFoundError(
                f"missing-bed-variable:{thickness_var}:{gdir_dir.name}"
            )
        thickness = np.asarray(ds[thickness_var].values, dtype=float)
    invalid_topography = mask & ~np.isfinite(topo)
    invalid_thickness = mask & ~np.isfinite(thickness)
    if invalid_topography.any():
        raise ValueError(
            f"modern-mask-topography-nodata:{gdir_dir.name}:"
            f"cell_count={int(invalid_topography.sum())}"
        )
    if invalid_thickness.any():
        raise ValueError(
            f"modern-mask-thickness-nodata:{bed_scenario}:{gdir_dir.name}:"
            f"cell_count={int(invalid_thickness.sum())}"
        )
    bed = topo - np.where(mask, thickness, 0.0)
    modern_bed = np.where(mask, bed, np.nan)
    return grid, topo, modern_bed, mask


def mosaic_system_bed(
    member_dirs: list[Path],
    target: SystemGrid,
    bed_scenario: str,
    *,
    system_topography_m: np.ndarray | None = None,
    overlap_tolerance_m: float = 5.0,
    overlap_resolution_policy: str = INTERIOR_DISTANCE_POLICY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    overlap_resolution_policy = normalize_overlap_resolution_policy(
        overlap_resolution_policy
    )
    if system_topography_m is not None:
        system_topography_m = np.asarray(system_topography_m, dtype=float)
        if system_topography_m.shape != target.shape:
            raise ValueError("system-topography-shape-mismatch")
    topo_mosaic = np.full(target.shape, np.nan, dtype=float)
    bed_mosaic = np.full(target.shape, np.nan, dtype=float)
    topo_quality = np.full(target.shape, -np.inf, dtype=float)
    bed_quality = np.full(target.shape, -np.inf, dtype=float)
    modern_mask = np.zeros(target.shape, dtype=bool)
    overlap_count = 0
    maximum_topo_difference = 0.0
    maximum_bed_difference = 0.0
    topo_conflict_cells = 0
    bed_conflict_cells = 0
    for gdir_dir in member_dirs:
        grid, topo, modern_bed, mask = _load_source(gdir_dir, bed_scenario)
        source_crs = str(grid["proj"])
        topo_target = _reproject(
            topo, grid, target, source_crs, resampling=Resampling.bilinear
        )
        bed_target = _reproject(
            modern_bed, grid, target, source_crs, resampling=Resampling.bilinear
        )
        mask_target = _reproject(
            mask.astype(float), grid, target, source_crs, resampling=Resampling.nearest
        ) > 0.5
        source_valid = np.isfinite(topo)
        source_topo_quality = _interior_distance(
            source_valid, abs(float(grid["dxdy"][0]))
        )
        source_bed_quality = _interior_distance(
            mask, abs(float(grid["dxdy"][0]))
        )
        topo_quality_target = _reproject(
            source_topo_quality,
            grid,
            target,
            source_crs,
            resampling=Resampling.bilinear,
            source_nodata=None,
        )
        bed_quality_target = _reproject(
            source_bed_quality,
            grid,
            target,
            source_crs,
            resampling=Resampling.bilinear,
            source_nodata=None,
        )
        if overlap_resolution_policy == INTERIOR_DISTANCE_POLICY:
            topo_mosaic, topo_quality, count, maximum, conflicts = _merge_by_quality(
                topo_mosaic,
                topo_quality,
                topo_target,
                topo_quality_target,
                tolerance_m=overlap_tolerance_m,
            )
        else:
            topo_mosaic, count, maximum, conflicts = _merge_checked(
                topo_mosaic,
                topo_target,
                tolerance_m=overlap_tolerance_m,
                label="topography",
            )
        overlap_count += count
        maximum_topo_difference = max(maximum_topo_difference, maximum)
        topo_conflict_cells += conflicts
        if overlap_resolution_policy == INTERIOR_DISTANCE_POLICY:
            bed_mosaic, bed_quality, count, maximum, conflicts = _merge_by_quality(
                bed_mosaic,
                bed_quality,
                bed_target,
                bed_quality_target,
                tolerance_m=overlap_tolerance_m,
            )
        else:
            bed_mosaic, count, maximum, conflicts = _merge_checked(
                bed_mosaic,
                bed_target,
                tolerance_m=overlap_tolerance_m,
                label="bed",
            )
        overlap_count += count
        maximum_bed_difference = max(maximum_bed_difference, maximum)
        bed_conflict_cells += conflicts
        modern_mask |= mask_target
    missing_modern_bed = modern_mask & ~np.isfinite(bed_mosaic)
    if missing_modern_bed.any():
        raise ValueError(
            f"system-bed-mosaic-hole:{bed_scenario}:"
            f"cell_count={int(missing_modern_bed.sum())}"
        )
    authoritative_topography = (
        topo_mosaic if system_topography_m is None else system_topography_m
    )
    final_bed = np.where(modern_mask, bed_mosaic, authoritative_topography)
    raw_comparison = np.isfinite(topo_mosaic) & np.isfinite(authoritative_topography)
    delta = topo_mosaic[raw_comparison] - authoritative_topography[raw_comparison]
    absolute_delta = np.abs(delta)
    metadata = {
        "overlap_resolution_policy": overlap_resolution_policy,
        "overlap_resolution_semantics": "single-source-selection-no-average",
        "member_count": len(member_dirs),
        "overlap_cell_count": int(overlap_count),
        "maximum_topography_overlap_difference_m": maximum_topo_difference,
        "maximum_bed_overlap_difference_m": maximum_bed_difference,
        "topography_overlap_conflict_cells": int(topo_conflict_cells),
        "bed_overlap_conflict_cells": int(bed_conflict_cells),
        "finite_topography_fraction": float(
            np.isfinite(authoritative_topography).mean()
        ),
        "gdir_topography_mosaic_finite_fraction": float(
            np.isfinite(topo_mosaic).mean()
        ),
        "gdir_minus_system_dem_cell_count": int(delta.size),
        "gdir_minus_system_dem_median_m": (
            float(np.median(delta)) if delta.size else np.nan
        ),
        "gdir_minus_system_dem_mean_m": (
            float(np.mean(delta)) if delta.size else np.nan
        ),
        "gdir_minus_system_dem_rmse_m": (
            float(np.sqrt(np.mean(delta**2))) if delta.size else np.nan
        ),
        "gdir_minus_system_dem_p95_absolute_m": (
            float(np.percentile(absolute_delta, 95.0)) if delta.size else np.nan
        ),
        "gdir_minus_system_dem_maximum_absolute_m": (
            float(np.max(absolute_delta)) if delta.size else np.nan
        ),
        "gdir_minus_system_dem_over_tolerance_cell_count": int(
            (absolute_delta > float(overlap_tolerance_m)).sum()
        ),
        "finite_bed_fraction": float(np.isfinite(final_bed).mean()),
    }
    return final_bed, authoritative_topography, modern_mask, metadata


def build_system_grid_context(config: dict, rgi_id: str, lia_unique_id: int):
    """Build a complete system grid and all configured bed scenarios."""
    gdir_root = resolve(config["stage0_gdir_root"])
    lia = gpd.read_file(resolve(config["lia_geometry_path"])).set_index("UniqueID")
    if lia_unique_id not in lia.index:
        raise ValueError(f"missing-lia-geometry:{lia_unique_id}")
    main_dir = _gdir_path(gdir_root, rgi_id)
    main_grid_json = _grid_from_gdir(main_dir)
    target_crs = CRS.from_user_input(main_grid_json["proj"])
    geometry = gpd.GeoSeries(
        [lia.loc[lia_unique_id].geometry], crs=lia.crs
    ).to_crs(target_crs).iloc[0]
    components = normalize_lia_components(geometry)
    grid_config = config.get("lia_system_grid", {})
    overlap_resolution_policy = normalize_overlap_resolution_policy(
        grid_config.get("overlap_resolution_policy")
    )
    buffer_m = float(grid_config.get("buffer_m", 500.0))
    resolution_factor = grid_config.get("resolution_factor", 1)
    output_root = resolve(config["output_dir"])
    grid_output_subdir = str(
        grid_config.get("grid_output_subdir", "lia_system_grids_v1")
    )
    grid_path = output_root / grid_output_subdir / str(lia_unique_id) / "system_grid.json"
    target = _target_grid(
        main_grid_json,
        geometry,
        grid_path,
        buffer_m,
        resolution_factor=resolution_factor,
    )
    if "surface_dem_path" not in config:
        raise KeyError("missing-required-config:surface_dem_path")
    topography, topography_metadata = _load_system_topography(
        resolve(config["surface_dem_path"]), target
    )
    lia_mask = geometry_mask(
        [geometry.__geo_interface__],
        out_shape=target.shape,
        transform=target.transform,
        invert=True,
        all_touched=True,
    )
    missing_lia_topography = lia_mask & ~np.isfinite(topography)
    if missing_lia_topography.any():
        raise ValueError(
            "system-surface-dem-lia-coverage-hole:"
            f"cell_count={int(missing_lia_topography.sum())}"
        )
    topography_metadata.update({
        "topography_lia_cell_count": int(lia_mask.sum()),
        "topography_lia_finite_cell_count": int(
            np.isfinite(topography[lia_mask]).sum()
        ),
        "topography_lia_finite_fraction": 1.0,
        "system_grid_resolution_factor": int(resolution_factor),
        "system_grid_dx_m": float(target.dx),
    })
    manifest_spec = config.get("system_manifest_path")
    manifest = None
    if manifest_spec:
        manifest = pd.read_csv(resolve(manifest_spec)).set_index("lia_unique_id")
    ids = _member_ids(manifest, lia_unique_id, rgi_id)
    member_dirs = [_gdir_path(gdir_root, member_id) for member_id in ids]
    missing = [str(path) for path in member_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing-system-member-gdir:" + ";".join(missing)
        )
    beds: dict[str, GriddedBed] = {}
    mosaic_metadata_by_scenario: dict[str, dict] = {}
    bed_errors_by_scenario: dict[str, str] = {}
    modern_mask = None
    for scenario in config["bed_scenarios"]:
        try:
            bed, topo, mask, metadata = mosaic_system_bed(
                member_dirs,
                target,
                scenario,
                system_topography_m=topography,
                overlap_tolerance_m=float(grid_config.get("overlap_tolerance_m", 5.0)),
                overlap_resolution_policy=overlap_resolution_policy,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            bed_errors_by_scenario[scenario] = f"{type(exc).__name__}: {exc}"
            continue
        if not np.isfinite(topo[lia_mask]).all():
            bed_errors_by_scenario[scenario] = (
                "system-topography-lia-coverage-hole:"
                f"cell_count={int((~np.isfinite(topo) & lia_mask).sum())}"
            )
            continue
        if not np.isfinite(bed[lia_mask]).all():
            bed_errors_by_scenario[scenario] = (
                "system-bed-lia-coverage-hole:"
                f"cell_count={int((~np.isfinite(bed) & lia_mask).sum())}"
            )
            continue
        metadata["lia_topography_coverage_fraction"] = 1.0
        metadata["lia_bed_coverage_fraction"] = 1.0
        metadata.update(topography_metadata)
        if modern_mask is None:
            modern_mask = mask.copy()
        else:
            if not np.array_equal(modern_mask, mask):
                raise ValueError("system-member-mask-disagreement")
        mosaic_metadata_by_scenario[scenario] = metadata
        gradient_y, gradient_x = np.gradient(bed, target.dx, target.dx)
        slope = np.arctan(np.hypot(gradient_x, gradient_y))
        beds[scenario] = GriddedBed(
            bed_m=bed,
            glacier_mask=mask,
            slope=slope,
            map_dx_m=target.dx,
            bed_scenario=scenario,
        )
    if modern_mask is None:
        details = ";".join(
            f"{scenario}={reason}"
            for scenario, reason in sorted(bed_errors_by_scenario.items())
        )
        raise ValueError(f"system-grid-no-bed-scenarios:{details}")
    return SystemGridContext(
        rgi_id=rgi_id,
        lia_unique_id=int(lia_unique_id),
        grid=target,
        lia_geometry=geometry,
        component_geometries=components,
        member_rgi_ids=ids,
        topography_m=topography,
        topography_metadata=topography_metadata,
        bed_by_scenario=beds,
        modern_mask=modern_mask,
        mosaic_metadata_by_scenario=mosaic_metadata_by_scenario,
        bed_errors_by_scenario=bed_errors_by_scenario,
    )


__all__ = [
    "INTERIOR_DISTANCE_POLICY",
    "MAIN_MEMBER_FIRST_POLICY",
    "SystemGrid",
    "SystemGridContext",
    "build_system_grid_context",
    "_load_system_topography",
    "mosaic_system_bed",
    "normalize_overlap_resolution_policy",
    "normalize_lia_components",
]
