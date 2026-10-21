"""Build fixed-resolution LIA OGGM networks and run zero-MB diagnostics.

This is an intentionally isolated pilot.  It reads the frozen v0.1 system-grid
NetCDF products, creates one dynamic network per connected LIA component, and
never modifies the reconstruction products or modern glacier directories.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import tempfile
import time
import traceback
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
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import label as connected_component_labels
from scipy.spatial import cKDTree

from ..paths import resolve
from .flux_mass_budget import budget_rows_as_dicts, run_zero_mb_budget
from .lia_centerline_pilot import _prepare_oggm_lia_geometry
from .lia_dynamic_flowlines import (
    FlowlineNodeFields,
    build_variable_lambda_flowlines,
    derive_fixed_bed_trapezoid_geometry,
    project_thickness_to_trapezoid_domain,
)
from .lia_system_grid import normalize_lia_components


@dataclass(frozen=True)
class SourceState:
    path: Path
    rgi_id: str
    lia_unique_id: int
    crs: CRS
    transform: Affine
    x: np.ndarray
    y: np.ndarray
    thickness_m: np.ndarray
    surface_h_m: np.ndarray
    bed_h_m: np.ndarray
    valid_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.thickness_m.shape

    @property
    def cell_area_m2(self) -> float:
        return abs(float(self.transform.a * self.transform.e))


class _PilotDirectory:
    """Small GlacierDirectory protocol required by OGGM centerline tasks."""

    def __init__(self, directory: Path, rgi_id: str, grid, geometry, area_m2: float):
        self.dir = str(directory)
        self.rgi_id = rgi_id
        self.grid = grid
        self.rgi_area_m2 = float(area_m2)
        self.rgi_area_km2 = float(area_m2) / 1e6
        self.is_icecap = False
        self.is_tidewater = False
        self._pickles = {"geometries": geometry}
        self.diagnostics = {}

    def get_filepath(self, basename: str, filesuffix: str = "") -> str:
        if basename != "gridded_data":
            raise ValueError(f"unsupported-pilot-gdir-file:{basename}")
        return str(Path(self.dir) / "gridded_data.nc")

    def read_pickle(self, basename: str):
        return self._pickles[basename]

    def write_pickle(self, obj, basename: str, filesuffix: str = "") -> None:
        self._pickles[basename] = obj

    def add_to_diagnostics(self, key: str, value) -> None:
        self.diagnostics[key] = value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_source_state(path: Path, bed_scenario: str = "farinotti2019") -> SourceState:
    with xr.open_dataset(path) as dataset:
        member_ids = [str(value) for value in dataset["member_id"].values]
        if bed_scenario not in member_ids:
            raise ValueError(f"missing-system-grid-member:{bed_scenario}")
        member_index = member_ids.index(bed_scenario)
        x = np.asarray(dataset["x"].values, dtype=float)
        y = np.asarray(dataset["y"].values, dtype=float)
        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        transform = Affine(dx, 0.0, x[0] - 0.5 * dx, 0.0, dy, y[0] - 0.5 * dy)
        crs = CRS.from_wkt(str(dataset["spatial_ref"].attrs["crs_wkt"]))
        thickness = np.asarray(
            dataset["thickness_m"].isel(member=member_index).values, dtype=float
        )
        surface = np.asarray(
            dataset["surface_elevation_m"].isel(member=member_index).values,
            dtype=float,
        )
        bed = np.asarray(
            dataset["bed_elevation_m"].isel(member=member_index).values,
            dtype=float,
        )
        valid = (
            np.asarray(dataset["valid_mask"].isel(member=member_index).values) > 0
        ) & (np.asarray(dataset["lia_mask"].values) > 0)
        valid &= np.isfinite(thickness) & np.isfinite(surface) & np.isfinite(bed)
        return SourceState(
            path=path,
            rgi_id=str(dataset.attrs["rgi_id"]),
            lia_unique_id=int(dataset.attrs["lia_unique_id"]),
            crs=crs,
            transform=transform,
            x=x,
            y=y,
            thickness_m=thickness,
            surface_h_m=surface,
            bed_h_m=bed,
            valid_mask=valid,
        )


def fixed_grid_for_geometry(geometry, crs: CRS, dx_m: float, buffer_m: float):
    import salem

    minx, miny, maxx, maxy = geometry.bounds
    left = np.floor((minx - buffer_m) / dx_m) * dx_m
    right = np.ceil((maxx + buffer_m) / dx_m) * dx_m
    bottom = np.floor((miny - buffer_m) / dx_m) * dx_m
    top = np.ceil((maxy + buffer_m) / dx_m) * dx_m
    nx = int(round((right - left) / dx_m))
    ny = int(round((top - bottom) / dx_m))
    if nx < 5 or ny < 5:
        raise ValueError("dynamic-pilot-grid-too-small")
    return salem.Grid(
        proj=crs.to_proj4(),
        nxny=(nx, ny),
        dxdy=(float(dx_m), -float(dx_m)),
        x0y0=(left + 0.5 * dx_m, top - 0.5 * dx_m),
        pixel_ref="center",
    )


def normalize_resolution_cases(
    pilot: dict, *, default_flowline_dx: float
) -> list[dict]:
    """Return validated map-grid/flowline-spacing experiment cases."""

    raw_cases = pilot.get("resolution_cases")
    if raw_cases is None:
        raw_cases = [
            {
                "map_dx_m": value,
                "flowline_dx": default_flowline_dx,
                "run_dynamics": True,
            }
            for value in pilot["map_dx_m"]
        ]
    cases = []
    seen = set()
    for raw in raw_cases:
        map_dx_m = float(raw["map_dx_m"])
        flowline_dx = float(raw.get("flowline_dx", default_flowline_dx))
        run_dynamics = bool(raw.get("run_dynamics", True))
        if map_dx_m <= 0:
            raise ValueError("map-dx-must-be-positive")
        if flowline_dx <= 0:
            raise ValueError("flowline-dx-must-be-positive")
        key = (map_dx_m, flowline_dx, run_dynamics)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            {
                "map_dx_m": map_dx_m,
                "flowline_dx": flowline_dx,
                "flowline_node_dx_m": map_dx_m * flowline_dx,
                "run_dynamics": run_dynamics,
            }
        )
    if not cases:
        raise ValueError("dynamic-pilot-has-no-resolution-cases")
    return cases


def resolution_case_directory(
    map_dx_m: float, flowline_dx: float, *, run_dynamics: bool = True
) -> str:
    name = f"map_{float(map_dx_m):g}m_flowline_{float(flowline_dx):g}"
    return name if run_dynamics else name + "_mapping_only"


def grid_transform(grid) -> Affine:
    return Affine(
        float(grid.dx),
        0.0,
        float(grid.x0) - 0.5 * float(grid.dx),
        0.0,
        -abs(float(grid.dy)),
        float(grid.y0) + 0.5 * abs(float(grid.dy)),
    )


def reproject_source_array(
    source: SourceState,
    values: np.ndarray,
    grid,
    *,
    resampling: Resampling,
) -> np.ndarray:
    destination = np.full((grid.ny, grid.nx), np.nan, dtype=np.float64)
    reproject(
        source=np.asarray(values, dtype=np.float64),
        destination=destination,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=np.nan,
        dst_transform=grid_transform(grid),
        dst_crs=CRS.from_user_input(grid.proj.srs),
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return destination


def read_dem_on_grid(path: Path, grid) -> np.ndarray:
    output = np.full((grid.ny, grid.nx), np.nan, dtype=np.float64)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=output,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=grid_transform(grid),
            dst_crs=CRS.from_user_input(grid.proj.srs),
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    if not np.isfinite(output).all():
        raise ValueError(
            "dynamic-pilot-dem-coverage-hole:"
            f"cell_count={int((~np.isfinite(output)).sum())}"
        )
    return output


def prepare_centerline_gdir(
    *,
    directory: Path,
    rgi_id: str,
    component_geometry,
    grid,
    routing_topography_m: np.ndarray,
    raster_island_policy: dict | None = None,
):
    from oggm.core.centerlines import (
        catchment_area,
        compute_centerlines,
        compute_downstream_bedshape,
        compute_downstream_line,
        initialize_flowlines,
    )

    pixel_geometry, glacier_mask, exterior = _prepare_oggm_lia_geometry(
        component_geometry, grid
    )
    labels, component_count = connected_component_labels(
        glacier_mask, structure=np.ones((3, 3), dtype=int)
    )
    raster_cleanup = {
        "raster_island_exclusion_applied": False,
        "raster_island_excluded_component_count": 0,
        "raster_island_excluded_cell_count": 0,
        "raster_island_excluded_area_km2": 0.0,
        "raster_island_excluded_area_fraction": 0.0,
    }
    if component_count != 1:
        sizes = np.bincount(labels.ravel())[1:]
        largest_label = int(np.argmax(sizes)) + 1
        excluded_cells = int(sizes.sum() - sizes[largest_label - 1])
        excluded_area_km2 = float(
            excluded_cells * abs(float(grid.dx) * float(grid.dy)) / 1e6
        )
        excluded_fraction = float(excluded_cells / max(int(sizes.sum()), 1))
        policy = raster_island_policy or {}
        eligible = bool(
            policy
            and excluded_area_km2
            <= float(policy.get("maximum_area_km2", 0.0))
            and excluded_fraction
            <= float(policy.get("maximum_system_area_fraction", 0.0))
        )
        if not eligible:
            raise ValueError(
                "lia-component-disconnected-after-rasterization:"
                f"component_count={component_count}:"
                "cell_counts="
                + ",".join(map(str, sorted(sizes.tolist(), reverse=True)))
            )
        glacier_mask = labels == largest_label
        exterior = glacier_mask & ~binary_erosion(
            glacier_mask, structure=np.ones((3, 3), dtype=bool), border_value=0
        )
        raster_cleanup = {
            "raster_island_exclusion_applied": True,
            "raster_island_excluded_component_count": int(component_count - 1),
            "raster_island_excluded_cell_count": excluded_cells,
            "raster_island_excluded_area_km2": excluded_area_km2,
            "raster_island_excluded_area_fraction": excluded_fraction,
        }
    if not np.isfinite(routing_topography_m[glacier_mask]).all():
        raise ValueError("nonfinite-routing-topography-inside-lia-component")
    dataset = xr.Dataset(
        data_vars={
            "glacier_mask": (("y", "x"), glacier_mask.astype(np.int16)),
            "glacier_ext": (("y", "x"), exterior.astype(np.int16)),
            "topo_smoothed": (
                ("y", "x"), np.asarray(routing_topography_m, dtype=np.float32)
            ),
            "topo": (("y", "x"), np.asarray(routing_topography_m, dtype=np.float32)),
        },
        coords={"x": np.arange(grid.nx), "y": np.arange(grid.ny)},
    )
    dataset.to_netcdf(directory / "gridded_data.nc", engine="netcdf4")
    geometry = {
        "polygon_pix": pixel_geometry,
        "polygon_area": float(component_geometry.area),
    }
    gdir = _PilotDirectory(
        directory, rgi_id, grid, geometry, float(component_geometry.area)
    )
    gdir.raster_component_cleanup = raster_cleanup
    compute_centerlines.__wrapped__(gdir)
    initialize_flowlines.__wrapped__(gdir)
    catchment_area.__wrapped__(gdir)
    compute_downstream_line.__wrapped__(gdir)
    compute_downstream_bedshape.__wrapped__(gdir)
    return gdir, glacier_mask


def _source_component_mask(source: SourceState, component_geometry) -> np.ndarray:
    return geometry_mask(
        [component_geometry.__geo_interface__],
        out_shape=source.shape,
        transform=source.transform,
        invert=True,
        all_touched=True,
    )


def _catchment_labels_on_source(source: SourceState, gdir) -> tuple[np.ndarray, int]:
    catchments = gdir.read_pickle("geometries")["catchment_indices"]
    target = np.full((gdir.grid.ny, gdir.grid.nx), -1, dtype=np.int16)
    for index, cells in enumerate(catchments):
        target[tuple(np.asarray(cells, dtype=int).T)] = index
    labels = np.full(source.shape, -1, dtype=np.int16)
    reproject(
        source=target,
        destination=labels,
        src_transform=grid_transform(gdir.grid),
        src_crs=CRS.from_user_input(gdir.grid.proj.srs),
        src_nodata=-1,
        dst_transform=source.transform,
        dst_crs=source.crs,
        dst_nodata=-1,
        resampling=Resampling.nearest,
    )
    missing = labels < 0
    if missing.all():
        raise ValueError("catchment-label-reprojection-produced-no-labels")
    missing_count = int(missing.sum())
    _, nearest = distance_transform_edt(missing, return_indices=True)
    labels[missing] = labels[tuple(nearest[:, missing])]
    return labels, missing_count


def sample_source_cross_sections(
    source: SourceState,
    component_geometry,
    gdir,
    *,
    cross_section_contract: str = "legacy-projected-width",
):
    """Sample transverse 2-D sections at every target flowline node.

    Catchment labels prevent a normal transect from crossing into a neighbouring
    tributary.  One documented global section factor closes the discretised
    flowline volume to the source-grid volume; centerline thickness and measured
    surface width are never scaled.
    """

    centerlines = gdir.read_pickle("inversion_flowlines")
    labels, initially_unlabelled = _catchment_labels_on_source(source, gdir)
    component_mask = _source_component_mask(source, component_geometry)
    valid = source.valid_mask & component_mask
    valid_rows, valid_cols = np.nonzero(valid)
    if len(valid_rows) == 0:
        raise ValueError("lia-component-has-no-valid-source-cells")
    valid_x = source.x[valid_cols]
    valid_y = source.y[valid_rows]
    source_tree = cKDTree(np.column_stack([valid_x, valid_y]))
    sample_step = min(abs(float(source.transform.a)), abs(float(source.transform.e)))
    bounds = component_geometry.bounds
    half_length = 0.6 * float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    offsets = np.arange(-half_length, half_length + 0.5 * sample_step, sample_step)
    inverse = ~source.transform
    raw_fields = []
    audits = []
    raw_flowline_volume = 0.0
    represented_area = 0.0
    for branch_index, centerline in enumerate(centerlines):
        coordinates = np.asarray(centerline.line.coords, dtype=float)
        station_x = float(gdir.grid.x0) + coordinates[:, 0] * float(gdir.grid.dx)
        station_y = float(gdir.grid.y0) + coordinates[:, 1] * float(gdir.grid.dy)
        station_points = np.column_stack([station_x, station_y])
        sample_distance, source_index = source_tree.query(station_points, k=1)
        center_rows = valid_rows[source_index]
        center_cols = valid_cols[source_index]
        reference_surface = source.surface_h_m[center_rows, center_cols]
        reference_bed = source.bed_h_m[center_rows, center_cols]
        center_thickness = source.thickness_m[center_rows, center_cols]
        if cross_section_contract == "fixed-bed-section" and not np.allclose(
            reference_surface - reference_bed,
            center_thickness,
            rtol=0.0,
            atol=1e-3,
        ):
            raise ValueError(
                f"fixed-bed-source-surface-thickness-mismatch:{branch_index}"
            )

        tangent_x = np.gradient(station_x)
        tangent_y = np.gradient(station_y)
        norm = np.hypot(tangent_x, tangent_y)
        if np.any(norm <= 0):
            raise ValueError(f"zero-flowline-tangent:{branch_index}")
        normal_x = -tangent_y / norm
        normal_y = tangent_x / norm
        widths = np.zeros(centerline.nx, dtype=float)
        sections = np.zeros(centerline.nx, dtype=float)
        thickness = np.zeros(centerline.nx, dtype=float)
        bed = np.zeros(centerline.nx, dtype=float)
        representative_offsets = np.zeros(centerline.nx, dtype=float)
        catchment_cut_side_count = np.zeros(centerline.nx, dtype=np.int8)
        for node_index in range(centerline.nx):
            xs = station_x[node_index] + offsets * normal_x[node_index]
            ys = station_y[node_index] + offsets * normal_y[node_index]
            edge_cols, edge_rows = inverse * (xs, ys)
            rr = np.floor(edge_rows).astype(int)
            cc = np.floor(edge_cols).astype(int)
            inside = (
                (rr >= 0)
                & (rr < source.shape[0])
                & (cc >= 0)
                & (cc < source.shape[1])
            )
            eligible = np.zeros(len(offsets), dtype=bool)
            valid_indices = np.flatnonzero(inside)
            eligible[valid_indices] = (
                valid[rr[valid_indices], cc[valid_indices]]
                & (labels[rr[valid_indices], cc[valid_indices]] == branch_index)
            )
            candidates = np.flatnonzero(eligible)
            if candidates.size == 0:
                raise ValueError(
                    f"cross-section-has-no-catchment-cells:{branch_index}:{node_index}"
                )
            center_sample = int(candidates[np.argmin(np.abs(offsets[candidates]))])
            start = center_sample
            stop = center_sample
            while start > 0 and eligible[start - 1]:
                start -= 1
            while stop + 1 < len(eligible) and eligible[stop + 1]:
                stop += 1
            selected = np.arange(start, stop + 1)
            cut_sides = 0
            for neighbour in (start - 1, stop + 1):
                if neighbour < 0 or neighbour >= len(offsets) or not inside[neighbour]:
                    continue
                neighbour_row = rr[neighbour]
                neighbour_col = cc[neighbour]
                if (
                    valid[neighbour_row, neighbour_col]
                    and labels[neighbour_row, neighbour_col] != branch_index
                ):
                    cut_sides += 1
            catchment_cut_side_count[node_index] = cut_sides
            values = source.thickness_m[rr[selected], cc[selected]]
            widths[node_index] = float(len(selected) * sample_step)
            sections[node_index] = float(np.sum(values) * sample_step)
            representative = int(selected[np.argmax(values)])
            thickness[node_index] = float(
                source.thickness_m[rr[representative], cc[representative]]
            )
            bed[node_index] = float(
                source.bed_h_m[rr[representative], cc[representative]]
            )
            representative_offsets[node_index] = abs(float(offsets[representative]))

        if cross_section_contract == "fixed-bed-section":
            selected_thickness = center_thickness
            positive = selected_thickness > 1e-8
        elif cross_section_contract == "legacy-projected-width":
            selected_thickness = thickness
            positive = selected_thickness > 0.1
        else:
            raise ValueError(
                f"unsupported-cross-section-sampling-contract:"
                f"{cross_section_contract}"
            )
        sections[~positive] = 0.0
        if not np.any(positive):
            raise ValueError(f"flowline-has-no-positive-source-thickness:{branch_index}")
        dx_meter = float(centerline.dx) * float(gdir.grid.dx)
        raw_flowline_volume += float(np.sum(sections) * dx_meter)
        represented_area += float(np.sum(widths[positive]) * dx_meter)
        raw_fields.append(
            (
                reference_surface,
                reference_bed,
                selected_thickness,
                sections,
                widths,
                thickness,
                representative_offsets,
                catchment_cut_side_count,
                station_x,
                station_y,
            )
        )
        raw_ratio = sections[positive] / (
            widths[positive] * selected_thickness[positive]
        )
        audits.append(
            {
                "branch_index": branch_index,
                "node_count": int(centerline.nx),
                "positive_node_count": int(positive.sum()),
                "maximum_station_sample_distance_m": float(np.max(sample_distance)),
                "representative_offset_median": float(
                    np.median(representative_offsets[positive])
                ),
                "representative_offset_maximum": float(
                    np.max(representative_offsets[positive])
                ),
                "raw_shape_factor_min": float(np.min(raw_ratio)),
                "raw_shape_factor_median": float(np.median(raw_ratio)),
                "raw_shape_factor_max": float(np.max(raw_ratio)),
                "raw_shape_factor_below_0p5_count": int((raw_ratio <= 0.5).sum()),
                "raw_shape_factor_above_1_count": int((raw_ratio > 1.0).sum()),
            }
        )

    cell_area = float(source.cell_area_m2)
    source_area = float(valid.sum() * cell_area)
    source_volume = float(np.sum(source.thickness_m[valid]) * cell_area)
    if raw_flowline_volume <= 0:
        raise ValueError("cross-section-quadrature-produced-zero-volume")
    quadrature_scale_factor = source_volume / raw_flowline_volume
    section_scale_factor = (
        1.0
        if cross_section_contract == "fixed-bed-section"
        else quadrature_scale_factor
    )
    node_fields = []
    projection_rows = []
    fixed_bed_node_rows = []
    for branch_index, (
        reference_surface,
        reference_bed,
        thickness,
        section,
        width,
        transverse_maximum_thickness,
        representative_offsets,
        catchment_cut_side_count,
        station_x,
        station_y,
    ) in enumerate(raw_fields):
        scaled_section = section * section_scale_factor
        if cross_section_contract == "fixed-bed-section":
            geometry = derive_fixed_bed_trapezoid_geometry(
                thickness, scaled_section, width
            )
            effective_thickness = thickness
            effective_bed = reference_bed
            effective_surface = effective_bed + effective_thickness
            adjusted = np.zeros_like(thickness, dtype=bool)
        else:
            projection = project_thickness_to_trapezoid_domain(
                thickness,
                scaled_section,
                width,
                minimum_bottom_width_m=10.0,
            )
            effective_thickness = projection.effective_thickness_m
            effective_bed = reference_surface - effective_thickness
            effective_surface = reference_surface
            adjusted = projection.adjusted
        positive = scaled_section > 0
        adjustment = effective_thickness - thickness
        row = {
                "branch_index": branch_index,
                "adjusted_node_count": int(adjusted.sum()),
                "adjusted_positive_node_fraction": (
                    float(adjusted[positive].mean())
                    if np.any(positive)
                    else 0.0
                ),
                "thickness_adjustment_median_m": (
                    float(np.median(adjustment[positive]))
                    if np.any(positive)
                    else 0.0
                ),
                "thickness_adjustment_maximum_absolute_m": (
                    float(np.max(np.abs(adjustment[positive])))
                    if np.any(positive)
                    else 0.0
                ),
                "equivalent_bed_adjustment_median_m": (
                    float(np.median((effective_bed - reference_bed)[positive]))
                    if np.any(positive)
                    else 0.0
                ),
                "equivalent_bed_adjustment_maximum_absolute_m": (
                    float(
                        np.max(
                            np.abs((effective_bed - reference_bed)[positive])
                        )
                    )
                    if np.any(positive)
                    else 0.0
                ),
                "cross_section_contract": cross_section_contract,
                "source_surface_closure_adjustment_maximum_absolute_m": float(
                    np.max(np.abs(effective_surface - reference_surface))
                ),
            }
        if cross_section_contract == "fixed-bed-section":
            positive_geometry = geometry.positive_ice
            row.update(
                {
                    "fixed_bed_low_shape_factor_count": int(
                        (geometry.regime_code == 1).sum()
                    ),
                    "fixed_bed_exact_width_count": int(
                        (geometry.regime_code == 2).sum()
                    ),
                    "fixed_bed_high_shape_factor_count": int(
                        (geometry.regime_code == 3).sum()
                    ),
                    "fixed_bed_f_over_1p5_count": int(
                        np.sum(geometry.area_shape_factor[positive_geometry] > 1.5)
                    ),
                    "fixed_bed_effective_width_error_mean_fraction": float(
                        np.mean(
                            geometry.width_error_fraction[positive_geometry]
                        )
                    ),
                    "fixed_bed_effective_width_error_maximum_absolute_fraction": float(
                        np.max(
                            np.abs(
                                geometry.width_error_fraction[positive_geometry]
                            )
                        )
                    ),
                    "fixed_bed_lambda_maximum": float(
                        np.max(geometry.lambdas[positive_geometry])
                    ),
                }
            )
            dx_meter = (
                float(centerlines[branch_index].dx) * float(gdir.grid.dx)
            )
            for node_index in np.flatnonzero(positive_geometry):
                fixed_bed_node_rows.append(
                    {
                        "branch_index": int(branch_index),
                        "node_index": int(node_index),
                        "dx_meter": float(dx_meter),
                        "surface_h_m": float(effective_surface[node_index]),
                        "bed_h_m": float(reference_bed[node_index]),
                        "thickness_m": float(thickness[node_index]),
                        "section_m2": float(scaled_section[node_index]),
                        "observed_width_m": float(width[node_index]),
                        "effective_width_m": float(
                            geometry.effective_surface_width_m[node_index]
                        ),
                        "width_error_fraction": float(
                            geometry.width_error_fraction[node_index]
                        ),
                        "bottom_width_m": float(
                            geometry.bottom_width_m[node_index]
                        ),
                        "lambda": float(geometry.lambdas[node_index]),
                        "shape_factor": float(
                            geometry.area_shape_factor[node_index]
                        ),
                        "regime_code": int(geometry.regime_code[node_index]),
                        "cross_section_mean_thickness_m": float(
                            scaled_section[node_index] / width[node_index]
                        ),
                        "cross_section_maximum_thickness_m": float(
                            transverse_maximum_thickness[node_index]
                        ),
                        "maximum_thickness_offset_m": float(
                            representative_offsets[node_index]
                        ),
                        "catchment_cut_side_count": int(
                            catchment_cut_side_count[node_index]
                        ),
                        "station_x_m": float(station_x[node_index]),
                        "station_y_m": float(station_y[node_index]),
                    }
                )
        projection_rows.append(row)
        node_fields.append(
            FlowlineNodeFields(
                surface_h_m=effective_surface,
                bed_h_m=effective_bed,
                section_m2=scaled_section,
                surface_width_m=width,
            )
        )
    return node_fields, {
        "branches": audits,
        "trapezoid_projection": projection_rows,
        "source_cell_count": int(valid.sum()),
        "source_area_m2": source_area,
        "source_volume_m3": source_volume,
        "raw_flowline_area_m2": represented_area,
        "raw_flowline_volume_m3": raw_flowline_volume,
        "section_scale_factor": float(section_scale_factor),
        "quadrature_scale_factor_not_applied": float(quadrature_scale_factor),
        "cross_section_contract": cross_section_contract,
        "fixed_bed_node_rows": fixed_bed_node_rows,
        "raw_area_error_fraction": (represented_area - source_area) / source_area,
        "raw_volume_error_fraction": (
            raw_flowline_volume - source_volume
        ) / source_volume,
        "initially_unlabelled_source_grid_cell_count": initially_unlabelled,
    }


def _trailing_zero_distance_m(flowline) -> float:
    positive = np.flatnonzero(np.asarray(flowline.thick) > 0.0)
    if positive.size == 0:
        return float(flowline.nx * flowline.dx_meter)
    return float((flowline.nx - positive[-1] - 1) * flowline.dx_meter)


def _run_dynamic_checkpoints(
    flowlines, checkpoints, glen_a, fs, *, smooth_trib_influx: bool
):
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    model = FluxBasedModel(
        flowlines,
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(glen_a),
        fs=float(fs),
        smooth_trib_influx=bool(smooth_trib_influx),
        check_for_boundaries=True,
    )
    initial_volume = float(model.volume_m3)
    previous = 0.0
    checkpoint_rows = []
    step_rows = []
    budget_names = (
        "actual_volume_change_m3",
        "upstream_gate_input_m3",
        "root_domain_outflow_m3",
        "negative_tributary_creation_m3",
        "negative_section_clip_creation_m3",
        "predicted_volume_change_m3",
        "closure_residual_m3",
    )
    cumulative = {name: 0.0 for name in budget_names}
    cumulative_step_count = 0
    minimum_timestep = np.inf
    for checkpoint in sorted(float(value) for value in checkpoints):
        rows, totals = run_zero_mb_budget(model, checkpoint - previous)
        step_rows.extend(budget_rows_as_dicts(rows))
        for name in budget_names:
            cumulative[name] += float(totals[name])
        cumulative_step_count += int(totals["step_count"])
        minimum_timestep = min(
            minimum_timestep, float(totals["minimum_timestep_seconds"])
        )
        checkpoint_rows.append(
            {
                "year": checkpoint,
                "volume_m3": float(model.volume_m3),
                "area_m2": float(model.area_m2),
                "length_m": float(model.length_m),
                "volume_change_fraction_from_initial": (
                    (float(model.volume_m3) - initial_volume) / initial_volume
                    if initial_volume > 0
                    else None
                ),
                **cumulative,
                "step_count": cumulative_step_count,
                "minimum_timestep_seconds": minimum_timestep,
                **{
                    f"interval_{name}": totals[name] for name in budget_names
                },
            }
        )
        previous = checkpoint
    return checkpoint_rows, step_rows


def run_case(
    *,
    source: SourceState,
    component_geometry,
    component_index: int,
    map_dx_m: float,
    flowline_dx: float,
    routing_surface: str,
    dem_path: Path,
    buffer_m: float,
    output_dir: Path,
    checkpoints,
    glen_a: float,
    fs: float,
    smooth_trib_influx: bool,
    run_dynamics: bool,
    cross_section_contract: str = "legacy-projected-width",
) -> dict:
    from oggm import cfg

    started = time.perf_counter()
    cfg.PARAMS["flowline_dx"] = float(flowline_dx)
    grid = fixed_grid_for_geometry(
        component_geometry, source.crs, float(map_dx_m), float(buffer_m)
    )
    dem = read_dem_on_grid(dem_path, grid)
    if routing_surface == "modern-dem":
        routing = dem
    elif routing_surface == "lia-surface":
        lia_surface = reproject_source_array(
            source, source.surface_h_m, grid, resampling=Resampling.bilinear
        )
        component_mask = geometry_mask(
            [component_geometry.__geo_interface__],
            out_shape=(grid.ny, grid.nx),
            transform=grid_transform(grid),
            invert=True,
            all_touched=True,
        )
        # Bilinear reprojection can leave a one-cell NaN fringe because the
        # frozen source product is masked outside the LIA outline.  At the
        # zero-thickness margin the regional DEM is the correct limiting value.
        routing = np.where(
            component_mask & np.isfinite(lia_surface), lia_surface, dem
        )
    elif routing_surface == "lia-bed":
        lia_bed = reproject_source_array(
            source, source.bed_h_m, grid, resampling=Resampling.bilinear
        )
        component_mask = geometry_mask(
            [component_geometry.__geo_interface__],
            out_shape=(grid.ny, grid.nx),
            transform=grid_transform(grid),
            invert=True,
            all_touched=True,
        )
        routing = np.where(component_mask & np.isfinite(lia_bed), lia_bed, dem)
    else:
        raise ValueError(f"unsupported-routing-surface:{routing_surface}")

    case_dir = (
        output_dir
        / str(source.lia_unique_id)
        / f"component_{component_index}"
        / resolution_case_directory(
            map_dx_m, flowline_dx, run_dynamics=run_dynamics
        )
        / routing_surface
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    geometry_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gdir_", dir=case_dir) as temporary:
        gdir, _ = prepare_centerline_gdir(
            directory=Path(temporary),
            rgi_id=f"LIA-{source.lia_unique_id}-C{component_index}",
            component_geometry=component_geometry,
            grid=grid,
            routing_topography_m=routing,
        )
        geometry_seconds = time.perf_counter() - geometry_started
        mapping_started = time.perf_counter()
        node_fields, cross_section_audit = sample_source_cross_sections(
            source,
            component_geometry,
            gdir,
            cross_section_contract=cross_section_contract,
        )
        build_contract = (
            "fixed-bed-section"
            if cross_section_contract == "fixed-bed-section"
            else "legacy-exact-width"
        )
        flowlines, build_audit = build_variable_lambda_flowlines(
            gdir.read_pickle("inversion_flowlines"),
            node_fields,
            map_dx_m=float(map_dx_m),
            downstream_line=gdir.read_pickle("downstream_line"),
            default_lambda=2.0,
            rgi_id=f"LIA-{source.lia_unique_id}-C{component_index}",
            smooth_trib_influx=bool(smooth_trib_influx),
            cross_section_contract=build_contract,
        )
        mapping_seconds = time.perf_counter() - mapping_started
    with gzip.open(case_dir / "model_flowlines_lia.pkl.gz", "wb") as stream:
        pickle.dump(flowlines, stream)

    flowline_area = float(sum(fl.area_m2 for fl in flowlines))
    flowline_volume = float(sum(fl.volume_m3 for fl in flowlines))
    source_area = float(cross_section_audit["source_area_m2"])
    source_volume = float(cross_section_audit["source_volume_m3"])
    checkpoints_out = []
    step_rows = []
    dynamics_seconds = 0.0
    if run_dynamics:
        dynamics_started = time.perf_counter()
        checkpoints_out, step_rows = _run_dynamic_checkpoints(
            flowlines,
            checkpoints,
            glen_a,
            fs,
            smooth_trib_influx=bool(smooth_trib_influx),
        )
        dynamics_seconds = time.perf_counter() - dynamics_started
    audit = {
        "lia_unique_id": source.lia_unique_id,
        "rgi_id": source.rgi_id,
        "component_index": component_index,
        "map_dx_m": float(map_dx_m),
        "flowline_dx": float(flowline_dx),
        "flowline_node_dx_m": float(map_dx_m) * float(flowline_dx),
        "run_dynamics": bool(run_dynamics),
        "cross_section_contract": cross_section_contract,
        "routing_surface": routing_surface,
        "grid_nx": int(grid.nx),
        "grid_ny": int(grid.ny),
        "buffer_m": float(buffer_m),
        "cross_section_mapping": cross_section_audit,
        "flowline_build": build_audit,
        "flowline_area_m2": flowline_area,
        "flowline_volume_m3": flowline_volume,
        "area_error_fraction": (flowline_area - source_area) / source_area,
        "volume_error_fraction": (flowline_volume - source_volume) / source_volume,
        "downstream_free_distance_m": _trailing_zero_distance_m(flowlines[-1]),
        "zero_mb_checkpoints": checkpoints_out,
        "geometry_preprocessing_seconds": float(geometry_seconds),
        "cross_section_mapping_seconds": float(mapping_seconds),
        "zero_mb_dynamics_seconds": float(dynamics_seconds),
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    write_json(case_dir / "case_audit.json", _json_safe(audit))
    if step_rows:
        pd.DataFrame(step_rows).to_csv(
            case_dir / "zero_mb_budget_steps.csv", index=False
        )
    return audit


def result_row_from_audit(audit: dict) -> dict:
    checkpoints = audit.get("zero_mb_checkpoints", [])
    checkpoint = checkpoints[-1] if checkpoints else None
    return {
        "lia_unique_id": int(audit["lia_unique_id"]),
        "rgi_id": str(audit["rgi_id"]),
        "component_index": int(audit["component_index"]),
        "map_dx_m": float(audit["map_dx_m"]),
        "flowline_dx": float(audit["flowline_dx"]),
        "flowline_node_dx_m": float(audit["flowline_node_dx_m"]),
        "run_dynamics": bool(audit["run_dynamics"]),
        "routing_surface": str(audit["routing_surface"]),
        "status": "ok",
        "flowline_count": audit["flowline_build"]["network"]["flowline_count"],
        "junction_count": audit["flowline_build"]["network"]["junction_count"],
        "negative_junction_surface_count": audit["flowline_build"]["network"][
            "negative_junction_surface_count"
        ],
        "area_error_fraction": audit["area_error_fraction"],
        "volume_error_fraction": audit["volume_error_fraction"],
        "downstream_free_distance_m": audit["downstream_free_distance_m"],
        "final_zero_mb_year": checkpoint["year"] if checkpoint else None,
        "final_zero_mb_volume_change_fraction": (
            checkpoint["volume_change_fraction_from_initial"]
            if checkpoint
            else None
        ),
        "negative_tributary_creation_m3": (
            checkpoint["negative_tributary_creation_m3"] if checkpoint else None
        ),
        "negative_section_clip_creation_m3": (
            checkpoint["negative_section_clip_creation_m3"]
            if checkpoint
            else None
        ),
        "closure_residual_m3": (
            checkpoint["closure_residual_m3"] if checkpoint else None
        ),
        "minimum_timestep_seconds": (
            checkpoint["minimum_timestep_seconds"] if checkpoint else None
        ),
        "geometry_preprocessing_seconds": audit.get(
            "geometry_preprocessing_seconds"
        ),
        "cross_section_mapping_seconds": audit.get(
            "cross_section_mapping_seconds"
        ),
        "zero_mb_dynamics_seconds": audit.get("zero_mb_dynamics_seconds"),
        "wall_time_seconds": audit["wall_time_seconds"],
    }


def write_matrix_progress(output_dir: Path, results: list[dict]) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "lia_dynamic_matrix_results.csv", index=False)
    failures = frame[frame["status"].eq("failed")] if not frame.empty else frame
    if not failures.empty:
        failures.to_csv(output_dir / "lia_dynamic_matrix_failures.csv", index=False)


def _configure_oggm(config: dict) -> str:
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    requested = config.get("lia_centerlines", {}).get("oggm_params", {})
    for key, value in requested.items():
        if key in cfg.PARAMS:
            cfg.PARAMS[key] = value
    cfg.PARAMS["downstream_line_shape"] = "trapezoidal"
    cfg.PARAMS["use_kcalving_for_run"] = False
    return str(oggm.__version__)


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    oggm_version = _configure_oggm(config)
    pilot = config["lia_dynamic_pilot"]
    source_dir = resolve(config["output_dir"]) / str(
        config["lia_3d_system_grid_pilot_v1"]["output_subdir"]
    )
    output_dir = resolve(config["output_dir"]) / str(pilot["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    lia_by_id = lia.set_index("UniqueID")
    default_flowline_dx = float(
        config.get("lia_centerlines", {})
        .get("oggm_params", {})
        .get("flowline_dx", 2.0)
    )
    resolution_cases = normalize_resolution_cases(
        pilot, default_flowline_dx=default_flowline_dx
    )
    results = []
    failures = []
    for unique_id in [int(value) for value in pilot["system_ids"]]:
        candidates = sorted(source_dir.glob(f"lia_{unique_id}_*_system_grid_v1.nc"))
        if len(candidates) != 1:
            raise ValueError(
                f"system-grid-file-count:{unique_id}:{len(candidates)}"
            )
        source = load_source_state(candidates[0], str(pilot["bed_scenario"]))
        geometry = gpd.GeoSeries(
            [lia_by_id.loc[unique_id].geometry], crs=lia.crs
        ).to_crs(source.crs).iloc[0]
        components = normalize_lia_components(geometry)
        for component_index, component in enumerate(components):
            for resolution_case in resolution_cases:
                map_dx_m = float(resolution_case["map_dx_m"])
                flowline_dx = float(resolution_case["flowline_dx"])
                run_dynamics = bool(resolution_case["run_dynamics"])
                for routing_surface in pilot["routing_surfaces"]:
                    key = {
                        "lia_unique_id": unique_id,
                        "rgi_id": source.rgi_id,
                        "component_index": component_index,
                        "map_dx_m": float(map_dx_m),
                        "flowline_dx": flowline_dx,
                        "flowline_node_dx_m": map_dx_m * flowline_dx,
                        "run_dynamics": run_dynamics,
                        "routing_surface": str(routing_surface),
                    }
                    try:
                        case_dir = (
                            output_dir
                            / str(source.lia_unique_id)
                            / f"component_{component_index}"
                            / resolution_case_directory(
                                map_dx_m,
                                flowline_dx,
                                run_dynamics=run_dynamics,
                            )
                            / str(routing_surface)
                        )
                        audit_path = case_dir / "case_audit.json"
                        requested_year = (
                            max(
                                float(value)
                                for value in pilot["zero_mb_checkpoints_years"]
                            )
                            if run_dynamics
                            else None
                        )
                        audit = None
                        if audit_path.exists():
                            cached = json.loads(audit_path.read_text(encoding="utf-8"))
                            cached_matches = (
                                float(cached.get("flowline_dx", -1.0))
                                == flowline_dx
                                and bool(cached.get("run_dynamics", True))
                                == run_dynamics
                            )
                            cached_checkpoints = cached.get(
                                "zero_mb_checkpoints", []
                            )
                            cached_year = (
                                float(cached_checkpoints[-1]["year"])
                                if cached_checkpoints
                                else None
                            )
                            if cached_matches and (
                                not run_dynamics
                                or (
                                    cached_year is not None
                                    and cached_year >= requested_year
                                )
                            ):
                                audit = cached
                        if audit is None:
                            audit = run_case(
                                source=source,
                                component_geometry=component,
                                component_index=component_index,
                                map_dx_m=float(map_dx_m),
                                flowline_dx=flowline_dx,
                                routing_surface=str(routing_surface),
                                dem_path=resolve(config["surface_dem_path"]),
                                buffer_m=float(pilot["buffer_m"]),
                                output_dir=output_dir,
                                checkpoints=pilot["zero_mb_checkpoints_years"],
                                glen_a=float(pilot["glen_a"]),
                                fs=float(pilot["fs"]),
                                smooth_trib_influx=bool(
                                    pilot.get("smooth_trib_influx", True)
                                ),
                                run_dynamics=run_dynamics,
                            )
                        results.append(result_row_from_audit(audit))
                    except Exception as exc:
                        failure = {
                            **key,
                            "status": "failed",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "failure_traceback": traceback.format_exc(limit=12),
                        }
                        failures.append(failure)
                        results.append(failure)
                    write_matrix_progress(output_dir, results)
    frame = pd.DataFrame(results)
    ok = frame["status"].eq("ok") if not frame.empty else pd.Series(dtype=bool)
    summary = {
        "status": "completed" if bool(ok.all()) else "completed-with-failures",
        "oggm_version": oggm_version,
        "case_count": int(len(frame)),
        "ok_count": int(ok.sum()),
        "failure_count": int((~ok).sum()),
        "system_ids": [int(value) for value in pilot["system_ids"]],
        "resolution_cases": resolution_cases,
        "routing_surfaces": [str(value) for value in pilot["routing_surfaces"]],
        "buffer_m": float(pilot["buffer_m"]),
        "zero_mb_checkpoints_years": [
            float(value) for value in pilot["zero_mb_checkpoints_years"]
        ],
    }
    write_json(output_dir / "lia_dynamic_matrix_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument("--maximum-years", type=float)
    parser.add_argument("--map-dx", action="append", type=float)
    parser.add_argument(
        "--resolution-case",
        action="append",
        help="Repeatable MAP_DX:FLOWLINE_DX pair, for example 100:1.",
    )
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--routing-surface", action="append")
    parser.add_argument("--output-subdir")
    args = parser.parse_args()
    config = copy.deepcopy(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    pilot = config["lia_dynamic_pilot"]
    if args.system_id:
        pilot["system_ids"] = list(dict.fromkeys(args.system_id))
    if args.maximum_years is not None:
        maximum = float(args.maximum_years)
        if maximum <= 0:
            raise ValueError("maximum-years-must-be-positive")
        checkpoints = [
            float(value)
            for value in pilot["zero_mb_checkpoints_years"]
            if float(value) <= maximum
        ]
        if maximum not in checkpoints:
            checkpoints.append(maximum)
        pilot["zero_mb_checkpoints_years"] = sorted(set(checkpoints))
    if args.map_dx:
        if any(value <= 0 for value in args.map_dx):
            raise ValueError("map-dx-must-be-positive")
        pilot["map_dx_m"] = list(dict.fromkeys(args.map_dx))
        pilot.pop("resolution_cases", None)
    if args.resolution_case:
        cases = []
        for value in args.resolution_case:
            try:
                map_dx, flowline_dx = (float(item) for item in value.split(":"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid-resolution-case:{value}:expected-MAP_DX:FLOWLINE_DX"
                ) from exc
            cases.append(
                {
                    "map_dx_m": map_dx,
                    "flowline_dx": flowline_dx,
                    "run_dynamics": not args.mapping_only,
                }
            )
        pilot["resolution_cases"] = cases
    elif args.mapping_only:
        default_flowline_dx = float(
            config.get("lia_centerlines", {})
            .get("oggm_params", {})
            .get("flowline_dx", 2.0)
        )
        cases = normalize_resolution_cases(
            pilot, default_flowline_dx=default_flowline_dx
        )
        pilot["resolution_cases"] = [
            {**case, "run_dynamics": False} for case in cases
        ]
    if args.routing_surface:
        unsupported = set(args.routing_surface) - {"modern-dem", "lia-surface"}
        if unsupported:
            raise ValueError(
                "unsupported-routing-surfaces:" + ",".join(sorted(unsupported))
            )
        pilot["routing_surfaces"] = list(dict.fromkeys(args.routing_surface))
    if args.output_subdir:
        pilot["output_subdir"] = str(args.output_subdir)
    _, summary = run(config)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
