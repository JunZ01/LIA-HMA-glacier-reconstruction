"""Generate flat-folder v3 thickness inspection maps and structural QC tables."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import patheffects
from pyproj import CRS, Transformer
from scipy.ndimage import binary_erosion
from shapely import wkb
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import transform as transform_geometry

from ..paths import resolve
from .lia_system_grid import normalize_lia_components


MATERIAL_NEGATIVE_TOLERANCE_M = -1e-6
AREA_CLOSURE_TOLERANCE_FRACTION = 0.02
THICKNESS_REVIEW_M = 800.0
THICKNESS_HIGH_REVIEW_M = 1000.0
SURFACE_SLOPE_P99_REVIEW_DEG = 60.0
BED_OVERLAP_REVIEW_M = 100.0


def _dataset_crs(dataset: xr.Dataset) -> CRS:
    attrs = dataset["spatial_ref"].attrs
    value = attrs.get("spatial_ref") or attrs.get("crs_wkt")
    if not value:
        raise ValueError("NetCDF spatial_ref has no CRS WKT")
    return CRS.from_user_input(value)


def _pairwise_gradient_values(
    values: np.ndarray,
    valid: np.ndarray,
    dx_m: float,
) -> np.ndarray:
    """Return absolute neighbour gradients where both cells are valid."""
    horizontal = np.abs(np.diff(values, axis=1)) / dx_m
    horizontal = horizontal[valid[:, 1:] & valid[:, :-1]]
    vertical = np.abs(np.diff(values, axis=0)) / dx_m
    vertical = vertical[valid[1:, :] & valid[:-1, :]]
    output = np.concatenate([horizontal, vertical])
    return output[np.isfinite(output)]


def thickness_qc_metrics(
    thickness_m: np.ndarray,
    surface_m: np.ndarray,
    bed_m: np.ndarray,
    valid_mask: np.ndarray,
    dx_m: float,
) -> dict:
    """Compute structural checks without claiming independent accuracy."""
    thickness = np.asarray(thickness_m, dtype=float)
    surface = np.asarray(surface_m, dtype=float)
    bed = np.asarray(bed_m, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(thickness)
    if not valid.any():
        raise ValueError("QC requires at least one valid thickness cell")
    values = thickness[valid]
    boundary = valid & ~binary_erosion(valid, border_value=0)
    # The imposed zero-thickness margin creates an intentionally steep rim.
    # Exclude two cells before screening interior gradients and surface slopes.
    gradient_mask = binary_erosion(valid, iterations=2, border_value=0)
    thickness_gradient = _pairwise_gradient_values(thickness, gradient_mask, dx_m)
    surface_gradient = _pairwise_gradient_values(surface, gradient_mask, dx_m)
    closure = np.abs((surface - bed) - thickness)
    quantiles = np.quantile(values, [0.05, 0.5, 0.95, 0.99])
    maximum_flat_index = int(np.nanargmax(np.where(valid, thickness, np.nan)))
    maximum_row, maximum_col = np.unravel_index(maximum_flat_index, thickness.shape)
    return {
        "finite_cell_count": int(valid.sum()),
        "minimum_thickness_m": float(values.min()),
        "mean_thickness_m": float(values.mean()),
        "median_thickness_m": float(quantiles[1]),
        "p05_thickness_m": float(quantiles[0]),
        "p95_thickness_m": float(quantiles[2]),
        "p99_thickness_m": float(quantiles[3]),
        "maximum_thickness_m": float(values.max()),
        "maximum_thickness_row": int(maximum_row),
        "maximum_thickness_col": int(maximum_col),
        "material_negative_cell_count": int(
            (values < MATERIAL_NEGATIVE_TOLERANCE_M).sum()
        ),
        "boundary_maximum_thickness_m": float(np.max(thickness[boundary])),
        "maximum_surface_closure_error_m": float(np.nanmax(closure[valid])),
        "p99_thickness_gradient_m_per_m": float(
            np.quantile(thickness_gradient, 0.99)
        ) if thickness_gradient.size else 0.0,
        "maximum_thickness_gradient_m_per_m": float(
            thickness_gradient.max()
        ) if thickness_gradient.size else 0.0,
        "p99_surface_slope_deg": float(np.degrees(np.arctan(
            np.quantile(surface_gradient, 0.99)
        ))) if surface_gradient.size else 0.0,
        "maximum_surface_slope_deg": float(np.degrees(np.arctan(
            surface_gradient.max()
        ))) if surface_gradient.size else 0.0,
        "area_fraction_thicker_than_800m": float(
            np.mean(values > THICKNESS_REVIEW_M)
        ),
        "area_fraction_thicker_than_1000m": float(
            np.mean(values > THICKNESS_HIGH_REVIEW_M)
        ),
    }


def _transform_array(
    coordinates: np.ndarray,
    source_crs: CRS,
    target_crs: CRS,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if source_crs == target_crs:
        return coordinates
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    return np.column_stack([x, y])


def _transform_shape(geometry, source_crs: CRS, target_crs: CRS):
    if source_crs == target_crs:
        return geometry
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transform_geometry(transformer.transform, geometry)


def _plot_outline(axis, geometry, **kwargs) -> None:
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    for polygon in polygons:
        x, y = polygon.exterior.xy
        axis.plot(x, y, **kwargs)
        for interior in polygon.interiors:
            ix, iy = interior.xy
            axis.plot(ix, iy, **kwargs)


def _load_network_lines(network_root: Path, unique_id: int, target_crs: CRS):
    output = []
    system_root = network_root / str(unique_id)
    for component_dir in sorted(system_root.glob("component_*")):
        path = component_dir / "lia_centerline_network.pkl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        source_crs = CRS.from_user_input(payload["crs"])
        main_index = int(payload.get(
            "main_branch_index", len(payload["branches"]) - 1
        ))
        for branch in payload["branches"]:
            output.append({
                "component_index": int(component_dir.name.split("_")[-1]),
                "branch_index": int(branch["branch_index"]),
                "is_main": int(branch["branch_index"]) == main_index,
                "coordinates": _transform_array(
                    branch["coords_projected_m"], source_crs, target_crs
                ),
            })
    return output


def _review_flags(record: dict) -> list[str]:
    flags = []
    if record["material_negative_cell_count"]:
        flags.append("material-negative-thickness")
    if record["boundary_maximum_thickness_m"] > 1e-6:
        flags.append("nonzero-boundary-thickness")
    if abs(record["area_closure_fraction"]) > AREA_CLOSURE_TOLERANCE_FRACTION:
        flags.append("raster-area-minus-vector-over-2pct")
    if record["maximum_thickness_m"] > THICKNESS_HIGH_REVIEW_M:
        flags.append("maximum-thickness-over-1000m")
    elif record["maximum_thickness_m"] > THICKNESS_REVIEW_M:
        flags.append("maximum-thickness-over-800m")
    if record["p99_surface_slope_deg"] > SURFACE_SLOPE_P99_REVIEW_DEG:
        flags.append("interior-p99-surface-slope-over-60deg")
    if record["maximum_bed_overlap_difference_m"] > BED_OVERLAP_REVIEW_M:
        flags.append("bed-overlap-difference-over-100m")
    if record["excluded_component_count"]:
        flags.append("documented-small-component-exclusion")
    return flags


def _run_system(payload: dict) -> dict:
    netcdf_path = Path(payload["netcdf_path"])
    output_path = Path(payload["output_path"])
    network_root = Path(payload["network_root"])
    source_geometry = wkb.loads(bytes.fromhex(payload["geometry_wkb_hex"]))
    source_crs = CRS.from_user_input(payload["geometry_crs"])
    excluded_indices = set(payload["excluded_component_indices"])

    with xr.open_dataset(netcdf_path) as dataset:
        target_crs = _dataset_crs(dataset)
        member = dataset["thickness_m"].isel(member=0)
        thickness = np.asarray(member.values, dtype=float)
        surface = np.asarray(
            dataset["surface_elevation_m"].isel(member=0).values, dtype=float
        )
        bed = np.asarray(
            dataset["bed_elevation_m"].isel(member=0).values, dtype=float
        )
        valid = np.asarray(
            dataset["valid_mask"].isel(member=0).values, dtype=bool
        )
        excluded_mask = np.asarray(dataset["excluded_component_mask"].values, bool)
        x = np.asarray(dataset["x"].values, dtype=float)
        y = np.asarray(dataset["y"].values, dtype=float)
        status = str(dataset["member_status"].isel(member=0).item())
        rgi_id = str(dataset.attrs["rgi_id"])
        unique_id = int(dataset.attrs["lia_unique_id"])

    dx_m = float(abs(np.median(np.diff(x))))
    geometry = _transform_shape(source_geometry, source_crs, target_crs)
    components = normalize_lia_components(geometry)
    active_geometry_area_m2 = float(sum(
        component.area for index, component in enumerate(components)
        if index not in excluded_indices
    ))
    metrics = thickness_qc_metrics(thickness, surface, bed, valid, dx_m)
    valid_area_m2 = float(valid.sum() * dx_m**2)
    record = {
        "rgi_id": rgi_id,
        "lia_unique_id": unique_id,
        "status": status,
        "system_grid_dx_m": dx_m,
        "vector_active_area_km2": active_geometry_area_m2 / 1e6,
        "raster_valid_area_km2": valid_area_m2 / 1e6,
        "area_closure_fraction": (
            valid_area_m2 - active_geometry_area_m2
        ) / active_geometry_area_m2,
        "volume_km3": float(np.nansum(thickness[valid]) * dx_m**2 / 1e9),
        "excluded_component_count": int(len(excluded_indices)),
        "excluded_component_area_km2": float(sum(
            component.area for index, component in enumerate(components)
            if index in excluded_indices
        ) / 1e6),
        "mosaic_bed_conflict_cells": int(payload["mosaic_bed_conflict_cells"]),
        "maximum_bed_overlap_difference_m": float(
            payload["maximum_bed_overlap_difference_m"]
        ),
        **metrics,
    }
    max_row = record["maximum_thickness_row"]
    max_col = record["maximum_thickness_col"]
    record["maximum_thickness_x_m"] = float(x[max_col])
    record["maximum_thickness_y_m"] = float(y[max_row])
    record["excluded_finite_cell_count"] = int(
        (excluded_mask & np.isfinite(thickness)).sum()
    )
    lines = _load_network_lines(network_root, unique_id, target_crs)
    record["flowline_branch_count"] = int(len(lines))
    record["main_flowline_count"] = int(sum(line["is_main"] for line in lines))
    record["tributary_flowline_count"] = int(sum(
        not line["is_main"] for line in lines
    ))
    flags = _review_flags(record)
    record["review_flag_count"] = len(flags)
    record["review_flags"] = ";".join(flags)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    data = [thickness, surface, bed]
    titles = ["LIA ice thickness (m)", "LIA surface elevation (m)", "Bed elevation (m)"]
    cmaps = ["viridis", "terrain", "terrain"]
    for index, (axis, values, title, cmap) in enumerate(zip(axes, data, titles, cmaps)):
        shown = np.where(valid, values, np.nan)
        finite = shown[np.isfinite(shown)]
        if index == 0:
            low, high = 0.0, float(np.quantile(finite, 0.995))
        else:
            low, high = np.quantile(finite, [0.01, 0.99])
        image = axis.pcolormesh(
            x, y, shown, cmap=cmap, vmin=float(low), vmax=float(high),
            shading="auto", rasterized=True,
        )
        _plot_outline(axis, geometry, color="black", linewidth=1.0)
        for component_index in excluded_indices:
            if component_index < len(components):
                _plot_outline(
                    axis, components[component_index], color="#c026d3",
                    linewidth=1.8, linestyle="--",
                )
        for line in lines:
            coordinates = line["coordinates"]
            plotted = axis.plot(
                coordinates[:, 0], coordinates[:, 1],
                color="white" if line["is_main"] else "#38bdf8",
                linewidth=1.5 if line["is_main"] else 0.8,
                alpha=0.95,
            )[0]
            if line["is_main"]:
                plotted.set_path_effects([
                    patheffects.Stroke(linewidth=2.7, foreground="black"),
                    patheffects.Normal(),
                ])
        if index == 0:
            axis.scatter(
                [record["maximum_thickness_x_m"]],
                [record["maximum_thickness_y_m"]],
                marker="x", s=55, linewidth=1.8, color="#ef4444", zorder=6,
            )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.ticklabel_format(style="plain", useOffset=False)
        axis.tick_params(labelsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
    fig.suptitle(
        f"LIA {unique_id} | {rgi_id} | V={record['volume_km3']:.2f} km³ | "
        f"Hmax={record['maximum_thickness_m']:.0f} m | "
        f"flags={record['review_flag_count']}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    record["image_file"] = output_path.name
    return record


def _overview_plot(frame: pd.DataFrame, output_path: Path) -> None:
    ordered = frame.sort_values("maximum_thickness_m", ascending=False).head(25)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    axes[0].hist(frame["mean_thickness_m"], bins=24, color="#2563a6")
    axes[0].set_xlabel("Mean thickness (m)")
    axes[0].set_ylabel("System count")
    axes[0].set_title("Mean thickness distribution")
    scatter = axes[1].scatter(
        frame["vector_active_area_km2"], frame["maximum_thickness_m"],
        c=frame["review_flag_count"], cmap="magma", s=26, alpha=0.85,
    )
    axes[1].set_xscale("log")
    axes[1].axhline(THICKNESS_REVIEW_M, color="#d97706", linewidth=1.0)
    axes[1].axhline(THICKNESS_HIGH_REVIEW_M, color="#b42318", linewidth=1.0)
    axes[1].set_xlabel("Active LIA area (km², log scale)")
    axes[1].set_ylabel("Maximum thickness (m)")
    axes[1].set_title("Thickness versus system size")
    fig.colorbar(scatter, ax=axes[1], label="Review flag count")
    labels = ordered["lia_unique_id"].astype(str)
    axes[2].barh(labels[::-1], ordered["maximum_thickness_m"][::-1], color="#16803a")
    axes[2].axvline(THICKNESS_REVIEW_M, color="#d97706", linewidth=1.0)
    axes[2].axvline(THICKNESS_HIGH_REVIEW_M, color="#b42318", linewidth=1.0)
    axes[2].set_xlabel("Maximum thickness (m)")
    axes[2].set_ylabel("LIA system ID")
    axes[2].set_title("Top 25 maximum thickness")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    output_root: Path,
    workers: int,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outlines = gpd.read_file(resolve(config["lia_geometry_path"]))
    outlines["UniqueID"] = outlines["UniqueID"].astype(int)
    geometry_by_id = outlines.set_index("UniqueID").geometry
    component_results = pd.read_csv(result_root / "lia_system_component_results_v1.csv")
    mosaic = pd.read_csv(result_root / "lia_system_grid_mosaic_audit_v1.csv")
    mosaic_by_id = mosaic.set_index("lia_unique_id")
    images = output_root / "images"
    images.mkdir(parents=True, exist_ok=True)

    payloads = []
    for netcdf_path in sorted(result_root.glob("lia_*_system_grid_v1.nc")):
        with xr.open_dataset(netcdf_path) as dataset:
            unique_id = int(dataset.attrs["lia_unique_id"])
            rgi_id = str(dataset.attrs["rgi_id"])
        excluded = component_results[
            component_results["lia_unique_id"].astype(int).eq(unique_id)
            & component_results["status"].astype(str).eq(
                "documented-small-component-exclusion"
            )
        ]
        mosaic_row = mosaic_by_id.loc[unique_id]
        payloads.append({
            "netcdf_path": str(netcdf_path),
            "network_root": str(network_root),
            "output_path": str(images / f"lia_{unique_id}_{rgi_id}_v3_qc.png"),
            "geometry_wkb_hex": geometry_by_id.loc[unique_id].wkb_hex,
            "geometry_crs": outlines.crs.to_string(),
            "excluded_component_indices": excluded["component_index"].astype(int).tolist(),
            "mosaic_bed_conflict_cells": int(mosaic_row["bed_overlap_conflict_cells"]),
            "maximum_bed_overlap_difference_m": float(
                mosaic_row["maximum_bed_overlap_difference_m"]
            ),
        })
    if len(payloads) != 191:
        raise ValueError(f"Expected 191 v3 systems, found {len(payloads)}")

    rows = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_run_system, payload): payload for payload in payloads}
        for future in as_completed(futures):
            rows.append(future.result())
    frame = pd.DataFrame(rows).sort_values(["review_flag_count", "maximum_thickness_m"], ascending=[False, False])
    metrics_path = output_root / "core191_v3_qc_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    overview_path = output_root / "core191_v3_qc_overview.png"
    _overview_plot(frame, overview_path)
    summary = {
        "status": "completed",
        "scientific_scope": "structural-plausibility-and-numerical-QC-not-independent-LIA-accuracy",
        "system_count": int(len(frame)),
        "image_count": int(len(list(images.glob("*.png")))),
        "systems_with_review_flags": int(frame["review_flag_count"].gt(0).sum()),
        "review_flag_counts": frame["review_flags"].str.split(";").explode().replace("", np.nan).value_counts().to_dict(),
        "material_negative_cell_count": int(frame["material_negative_cell_count"].sum()),
        "maximum_boundary_thickness_m": float(frame["boundary_maximum_thickness_m"].max()),
        "maximum_surface_closure_error_m": float(frame["maximum_surface_closure_error_m"].max()),
        "maximum_absolute_area_closure_fraction": float(frame["area_closure_fraction"].abs().max()),
        "metrics_csv": metrics_path.name,
        "overview_png": overview_path.name,
        "images_directory": images.name,
    }
    (output_root / "core191_v3_qc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        output_root=args.output_root,
        workers=args.workers,
    ), indent=2))


if __name__ == "__main__":
    main()
