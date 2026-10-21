"""Add topographic ridge evidence to candidate LIA ice-divide segments.

The resulting classes are review priorities, not boundary-condition decisions.
Current topography is only a proxy for the unknown LIA surface, so even strong
ridge support cannot by itself freeze a zero-flux boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from scipy.ndimage import map_coordinates
from shapely.geometry import Point
from shapely.ops import unary_union

from ..paths import resolve
from .lia_geometry_preflight import _gdir_path


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _sample_raster(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    col = (np.asarray(x, dtype=float) - x0) / dx
    row = (np.asarray(y, dtype=float) - y0) / dy
    return map_coordinates(
        values,
        np.vstack([row, col]),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def sample_segment_ridge_evidence(
    segment,
    topo_m: np.ndarray,
    grid: dict,
    target_polygon,
    neighbor_union,
    sample_spacing_m: float,
    offset_pixels: list[float],
    minimum_relief_m: float,
) -> pd.DataFrame:
    """Sample cross-boundary DEM profiles along one candidate segment."""
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    map_dx_m = abs(dx)
    if segment.length <= 0:
        return pd.DataFrame()
    positions = np.arange(0.5 * sample_spacing_m, segment.length, sample_spacing_m)
    if not positions.size:
        positions = np.asarray([0.5 * segment.length])
    tangent_step = min(map_dx_m, max(segment.length * 0.25, 1.0))
    rows: list[dict] = []
    for sample_index, position in enumerate(positions):
        center = segment.interpolate(float(position))
        before = segment.interpolate(max(float(position) - tangent_step, 0.0))
        after = segment.interpolate(min(float(position) + tangent_step, segment.length))
        tangent = np.asarray([after.x - before.x, after.y - before.y], dtype=float)
        norm = float(np.linalg.norm(tangent))
        if norm <= 0:
            continue
        tangent /= norm
        normal = np.asarray([-tangent[1], tangent[0]])
        z_center = float(_sample_raster(
            topo_m, np.asarray([center.x]), np.asarray([center.y]),
            x0, y0, dx, dy,
        )[0])
        record: dict[str, float | int | bool] = {
            "sample_index": sample_index,
            "distance_along_segment_m": float(position),
            "x_m": float(center.x),
            "y_m": float(center.y),
            "center_elevation_m": z_center,
        }
        for offset in offset_pixels:
            distance = float(offset) * map_dx_m
            plus = Point(center.x + normal[0] * distance,
                         center.y + normal[1] * distance)
            minus = Point(center.x - normal[0] * distance,
                          center.y - normal[1] * distance)
            z_plus, z_minus = _sample_raster(
                topo_m,
                np.asarray([plus.x, minus.x]),
                np.asarray([plus.y, minus.y]),
                x0, y0, dx, dy,
            )
            key = f"offset_{offset:g}px"
            record[f"{key}_plus_elevation_m"] = float(z_plus)
            record[f"{key}_minus_elevation_m"] = float(z_minus)
            record[f"{key}_ridge_relief_m"] = float(
                z_center - max(z_plus, z_minus)
            )
            record[f"{key}_two_sided_descent"] = bool(
                np.isfinite([z_center, z_plus, z_minus]).all()
                and z_center - z_plus >= minimum_relief_m
                and z_center - z_minus >= minimum_relief_m
            )
            plus_target = target_polygon.covers(plus)
            minus_target = target_polygon.covers(minus)
            plus_neighbor = neighbor_union.covers(plus)
            minus_neighbor = neighbor_union.covers(minus)
            record[f"{key}_opposite_polygon_sides"] = bool(
                (plus_target and minus_neighbor)
                or (minus_target and plus_neighbor)
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _classify_segment(
    samples: pd.DataFrame,
    classification_offset_pixels: float,
    geometry_threshold: float,
    ridge_likely_threshold: float,
    ridge_weak_threshold: float,
) -> tuple[str, float, float]:
    if samples.empty:
        return "insufficient-samples-review", np.nan, np.nan
    key = f"offset_{classification_offset_pixels:g}px"
    geometry_support = float(samples[f"{key}_opposite_polygon_sides"].mean())
    ridge_support = float(samples[f"{key}_two_sided_descent"].mean())
    if geometry_support < geometry_threshold:
        label = "geometry-ambiguous-review"
    elif ridge_support >= ridge_likely_threshold:
        label = "ridge-supported-divide-candidate"
    elif ridge_support <= ridge_weak_threshold:
        label = "weak-ridge-divide-review"
    else:
        label = "intermediate-ridge-divide-review"
    return label, geometry_support, ridge_support


def _plot_review_map(
    uid: int,
    rgi_id: str,
    polygon,
    neighbor_geometries: list,
    segments: gpd.GeoDataFrame,
    centerlines: gpd.GeoDataFrame,
    topo_m: np.ndarray,
    grid: dict,
    output_path: Path,
) -> None:
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    ny, nx = topo_m.shape
    extent = [
        x0 - 0.5 * dx,
        x0 + (nx - 0.5) * dx,
        y0 + (ny - 0.5) * dy,
        y0 - 0.5 * dy,
    ]
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = ax.imshow(topo_m, extent=extent, origin="upper", cmap="terrain")
    ax.plot(*polygon.exterior.xy, color="black", linewidth=1.2, label="LIA boundary")
    for neighbor in neighbor_geometries:
        if neighbor.geom_type == "Polygon":
            ax.plot(*neighbor.exterior.xy, color="#1287a8", linewidth=1.0,
                    linestyle="--", label="Adjacent LIA system")
    colors = {
        "ice-divide-candidate-review": "#d62728",
        "outer-margin-zero-thickness-provisional": "#333333",
        "nunatak-zero-thickness-provisional": "#7f7f7f",
    }
    for row in segments.itertuples():
        ax.plot(*row.geometry.xy, color=colors[row.boundary_class],
                linewidth=2.4 if row.review_required else 0.8)
    for line in centerlines.geometry:
        ax.plot(*line.xy, color="#1464a5", linewidth=1.1)
    minx, miny, maxx, maxy = polygon.bounds
    pad = max(abs(dx) * 8, 500.0)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_title(f"LIA {uid} / {rgi_id}: boundary review")
    ax.set_xlabel("Projected x (m)")
    ax.set_ylabel("Projected y (m)")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Elevation (m)")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = resolve(config["output_dir"])
    boundary_segments = gpd.read_file(output_dir / "lia_boundary_segments_v0.geojson")
    candidate_segments = boundary_segments[
        boundary_segments["boundary_class"] == "ice-divide-candidate-review"
    ].copy()
    centerlines = gpd.read_file(output_dir / "lia_centerline_branches_v0.geojson")
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    gdir_root = resolve(config["stage0_gdir_root"])
    evidence_config = config.get("boundary_ridge_evidence", {})
    spacing_pixels = float(evidence_config.get("sample_spacing_pixels", 1.0))
    offset_pixels = [float(value) for value in evidence_config.get(
        "normal_offset_pixels", [1.0, 2.0, 3.0]
    )]
    classification_offset = float(evidence_config.get(
        "classification_offset_pixels", 2.0
    ))
    minimum_relief = float(evidence_config.get("minimum_relief_m", 5.0))
    geometry_threshold = float(evidence_config.get(
        "geometry_side_support_threshold", 0.7
    ))
    ridge_likely = float(evidence_config.get("ridge_likely_threshold", 0.5))
    ridge_weak = float(evidence_config.get("ridge_weak_threshold", 0.2))
    sample_rows: list[dict] = []
    summary_rows: list[dict] = []

    for (rgi_id, uid), group_wgs84 in candidate_segments.groupby(
        ["rgi_id", "lia_unique_id"], sort=True
    ):
        uid = int(uid)
        gdir_dir = _gdir_path(gdir_root, str(rgi_id))
        grid = json.loads((gdir_dir / "glacier_grid.json").read_text())
        grid_crs = CRS.from_user_input(grid["proj"])
        map_dx_m = abs(float(grid["dxdy"][0]))
        with xr.open_dataset(gdir_dir / "gridded_data.nc") as dataset:
            topo_m = np.asarray(dataset["topo"].values, dtype=float)
        lia_grid = lia.to_crs(grid_crs)
        polygon = lia_grid.loc[lia_grid["UniqueID"] == uid].geometry.iloc[0]
        group = group_wgs84.to_crs(grid_crs)
        neighbor_ids = sorted({
            int(value)
            for text in group["neighbor_lia_unique_ids"]
            for value in str(text).split(";")
            if value not in {"", "none", "nan"}
        })
        neighbor_geometries = list(
            lia_grid.loc[lia_grid["UniqueID"].isin(neighbor_ids)].geometry
        )
        neighbor_union = unary_union(neighbor_geometries)
        for segment in group.itertuples():
            samples = sample_segment_ridge_evidence(
                segment.geometry,
                topo_m,
                grid,
                polygon,
                neighbor_union,
                sample_spacing_m=spacing_pixels * map_dx_m,
                offset_pixels=offset_pixels,
                minimum_relief_m=minimum_relief,
            )
            label, geometry_support, ridge_support = _classify_segment(
                samples,
                classification_offset,
                geometry_threshold,
                ridge_likely,
                ridge_weak,
            )
            key = f"offset_{classification_offset:g}px"
            summary_rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "segment_index": int(segment.segment_index),
                "length_m": float(segment.length_m),
                "neighbor_lia_unique_ids": segment.neighbor_lia_unique_ids,
                "sample_count": int(len(samples)),
                "geometry_side_support_fraction": geometry_support,
                "ridge_support_fraction": ridge_support,
                "ridge_relief_median_m": (
                    float(samples[f"{key}_ridge_relief_m"].median())
                    if not samples.empty else np.nan
                ),
                "evidence_class": label,
                "boundary_condition_status": "not-frozen-review",
            })
            for sample in samples.to_dict(orient="records"):
                sample_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "segment_index": int(segment.segment_index),
                    **sample,
                })
        system_segments = boundary_segments[
            (boundary_segments["rgi_id"] == rgi_id)
            & (boundary_segments["lia_unique_id"].astype(int) == uid)
        ].to_crs(grid_crs)
        system_centerlines = centerlines[
            (centerlines["rgi_id"] == rgi_id)
            & (centerlines["lia_unique_id"].astype(int) == uid)
            & (centerlines["elevation_field"] == "raw-bed-farinotti2019")
        ].to_crs(grid_crs)
        _plot_review_map(
            uid,
            str(rgi_id),
            polygon,
            neighbor_geometries,
            system_segments,
            system_centerlines,
            topo_m,
            grid,
            output_dir / f"lia_boundary_review_{uid}_v0.png",
        )

    summaries = pd.DataFrame(summary_rows)
    samples = pd.DataFrame(sample_rows)
    summaries.to_csv(output_dir / "lia_boundary_ridge_evidence_v0.csv", index=False)
    samples.to_csv(output_dir / "lia_boundary_ridge_samples_v0.csv", index=False)
    summary = {
        "status": "review-evidence-not-boundary-condition-freeze",
        "candidate_segments": int(len(summaries)),
        "candidate_systems": int(summaries["rgi_id"].nunique()),
        "evidence_class_counts": summaries["evidence_class"].value_counts(
            dropna=False
        ).to_dict(),
        "classification_offset_pixels": classification_offset,
        "minimum_relief_m": minimum_relief,
        "warning": "current DEM ridge support is a proxy, not LIA surface evidence",
    }
    (output_dir / "lia_boundary_ridge_evidence_summary_v0.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summaries, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summaries, _ = run(json.loads(args.config.read_text(encoding="utf-8")))
    print(summaries["evidence_class"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
