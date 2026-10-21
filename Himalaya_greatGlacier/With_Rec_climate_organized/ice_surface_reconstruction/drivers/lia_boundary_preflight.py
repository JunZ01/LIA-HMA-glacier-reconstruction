"""Classify pilot LIA polygon boundaries before imposing bed/surface BCs.

This is a geometric review queue, not an automatic ice-divide detector.
Exterior segments close to another mapped LIA system are marked as candidates
requiring review. Remaining exterior segments and nunatak rings receive the
provisional zero-thickness policy registered in the production config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString
from shapely.ops import unary_union

from ..paths import resolve
from .lia_geometry_preflight import _gdir_path


def _line_parts(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [LineString(geometry)]
    parts: list[LineString] = []
    if hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            parts.extend(_line_parts(item))
    return parts


def classify_boundary_segments(
    polygon,
    neighbor_boundaries: list,
    tolerance_m: float,
) -> tuple[list[dict], dict[str, float]]:
    """Partition exterior and nunatak rings into explicit boundary classes."""
    if polygon.geom_type != "Polygon":
        raise ValueError(f"Expected Polygon, got {polygon.geom_type}")
    exterior = LineString(polygon.exterior.coords)
    if neighbor_boundaries:
        review_zone = unary_union(neighbor_boundaries).buffer(
            tolerance_m, cap_style=2, join_style=2
        )
        candidate = exterior.intersection(review_zone)
        ordinary = exterior.difference(review_zone)
    else:
        candidate = LineString()
        ordinary = exterior

    rows: list[dict] = []
    for part in _line_parts(ordinary):
        rows.append({
            "boundary_class": "outer-margin-zero-thickness-provisional",
            "review_required": False,
            "geometry": part,
        })
    for part in _line_parts(candidate):
        rows.append({
            "boundary_class": "ice-divide-candidate-review",
            "review_required": True,
            "geometry": part,
        })
    for ring in polygon.interiors:
        rows.append({
            "boundary_class": "nunatak-zero-thickness-provisional",
            "review_required": False,
            "geometry": LineString(ring.coords),
        })

    candidate_length = float(sum(
        row["geometry"].length for row in rows
        if row["boundary_class"] == "ice-divide-candidate-review"
    ))
    ordinary_length = float(sum(
        row["geometry"].length for row in rows
        if row["boundary_class"] == "outer-margin-zero-thickness-provisional"
    ))
    nunatak_length = float(sum(
        row["geometry"].length for row in rows
        if row["boundary_class"] == "nunatak-zero-thickness-provisional"
    ))
    metrics = {
        "exterior_length_m": float(exterior.length),
        "candidate_divide_length_m": candidate_length,
        "ordinary_outer_length_m": ordinary_length,
        "nunatak_boundary_length_m": nunatak_length,
        "exterior_partition_closure_error_m": float(
            ordinary_length + candidate_length - exterior.length
        ),
    }
    return rows, metrics


def run(config: dict) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    gdir_root = resolve(config["stage0_gdir_root"])
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    lia_by_id = lia.set_index("UniqueID")
    tolerance_pixels = float(
        config.get("boundary_policy", {}).get(
            "adjacent_lia_review_tolerance_pixels", 1.0
        )
    )
    system_rows: list[dict] = []
    segment_rows: list[dict] = []

    for item in selection.itertuples(index=False):
        rgi_id = str(item.main_rgi_id)
        uid = int(item.lia_unique_id)
        gdir_dir = _gdir_path(gdir_root, rgi_id)
        grid = json.loads((gdir_dir / "glacier_grid.json").read_text())
        grid_crs = CRS.from_user_input(grid["proj"])
        map_dx_m = abs(float(grid["dxdy"][0]))
        tolerance_m = map_dx_m * tolerance_pixels
        lia_grid = lia.to_crs(grid_crs)
        polygon = lia_grid.loc[lia_grid["UniqueID"] == uid].geometry.iloc[0]
        if polygon.geom_type != "Polygon":
            system_rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "status": "unsupported-multipart-boundary-review",
                "geometry_type": polygon.geom_type,
            })
            continue
        other = lia_grid.loc[lia_grid["UniqueID"] != uid, ["UniqueID", "geometry"]]
        distances = other.geometry.distance(polygon)
        neighbors = other.loc[distances <= tolerance_m].copy()
        rows, metrics = classify_boundary_segments(
            polygon,
            [geometry.boundary for geometry in neighbors.geometry],
            tolerance_m,
        )
        exterior_length = metrics["exterior_length_m"]
        candidate_fraction = (
            metrics["candidate_divide_length_m"] / exterior_length
            if exterior_length > 0 else np.nan
        )
        status = (
            "candidate-divide-review"
            if metrics["candidate_divide_length_m"] > 0
            else "provisional-zero-boundaries"
        )
        system_rows.append({
            "rgi_id": rgi_id,
            "lia_unique_id": uid,
            "status": status,
            "map_dx_m": map_dx_m,
            "review_tolerance_m": tolerance_m,
            "neighbor_count": int(len(neighbors)),
            "neighbor_lia_unique_ids": ";".join(
                str(int(value)) for value in sorted(neighbors["UniqueID"])
            ) or "none",
            "hole_count": int(len(polygon.interiors)),
            "candidate_divide_fraction": candidate_fraction,
            **metrics,
        })
        for segment_index, row in enumerate(rows):
            geometry = row["geometry"]
            nearby_ids = "none"
            if row["review_required"] and not neighbors.empty:
                ids = neighbors.loc[
                    neighbors.geometry.boundary.distance(geometry) <= tolerance_m,
                    "UniqueID",
                ]
                nearby_ids = ";".join(str(int(value)) for value in sorted(ids)) or "none"
            geometry_wgs84 = gpd.GeoSeries(
                [geometry], crs=grid_crs
            ).to_crs("EPSG:4326").iloc[0]
            segment_rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "segment_index": segment_index,
                "boundary_class": row["boundary_class"],
                "review_required": row["review_required"],
                "neighbor_lia_unique_ids": nearby_ids,
                "length_m": float(geometry.length),
                "geometry": geometry_wgs84,
            })

    systems = pd.DataFrame(system_rows)
    segments = gpd.GeoDataFrame(segment_rows, geometry="geometry", crs="EPSG:4326")
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    systems.to_csv(output_dir / "lia_boundary_preflight_v0.csv", index=False)
    segments.drop(columns="geometry").to_csv(
        output_dir / "lia_boundary_segments_v0.csv", index=False
    )
    segments.to_file(
        output_dir / "lia_boundary_segments_v0.geojson", driver="GeoJSON"
    )
    summary = {
        "status": "geometric-candidate-classification-not-physical-freeze",
        "systems_requested": int(len(selection)),
        "status_counts": systems["status"].value_counts(dropna=False).to_dict(),
        "boundary_class_counts": segments["boundary_class"].value_counts(
            dropna=False
        ).to_dict(),
        "review_tolerance_pixels": tolerance_pixels,
        "candidate_divide_systems": int(
            (systems["status"] == "candidate-divide-review").sum()
        ),
        "classification_warning": (
            "adjacency is only a review trigger; it does not prove an ice divide"
        ),
    }
    (output_dir / "lia_boundary_preflight_summary_v0.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return systems, segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    systems, _ = run(json.loads(args.config.read_text(encoding="utf-8")))
    print(systems["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
