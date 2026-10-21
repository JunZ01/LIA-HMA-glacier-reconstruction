"""Read-only component audit for the remaining MultiPolygon LIA system 2898."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize


def _component_raster_cells(components, dx_m: float) -> list[int]:
    bounds = np.asarray([part.bounds for part in components], dtype=float)
    minx = np.floor(bounds[:, 0].min() / dx_m) * dx_m
    miny = np.floor(bounds[:, 1].min() / dx_m) * dx_m
    maxx = np.ceil(bounds[:, 2].max() / dx_m) * dx_m
    maxy = np.ceil(bounds[:, 3].max() / dx_m) * dx_m
    width = max(1, int(round((maxx - minx) / dx_m)))
    height = max(1, int(round((maxy - miny) / dx_m)))
    transform = Affine(dx_m, 0, minx, 0, -dx_m, maxy)
    kwargs = dict(out_shape=(height, width), transform=transform, fill=0,
                  all_touched=False, dtype="uint8")
    return [int(rasterize([(part, 1)], **kwargs).sum()) for part in components]


def run(outline: Path, route_csv: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    outlines = gpd.read_file(outline)
    selected = outlines.loc[outlines["UniqueID"].astype(int).eq(2898)]
    if len(selected) != 1:
        raise RuntimeError(f"outline-row-count:{len(selected)}")
    geometry = selected.iloc[0].geometry
    if geometry.geom_type != "MultiPolygon" or not geometry.is_valid:
        raise RuntimeError(f"unexpected-geometry:{geometry.geom_type}:{geometry.is_valid}")
    components = sorted(list(geometry.geoms), key=lambda part: part.area, reverse=True)
    cells = _component_raster_cells(components, 25.0)
    total_area = float(geometry.area)
    rows = []
    for index, (part, cell_count) in enumerate(zip(components, cells)):
        nearest_other = min(
            (float(part.distance(other)) for j, other in enumerate(components)
             if j != index), default=float("nan")
        )
        rows.append({
            "component_rank": index + 1,
            "area_m2": float(part.area),
            "area_fraction": float(part.area / total_area),
            "hole_count": len(part.interiors),
            "is_valid": bool(part.is_valid),
            "raster_cells_25m": cell_count,
            "raster_area_m2": float(cell_count * 625.0),
            "nearest_other_distance_m": nearest_other,
            "bounds": [float(value) for value in part.bounds],
        })
    import pandas as pd
    route = pd.read_csv(route_csv)
    route_row = route.loc[route["lia_unique_id"].eq(2898)]
    if len(route_row) != 1:
        raise RuntimeError(f"route-row-count:{len(route_row)}")
    route_row = route_row.iloc[0]
    payload = {
        "schema_version": 1,
        "experiment": "multipolygon-system-2898-read-only-component-audit-v1",
        "status": "read-only-audit-complete",
        "source_outline": str(outline.resolve()),
        "source_crs": str(outlines.crs),
        "source_geometry_type": geometry.geom_type,
        "source_is_valid": bool(geometry.is_valid),
        "component_count": len(components),
        "total_area_m2": total_area,
        "registered_lia_length_m": float(route_row["lia_registered_length_m"]),
        "registered_rgi_length_m": float(route_row["rgi_registered_length_m"]),
        "components": rows,
        "largest_component_area_fraction": rows[0]["area_fraction"],
        "largest_component_raster_fraction": float(cells[0] / sum(cells)),
        "largest_component_only_semantically_lossless": bool(
            len(components) == 1 or sum(cells[1:]) == 0
        ),
        "allowed_interpretations": {
            "drop_minor_components": bool(sum(cells[1:]) == 0),
            "component_aggregate_required": bool(sum(cells[1:]) > 0),
        },
        "oggm_run": false,
        "source_modified": false
    }
    (output_root / "component_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outline", type=Path, required=True)
    parser.add_argument("--route-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.outline.resolve(), args.route_csv.resolve(),
                         args.output_root.resolve()), indent=2, ensure_ascii=False,
                     allow_nan=False))


if __name__ == "__main__":
    main()
