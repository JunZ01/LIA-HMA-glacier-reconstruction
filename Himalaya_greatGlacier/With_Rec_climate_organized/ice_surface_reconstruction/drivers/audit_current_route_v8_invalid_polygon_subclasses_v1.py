"""Read-only subclass audit for current-route-v8 invalid LIA Polygons.

This driver never writes a repaired outline.  It measures whether a deterministic
``shapely.make_valid`` operation preserves the semantic support closely enough to
define a representative-plus-confirmation pilot contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import rasterize
from shapely import make_valid
from shapely.validation import explain_validity


INVALID_IDS = (
    14, 334, 516, 1014, 1120, 1133, 1499, 1639, 1653, 2535,
    2692, 2919, 3075, 4637, 5014, 5231, 5353, 5398, 5508, 5598,
)


def _parts(geometry):
    return list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]


def _polygon_components(geometry) -> int:
    return sum(part.geom_type == "Polygon" for part in _parts(geometry))


def _holes(geometry) -> int:
    return sum(len(part.interiors) for part in _parts(geometry)
               if part.geom_type == "Polygon")


def _raster_support(source, repaired, dx_m: float) -> dict:
    minx = np.floor(min(source.bounds[0], repaired.bounds[0]) / dx_m) * dx_m
    miny = np.floor(min(source.bounds[1], repaired.bounds[1]) / dx_m) * dx_m
    maxx = np.ceil(max(source.bounds[2], repaired.bounds[2]) / dx_m) * dx_m
    maxy = np.ceil(max(source.bounds[3], repaired.bounds[3]) / dx_m) * dx_m
    width = max(1, int(round((maxx - minx) / dx_m)))
    height = max(1, int(round((maxy - miny) / dx_m)))
    transform = Affine(dx_m, 0, minx, 0, -dx_m, maxy)
    kwargs = dict(out_shape=(height, width), transform=transform, fill=0,
                  all_touched=False, dtype="uint8")
    source_mask = rasterize([(source, 1)], **kwargs)
    repaired_mask = rasterize([(repaired, 1)], **kwargs)
    return {
        "source_raster_cells": int(source_mask.sum()),
        "repaired_raster_cells": int(repaired_mask.sum()),
        "raster_changed_cells": int(np.count_nonzero(source_mask != repaired_mask)),
    }


def run(outline: Path, route_csv: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    outlines = gpd.read_file(outline)
    route = pd.read_csv(route_csv)
    expected = route.loc[
        route["lia_unique_id"].isin(INVALID_IDS)
        & route["outline_state"].eq("invalid-polygon")
        & route["current_route"].eq("input-geometry-blocked")
    ].copy()
    if set(expected["lia_unique_id"].astype(int)) != set(INVALID_IDS):
        raise RuntimeError("current-route-v8 invalid-polygon membership changed")

    rows = []
    for system_id in INVALID_IDS:
        selected = outlines.loc[outlines["UniqueID"].astype(int).eq(system_id)]
        if len(selected) != 1:
            raise RuntimeError(f"outline-row-count:{system_id}:{len(selected)}")
        source = selected.iloc[0].geometry
        repaired = make_valid(source)
        raster = _raster_support(source, repaired, 25.0)
        source_area = float(source.area)
        repaired_area = float(repaired.area)
        row = {
            "system_id": system_id,
            "source_type": source.geom_type,
            "source_valid": bool(source.is_valid),
            "validity_reason": explain_validity(source),
            "source_polygon_components": _polygon_components(source),
            "source_holes": _holes(source),
            "repaired_type": repaired.geom_type,
            "repaired_valid": bool(repaired.is_valid),
            "repaired_polygon_components": _polygon_components(repaired),
            "repaired_holes": _holes(repaired),
            "hole_delta": _holes(repaired) - _holes(source),
            "source_area_m2": source_area,
            "area_relative_change": abs(repaired_area - source_area) / source_area,
            "boundary_hausdorff_m": float(
                source.boundary.hausdorff_distance(repaired.boundary)
            ),
            **raster,
        }
        row["strict_semantic_support_pass"] = bool(
            row["source_type"] == "Polygon"
            and not row["source_valid"]
            and row["source_polygon_components"] == 1
            and row["repaired_type"] == "Polygon"
            and row["repaired_valid"]
            and row["repaired_polygon_components"] == 1
            and row["hole_delta"] == 1
            and row["area_relative_change"] <= 1e-12
            and row["boundary_hausdorff_m"] <= 1e-6
            and row["raster_changed_cells"] == 0
        )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)
    frame.to_csv(output_root / "invalid_polygon_classification.csv", index=False)
    grouped = (
        frame.groupby([
            "source_type", "repaired_type", "repaired_polygon_components",
            "hole_delta", "strict_semantic_support_pass",
        ], dropna=False)
        .agg(system_count=("system_id", "size"),
             systems=("system_id", lambda values: ";".join(map(str, values))))
        .reset_index()
        .sort_values(["system_count", "strict_semantic_support_pass"],
                     ascending=[False, False])
    )
    grouped.to_csv(output_root / "subclass_summary.csv", index=False)

    eligible = frame.loc[frame["strict_semantic_support_pass"]].copy()
    if len(eligible) >= 2:
        # Representative nearest the median source area; confirmation is the
        # largest-area member, exercising a materially different network scale.
        median_area = float(eligible["source_area_m2"].median())
        representative = int(
            eligible.assign(distance=(eligible["source_area_m2"] - median_area).abs())
            .sort_values(["distance", "system_id"]).iloc[0]["system_id"]
        )
        confirmation = int(
            eligible.loc[eligible["system_id"].ne(representative)]
            .sort_values(["source_area_m2", "system_id"], ascending=[False, True])
            .iloc[0]["system_id"]
        )
        selected = [representative, confirmation]
    else:
        selected = []

    contract = {
        "schema_version": 1,
        "experiment": "current-route-v8-invalid-polygon-subclass-audit-v1",
        "source_outline": str(outline.resolve()),
        "source_route": str(route_csv.resolve()),
        "route_member_count": len(frame),
        "strict_semantic_support_pass_count": int(
            frame["strict_semantic_support_pass"].sum()
        ),
        "pilot_selection_rule": (
            "largest strict semantic-support subclass; representative nearest "
            "median source area plus largest-area confirmation"
        ),
        "pilot_systems": selected,
        "promotion_contract": {
            "phase_1": "deterministic make_valid plus unchanged static geometry gates",
            "phase_2": "only static-pass cases may enter unchanged formal inversion gates",
            "phase_3": "only inversion-pass cases may enter formal zero-MB gates",
            "batch_promotion": "both representative and confirmation must pass each phase",
            "forbidden": [
                "source-outline mutation", "25m root-length stretching",
                "closed root-length mechanism", "closed surface-regularization class",
                "physics-parameter change", "scoring change",
            ],
        },
    }
    (output_root / "pilot_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outline", required=True, type=Path)
    parser.add_argument("--route-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.outline.resolve(), args.route_csv.resolve(),
                         args.output_root.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
