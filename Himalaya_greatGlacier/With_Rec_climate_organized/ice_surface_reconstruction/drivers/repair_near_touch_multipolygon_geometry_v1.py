"""Deterministic sub-grid closure for the near-touching MultiPolygon system 2898.

The source outline is immutable.  Both polygon components are retained; a fixed
12.5 mm morphological closing bridges only a source gap no larger than 25 mm.
The derived single Polygon may enter unchanged synthetic-system static gates only
when area, holes, boundary distance, and 25 m raster support all pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

from ..paths import resolve
from . import repair_invalid_lia_polygon_geometry_v1 as common
from . import repair_invalid_lia_polygon_geometry_v2 as support
from . import synthetic_lia_system_geometry_preflight_v4 as geometry_v4


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    source_config_path = resolve(config["source_geometry_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    system_id = int(config["system_id"])
    if int(source_config["system_id"]) != system_id:
        raise RuntimeError("source-config-system-mismatch")
    outline_path = resolve(source_config["lia_outline"])
    hashes_before = support._canonical_source_hashes(outline_path)
    outlines = gpd.read_file(outline_path)
    selected = outlines.loc[outlines["UniqueID"].astype(int).eq(system_id)].copy()
    if len(selected) != 1:
        raise RuntimeError(f"outline-row-count:{len(selected)}")
    source = selected.iloc[0].geometry
    if source.geom_type != "MultiPolygon" or not source.is_valid:
        raise RuntimeError(f"unexpected-source:{source.geom_type}:{source.is_valid}")
    components = sorted(source.geoms, key=lambda part: part.area, reverse=True)
    if len(components) != int(config["gates"]["required_source_component_count"]):
        raise RuntimeError(f"component-count:{len(components)}")
    gap_m = float(components[0].distance(components[1]))
    radius_m = float(config["closing_radius_m"])
    repaired = source.buffer(radius_m).buffer(-radius_m)
    source_area = float(source.area)
    repaired_area = float(repaired.area)
    raster = support._raster_support(
        source, repaired, float(config["gates"]["raster_support_dx_m"])
    )
    component_coverage = [
        float(component.intersection(repaired).area / component.area)
        for component in components
    ]
    audit = {
        "schema_version": 1,
        "experiment": "near-touch-multipolygon-subgrid-closing-v1",
        "system_id": system_id,
        "source_outline": str(outline_path.resolve()),
        "source_geometry_type": source.geom_type,
        "source_is_valid": bool(source.is_valid),
        "source_component_count": len(components),
        "source_hole_count": common._hole_count(source),
        "source_area_m2": source_area,
        "nearest_component_gap_m": gap_m,
        "repair_method": "fixed morphological closing: buffer(+r).buffer(-r)",
        "closing_radius_m": radius_m,
        "repaired_geometry_type": repaired.geom_type,
        "repaired_is_valid": bool(repaired.is_valid),
        "repaired_component_count": common._component_count(repaired),
        "repaired_hole_count": common._hole_count(repaired),
        "repaired_area_m2": repaired_area,
        "area_relative_change": abs(repaired_area - source_area) / source_area,
        "boundary_hausdorff_distance_m": float(
            source.boundary.hausdorff_distance(repaired.boundary)
        ),
        "component_area_coverage_fractions": component_coverage,
        **raster,
        "gates": config["gates"],
        "source_sha256_before": hashes_before,
    }
    gates = config["gates"]
    repair_pass = bool(
        gap_m <= float(gates["maximum_source_gap_m"])
        and repaired.geom_type == "Polygon"
        and repaired.is_valid
        and common._component_count(repaired) == 1
        and common._hole_count(repaired) == common._hole_count(source)
        and audit["area_relative_change"] <= float(gates["maximum_area_relative_change"])
        and audit["boundary_hausdorff_distance_m"] <= float(
            gates["maximum_boundary_hausdorff_distance_m"]
        )
        and min(component_coverage) >= float(gates["minimum_component_area_coverage"])
        and raster["raster_support_changed_cell_count"] <= int(
            gates["maximum_raster_support_changed_cells"]
        )
    )
    audit["subgrid_closure_pass"] = repair_pass
    common._write_json(output_root / "subgrid_closure_audit.json", audit)
    if not repair_pass:
        summary = {
            "status": "subgrid-closure-fail", "system_id": system_id,
            "static_run": False, "inversion_run": False, "zero_mb_run": False,
            "audit": str((output_root / "subgrid_closure_audit.json").resolve()),
        }
        common._write_json(output_root / "summary.json", summary)
        return summary

    derived_path = output_root / "derived_outline" / "lia_2898_subgrid_closed_v1.gpkg"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    selected.geometry = [repaired]
    selected.to_file(derived_path, layer="lia_outline", driver="GPKG")
    child = dict(source_config)
    child.update({
        "experiment": "synthetic-lia-system-near-touch-closed-standard-geometry-v1",
        "stage": "multipolygon-2898-subgrid-closure-static-v1",
        "lia_outline": str(derived_path.resolve()),
        "output_root": str((output_root / "geometry").resolve()),
        "authorization": {"inversion": False, "zero_mb": False,
                          "steady_50y": False, "historical": False},
    })
    child_path = output_root / "geometry_config.json"
    common._write_json(child_path, child)
    geometry_summary = geometry_v4.v1.run(child_path)
    hashes_after = support._canonical_source_hashes(outline_path)
    audit.update({
        "source_sha256_after": hashes_after,
        "source_products_unchanged": hashes_before == hashes_after,
        "derived_outline": str(derived_path.resolve()),
        "derived_outline_sha256": common._sha256(derived_path),
    })
    common._write_json(output_root / "subgrid_closure_audit.json", audit)
    if hashes_before != hashes_after:
        raise RuntimeError("source-outline-mutation")
    summary = {
        "status": geometry_summary["status"],
        "system_id": system_id,
        "subgrid_closure_pass": True,
        "static_run": True,
        "geometry_review_flags": geometry_summary.get("review_flags", []),
        "geometry_summary": str((output_root / "geometry" / "geometry_summary.json").resolve()),
        "inversion_run": False,
        "zero_mb_run": False,
    }
    common._write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
