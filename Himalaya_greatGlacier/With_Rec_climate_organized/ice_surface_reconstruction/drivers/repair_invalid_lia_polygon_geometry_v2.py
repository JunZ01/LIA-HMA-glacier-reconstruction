"""Semantic-support topology repair for singly self-touching LIA Polygons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from affine import Affine
from rasterio.features import rasterize
from shapely import make_valid
from shapely.validation import explain_validity

from ..paths import resolve
from . import repair_invalid_lia_polygon_geometry_v1 as common
from . import synthetic_lia_system_geometry_preflight_v4 as geometry_v4


def _canonical_source_hashes(outline_path: Path) -> dict[str, str]:
    if outline_path.suffix.lower() == ".shp":
        allowed = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"}
        paths = sorted(
            path for path in outline_path.parent.glob(f"{outline_path.stem}.*")
            if path.suffix.lower() in allowed
        )
    else:
        paths = [outline_path]
    return {str(path.resolve()): common._sha256(path) for path in paths if path.is_file()}


def _raster_support(source_geometry, repaired_geometry, dx_m: float) -> dict:
    minx, miny, maxx, maxy = source_geometry.bounds
    minx, miny = np.floor(minx / dx_m) * dx_m, np.floor(miny / dx_m) * dx_m
    maxx, maxy = np.ceil(maxx / dx_m) * dx_m, np.ceil(maxy / dx_m) * dx_m
    width = int(round((maxx - minx) / dx_m))
    height = int(round((maxy - miny) / dx_m))
    transform = Affine(dx_m, 0, minx, 0, -dx_m, maxy)
    kwargs = dict(out_shape=(height, width), transform=transform, fill=0,
                  all_touched=False, dtype="uint8")
    source_mask = rasterize([(source_geometry, 1)], **kwargs)
    repaired_mask = rasterize([(repaired_geometry, 1)], **kwargs)
    changed = int(np.count_nonzero(source_mask != repaired_mask))
    return {
        "raster_support_dx_m": dx_m,
        "source_raster_cell_count": int(source_mask.sum()),
        "repaired_raster_cell_count": int(repaired_mask.sum()),
        "raster_support_changed_cell_count": changed,
        "raster_support_changed_area_m2": float(changed * dx_m * dx_m),
    }


def _repair_one(config: dict, case: dict, output_root: Path) -> dict:
    source_config_path = resolve(case["source_geometry_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    system_id = int(case["system_id"])
    if int(source_config["system_id"]) != system_id:
        raise ValueError(f"source-config-system-mismatch:{system_id}")
    outline_path = resolve(source_config["lia_outline"])
    hashes_before = _canonical_source_hashes(outline_path)
    outlines = gpd.read_file(outline_path)
    selected = outlines.loc[outlines["UniqueID"].astype(int).eq(system_id)].copy()
    if len(selected) != 1:
        raise ValueError(f"lia-outline-row-count:{system_id}:{len(selected)}")
    source_geometry = selected.iloc[0].geometry
    repaired_geometry = make_valid(source_geometry)
    source_area, repaired_area = float(source_geometry.area), float(repaired_geometry.area)
    gates = config["topology_repair_gates"]
    raster = _raster_support(source_geometry, repaired_geometry,
                             float(gates["raster_support_dx_m"]))
    hole_delta = common._hole_count(repaired_geometry) - common._hole_count(source_geometry)
    audit = {
        "schema_version": 2,
        "experiment": "deterministic-invalid-lia-polygon-topology-repair-v2",
        "system_id": system_id,
        "source_outline": str(outline_path.resolve()),
        "source_geometry_type": source_geometry.geom_type,
        "source_is_valid": bool(source_geometry.is_valid),
        "source_validity_reason": explain_validity(source_geometry),
        "source_area_m2": source_area,
        "source_component_count": common._component_count(source_geometry),
        "source_hole_count": common._hole_count(source_geometry),
        "repair_method": "shapely.make_valid",
        "shapely_version": shapely.__version__,
        "repaired_geometry_type": repaired_geometry.geom_type,
        "repaired_is_valid": bool(repaired_geometry.is_valid),
        "repaired_validity_reason": explain_validity(repaired_geometry),
        "repaired_area_m2": repaired_area,
        "area_relative_change": abs(repaired_area - source_area) / source_area,
        "repaired_component_count": common._component_count(repaired_geometry),
        "repaired_hole_count": common._hole_count(repaired_geometry),
        "explicit_hole_count_change": int(hole_delta),
        "hole_semantics": "one implicit self-touch loop becomes one explicit interior ring",
        "boundary_hausdorff_distance_m": float(source_geometry.boundary.hausdorff_distance(repaired_geometry.boundary)),
        **raster,
        "source_sha256_before": hashes_before,
        "gates": gates,
    }
    repair_pass = bool(
        source_geometry.geom_type == "Polygon" and not source_geometry.is_valid
        and repaired_geometry.geom_type == "Polygon" and repaired_geometry.is_valid
        and common._component_count(source_geometry) == 1
        and common._component_count(repaired_geometry) == 1
        and hole_delta == int(gates["required_explicit_hole_count_change"])
        and audit["area_relative_change"] <= float(gates["maximum_area_relative_change"])
        and audit["boundary_hausdorff_distance_m"] <= float(gates["maximum_boundary_hausdorff_distance_m"])
        and raster["raster_support_changed_cell_count"] <= int(gates["maximum_raster_support_changed_cells"])
    )
    audit["topology_repair_pass"] = repair_pass
    case_root = output_root / f"system_{system_id}"
    common._write_json(case_root / "topology_repair_audit.json", audit)
    if not repair_pass:
        return {"system_id": system_id, "status": "topology-repair-fail",
                "topology_repair_pass": False, "geometry_run": False,
                "audit": str(case_root / "topology_repair_audit.json")}

    derived_path = case_root / "derived_outline" / f"lia_{system_id}_make_valid_v2.gpkg"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    selected.geometry = [repaired_geometry]
    selected.to_file(derived_path, layer="lia_outline", driver="GPKG")
    child_config = dict(source_config)
    child_config.update({
        "experiment": "synthetic-lia-system-topology-repaired-standard-geometry-v2",
        "stage": "invalid-polygon-topology-repair-static-pilot-v2",
        "lia_outline": str(derived_path.resolve()),
        "output_root": str((case_root / "geometry").resolve()),
        "authorization": {"inversion": False, "zero_mb": False,
                          "steady_50y": False, "historical": False},
    })
    child_config_path = case_root / "geometry_config.json"
    common._write_json(child_config_path, child_config)
    geometry_summary = geometry_v4.v1.run(child_config_path)
    hashes_after = _canonical_source_hashes(outline_path)
    audit.update({
        "source_sha256_after": hashes_after,
        "source_products_unchanged": hashes_before == hashes_after,
        "derived_outline": str(derived_path.resolve()),
        "derived_outline_sha256": common._sha256(derived_path),
        "geometry_summary": str((case_root / "geometry" / "geometry_summary.json").resolve()),
    })
    common._write_json(case_root / "topology_repair_audit.json", audit)
    if hashes_before != hashes_after:
        raise RuntimeError(f"source-outline-mutation:{system_id}")
    return {
        "system_id": system_id, "status": geometry_summary["status"],
        "topology_repair_pass": True, "geometry_run": True,
        "geometry_review_flags": geometry_summary.get("review_flags", []),
        "root_length_relative_error": geometry_summary.get("root_length_relative_error"),
        "outline_area_relative_error": geometry_summary.get("outline_area_relative_error"),
        "downstream_free_length_m": geometry_summary.get("downstream_free_length_m"),
        "branch_count": geometry_summary.get("branch_count"),
        "audit": str(case_root / "topology_repair_audit.json"),
        "geometry_summary": audit["geometry_summary"],
    }


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [_repair_one(config, case, output_root) for case in config["cases"]]
    frame = pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)
    frame.to_csv(output_root / "system_summary.csv", index=False)
    payload = {
        "schema_version": 2,
        "experiment": "deterministic-invalid-lia-polygon-topology-repair-v2",
        "status": "static-pilot-pass" if frame["status"].eq("geometry-pass").all() else "static-pilot-partial-or-fail",
        "system_count": len(frame),
        "topology_repair_pass_count": int(frame["topology_repair_pass"].sum()),
        "geometry_pass_count": int(frame["status"].eq("geometry-pass").sum()),
        "inversion_run": False, "zero_mb_run": False,
        "source_products_mutated": False, "systems": rows,
        "config": str(config_path.resolve()),
    }
    common._write_json(output_root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
