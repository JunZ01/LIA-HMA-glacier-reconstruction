"""Deterministic topology-only repair for invalid synthetic LIA outlines.

The immutable source outline is never edited.  Each requested invalid Polygon
is repaired in memory with Shapely ``make_valid`` and written as a one-feature
derived GeoPackage only after strict topology/area/hole/component gates pass.
The normal 25 m synthetic-system geometry hard gate is then run from that
derived feature.  This stage does not authorize inversion or dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely import make_valid
from shapely.validation import explain_validity

from ..paths import resolve
from . import synthetic_lia_system_geometry_preflight_v4 as geometry_v4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _component_count(geometry) -> int:
    return len(geometry.geoms) if hasattr(geometry, "geoms") else 1


def _hole_count(geometry) -> int:
    if geometry.geom_type == "Polygon":
        return len(geometry.interiors)
    if geometry.geom_type == "MultiPolygon":
        return sum(len(part.interiors) for part in geometry.geoms)
    return 0


def _source_hashes(outline_path: Path) -> dict[str, str]:
    if outline_path.suffix.lower() == ".shp":
        paths = sorted(outline_path.parent.glob(f"{outline_path.stem}.*"))
    else:
        paths = [outline_path]
    return {str(path.resolve()): _sha256(path) for path in paths if path.is_file()}


def _repair_one(config: dict, case: dict, output_root: Path) -> dict:
    source_config_path = resolve(case["source_geometry_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    system_id = int(case["system_id"])
    if int(source_config["system_id"]) != system_id:
        raise ValueError(f"source-config-system-mismatch:{system_id}")

    outline_path = resolve(source_config["lia_outline"])
    hashes_before = _source_hashes(outline_path)
    outlines = gpd.read_file(outline_path)
    selected = outlines.loc[outlines["UniqueID"].astype(int).eq(system_id)].copy()
    if len(selected) != 1:
        raise ValueError(f"lia-outline-row-count:{system_id}:{len(selected)}")
    source_geometry = selected.iloc[0].geometry
    repaired_geometry = make_valid(source_geometry)

    source_area = float(source_geometry.area)
    repaired_area = float(repaired_geometry.area)
    area_change = abs(repaired_area - source_area) / source_area
    audit = {
        "schema_version": 1,
        "experiment": "deterministic-invalid-lia-polygon-topology-repair-v1",
        "system_id": system_id,
        "source_outline": str(outline_path.resolve()),
        "source_geometry_type": source_geometry.geom_type,
        "source_is_valid": bool(source_geometry.is_valid),
        "source_validity_reason": explain_validity(source_geometry),
        "source_area_m2": source_area,
        "source_component_count": _component_count(source_geometry),
        "source_hole_count": _hole_count(source_geometry),
        "repair_method": "shapely.make_valid",
        "shapely_version": shapely.__version__,
        "repaired_geometry_type": repaired_geometry.geom_type,
        "repaired_is_valid": bool(repaired_geometry.is_valid),
        "repaired_validity_reason": explain_validity(repaired_geometry),
        "repaired_area_m2": repaired_area,
        "area_relative_change": area_change,
        "repaired_component_count": _component_count(repaired_geometry),
        "repaired_hole_count": _hole_count(repaired_geometry),
        "boundary_hausdorff_distance_m": float(
            source_geometry.boundary.hausdorff_distance(repaired_geometry.boundary)
        ),
        "source_sha256_before": hashes_before,
        "gates": config["topology_repair_gates"],
    }
    gates = config["topology_repair_gates"]
    repair_pass = bool(
        source_geometry.geom_type == "Polygon"
        and not source_geometry.is_valid
        and repaired_geometry.geom_type == "Polygon"
        and repaired_geometry.is_valid
        and _component_count(repaired_geometry) == 1
        and _component_count(source_geometry) == 1
        and _hole_count(repaired_geometry) == _hole_count(source_geometry)
        and area_change <= float(gates["maximum_area_relative_change"])
        and audit["boundary_hausdorff_distance_m"]
        <= float(gates["maximum_boundary_hausdorff_distance_m"])
    )
    audit["topology_repair_pass"] = repair_pass
    case_root = output_root / f"system_{system_id}"
    _write_json(case_root / "topology_repair_audit.json", audit)
    if not repair_pass:
        return {
            "system_id": system_id,
            "status": "topology-repair-fail",
            "topology_repair_pass": False,
            "geometry_run": False,
            "audit": str(case_root / "topology_repair_audit.json"),
        }

    derived_path = case_root / "derived_outline" / f"lia_{system_id}_make_valid_v1.gpkg"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    selected.geometry = [repaired_geometry]
    selected.to_file(derived_path, layer="lia_outline", driver="GPKG")

    child_config = dict(source_config)
    child_config["experiment"] = "synthetic-lia-system-topology-repaired-standard-geometry-v1"
    child_config["stage"] = "invalid-polygon-topology-repair-static-pilot"
    child_config["lia_outline"] = str(derived_path.resolve())
    child_config["output_root"] = str((case_root / "geometry").resolve())
    child_config["authorization"] = {
        "inversion": False,
        "zero_mb": False,
        "steady_50y": False,
        "historical": False,
    }
    child_config_path = case_root / "geometry_config.json"
    _write_json(child_config_path, child_config)
    geometry_summary = geometry_v4.v1.run(child_config_path)

    hashes_after = _source_hashes(outline_path)
    audit["source_sha256_after"] = hashes_after
    audit["source_products_unchanged"] = hashes_before == hashes_after
    audit["derived_outline"] = str(derived_path.resolve())
    audit["derived_outline_sha256"] = _sha256(derived_path)
    audit["geometry_summary"] = str(
        (case_root / "geometry" / "geometry_summary.json").resolve()
    )
    _write_json(case_root / "topology_repair_audit.json", audit)
    if hashes_before != hashes_after:
        raise RuntimeError(f"source-outline-mutation:{system_id}")
    return {
        "system_id": system_id,
        "status": geometry_summary["status"],
        "topology_repair_pass": True,
        "geometry_run": True,
        "geometry_review_flags": geometry_summary.get("review_flags", []),
        "root_length_relative_error": geometry_summary.get(
            "root_length_relative_error"
        ),
        "outline_area_relative_error": geometry_summary.get(
            "outline_area_relative_error"
        ),
        "downstream_free_length_m": geometry_summary.get(
            "downstream_free_length_m"
        ),
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
        "schema_version": 1,
        "experiment": "deterministic-invalid-lia-polygon-topology-repair-v1",
        "status": (
            "static-pilot-pass"
            if frame["status"].eq("geometry-pass").all()
            else "static-pilot-partial-or-fail"
        ),
        "system_count": len(frame),
        "topology_repair_pass_count": int(frame["topology_repair_pass"].sum()),
        "geometry_pass_count": int(frame["status"].eq("geometry-pass").sum()),
        "inversion_run": False,
        "zero_mb_run": False,
        "source_products_mutated": False,
        "systems": rows,
        "config": str(config_path.resolve()),
    }
    _write_json(output_root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
