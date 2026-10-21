"""Build a deterministic 40-system scale-out batch from the core-191 set.

Batch 2 samples ten systems from each HMA region after excluding the original
pilot and Batch 1.  The slots deliberately span ordinary scale extremes and
geometry/system-grid risks; this driver only builds the batch artifacts and
does not run any reconstruction stage.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from ..paths import repo_root, resolve
from .lia_geometry_preflight import _gdir_path


ROUTING_PATH = (
    "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/"
    "stage2_freeze_stage3_routing_v0/stage3_routing_245.csv"
)
MANIFEST_PATH = (
    "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/"
    "system_manifest_work/system_manifest_245_v0.csv"
)
BATCH1_SELECTION = (
    "validation_data/results/paleo_surface_validation_v0/scaleout_batch1_v1/"
    "scaleout_batch1_selection_16.csv"
)
BATCH_SIZE_PER_REGION = 10


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _repair_flag(frame: pd.DataFrame) -> pd.Series:
    repair = frame["derived_geometry_repair"].fillna("none").astype(str).str.lower()
    return _truthy(frame["lia_source_invalid"]) | ~repair.isin({"", "none", "nan"})


def _main_grid_metrics(gdir_root: Path, rgi_id: str, geometry, source_crs) -> dict:
    grid = json.loads(
        (_gdir_path(gdir_root, rgi_id) / "glacier_grid.json").read_text(
            encoding="utf-8"
        )
    )
    grid_crs = CRS.from_user_input(grid["proj"])
    transformer = Transformer.from_crs(source_crs, grid_crs, always_xy=True)
    projected = shapely_transform(transformer.transform, geometry)
    repaired = projected.buffer(0)
    if repaired.is_empty or repaired.area <= 0:
        raise ValueError(f"empty-lia-geometry-after-repair:{rgi_id}")
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    nx, ny = map(int, grid["nxny"])
    x1 = x0 + (nx - 1) * dx
    y1 = y0 + (ny - 1) * dy
    footprint = box(
        min(x0, x1) - abs(dx) / 2,
        min(y0, y1) - abs(dy) / 2,
        max(x0, x1) + abs(dx) / 2,
        max(y0, y1) + abs(dy) / 2,
    )
    covered_area = repaired.intersection(footprint).area
    coverage = float(np.clip(covered_area / repaired.area, 0.0, 1.0))
    component_count = len(repaired.geoms) if repaired.geom_type == "MultiPolygon" else 1
    return {
        "map_dx_m": abs(dx),
        "lia_geometry_type": repaired.geom_type,
        "lia_component_count": int(component_count),
        "main_gdir_polygon_coverage_fraction": coverage,
        "main_gdir_outside_area_km2": float((repaired.area - covered_area) / 1e6),
    }


def _fallback_order(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "main_gdir_polygon_coverage_fraction",
            "main_gdir_outside_area_km2",
            "lia_component_count",
            "geometry_repair_flag",
            "declared_member_count",
            "lia_area_km2",
            "main_rgi_id",
        ],
        ascending=[True, False, False, False, False, False, True],
        kind="mergesort",
    )


def _select_region(group: pd.DataFrame, count: int = BATCH_SIZE_PER_REGION) -> pd.DataFrame:
    """Select unique systems for one region using fixed, auditable slots."""
    remaining = group.copy()
    chosen: list[dict] = []

    def pick(slot: str, candidates: pd.DataFrame, criterion: str) -> None:
        nonlocal remaining
        status = "matched"
        available = candidates[
            candidates.main_rgi_id.astype(str).isin(remaining.main_rgi_id.astype(str))
        ]
        if available.empty:
            available = _fallback_order(remaining)
            status = "fallback-no-eligible-special-case"
        if available.empty:
            return
        row = available.iloc[0].to_dict()
        row["batch_selection_slot"] = slot
        row["selection_criterion"] = criterion
        row["selection_criterion_status"] = status
        chosen.append(row)
        remaining = remaining[
            remaining.main_rgi_id.astype(str) != str(row["main_rgi_id"])
        ]

    pick(
        "multipolygon-risk",
        group[group["lia_component_count"] > 1].sort_values(
            ["lia_component_count", "lia_area_km2", "main_rgi_id"],
            ascending=[False, False, True], kind="mergesort",
        ),
        "largest repaired LIA component count",
    )
    pick(
        "geometry-repair-risk",
        group[group["geometry_repair_flag"]].sort_values(
            ["lia_source_invalid", "lia_polygon_parts", "lia_area_km2", "main_rgi_id"],
            ascending=[False, False, False, True], kind="mergesort",
        ),
        "manifest reports invalid source or derived geometry repair",
    )
    pick(
        "lowest-main-gdir-coverage",
        group.sort_values(
            ["main_gdir_polygon_coverage_fraction", "main_rgi_id"],
            ascending=[True, True], kind="mergesort",
        ),
        "lowest fraction of LIA polygon represented by main gdir",
    )
    pick(
        "largest-main-gdir-outside-area",
        group.sort_values(
            ["main_gdir_outside_area_km2", "main_rgi_id"],
            ascending=[False, True], kind="mergesort",
        ),
        "largest LIA area outside main gdir",
    )
    pick(
        "largest-modern-membership",
        group.sort_values(
            ["declared_member_count", "main_rgi_id"],
            ascending=[False, True], kind="mergesort",
        ),
        "largest declared modern member count",
    )
    pick(
        "largest-area",
        group.sort_values(
            ["lia_area_km2", "main_rgi_id"],
            ascending=[False, True], kind="mergesort",
        ),
        "largest LIA area",
    )
    pick(
        "smallest-area",
        group.sort_values(
            ["lia_area_km2", "main_rgi_id"],
            ascending=[True, True], kind="mergesort",
        ),
        "smallest LIA area",
    )
    pick(
        "coarsest-grid",
        group.sort_values(
            ["map_dx_m", "main_rgi_id"],
            ascending=[False, True], kind="mergesort",
        ),
        "coarsest main-gdir grid",
    )
    pick(
        "finest-grid",
        group.sort_values(
            ["map_dx_m", "main_rgi_id"],
            ascending=[True, True], kind="mergesort",
        ),
        "finest main-gdir grid",
    )
    median_area = float(group["lia_area_km2"].median())
    median_candidates = group.assign(
        _median_distance=(group["lia_area_km2"] - median_area).abs()
    ).sort_values(
        ["_median_distance", "main_rgi_id"], ascending=[True, True], kind="mergesort"
    )
    pick("median-area", median_candidates, "nearest to regional median LIA area")

    while len(chosen) < count and not remaining.empty:
        index = len(chosen) + 1
        pick(
            f"fallback-risk-{index:02d}",
            _fallback_order(remaining),
            "deterministic composite-risk fallback",
        )
    if len(chosen) != count:
        raise ValueError(
            f"region-has-insufficient-unseen-core-systems:{len(chosen)}<{count}"
        )
    return pd.DataFrame(chosen)


def _summary(selection: pd.DataFrame, selection_path: Path, config_path: Path) -> dict:
    per_region = {}
    for region, group in selection.groupby("region_code", sort=True):
        per_region[str(int(region))] = {
            "systems": int(len(group)),
            "lia_area_km2_min": float(group["lia_area_km2"].min()),
            "lia_area_km2_max": float(group["lia_area_km2"].max()),
            "map_dx_m_min": float(group["map_dx_m"].min()),
            "map_dx_m_max": float(group["map_dx_m"].max()),
            "declared_member_count_max": int(group["declared_member_count"].max()),
            "multipolygon_systems": int((group["lia_component_count"] > 1).sum()),
            "geometry_repair_systems": int(group["geometry_repair_flag"].sum()),
            "main_gdir_coverage_below_99pct": int(
                (group["main_gdir_polygon_coverage_fraction"] < 0.99).sum()
            ),
            "main_gdir_coverage_min": float(
                group["main_gdir_polygon_coverage_fraction"].min()
            ),
        }
    return {
        "status": "batch-built-not-run",
        "systems": int(len(selection)),
        "systems_per_region": {
            str(int(key)): int(value)
            for key, value in selection.groupby("region_code").size().items()
        },
        "selection_path": str(selection_path),
        "config_path": str(config_path),
        "original_pilot_excluded": True,
        "scaleout_batch1_excluded": True,
        "selection_is_unique": bool(selection["main_rgi_id"].is_unique),
        "selection_slots": selection["batch_selection_slot"].value_counts().to_dict(),
        "criterion_status_counts": selection[
            "selection_criterion_status"
        ].value_counts().to_dict(),
        "overall": {
            "lia_area_km2_range": [
                float(selection["lia_area_km2"].min()),
                float(selection["lia_area_km2"].max()),
            ],
            "map_dx_m_range": [
                float(selection["map_dx_m"].min()),
                float(selection["map_dx_m"].max()),
            ],
            "declared_member_count_range": [
                int(selection["declared_member_count"].min()),
                int(selection["declared_member_count"].max()),
            ],
            "multipolygon_systems": int((selection["lia_component_count"] > 1).sum()),
            "geometry_repair_systems": int(selection["geometry_repair_flag"].sum()),
            "main_gdir_coverage_below_99pct": int(
                (selection["main_gdir_polygon_coverage_fraction"] < 0.99).sum()
            ),
            "main_gdir_coverage_min": float(
                selection["main_gdir_polygon_coverage_fraction"].min()
            ),
            "main_gdir_outside_area_km2_max": float(
                selection["main_gdir_outside_area_km2"].max()
            ),
        },
        "per_region": per_region,
    }


def run(config: dict) -> dict:
    routing = pd.read_csv(repo_root() / ROUTING_PATH)
    manifest = pd.read_csv(repo_root() / MANIFEST_PATH)
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    batch1 = pd.read_csv(repo_root().parent.parent / BATCH1_SELECTION)
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    if "UniqueID" not in lia.columns:
        raise ValueError("LIA geometry must contain UniqueID")
    lia_by_id = lia.set_index("UniqueID")

    manifest_columns = [
        "lia_unique_id", "main_rgi_id", "lia_area_km2", "centroid_lon",
        "centroid_lat", "lia_source_invalid", "derived_geometry_repair",
        "lia_polygon_parts", "geometry_qc_status",
    ]
    core = routing[
        routing.farinotti_core_191.astype(str).str.lower().eq("true")
    ].merge(
        manifest[manifest_columns],
        on=["lia_unique_id", "main_rgi_id"],
        how="inner",
    )
    excluded_ids = set(pilot.main_rgi_id.astype(str)) | set(
        batch1.main_rgi_id.astype(str)
    )
    core = core[~core.main_rgi_id.astype(str).isin(excluded_ids)].copy()
    core["geometry_repair_flag"] = _repair_flag(core)

    gdir_root = resolve(config["stage0_gdir_root"])
    metrics: list[dict] = []
    for item in core.itertuples(index=False):
        uid = int(item.lia_unique_id)
        if uid not in lia_by_id.index:
            raise ValueError(f"missing-lia-polygon:{uid}")
        geometry = lia_by_id.loc[uid].geometry
        metrics.append(
            _main_grid_metrics(gdir_root, str(item.main_rgi_id), geometry, lia.crs)
        )
    core = pd.concat(
        [core.reset_index(drop=True), pd.DataFrame(metrics)], axis=1
    )
    core["main_gdir_coverage_risk"] = np.select(
        [
            core["main_gdir_polygon_coverage_fraction"] < 0.99,
            core["main_gdir_polygon_coverage_fraction"] < 0.999,
        ],
        ["high-below-99pct", "moderate-below-99.9pct"],
        default="low",
    )

    selected = pd.concat(
        [
            _select_region(group)
            for _, group in core.groupby("region_code", sort=True)
        ],
        ignore_index=True,
    )
    selected["selection_reason"] = (
        "core-191-scaleout-batch2-" + selected["batch_selection_slot"]
    )
    columns = [
        "lia_unique_id", "region_code", "batch_selection_slot",
        "selection_reason", "selection_criterion", "selection_criterion_status",
        "main_rgi_id", "region_label", "declared_member_count", "lia_area_km2",
        "map_dx_m", "centroid_lon", "centroid_lat", "lia_geometry_type",
        "lia_component_count", "lia_polygon_parts", "lia_source_invalid",
        "derived_geometry_repair", "geometry_repair_flag", "geometry_qc_status",
        "main_gdir_polygon_coverage_fraction", "main_gdir_outside_area_km2",
        "main_gdir_coverage_risk", "farinotti_core_191",
        "millan_scenario_available", "stage1_identifiability_status",
    ]
    selection = selected[columns].sort_values(
        ["region_code", "batch_selection_slot", "main_rgi_id"], kind="mergesort"
    )
    if len(selection) != 40 or not selection["main_rgi_id"].is_unique:
        raise ValueError("batch2-selection-must-contain-40-unique-systems")

    output_dir = resolve(config["output_dir"]) / "scaleout_batch2_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "scaleout_batch2_selection_40.csv"
    selection.to_csv(selection_path, index=False)

    batch_config = copy.deepcopy(config)
    batch_config["run_id"] = "lia-surface-scaleout-batch2-v1"
    batch_config["pilot_selection_path"] = str(selection_path)
    batch_config["output_dir"] = str(output_dir)
    batch_config["modern_tau_calibration"]["output_subdir"] = "modern_tau_calibration_v1"
    batch_config["lia_3d_zero_boundary_pilot"]["output_subdir"] = (
        "lia_3d_refined_flowline_pilot_v1"
    )
    batch_config["lia_3d_refined_flowline_pilot_v1"]["output_subdir"] = (
        "lia_3d_refined_flowline_pilot_v1"
    )
    batch_config["lia_3d_system_grid_pilot_v1"]["output_subdir"] = (
        "lia_3d_system_grid_pilot_v1"
    )
    config_path = output_dir / "lia_surface_scaleout_batch2_v1.json"
    config_path.write_text(json.dumps(batch_config, indent=2), encoding="utf-8")

    payload = _summary(selection, selection_path, config_path)
    (output_dir / "scaleout_batch2_selection_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    payload = run(json.loads(args.config.read_text(encoding="utf-8")))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
