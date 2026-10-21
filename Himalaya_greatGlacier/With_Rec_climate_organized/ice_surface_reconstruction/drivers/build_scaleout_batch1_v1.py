"""Build a deterministic 16-system scale-out batch from the core-191 set."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

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
BATCH_SIZE_PER_REGION = 4


def _select_region(group: pd.DataFrame) -> pd.DataFrame:
    chosen: list[dict] = []
    remaining = group.copy()
    criteria = [
        ("largest-area", "lia_area_km2", False),
        ("coarsest-grid", "map_dx_m", False),
        ("largest-modern-membership", "declared_member_count", False),
        ("smallest-area", "lia_area_km2", True),
    ]
    for slot, column, ascending in criteria:
        candidates = remaining.sort_values(
            [column, "main_rgi_id"], ascending=[ascending, True]
        )
        if candidates.empty:
            continue
        row = candidates.iloc[0].to_dict()
        row["batch_selection_slot"] = slot
        chosen.append(row)
        remaining = remaining[
            remaining.main_rgi_id.astype(str) != str(row["main_rgi_id"])
        ]
    return pd.DataFrame(chosen)


def run(config: dict) -> dict:
    routing = pd.read_csv(repo_root() / ROUTING_PATH)
    manifest = pd.read_csv(repo_root() / MANIFEST_PATH)
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    core = routing[
        routing.farinotti_core_191.astype(str).str.lower().eq("true")
    ].merge(
        manifest[[
            "lia_unique_id", "main_rgi_id", "lia_area_km2",
            "centroid_lon", "centroid_lat",
        ]],
        on=["lia_unique_id", "main_rgi_id"],
        how="inner",
    )
    core = core[
        ~core.main_rgi_id.astype(str).isin(pilot.main_rgi_id.astype(str))
    ].copy()
    gdir_root = resolve(config["stage0_gdir_root"])
    dx = []
    for item in core.itertuples(index=False):
        grid = json.loads(
            (_gdir_path(gdir_root, str(item.main_rgi_id)) / "glacier_grid.json").read_text()
        )
        dx.append(abs(float(grid["dxdy"][0])))
    core["map_dx_m"] = dx
    selected = pd.concat(
        [_select_region(group) for _, group in core.groupby("region_code", sort=True)],
        ignore_index=True,
    )
    selected["selection_reason"] = (
        "core-191-scaleout-batch1-" + selected["batch_selection_slot"]
    )
    selection_columns = [
        "lia_unique_id", "region_code", "batch_selection_slot",
        "selection_reason", "main_rgi_id", "region_label",
        "declared_member_count", "lia_area_km2", "map_dx_m",
        "centroid_lon", "centroid_lat", "farinotti_core_191",
        "millan_scenario_available", "stage1_identifiability_status",
    ]
    selection = selected[selection_columns].sort_values(
        ["region_code", "batch_selection_slot", "main_rgi_id"]
    )
    output_dir = resolve(config["output_dir"]) / "scaleout_batch1_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "scaleout_batch1_selection_16.csv"
    selection.to_csv(selection_path, index=False)

    batch_config = copy.deepcopy(config)
    batch_config["run_id"] = "lia-surface-scaleout-batch1-v1"
    batch_config["pilot_selection_path"] = str(selection_path)
    batch_config["output_dir"] = str(output_dir)
    batch_config["modern_tau_calibration"]["output_subdir"] = "modern_tau_calibration_v1"
    batch_config["lia_3d_zero_boundary_pilot"]["output_subdir"] = "lia_3d_refined_flowline_pilot_v1"
    batch_config["lia_3d_refined_flowline_pilot_v1"]["output_subdir"] = "lia_3d_refined_flowline_pilot_v1"
    config_path = output_dir / "lia_surface_scaleout_batch1_v1.json"
    config_path.write_text(json.dumps(batch_config, indent=2), encoding="utf-8")
    payload = {
        "status": "batch-built-not-run",
        "systems": int(len(selection)),
        "systems_per_region": selection.groupby("region_code").size().to_dict(),
        "selection_path": str(selection_path),
        "config_path": str(config_path),
        "pilot_systems_excluded": True,
    }
    (output_dir / "scaleout_batch1_build_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
