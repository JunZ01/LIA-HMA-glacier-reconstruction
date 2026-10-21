"""Scan grid-aware LIA head exclusion rules on the refined-flowline pilot."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .lia_3d_zero_boundary_pilot import run as run_base_pilot


MULTIPLIERS = (0.75, 1.0, 1.25)
REFINEMENT_FACTOR = 4


def _label(value: float) -> str:
    return str(value).replace(".", "p")


def run(config: dict) -> dict:
    root = resolve(config["output_dir"])
    reference_path = (
        root / "lia_3d_refined_flowline_pilot_v1"
        / "lia_3d_calibrated_results_v1.csv"
    )
    reference = pd.read_csv(reference_path)
    reference = reference[
        (reference["head_exclusion_distance_m"] == 100.0)
        & reference.status.astype(str).eq("success")
    ][["rgi_id", "bed_scenario", "volume_km3"]].rename(
        columns={"volume_km3": "reference_100m_volume_km3"}
    )
    rows: list[dict] = []
    for multiplier in MULTIPLIERS:
        scan_config = copy.deepcopy(config)
        pilot = scan_config["lia_3d_zero_boundary_pilot"]
        subdir = f"lia_3d_grid_boundary_c{_label(multiplier)}_v1"
        pilot["output_subdir"] = subdir
        pilot["result_version"] = f"v1_c{_label(multiplier)}"
        pilot["flowline_integration_refinement_factor"] = REFINEMENT_FACTOR
        pilot["head_exclusion_grid_multiplier"] = multiplier
        results, _ = run_base_pilot(scan_config)
        for item in results.itertuples(index=False):
            if not np.isfinite(getattr(item, "head_exclusion_distance_m", np.nan)):
                continue
            rows.append({
                "multiplier": multiplier,
                "rgi_id": str(item.rgi_id),
                "lia_unique_id": int(item.lia_unique_id),
                "bed_scenario": str(item.bed_scenario),
                "map_dx_m": float(getattr(item, "map_dx_m", np.nan)),
                "effective_head_exclusion_m": float(item.head_exclusion_distance_m),
                "status": str(item.status),
                "volume_km3": float(item.volume_km3) if np.isfinite(getattr(item, "volume_km3", np.nan)) else np.nan,
                "excluded_constraint_count": getattr(item, "excluded_constraint_count", np.nan),
                "retained_constraint_count": getattr(item, "retained_constraint_count", np.nan),
                "minimum_retained_boundary_distance_m": getattr(item, "minimum_retained_boundary_distance_m", np.nan),
                "maximum_inner_edge_implied_slope_deg": getattr(item, "maximum_inner_edge_implied_slope_deg", np.nan),
                "conflict_cell_count": getattr(item, "conflict_cell_count", np.nan),
                "maximum_boundary_constraint_thickness_m": getattr(item, "maximum_boundary_constraint_thickness_m", np.nan),
            })
    frame = pd.DataFrame(rows).merge(
        reference, on=["rgi_id", "bed_scenario"], how="left"
    )
    frame["volume_relative_change_vs_100m"] = (
        frame["volume_km3"] - frame["reference_100m_volume_km3"]
    ) / frame["reference_100m_volume_km3"]
    output_dir = root / "lia_failure_diagnostics_v1"
    frame.to_csv(output_dir / "grid_aware_boundary_scan_v1.csv", index=False)
    aggregates = []
    for multiplier, group in frame.groupby("multiplier", sort=True):
        success = group[group.status.astype(str).eq("success")]
        science = success[np.isfinite(success.volume_relative_change_vs_100m)]
        aggregates.append({
            "multiplier": float(multiplier),
            "success_members": int(len(success)),
            "centerline_boundary_conflicts": int(group.status.astype(str).str.contains("centerline-boundary-conflict").sum()),
            "negative_start_thickness_failures": int(group.status.astype(str).str.contains("negative-start-thickness").sum()),
            "volume_change_comparison_members": int(len(science)),
            "volume_relative_change_median": float(science.volume_relative_change_vs_100m.median()) if not science.empty else np.nan,
            "volume_relative_change_max_abs": float(science.volume_relative_change_vs_100m.abs().max()) if not science.empty else np.nan,
            "maximum_inner_edge_implied_slope_deg": float(success.maximum_inner_edge_implied_slope_deg.max()) if not success.empty else np.nan,
        })
    eligible = [
        item for item in aggregates
        if item["centerline_boundary_conflicts"] == 0
    ]
    selected = min(eligible, key=lambda item: item["multiplier"]) if eligible else None
    payload = {
        "status": "completed-diagnostic-only",
        "rule": "max(100 m, multiplier * map_dx_m)",
        "multipliers": list(MULTIPLIERS),
        "aggregates": aggregates,
        "smallest_zero-conflict_multiplier": (
            selected["multiplier"] if selected is not None else None
        ),
        "production_fields_unchanged": True,
    }
    (output_dir / "grid_aware_boundary_scan_summary_v1.json").write_text(
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
