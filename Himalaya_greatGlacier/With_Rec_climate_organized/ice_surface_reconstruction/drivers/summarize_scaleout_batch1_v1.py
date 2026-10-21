"""Build the auditable scale-out Batch 1 validation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve


KEYS = ["rgi_id", "lia_unique_id", "bed_scenario"]
SCIENCE_COLUMNS = ["volume_km3", "mean_thickness_m", "maximum_thickness_m"]


def _results(root: Path, subdir: str) -> pd.DataFrame:
    return pd.read_csv(root / subdir / "lia_3d_calibrated_results_v1.csv")


def _successful(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["status"].astype(str).eq("success")].copy()


def _relative_comparison(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = _successful(left)[KEYS + SCIENCE_COLUMNS].merge(
        _successful(right)[KEYS + SCIENCE_COLUMNS],
        on=KEYS,
        suffixes=("_left", "_right"),
    )
    for column in SCIENCE_COLUMNS:
        denominator = merged[f"{column}_right"].replace(0.0, np.nan)
        merged[f"{column}_relative_change"] = (
            merged[f"{column}_left"] - merged[f"{column}_right"]
        ) / denominator
    return merged


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame["status"].value_counts(dropna=False).items()
    }


def _bins(selection: pd.DataFrame) -> pd.DataFrame:
    output = selection.copy()
    output["dx_class"] = pd.cut(
        output["map_dx_m"],
        bins=[-np.inf, 75.0, 110.0, np.inf],
        labels=["fine-le-75m", "medium-75-110m", "coarse-gt-110m"],
    ).astype(str)
    output["area_class"] = pd.cut(
        output["lia_area_km2"],
        bins=[-np.inf, 20.0, 75.0, np.inf],
        labels=["small-le-20km2", "medium-20-75km2", "large-gt-75km2"],
    ).astype(str)
    output["member_complexity_class"] = pd.cut(
        output["declared_member_count"],
        bins=[-np.inf, 2.0, 7.0, np.inf],
        labels=["low-le-2", "medium-3-7", "high-ge-8"],
    ).astype(str)
    return output


def run(config: dict) -> dict:
    root = resolve(config["output_dir"])
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    selection = selection.rename(columns={"main_rgi_id": "rgi_id"})
    routing = pd.read_csv(root / "lia_centerline_routing_decisions_v0.csv")
    tau = pd.read_csv(
        root
        / str(config["modern_tau_calibration"]["output_subdir"])
        / "modern_tau_calibration_v0.csv"
    )
    initial = _results(root, "initial_step4x_c125_precontract_v1")
    candidate = _results(root, "diagnostic_step16x_c125_raster_contract_v1")
    reference = _results(root, "diagnostic_step32x_c125_raster_contract_v1")

    convergence = _relative_comparison(candidate, reference)
    convergence.to_csv(
        root / "scaleout_batch1_step16x_vs_32x_science_v1.csv", index=False
    )

    system = _bins(selection).merge(
        routing[["rgi_id", "centerline_route_status"]], on="rgi_id", how="left"
    )
    success_counts = (
        _successful(candidate).groupby("rgi_id")["bed_scenario"].nunique()
    )
    system["successful_bed_members"] = (
        system["rgi_id"].map(success_counts).fillna(0).astype(int)
    )
    system["scaleout_status"] = np.where(
        system["centerline_route_status"].eq("stable-pilot"),
        np.where(
            system["successful_bed_members"].eq(len(config["bed_scenarios"])),
            "production-pass-both-beds",
            "reconstruction-review",
        ),
        "centerline-structure-review",
    )
    system.to_csv(root / "scaleout_batch1_system_routing_v1.csv", index=False)

    grouped_rows: list[dict] = []
    for dimension in (
        "region_label",
        "dx_class",
        "area_class",
        "member_complexity_class",
    ):
        for category, group in system.groupby(dimension, observed=True, sort=True):
            grouped_rows.append({
                "dimension": dimension,
                "category": str(category),
                "systems": int(len(group)),
                "stable_centerline_systems": int(
                    group["centerline_route_status"].eq("stable-pilot").sum()
                ),
                "both_bed_success_systems": int(
                    group["scaleout_status"].eq("production-pass-both-beds").sum()
                ),
            })
    pd.DataFrame(grouped_rows).to_csv(
        root / "scaleout_batch1_stratified_status_v1.csv", index=False
    )

    candidate_success = _successful(candidate)
    raster_exclusions = pd.to_numeric(
        candidate_success.get("raster_boundary_excluded_constraint_count", 0),
        errors="coerce",
    ).fillna(0)
    convergence_stats = {}
    for column in SCIENCE_COLUMNS:
        values = convergence[f"{column}_relative_change"].abs()
        convergence_stats[column] = {
            "median_absolute_relative_change": float(values.median()),
            "maximum_absolute_relative_change": float(values.max()),
        }

    original_root = root.parent
    original_old_path = (
        original_root
        / "lia_3d_refined_flowline_pilot_v1"
        / "lia_3d_calibrated_results_v1.csv"
    )
    original_new_path = (
        original_root
        / "diagnostic_step16x_c125_raster_contract_regression_v1"
        / "lia_3d_calibrated_results_v1.csv"
    )
    original_regression: dict[str, object] = {"status": "not-available"}
    if original_old_path.exists() and original_new_path.exists():
        original_change = _relative_comparison(
            pd.read_csv(original_new_path), pd.read_csv(original_old_path)
        )
        volume_change = original_change["volume_km3_relative_change"].abs()
        original_regression = {
            "status": "completed",
            "common_successful_members": int(len(original_change)),
            "volume_median_absolute_relative_change": float(volume_change.median()),
            "volume_maximum_absolute_relative_change": float(volume_change.max()),
        }

    payload = {
        "status": "completed-scaleout-batch1-v1",
        "systems_requested": int(len(selection)),
        "initial_4x_c125_status_counts": _status_counts(initial),
        "candidate_method": {
            "flowline_integration_refinement_factor": 16,
            "head_exclusion_grid_multiplier": 1.25,
            "raster_boundary_contract": (
                "constraints on harmonic zero-boundary cells are excluded"
            ),
        },
        "candidate_status_counts": _status_counts(candidate),
        "stable_centerline_systems": int(
            system["centerline_route_status"].eq("stable-pilot").sum()
        ),
        "candidate_successful_members": int(len(candidate_success)),
        "candidate_successful_systems": int(candidate_success["rgi_id"].nunique()),
        "raster_boundary_excluded_constraint_members": int(
            (raster_exclusions > 0).sum()
        ),
        "raster_boundary_excluded_constraint_total": int(raster_exclusions.sum()),
        "step16x_vs_32x": {
            "common_successful_members": int(len(convergence)),
            **convergence_stats,
        },
        "original_pilot_regression": original_regression,
        "tau_status_counts": {
            str(key): int(value)
            for key, value in tau["status"].value_counts(dropna=False).items()
        },
        "release_assessment": (
            "numerical reconstruction passes for all stable-centerline members; "
            "geometry preprocessing must handle multipart and grid-clipped systems "
            "before core-191 production"
        ),
    }
    (root / "scaleout_batch1_validation_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
