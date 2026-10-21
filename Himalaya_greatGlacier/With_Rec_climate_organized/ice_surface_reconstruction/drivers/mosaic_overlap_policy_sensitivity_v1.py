"""Compare two non-averaging system-grid mosaic overlap policies."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..paths import resolve
from .lia_3d_system_grid_pilot_v1 import run as run_system_grid
from .lia_system_grid import (
    INTERIOR_DISTANCE_POLICY,
    MAIN_MEMBER_FIRST_POLICY,
)


DEFAULT_SYSTEMS = {
    "RGI60-15.02709",
    "RGI60-15.07500",
    "RGI60-14.11844",
    "RGI60-15.02846",
}
POLICIES = (INTERIOR_DISTANCE_POLICY, MAIN_MEMBER_FIRST_POLICY)


def _slug(policy: str) -> str:
    return policy.replace("-", "_")


def _correct_policy_metadata(output_dir: Path, policy: str) -> None:
    """Make policy-specific copies self-describing after using the v1 runner."""
    audit_path = output_dir / "lia_system_grid_mosaic_audit_v1.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        audit["merge_policy"] = policy
        audit["overlap_resolution_policy"] = policy
        audit.to_csv(audit_path, index=False)
    summary_path = output_dir / "lia_system_grid_summary_v1.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["mosaic_overlap_policy"] = policy
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _collect_policy_metrics(
    output_dir: Path,
    policy: str,
) -> pd.DataFrame:
    results = pd.read_csv(output_dir / "lia_system_grid_results_v1.csv")
    records: list[dict] = []
    for row in results.itertuples(index=False):
        record = {
            "rgi_id": str(row.rgi_id),
            "lia_unique_id": int(row.lia_unique_id),
            "bed_scenario": str(row.bed_scenario),
            "policy": policy,
            "status": str(row.status),
            "volume_km3": float(
                getattr(row, "volume_km3_finite_components", np.nan)
            ),
            "mean_thickness_m": np.nan,
            "maximum_thickness_m": np.nan,
        }
        nc_path = output_dir / (
            f"lia_{record['lia_unique_id']}_{record['rgi_id']}_system_grid_v1.nc"
        )
        if nc_path.exists():
            with xr.open_dataset(nc_path) as dataset:
                member_ids = dataset["member_id"].astype(str).values.tolist()
                if record["bed_scenario"] in member_ids:
                    index = member_ids.index(record["bed_scenario"])
                    thickness = np.asarray(
                        dataset["thickness_m"].isel(member=index).values,
                        dtype=float,
                    )
                    mask = np.asarray(dataset["lia_mask"].values, dtype=bool)
                    values = thickness[mask & np.isfinite(thickness)]
                    if values.size:
                        record["mean_thickness_m"] = float(np.mean(values))
                        record["maximum_thickness_m"] = float(np.max(values))
        records.append(record)
    return pd.DataFrame(records)


def _comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["rgi_id", "lia_unique_id", "bed_scenario"]
    frames = []
    for policy in POLICIES:
        subset = metrics[metrics["policy"].eq(policy)].drop(columns="policy")
        frames.append(
            subset.rename(columns={
                column: f"{_slug(policy)}_{column}"
                for column in subset.columns
                if column not in keys
            })
        )
    comparison = frames[0].merge(frames[1], on=keys, how="outer", validate="one_to_one")
    primary = _slug(INTERIOR_DISTANCE_POLICY)
    alternate = _slug(MAIN_MEMBER_FIRST_POLICY)
    for quantity in ("volume_km3", "mean_thickness_m", "maximum_thickness_m"):
        left = comparison[f"{primary}_{quantity}"]
        right = comparison[f"{alternate}_{quantity}"]
        comparison[f"main_member_first_minus_interior_distance_{quantity}"] = right - left
        comparison[f"relative_change_{quantity}"] = (
            (right - left) / left.replace(0.0, np.nan)
        )
    comparison["status_match"] = (
        comparison[f"{primary}_status"] == comparison[f"{alternate}_status"]
    )
    return comparison


def run(config: dict, selected_rgi_ids: set[str] | None = None) -> dict:
    selected = selected_rgi_ids or DEFAULT_SYSTEMS
    root = resolve(config["output_dir"])
    sensitivity_dir = root / "mosaic_overlap_policy_sensitivity_v1"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pd.DataFrame] = []
    run_summaries: dict[str, dict] = {}
    for policy in POLICIES:
        policy_config = copy.deepcopy(config)
        policy_config.setdefault("lia_system_grid", {})[
            "overlap_resolution_policy"
        ] = policy
        relative_output = (
            f"mosaic_overlap_policy_sensitivity_v1/policy_{_slug(policy)}"
        )
        policy_config.setdefault("lia_3d_system_grid_pilot_v1", {})[
            "output_subdir"
        ] = relative_output
        run_summaries[policy] = run_system_grid(policy_config, selected)
        run_summaries[policy]["mosaic_overlap_policy"] = policy
        policy_output = root / relative_output
        _correct_policy_metadata(policy_output, policy)
        metric_frames.append(_collect_policy_metrics(policy_output, policy))

    metrics = pd.concat(metric_frames, ignore_index=True)
    comparison = _comparison_table(metrics)
    metrics.to_csv(
        sensitivity_dir / "mosaic_overlap_policy_metrics_v1.csv", index=False
    )
    comparison.to_csv(
        sensitivity_dir / "mosaic_overlap_policy_comparison_v1.csv", index=False
    )
    relative_volume = comparison["relative_change_volume_km3"].abs()
    statuses_match = bool(comparison["status_match"].fillna(False).all())
    successful_statuses = {"success", "success-with-documented-exclusion"}
    status_columns = [
        f"{_slug(policy)}_status" for policy in POLICIES
    ]
    all_successful = bool(
        comparison[status_columns]
        .apply(lambda column: column.astype(str).isin(successful_statuses))
        .all(axis=None)
    )
    maximum_relative_volume_change = (
        float(relative_volume.max()) if relative_volume.notna().any() else np.nan
    )
    payload = {
        "status": "completed-mosaic-overlap-policy-sensitivity-v1",
        "systems_requested": sorted(selected),
        "policies": list(POLICIES),
        "member_comparisons": int(len(comparison)),
        "all_statuses_match": statuses_match,
        "all_members_successful_under_both_policies": all_successful,
        "maximum_absolute_relative_volume_change": maximum_relative_volume_change,
        "median_absolute_relative_volume_change": (
            float(relative_volume.median()) if relative_volume.notna().any() else np.nan
        ),
        "maximum_absolute_mean_thickness_change_m": float(
            comparison["main_member_first_minus_interior_distance_mean_thickness_m"]
            .abs()
            .max()
        ),
        "maximum_absolute_maximum_thickness_change_m": float(
            comparison[
                "main_member_first_minus_interior_distance_maximum_thickness_m"
            ]
            .abs()
            .max()
        ),
        "candidate_freeze_threshold_relative_volume": 0.005,
        "candidate_freeze_result": (
            "supports-interior-distance-policy"
            if all_successful
            and statuses_match
            and np.isfinite(maximum_relative_volume_change)
            and maximum_relative_volume_change <= 0.005
            else "review-required"
        ),
        "run_summaries": run_summaries,
        "comparison_csv": "mosaic_overlap_policy_comparison_v1.csv",
        "metrics_csv": "mosaic_overlap_policy_metrics_v1.csv",
    }
    (sensitivity_dir / "mosaic_overlap_policy_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rgi-id", action="append", default=[])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, set(args.rgi_id) or None), indent=2))


if __name__ == "__main__":
    main()
