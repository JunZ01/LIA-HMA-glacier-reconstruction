"""Freeze a morphology-stratified sample for the fixed-bed geometry audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ..paths import resolve


def _rectangle_elongation(geometry) -> float:
    coordinates = np.asarray(geometry.minimum_rotated_rectangle.exterior.coords)
    lengths = np.hypot(
        np.diff(coordinates[:, 0]), np.diff(coordinates[:, 1])
    )
    positive = lengths[lengths > 0]
    return float(np.max(positive) / np.min(positive))


def _standardized_features(frame: pd.DataFrame) -> np.ndarray:
    raw = np.column_stack(
        [
            np.log(frame["lia_area_km2"].to_numpy()),
            np.log1p(frame["branch_count"].to_numpy()),
            frame["compactness"].to_numpy(),
            np.log(frame["elongation"].to_numpy()),
        ]
    )
    median = np.median(raw, axis=0)
    scale = np.subtract(*np.percentile(raw, [75, 25], axis=0))
    scale[scale <= 0] = 1.0
    return (raw - median) / scale


def _maximin_region_sample(
    frame: pd.DataFrame,
    feature_matrix: np.ndarray,
    *,
    region: int,
    quota: int,
    forced_ids: set[int],
) -> list[int]:
    region_positions = np.flatnonzero(frame["rgi_region"].to_numpy() == region)
    if len(region_positions) < quota:
        raise ValueError(f"fixed-bed-sample-region-too-small:{region}")
    selected = [
        int(position)
        for position in region_positions
        if int(frame.iloc[position]["lia_unique_id"]) in forced_ids
    ]
    if len(selected) > quota:
        raise ValueError(f"too-many-forced-fixed-bed-systems-in-region:{region}")
    if not selected:
        distances_from_center = np.linalg.norm(
            feature_matrix[region_positions], axis=1
        )
        selected.append(
            int(region_positions[int(np.argmax(distances_from_center))])
        )
    while len(selected) < quota:
        candidates = np.array(
            [position for position in region_positions if position not in selected],
            dtype=int,
        )
        distances = np.linalg.norm(
            feature_matrix[candidates, None, :]
            - feature_matrix[np.asarray(selected), :][None, :, :],
            axis=2,
        )
        nearest = np.min(distances, axis=1)
        best_distance = np.max(nearest)
        tied = candidates[np.isclose(nearest, best_distance)]
        if len(tied) > 1:
            ids = frame.iloc[tied]["lia_unique_id"].to_numpy()
            selected.append(int(tied[int(np.argmin(ids))]))
        else:
            selected.append(int(tied[0]))
    return selected


def _geodetic_system_fields(row, geodetic: pd.DataFrame) -> dict:
    member_ids = [value.strip() for value in str(row.AllRGIs).split(";") if value]
    available = geodetic.reindex(member_ids).dropna(subset=["err_dmdtda", "area"])
    if available.empty:
        return {
            "geodetic_member_count": 0,
            "geodetic_member_coverage_fraction": 0.0,
            "geodetic_mb_mwea": np.nan,
            "geodetic_sigma_mwea": np.nan,
        }
    weights = available["area"].to_numpy(dtype=float)
    weights /= np.sum(weights)
    return {
        "geodetic_member_count": int(len(available)),
        "geodetic_member_coverage_fraction": len(available) / len(member_ids),
        "geodetic_mb_mwea": float(
            np.sum(weights * available["dmdtda"].to_numpy(dtype=float))
        ),
        # Conservative aggregation: member errors may share DEM systematics.
        "geodetic_sigma_mwea": float(
            np.sum(weights * available["err_dmdtda"].to_numpy(dtype=float))
        ),
    }


def build_sample(config: dict) -> tuple[pd.DataFrame, dict]:
    from oggm import cfg, utils

    audit = config["lia_dynamic_fixed_bed_audit"]
    candidates = pd.read_csv(resolve(config["pilot_selection_path"]))
    component_results = pd.read_csv(
        resolve(config["output_dir"])
        / str(config["lia_3d_system_grid_pilot_v1"]["output_subdir"])
        / "lia_system_component_results_v1.csv"
    )
    component_results = component_results[
        component_results["bed_scenario"].eq(audit["bed_scenario"])
        & component_results["status"].eq("success")
    ]
    branch_counts = component_results.groupby("lia_unique_id")["branch_count"].sum()

    outlines = gpd.read_file(resolve(config["lia_geometry_path"]))
    outlines = outlines[outlines["UniqueID"].isin(candidates["lia_unique_id"])].copy()
    outlines = outlines.to_crs(6933)
    outlines["lia_unique_id"] = outlines["UniqueID"].astype(int)
    outlines["lia_area_km2"] = outlines.geometry.area / 1e6
    outlines["compactness"] = (
        4.0 * np.pi * outlines.geometry.area / outlines.geometry.length**2
    )
    outlines["elongation"] = [
        _rectangle_elongation(geometry) for geometry in outlines.geometry
    ]
    outlines["branch_count"] = outlines["lia_unique_id"].map(branch_counts)
    outlines["rgi_region"] = (
        outlines["MainRGI"].str.split("-").str[1].str.split(".").str[0].astype(int)
    )
    if outlines["branch_count"].isna().any():
        raise ValueError("fixed-bed-sample-missing-branch-count")
    outlines = outlines.sort_values("lia_unique_id").reset_index(drop=True)
    features = _standardized_features(outlines)

    quotas = {int(key): int(value) for key, value in audit["region_quotas"].items()}
    forced_ids = {int(value) for value in audit["forced_system_ids"]}
    positions = []
    for region, quota in sorted(quotas.items()):
        positions.extend(
            _maximin_region_sample(
                outlines,
                features,
                region=region,
                quota=quota,
                forced_ids=forced_ids,
            )
        )
    sample = outlines.iloc[positions].copy()
    if len(sample) != int(audit["sample_size"]):
        raise ValueError("fixed-bed-sample-size-mismatch")
    if not forced_ids.issubset(set(sample["lia_unique_id"])):
        raise ValueError("fixed-bed-forced-system-missing")

    cfg.initialize(logging_level="ERROR")
    geodetic = utils.get_geodetic_mb_dataframe()
    geodetic = geodetic[geodetic["period"].eq(audit["geodetic_period"])]
    if geodetic.index.duplicated().any():
        raise ValueError("duplicate-geodetic-rgi-period")
    geodetic_rows = [
        _geodetic_system_fields(row, geodetic) for _, row in sample.iterrows()
    ]
    geodetic_frame = pd.DataFrame(geodetic_rows, index=sample.index)
    for column in geodetic_frame:
        sample[column] = geodetic_frame[column]

    sample["area_quantile"] = pd.qcut(
        outlines.set_index("lia_unique_id").loc[sample["lia_unique_id"]]["lia_area_km2"],
        4,
        labels=False,
        duplicates="drop",
    ).to_numpy()
    sample["selection_role"] = np.where(
        sample["lia_unique_id"].isin(forced_ids),
        "forced-dynamic-pilot",
        "regional-morphology-maximin",
    )
    columns = [
        "lia_unique_id",
        "MainRGI",
        "AllRGIs",
        "rgi_region",
        "lia_area_km2",
        "branch_count",
        "compactness",
        "elongation",
        "area_quantile",
        "geodetic_member_count",
        "geodetic_member_coverage_fraction",
        "geodetic_mb_mwea",
        "geodetic_sigma_mwea",
        "selection_role",
    ]
    sample = sample[columns].sort_values(["rgi_region", "lia_area_km2"])
    summary = {
        "status": "frozen-before-fixed-bed-geometry-audit",
        "sample_size": int(len(sample)),
        "selection_method": "regional-quota-maximin-standardized-log-area-log-branch-compactness-log-elongation",
        "random_sampling": False,
        "region_quotas": quotas,
        "forced_system_ids": sorted(forced_ids),
        "geodetic_period": audit["geodetic_period"],
        "geodetic_error_aggregation": "area-weighted-conservative-no-independence-assumption",
        "green_rule": "abs(width_induced_net_mb_bias) <= system_geodetic_sigma",
        "yellow_rule": "system_geodetic_sigma < abs_bias <= 2 * system_geodetic_sigma",
        "red_rule": "abs_bias > 2 * system_geodetic_sigma",
        "gate_semantics": "observational-relevance-not-guaranteed-melt-factor-absorption",
        "monthly_ti_status": "pending-modern-mb-calibration-input",
    }
    return sample, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample, summary = build_sample(config)
    output_dir = resolve(config["output_dir"]) / str(
        config["lia_dynamic_fixed_bed_audit"]["output_subdir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_dir / "fixed_bed_audit_sample_v03.csv", index=False)
    (output_dir / "fixed_bed_audit_gate_v03.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
