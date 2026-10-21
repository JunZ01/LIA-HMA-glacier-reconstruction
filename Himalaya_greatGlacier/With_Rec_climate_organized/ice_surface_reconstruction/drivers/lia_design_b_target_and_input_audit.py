"""Freeze LIA design-B targets and audit modern climate/MB prerequisites."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from ..paths import resolve
from .lia_dynamic_matrix_pilot import load_source_state


QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
REQUIRED_GDIR_FILES = (
    "glacier_grid.json",
    "climate_historical.nc",
    "mb_calib.json",
    "inversion_flowlines.pkl",
)


def weighted_quantiles(values, weights, quantiles=QUANTILES) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError("design-b-target-has-no-valid-hypsometry")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(np.asarray(quantiles, dtype=float), cumulative, values)


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:8] / rgi_id[:11] / rgi_id


def _main_network_metrics(network_root: Path, unique_id: int) -> dict:
    candidates = sorted((network_root / str(unique_id)).glob("component_*/lia_centerline_network.pkl.gz"))
    if not candidates:
        raise FileNotFoundError(f"design-b-network-missing:{unique_id}")
    networks = []
    for path in candidates:
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        main_index = int(payload.get("main_branch_index", len(payload["branches"]) - 1))
        branch = next(
            item for item in payload["branches"]
            if int(item["branch_index"]) == main_index
        )
        coordinates = np.asarray(branch["coords_projected_m"], dtype=float)
        networks.append((LineString(coordinates).length, coordinates[-1]))
    length, outlet = max(networks, key=lambda item: item[0])
    return {
        "target_main_flowline_length_m": float(length),
        "target_outlet_x_m": float(outlet[0]),
        "target_outlet_y_m": float(outlet[1]),
    }


def _target_row(path: Path, network_root: Path) -> dict:
    source = load_source_state(path, "farinotti2019")
    positive = source.valid_mask & (source.thickness_m > 1e-6)
    surface = source.surface_h_m[positive]
    weights = np.full(surface.shape, float(source.cell_area_m2), dtype=float)
    quantile_values = weighted_quantiles(surface, weights)
    return {
        "lia_unique_id": int(source.lia_unique_id),
        "rgi_id": source.rgi_id,
        "target_area_km2": float(positive.sum() * source.cell_area_m2 / 1e6),
        "target_volume_km3": float(
            np.sum(source.thickness_m[positive]) * source.cell_area_m2 / 1e9
        ),
        "target_mean_surface_m": float(np.average(surface, weights=weights)),
        "target_minimum_surface_m": float(np.min(surface)),
        "target_maximum_surface_m": float(np.max(surface)),
        **{
            f"target_surface_q{int(q * 100):02d}_m": float(value)
            for q, value in zip(QUANTILES, quantile_values)
        },
        **_main_network_metrics(network_root, int(source.lia_unique_id)),
    }


def run(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    contract_nodes_path: Path,
    stage2_members_path: Path,
    timing_path: Path,
    climate_registry_path: Path,
    climate_pixels_path: Path,
    output_root: Path,
    climate_id: str = "ensemble_rec_ref",
    ensemble_role: str = "mid-turnover",
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = [
        _target_row(path, network_root)
        for path in sorted(result_root.glob("lia_*_*_system_grid_v1.nc"))
    ]
    target_frame = pd.DataFrame(targets).sort_values("lia_unique_id")
    if len(target_frame) != 191:
        raise ValueError(f"design-b-target-count:{len(target_frame)}:expected=191")

    nodes = pd.read_csv(contract_nodes_path)
    active = nodes[nodes["mapping_active"].astype(bool)].copy()
    branch_frame = (
        active.groupby(
            ["lia_unique_id", "rgi_id", "component_index", "branch_index", "is_main_branch"],
            as_index=False,
        )
        .agg(
            target_branch_area_m2=("assigned_area_m2", "sum"),
            target_branch_volume_m3=("assigned_volume_m3", "sum"),
            target_branch_node_count=("node_index", "size"),
            target_branch_minimum_surface_m=("surface_h_m", "min"),
            target_branch_maximum_surface_m=("surface_h_m", "max"),
        )
        .sort_values(["lia_unique_id", "component_index", "branch_index"])
    )
    branch_frame["target_branch_area_km2"] = branch_frame["target_branch_area_m2"] / 1e6
    branch_frame["target_branch_volume_km3"] = branch_frame["target_branch_volume_m3"] / 1e9

    stage2 = pd.read_csv(stage2_members_path)
    timing = pd.read_csv(timing_path)
    climate_registry = pd.read_csv(climate_registry_path)
    pixels = pd.read_csv(climate_pixels_path)
    climate_rows = climate_registry[climate_registry["climate_id"].astype(str).eq(climate_id)]
    if len(climate_rows) != 1:
        raise ValueError(f"design-b-climate-registry-count:{climate_id}:{len(climate_rows)}")
    climate = climate_rows.iloc[0]
    climate_path = Path(str(climate["path"]))
    modern_root = resolve(config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"])

    audit_rows = []
    for target in target_frame.itertuples(index=False):
        unique_id = int(target.lia_unique_id)
        parameters = stage2[
            stage2["lia_unique_id"].astype(int).eq(unique_id)
            & stage2["ensemble_role"].astype(str).eq(ensemble_role)
        ].copy()
        rgi_ids = sorted(parameters["rgi_id"].astype(str).unique())
        missing_gdir_files = []
        for rgi_id in rgi_ids:
            directory = gdir_path(modern_root, rgi_id)
            for name in REQUIRED_GDIR_FILES:
                if not (directory / name).exists():
                    missing_gdir_files.append(f"{rgi_id}:{name}")
        pixel_count = int(
            pixels[
                pixels["climate_id"].astype(str).eq(climate_id)
                & pixels["rgi_id"].astype(str).isin(rgi_ids)
            ]["rgi_id"].astype(str).nunique()
        )
        timing_rows = timing[timing["lia_unique_id"].astype(int).eq(unique_id)]
        target_year = None
        if len(timing_rows) == 1:
            target_year = int(round(0.5 * (
                float(timing_rows.iloc[0]["core_start_ce"])
                + float(timing_rows.iloc[0]["core_end_ce"])
            )))
        parameter_columns = ["melt_f", "temp_bias", "prcp_fac", "glen_a_pa_3_s_1", "fs"]
        finite_parameters = bool(
            len(parameters) > 0
            and np.isfinite(parameters[parameter_columns].to_numpy(dtype=float)).all()
        )
        flags = []
        if not rgi_ids:
            flags.append("stage2-mid-turnover-members-missing")
        if not finite_parameters:
            flags.append("stage2-parameters-nonfinite")
        if pixel_count != len(rgi_ids):
            flags.append("climate-pixel-coverage-incomplete")
        if missing_gdir_files:
            flags.append("modern-gdir-input-incomplete")
        if target_year is None:
            flags.append("lia-timing-missing")
        if not climate_path.exists():
            flags.append("climate-file-missing")
        audit_rows.append({
            "lia_unique_id": unique_id,
            "rgi_id": target.rgi_id,
            "ensemble_role": ensemble_role,
            "climate_id": climate_id,
            "member_rgi_count": len(rgi_ids),
            "member_rgi_ids": ";".join(rgi_ids),
            "stage2_parameter_row_count": len(parameters),
            "stage2_parameters_finite": finite_parameters,
            "climate_pixel_member_count": pixel_count,
            "modern_gdir_missing_file_count": len(missing_gdir_files),
            "modern_gdir_missing_files": ";".join(missing_gdir_files),
            "lia_target_year_ce": target_year,
            "climate_path": str(climate_path),
            "climate_file_exists": climate_path.exists(),
            "ready_for_design_b": not flags,
            "review_flags": ";".join(flags),
        })
    audit_frame = pd.DataFrame(audit_rows).sort_values("lia_unique_id")

    target_frame = target_frame.merge(
        audit_frame[["lia_unique_id", "lia_target_year_ce"]],
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    target_frame.to_csv(output_root / "lia_design_b_system_targets.csv", index=False)
    branch_frame.to_csv(output_root / "lia_design_b_branch_targets.csv", index=False)
    audit_frame.to_csv(output_root / "lia_design_b_input_audit.csv", index=False)
    flag_counts = (
        audit_frame["review_flags"].str.split(";").explode().loc[lambda values: values.ne("")].value_counts().to_dict()
    )
    summary = {
        "status": "completed",
        "system_target_count": int(len(target_frame)),
        "branch_target_count": int(len(branch_frame)),
        "design_b_ready_system_count": int(audit_frame["ready_for_design_b"].sum()),
        "design_b_review_system_count": int((~audit_frame["ready_for_design_b"]).sum()),
        "review_flag_counts": flag_counts,
        "climate_id": climate_id,
        "ensemble_role": ensemble_role,
        "climate_file": str(climate_path),
        "climate_file_exists": climate_path.exists(),
        "modern_gdir_root": str(modern_root),
        "source_products_mutated": False,
    }
    (output_root / "lia_design_b_target_input_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--contract-nodes", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--climate-registry", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        contract_nodes_path=args.contract_nodes,
        stage2_members_path=args.stage2_members,
        timing_path=args.timing,
        climate_registry_path=args.climate_registry,
        climate_pixels_path=args.climate_pixels,
        output_root=args.output_root,
    ), indent=2))


if __name__ == "__main__":
    main()
