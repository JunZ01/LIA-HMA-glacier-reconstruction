"""Measure elevation bias introduced by the frozen v3 cell-to-node mapping."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd

from ..paths import resolve
from .v3_flowline_geometry_audit import _run_system


HYPSOMETRY_GATE_FRACTION = 0.02


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...],
) -> np.ndarray:
    """Return deterministic weighted quantiles for positive finite weights."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError("weighted-quantiles-have-no-valid-values")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(np.asarray(probabilities, dtype=float), cumulative, values)


def summarize_node_bias(node_rows: list[dict]) -> dict:
    """Summarize signed node-minus-strip-mean surface bias by mapped area."""
    frame = pd.DataFrame(node_rows)
    frame = frame[frame["mapping_active"].astype(bool)].copy()
    if frame.empty:
        raise ValueError("elevation-bias-has-no-mapped-nodes")
    area = frame["assigned_area_m2"].to_numpy(dtype=float)
    delta = frame["node_minus_strip_mean_surface_m"].to_numpy(dtype=float)
    strip_range = frame["strip_surface_range_m"].to_numpy(dtype=float)
    strip_std = frame["strip_surface_standard_deviation_m"].to_numpy(dtype=float)
    if not np.isfinite(area).all() or not np.isfinite(delta).all() or np.any(area <= 0):
        raise ValueError("elevation-bias-node-values-invalid")
    quantile_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    delta_quantiles = weighted_quantiles(delta, area, quantile_levels)
    range_quantiles = weighted_quantiles(strip_range, area, quantile_levels)
    standard_deviation_quantiles = weighted_quantiles(
        strip_std, area, quantile_levels
    )
    total_area = float(area.sum())
    return {
        "mapped_area_km2": total_area / 1e6,
        "node_minus_strip_mean_area_weighted_mean_m": float(
            np.average(delta, weights=area)
        ),
        "node_minus_strip_mean_area_weighted_absolute_mean_m": float(
            np.average(np.abs(delta), weights=area)
        ),
        "node_minus_strip_mean_area_weighted_rms_m": float(
            np.sqrt(np.average(np.square(delta), weights=area))
        ),
        "node_lower_than_strip_mean_area_fraction": float(area[delta < 0].sum() / total_area),
        "node_higher_than_strip_mean_area_fraction": float(area[delta > 0].sum() / total_area),
        **{
            f"node_minus_strip_mean_area_weighted_p{int(level * 100):02d}_m": float(value)
            for level, value in zip(quantile_levels, delta_quantiles)
        },
        **{
            f"strip_surface_range_area_weighted_p{int(level * 100):02d}_m": float(value)
            for level, value in zip(quantile_levels, range_quantiles)
        },
        **{
            f"strip_surface_standard_deviation_area_weighted_p{int(level * 100):02d}_m": float(value)
            for level, value in zip(quantile_levels, standard_deviation_quantiles)
        },
    }


def _audit_system(payload: dict) -> dict:
    result = _run_system(payload)
    component_rows = result["components"]
    baseline_error = max(
        row["observed_hypsometry_maximum_cumulative_error_fraction"]
        for row in component_rows
    )
    strip_mean_error = max(
        row["strip_mean_hypsometry_maximum_cumulative_error_fraction"]
        for row in component_rows
    )
    bias = summarize_node_bias(result["nodes"])
    system = result["system"]
    system_row = {
        "rgi_id": system["rgi_id"],
        "lia_unique_id": int(system["lia_unique_id"]),
        "component_count": int(system["component_count"]),
        "branch_count": int(system["branch_count"]),
        "mapped_node_count": int(system["mapped_node_count"]),
        "source_positive_area_km2": float(system["source_positive_area_km2"]),
        "centerline_node_hypsometry_error_fraction": float(baseline_error),
        "strip_mean_node_hypsometry_error_fraction": float(strip_mean_error),
        "strip_mean_error_change_fraction": float(strip_mean_error - baseline_error),
        "strip_mean_error_reduction_fraction": float(baseline_error - strip_mean_error),
        "centerline_node_passes_2pct": bool(baseline_error <= HYPSOMETRY_GATE_FRACTION),
        "strip_mean_node_passes_2pct": bool(strip_mean_error <= HYPSOMETRY_GATE_FRACTION),
        **bias,
    }
    node_columns = [
        "rgi_id",
        "lia_unique_id",
        "component_index",
        "branch_index",
        "node_index",
        "is_main_branch",
        "x_m",
        "y_m",
        "control_length_m",
        "assigned_cell_count",
        "assigned_area_m2",
        "source_surface_m",
        "strip_mean_surface_m",
        "node_minus_strip_mean_surface_m",
        "strip_surface_standard_deviation_m",
        "strip_surface_minimum_m",
        "strip_surface_maximum_m",
        "strip_surface_range_m",
        "distance_to_lia_boundary_m",
    ]
    mapped_nodes = [
        {column: row[column] for column in node_columns}
        for row in result["nodes"]
        if row["mapping_active"]
    ]
    return {"system": system_row, "nodes": mapped_nodes}


def _build_payloads(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    selected_system_ids: set[int] | None = None,
) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outlines = gpd.read_file(resolve(config["lia_geometry_path"]))
    outlines["UniqueID"] = outlines["UniqueID"].astype(int)
    geometry_by_id = outlines.set_index("UniqueID").geometry
    components = pd.read_csv(result_root / "lia_system_component_results_v1.csv")
    payloads = []
    for path in sorted(result_root.glob("lia_*_*_system_grid_v1.nc")):
        unique_id = int(path.name.split("_")[1])
        if selected_system_ids and unique_id not in selected_system_ids:
            continue
        excluded = components[
            components["lia_unique_id"].astype(int).eq(unique_id)
            & components["status"].astype(str).eq(
                "documented-small-component-exclusion"
            )
        ]
        payloads.append({
            "netcdf_path": str(path),
            "network_root": str(network_root),
            "geometry_wkb_hex": geometry_by_id.loc[unique_id].wkb_hex,
            "geometry_crs": outlines.crs.to_string(),
            "excluded_component_indices": excluded["component_index"].astype(int).tolist(),
        })
    expected = len(selected_system_ids) if selected_system_ids else 191
    if len(payloads) != expected:
        raise ValueError(
            f"elevation-bias-source-count:{len(payloads)}:expected={expected}"
        )
    return payloads


def _column_quantiles(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].dropna()
    return {
        label: float(values.quantile(probability))
        for label, probability in (
            ("minimum", 0.0),
            ("p10", 0.1),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p90", 0.9),
            ("maximum", 1.0),
        )
    }


def run(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    output_root: Path,
    workers: int,
    selected_system_ids: set[int] | None = None,
) -> dict:
    payloads = _build_payloads(
        config_path=config_path,
        result_root=result_root,
        network_root=network_root,
        selected_system_ids=selected_system_ids,
    )
    systems = []
    nodes = []
    failures = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_audit_system, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
                systems.append(result["system"])
                nodes.extend(result["nodes"])
            except Exception as exc:
                failures.append({
                    "netcdf_path": payload["netcdf_path"],
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })

    output_root.mkdir(parents=True, exist_ok=True)
    system_frame = pd.DataFrame(systems).sort_values(
        "strip_mean_error_reduction_fraction", ascending=False
    )
    node_frame = pd.DataFrame(nodes).sort_values(
        ["lia_unique_id", "component_index", "branch_index", "node_index"]
    )
    failure_frame = pd.DataFrame(failures)
    system_frame.to_csv(output_root / "v3_elevation_bias_systems.csv", index=False)
    node_frame.to_csv(output_root / "v3_elevation_bias_nodes.csv", index=False)
    if not failure_frame.empty:
        failure_frame.to_csv(output_root / "v3_elevation_bias_failures.csv", index=False)

    global_area = node_frame["assigned_area_m2"].to_numpy(dtype=float)
    global_delta = node_frame["node_minus_strip_mean_surface_m"].to_numpy(dtype=float)
    summary = {
        "status": "completed" if not failures else "completed-with-failures",
        "scientific_scope": "static-v3-elevation-bias-diagnostic-no-dynamics-no-source-mutation",
        "system_count": int(len(payloads)),
        "completed_system_count": int(len(system_frame)),
        "failure_count": int(len(failures)),
        "mapped_node_count": int(len(node_frame)),
        "hypsometry_bin_size_m": 100.0,
        "centerline_node_2pct_pass_count": int(
            system_frame["centerline_node_passes_2pct"].sum()
        ),
        "strip_mean_node_2pct_pass_count": int(
            system_frame["strip_mean_node_passes_2pct"].sum()
        ),
        "systems_improved_by_strip_mean_count": int(
            system_frame["strip_mean_error_reduction_fraction"].gt(0).sum()
        ),
        "systems_worsened_by_strip_mean_count": int(
            system_frame["strip_mean_error_reduction_fraction"].lt(0).sum()
        ),
        "global_node_minus_strip_mean_area_weighted_mean_m": float(
            np.average(global_delta, weights=global_area)
        ),
        "global_node_minus_strip_mean_area_weighted_absolute_mean_m": float(
            np.average(np.abs(global_delta), weights=global_area)
        ),
        "centerline_node_hypsometry_error_quantiles": _column_quantiles(
            system_frame, "centerline_node_hypsometry_error_fraction"
        ),
        "strip_mean_node_hypsometry_error_quantiles": _column_quantiles(
            system_frame, "strip_mean_node_hypsometry_error_fraction"
        ),
        "strip_mean_error_reduction_quantiles": _column_quantiles(
            system_frame, "strip_mean_error_reduction_fraction"
        ),
        "node_minus_strip_mean_area_weighted_mean_quantiles_m": _column_quantiles(
            system_frame, "node_minus_strip_mean_area_weighted_mean_m"
        ),
        "node_minus_strip_mean_area_weighted_absolute_mean_quantiles_m": _column_quantiles(
            system_frame, "node_minus_strip_mean_area_weighted_absolute_mean_m"
        ),
        "strip_surface_range_area_weighted_median_quantiles_m": _column_quantiles(
            system_frame, "strip_surface_range_area_weighted_p50_m"
        ),
        "source_products_mutated": False,
        "interpretation_limit": (
            "Replacing each strip by its mean elevation removes the signed first "
            "moment bias only; it does not preserve within-strip hypsometry."
        ),
    }
    (output_root / "v3_elevation_bias_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--system-id", action="append", type=int)
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        output_root=args.output_root,
        workers=args.workers,
        selected_system_ids=set(args.system_id) if args.system_id else None,
    ), indent=2))


if __name__ == "__main__":
    main()
