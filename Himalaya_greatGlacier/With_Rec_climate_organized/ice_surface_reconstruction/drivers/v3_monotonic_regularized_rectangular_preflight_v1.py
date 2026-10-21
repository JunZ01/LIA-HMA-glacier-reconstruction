"""Static Q1 preflight for a monotonic conservative v3 flowline geometry.

This driver is deliberately separate from the rejected direct strip-mean
pipeline.  It changes no source product and runs no OGGM dynamics.  For one
connected flowline component it projects the strip-mean node surfaces onto
the joint branch/junction monotonic domain while retaining every node's
assigned area and volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from .lia_dynamic_matrix_pilot import load_source_state
from .v3_flowline_elevation_bias_audit import _build_payloads
from .v3_flowline_geometry_audit import (
    _run_system,
    maximum_cumulative_hypsometry_error_fraction,
)
from .v3_strip_mean_rectangular_pipeline import build_system_contract


SCHEMA_VERSION = 1
SUPPORTED_SYSTEM_ID = 3850
MONOTONIC_TOLERANCE_M = 1e-7
AREA_GATE_FRACTION = 0.005
VOLUME_GATE_FRACTION = 0.005
HYPSOMETRY_GATE_FRACTION = 0.02
HYPSOMETRY_BIN_SIZE_M = 100.0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _network_paths(payload: dict, component_indices: list[int]) -> list[Path]:
    source_id = int(
        Path(payload["netcdf_path"]).name.split("_")[1]
    )
    return [
        Path(payload["network_root"])
        / str(source_id)
        / f"component_{component_index}"
        / "lia_centerline_network.pkl.gz"
        for component_index in component_indices
    ]


def _solve_component(
    nodes: pd.DataFrame,
    junctions: pd.DataFrame,
    *,
    inactive_node_weight_m2: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Jointly enforce branch and tributary-to-receiver monotonicity."""
    if inactive_node_weight_m2 <= 0:
        raise ValueError("inactive-node-weight-must-be-positive")
    ordered = nodes.sort_values(["branch_index", "node_index"]).reset_index(
        drop=True
    )
    surface = ordered["surface_h_m"].to_numpy(dtype=float)
    area = ordered["assigned_area_m2"].to_numpy(dtype=float)
    active = ordered["mapping_active"].astype(bool).to_numpy()
    if not np.array_equal(active, area > 0):
        raise ValueError("mapping-active-area-state-mismatch")
    weights = np.where(active, area, float(inactive_node_weight_m2))
    weight_scale = float(np.median(weights[active]))
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        raise ValueError("active-node-weight-scale-is-invalid")
    weights /= weight_scale

    lookup = {
        (int(row.branch_index), int(row.node_index)): int(index)
        for index, row in ordered.iterrows()
    }
    edge_rows: list[dict] = []
    for branch_index, branch in ordered.groupby("branch_index", sort=True):
        indices = branch.index.to_numpy(dtype=int)
        for upstream, downstream in zip(indices[:-1], indices[1:]):
            edge_rows.append({
                "constraint_type": "within_branch",
                "branch_index": int(branch_index),
                "upstream_global_index": int(upstream),
                "downstream_global_index": int(downstream),
                "upstream_node_index": int(ordered.loc[upstream, "node_index"]),
                "downstream_node_index": int(ordered.loc[downstream, "node_index"]),
            })
    for _, row in junctions.sort_values("tributary_branch_index").iterrows():
        tributary = ordered[
            ordered["branch_index"].astype(int).eq(
                int(row["tributary_branch_index"])
            )
        ]
        if tributary.empty:
            raise ValueError("junction-tributary-is-missing")
        upstream = int(tributary.index[-1])
        downstream = lookup[
            (
                int(row["receiver_branch_index"]),
                int(row["receiver_node_index"]),
            )
        ]
        edge_rows.append({
            "constraint_type": "junction",
            "branch_index": int(row["tributary_branch_index"]),
            "receiver_branch_index": int(row["receiver_branch_index"]),
            "upstream_global_index": upstream,
            "downstream_global_index": downstream,
            "upstream_node_index": int(ordered.loc[upstream, "node_index"]),
            "downstream_node_index": int(ordered.loc[downstream, "node_index"]),
        })
    if not edge_rows:
        raise ValueError("monotonic-problem-has-no-constraints")

    matrix = np.zeros((len(edge_rows), len(ordered)), dtype=float)
    for edge_index, row in enumerate(edge_rows):
        matrix[edge_index, row["upstream_global_index"]] = 1.0
        matrix[edge_index, row["downstream_global_index"]] = -1.0
    result = minimize(
        lambda candidate: 0.5
        * float(np.sum(weights * np.square(candidate - surface))),
        surface,
        jac=lambda candidate: weights * (candidate - surface),
        constraints=[LinearConstraint(matrix, 0.0, np.inf)],
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )
    regularized = np.asarray(result.x, dtype=float)
    constraint_values = matrix @ regularized
    minimum_constraint = float(np.min(constraint_values))
    if not result.success:
        raise RuntimeError(
            f"monotonic-optimization-failed:{result.status}:{result.message}"
        )
    if minimum_constraint < -MONOTONIC_TOLERANCE_M:
        raise RuntimeError(
            "monotonic-optimization-constraint-violation:"
            f"{minimum_constraint}"
        )

    output = ordered.copy()
    output["original_surface_h_m"] = surface
    output["regularized_surface_h_m"] = regularized
    output["surface_adjustment_m"] = regularized - surface
    thickness = np.zeros(len(output), dtype=float)
    thickness[active] = (
        output.loc[active, "assigned_volume_m3"].to_numpy(dtype=float)
        / area[active]
    )
    output["thickness_m"] = thickness
    output["regularized_bed_h_m"] = regularized - thickness
    output["section_m2"] = output["assigned_volume_m3"].to_numpy(dtype=float) / output[
        "node_dx_m"
    ].to_numpy(dtype=float)
    original_width = output["surface_width_m"].to_numpy(dtype=float)
    regularized_width = original_width.copy()
    regularized_width[active] = area[active] / output.loc[
        active, "node_dx_m"
    ].to_numpy(dtype=float)
    output["surface_width_m"] = regularized_width

    edge_frame = pd.DataFrame(edge_rows)
    upstream = edge_frame["upstream_global_index"].to_numpy(dtype=int)
    downstream = edge_frame["downstream_global_index"].to_numpy(dtype=int)
    edge_frame["original_upstream_minus_downstream_surface_m"] = (
        surface[upstream] - surface[downstream]
    )
    edge_frame["regularized_upstream_minus_downstream_surface_m"] = (
        regularized[upstream] - regularized[downstream]
    )
    edge_frame["original_reverse_slope"] = edge_frame[
        "original_upstream_minus_downstream_surface_m"
    ].lt(-MONOTONIC_TOLERANCE_M)
    edge_frame["regularized_reverse_slope"] = edge_frame[
        "regularized_upstream_minus_downstream_surface_m"
    ].lt(-MONOTONIC_TOLERANCE_M)
    branch_edges = edge_frame[
        edge_frame["constraint_type"].eq("within_branch")
    ].copy()
    junction_edges = edge_frame[
        edge_frame["constraint_type"].eq("junction")
    ].copy()
    solver = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iteration_count": int(result.nit),
        "objective_value": float(result.fun),
        "constraint_count": int(len(edge_rows)),
        "minimum_constraint_value_m": minimum_constraint,
        "inactive_node_weight_m2": float(inactive_node_weight_m2),
        "active_weight_scale_m2": weight_scale,
    }
    return output, branch_edges, junction_edges, solver


def run(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    output_root: Path,
    system_id: int = SUPPORTED_SYSTEM_ID,
    bed_scenario: str = "farinotti2019",
) -> dict:
    if int(system_id) != SUPPORTED_SYSTEM_ID:
        raise ValueError(
            f"q1-v1-supports-only-system-{SUPPORTED_SYSTEM_ID}"
        )
    payloads = _build_payloads(
        config_path=config_path,
        result_root=result_root,
        network_root=network_root,
        selected_system_ids={int(system_id)},
    )
    payload = payloads[0]
    mapping = _run_system(payload)
    source = load_source_state(Path(payload["netcdf_path"]), bed_scenario)
    if int(mapping["system"]["lia_unique_id"]) != int(system_id):
        raise ValueError("q1-source-system-id-mismatch")
    base = build_system_contract(
        mapping,
        payload,
        build_oggm_flowlines=False,
    )
    if int(base["system"]["component_count"]) != 1:
        raise ValueError("q1-v1-requires-one-active-component")

    node_frame = pd.DataFrame(base["nodes"])
    junction_frame = pd.DataFrame(base["junctions"])
    component_index = int(node_frame["component_index"].iloc[0])
    component_nodes = node_frame[
        node_frame["component_index"].astype(int).eq(component_index)
    ].copy()
    component_junctions = junction_frame[
        junction_frame["component_index"].astype(int).eq(component_index)
    ].copy()
    regularized_nodes, branch_edges, junction_edges, solver = _solve_component(
        component_nodes,
        component_junctions,
        inactive_node_weight_m2=source.cell_area_m2,
    )

    active = regularized_nodes["mapping_active"].astype(bool).to_numpy()
    positive_source = source.valid_mask & (source.thickness_m > 1e-6)
    source_surface = source.surface_h_m[positive_source]
    source_cell_areas = np.full(
        source_surface.shape, source.cell_area_m2, dtype=float
    )
    represented_areas = regularized_nodes.loc[
        active, "assigned_area_m2"
    ].to_numpy(dtype=float)
    original_hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source_surface,
        source_area_m2=source_cell_areas,
        node_surface_m=regularized_nodes.loc[
            active, "original_surface_h_m"
        ].to_numpy(dtype=float),
        model_area_m2=represented_areas,
        bin_size_m=HYPSOMETRY_BIN_SIZE_M,
    )
    regularized_hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source_surface,
        source_area_m2=source_cell_areas,
        node_surface_m=regularized_nodes.loc[
            active, "regularized_surface_h_m"
        ].to_numpy(dtype=float),
        model_area_m2=represented_areas,
        bin_size_m=HYPSOMETRY_BIN_SIZE_M,
    )

    source_area_m2 = float(source_cell_areas.sum())
    source_volume_m3 = float(
        np.sum(source.thickness_m[positive_source]) * source.cell_area_m2
    )
    original_area_m2 = float(base["system"]["represented_area_km2"] * 1e6)
    original_volume_m3 = float(base["system"]["represented_volume_km3"] * 1e9)
    node_dx = regularized_nodes["node_dx_m"].to_numpy(dtype=float)
    new_area_m2 = float(np.sum(
        regularized_nodes.loc[active, "surface_width_m"].to_numpy(dtype=float)
        * node_dx[active]
    ))
    new_volume_m3 = float(np.sum(
        regularized_nodes["section_m2"].to_numpy(dtype=float) * node_dx
    ))
    area_error = (new_area_m2 - source_area_m2) / source_area_m2
    volume_error = (new_volume_m3 - source_volume_m3) / source_volume_m3
    adjustments = regularized_nodes.loc[
        active, "surface_adjustment_m"
    ].to_numpy(dtype=float)
    negative_thickness_count = int(
        regularized_nodes["thickness_m"].lt(-1e-12).sum()
    )
    branch_reverse_count = int(branch_edges["regularized_reverse_slope"].sum())
    junction_reverse_count = int(
        junction_edges["regularized_reverse_slope"].sum()
    )
    flags = []
    if abs(area_error) > AREA_GATE_FRACTION:
        flags.append("area-error-over-0.5pct")
    if abs(volume_error) > VOLUME_GATE_FRACTION:
        flags.append("volume-error-over-0.5pct")
    if regularized_hypsometry_error > HYPSOMETRY_GATE_FRACTION:
        flags.append("hypsometry-error-over-2pct")
    if branch_reverse_count:
        flags.append("within-branch-reverse-slope")
    if junction_reverse_count:
        flags.append("junction-reverse-slope")
    if negative_thickness_count:
        flags.append("negative-thickness")

    component_indices = sorted(
        regularized_nodes["component_index"].astype(int).unique().tolist()
    )
    input_paths = [config_path, Path(payload["netcdf_path"])] + _network_paths(
        payload, component_indices
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "q1-monotonic-regularized-rectangular-static-preflight-v1",
        "system_id": int(system_id),
        "bed_scenario": bed_scenario,
        "source_products_mutated": False,
        "oggm_dynamics_run": False,
        "inputs": [
            {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "static-pass" if not flags else "static-fail",
        "review_flags": flags,
        "system_id": int(system_id),
        "rgi_id": str(source.rgi_id),
        "component_count": 1,
        "branch_count": int(regularized_nodes["branch_index"].nunique()),
        "mapped_node_count": int(active.sum()),
        "source_area_m2": source_area_m2,
        "original_represented_area_m2": original_area_m2,
        "regularized_represented_area_m2": new_area_m2,
        "regularized_area_error_fraction": float(area_error),
        "source_volume_m3": source_volume_m3,
        "original_represented_volume_m3": original_volume_m3,
        "regularized_represented_volume_m3": new_volume_m3,
        "regularized_volume_error_fraction": float(volume_error),
        "hypsometry_bin_size_m": HYPSOMETRY_BIN_SIZE_M,
        "original_hypsometry_maximum_cumulative_error_fraction": float(
            original_hypsometry_error
        ),
        "regularized_hypsometry_maximum_cumulative_error_fraction": float(
            regularized_hypsometry_error
        ),
        "original_within_branch_reverse_slope_count": int(
            branch_edges["original_reverse_slope"].sum()
        ),
        "regularized_within_branch_reverse_slope_count": branch_reverse_count,
        "original_junction_reverse_slope_count": int(
            junction_edges["original_reverse_slope"].sum()
        ),
        "regularized_junction_reverse_slope_count": junction_reverse_count,
        "negative_thickness_count": negative_thickness_count,
        "maximum_absolute_surface_adjustment_m": float(
            np.max(np.abs(regularized_nodes["surface_adjustment_m"]))
        ),
        "active_area_weighted_mean_absolute_surface_adjustment_m": float(
            np.average(np.abs(adjustments), weights=represented_areas)
        ),
        "solver": solver,
        "gates": {
            "area_relative_error_max": AREA_GATE_FRACTION,
            "volume_relative_error_max": VOLUME_GATE_FRACTION,
            "effective_elevation_distribution_error_max": HYPSOMETRY_GATE_FRACTION,
            "downstream_reverse_slope_count": 0,
            "junction_reverse_slope_count": 0,
            "negative_thickness_count": 0,
            "monotonic_tolerance_m": MONOTONIC_TOLERANCE_M,
        },
        "source_products_mutated": False,
        "oggm_dynamics_run": False,
        "historical_chain_authorized": False,
        "next_gate": (
            "five-year-zero-mb-preflight"
            if not flags
            else "stop-q1-static-preflight-failed"
        ),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    regularized_nodes.to_csv(output_root / "regularized_nodes.csv", index=False)
    branch_edges.to_csv(output_root / "within_branch_slopes.csv", index=False)
    junction_edges.to_csv(output_root / "junction_slopes.csv", index=False)
    _write_json(output_root / "input_manifest.json", manifest)
    _write_json(output_root / "static_preflight_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--system-id", type=int, default=SUPPORTED_SYSTEM_ID)
    parser.add_argument("--bed-scenario", default="farinotti2019")
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        output_root=args.output_root,
        system_id=args.system_id,
        bed_scenario=args.bed_scenario,
    ), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
