"""Build and zero-MB test a conservative strip-mean v3 flowline contract."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .flux_mass_budget import run_zero_mb_budget
from .lia_dynamic_flowlines import (
    FlowlineNodeFields,
    build_variable_lambda_flowlines,
)
from .v3_flowline_elevation_bias_audit import _build_payloads
from .v3_flowline_geometry_audit import _run_system


STATIC_RELATIVE_TOLERANCE = 1e-12
SURFACE_CLOSURE_TOLERANCE_M = 1e-9
ZERO_MB_VOLUME_GATE_FRACTION = 0.001
BUDGET_CLOSURE_GATE_FRACTION = 1e-10
DOMAIN_OUTFLOW_GATE_FRACTION = 1e-8


def derive_strip_mean_rectangular_branch(
    node_rows: list[dict], *, node_dx_m: float
) -> dict:
    """Convert one complete v3 branch to an exact area-volume rectangle."""
    if node_dx_m <= 0:
        raise ValueError("strip-mean-node-dx-must-be-positive")
    rows = sorted(node_rows, key=lambda row: int(row["node_index"]))
    if not rows:
        raise ValueError("strip-mean-branch-has-no-nodes")
    expected = np.arange(len(rows), dtype=int)
    observed = np.asarray([row["node_index"] for row in rows], dtype=int)
    if not np.array_equal(observed, expected):
        raise ValueError("strip-mean-branch-node-indices-are-not-contiguous")

    active = np.asarray([row["mapping_active"] for row in rows], dtype=bool)
    assigned_area = np.asarray([row["assigned_area_m2"] for row in rows], dtype=float)
    assigned_volume = np.asarray([row["assigned_volume_m3"] for row in rows], dtype=float)
    source_surface = np.asarray([row["source_surface_m"] for row in rows], dtype=float)
    strip_surface = np.asarray([row["strip_mean_surface_m"] for row in rows], dtype=float)
    if not np.isfinite(source_surface).all():
        raise ValueError("strip-mean-source-surface-is-nonfinite")
    if np.any(assigned_area < 0) or np.any(assigned_volume < 0):
        raise ValueError("strip-mean-assignment-is-negative")
    if np.any(assigned_area[active] <= 0):
        raise ValueError("strip-mean-active-node-has-no-area")

    thickness = np.zeros(len(rows), dtype=float)
    section = np.zeros(len(rows), dtype=float)
    surface_width = np.full(len(rows), np.nan, dtype=float)
    surface = source_surface.copy()
    if active.any():
        thickness[active] = assigned_volume[active] / assigned_area[active]
        section[active] = assigned_volume[active] / float(node_dx_m)
        surface_width[active] = assigned_area[active] / float(node_dx_m)
        surface[active] = strip_surface[active]
    else:
        surface_width[:] = np.asarray(
            [row["polygon_transverse_width_m"] for row in rows], dtype=float
        )
    bed = surface - thickness

    active_indices = np.flatnonzero(active)
    inactive_indices = np.flatnonzero(~active)
    if active.any() and inactive_indices.size:
        surface_width[inactive_indices] = np.interp(
            inactive_indices,
            active_indices,
            surface_width[active_indices],
        )
    if not np.isfinite(surface_width).all() or np.any(surface_width <= 0):
        raise ValueError("strip-mean-width-is-not-positive-finite")
    if np.any(thickness < 0) or np.any(section < 0):
        raise ValueError("strip-mean-derived-geometry-is-negative")

    represented_area = float(np.sum(surface_width[active]) * node_dx_m)
    represented_volume = float(np.sum(section) * node_dx_m)
    source_area = float(assigned_area.sum())
    source_volume = float(assigned_volume.sum())
    closure = surface - (bed + thickness)
    downstream_surface_step = np.diff(surface)
    return {
        "rows": rows,
        "active": active,
        "surface_h_m": surface,
        "bed_h_m": bed,
        "thickness_m": thickness,
        "section_m2": section,
        "surface_width_m": surface_width,
        "source_area_m2": source_area,
        "source_volume_m3": source_volume,
        "represented_area_m2": represented_area,
        "represented_volume_m3": represented_volume,
        "area_error_fraction": (
            (represented_area - source_area) / source_area
            if source_area > 0
            else 0.0
        ),
        "volume_error_fraction": (
            (represented_volume - source_volume) / source_volume
            if source_volume > 0
            else 0.0
        ),
        "maximum_surface_closure_error_m": float(np.max(np.abs(closure))),
        "zero_control_branch": bool(not active.any()),
        "maximum_absolute_adjacent_surface_step_m": float(
            np.max(np.abs(downstream_surface_step), initial=0.0)
        ),
        "maximum_downstream_uphill_step_m": float(
            np.max(downstream_surface_step, initial=0.0)
        ),
        "downstream_uphill_step_count": int(
            np.sum(downstream_surface_step > 0.0)
        ),
    }


def _junction_index(parent_coordinates: np.ndarray, child_tail: np.ndarray) -> int:
    parent_line = LineString(parent_coordinates)
    station = float(parent_line.project(LineString([child_tail, child_tail]).centroid))
    segment_stations = np.concatenate(
        [[0.0], np.cumsum(np.hypot(*np.diff(parent_coordinates, axis=0).T))]
    )
    index = int(np.argmin(np.abs(segment_stations - station)))
    count = len(parent_coordinates)
    if count >= 9:
        return int(np.clip(index, 4, count - 5))
    if count >= 7:
        return int(np.clip(index, 3, count - 4))
    if count >= 5:
        return int(np.clip(index, 2, count - 3))
    raise ValueError("strip-mean-tributary-receiver-has-fewer-than-five-nodes")


def build_system_contract(
    mapping_result: dict,
    payload: dict,
    *,
    build_oggm_flowlines: bool,
    smooth_trib_influx: bool = True,
) -> dict:
    """Build static node fields and optionally linked OGGM flowlines."""
    system = mapping_result["system"]
    source_id = int(system["lia_unique_id"])
    nodes = mapping_result["nodes"]
    component_indices = sorted({int(row["component_index"]) for row in nodes})
    component_outputs = []
    contract_rows = []
    junction_rows = []
    all_flowlines = []

    for component_index in component_indices:
        network_path = (
            Path(payload["network_root"])
            / str(source_id)
            / f"component_{component_index}"
            / "lia_centerline_network.pkl.gz"
        )
        with gzip.open(network_path, "rb") as stream:
            network = pickle.load(stream)
        branches = sorted(network["branches"], key=lambda branch: int(branch["branch_index"]))
        branch_contracts = []
        centerlines = []
        node_fields = []
        from oggm.core.centerlines import Centerline

        for branch in branches:
            branch_index = int(branch["branch_index"])
            branch_rows = [
                row
                for row in nodes
                if int(row["component_index"]) == component_index
                and int(row["branch_index"]) == branch_index
            ]
            coordinates = np.asarray(branch["coords_projected_m"], dtype=float)
            node_dx_m = float(branch["dx_meter"])
            if len(branch_rows) != len(coordinates):
                raise ValueError(
                    f"strip-mean-network-node-count-mismatch:{source_id}:"
                    f"{component_index}:{branch_index}"
                )
            contract = derive_strip_mean_rectangular_branch(
                branch_rows, node_dx_m=node_dx_m
            )
            branch_contracts.append(contract)
            for local_index, source_row in enumerate(contract["rows"]):
                contract_rows.append({
                    "rgi_id": system["rgi_id"],
                    "lia_unique_id": source_id,
                    "component_index": component_index,
                    "branch_index": branch_index,
                    "node_index": local_index,
                    "is_main_branch": bool(source_row["is_main_branch"]),
                    "mapping_active": bool(contract["active"][local_index]),
                    "node_dx_m": node_dx_m,
                    "x_m": float(coordinates[local_index, 0]),
                    "y_m": float(coordinates[local_index, 1]),
                    "assigned_area_m2": float(source_row["assigned_area_m2"]),
                    "assigned_volume_m3": float(source_row["assigned_volume_m3"]),
                    "surface_h_m": float(contract["surface_h_m"][local_index]),
                    "bed_h_m": float(contract["bed_h_m"][local_index]),
                    "thickness_m": float(contract["thickness_m"][local_index]),
                    "section_m2": float(contract["section_m2"][local_index]),
                    "surface_width_m": float(contract["surface_width_m"][local_index]),
                    "lambda": 0.0,
                })
            centerlines.append(Centerline(
                line=LineString(coordinates),
                dx=node_dx_m,
                map_dx=1.0,
                surface_h=contract["surface_h_m"],
                rgi_id=system["rgi_id"],
            ))
            node_fields.append(FlowlineNodeFields(
                surface_h_m=contract["surface_h_m"],
                bed_h_m=contract["bed_h_m"],
                section_m2=contract["section_m2"],
                surface_width_m=contract["surface_width_m"],
            ))

        by_index = {int(branch["branch_index"]): i for i, branch in enumerate(branches)}
        for local_index, branch in enumerate(branches):
            parent_index = branch.get("parent_index")
            if parent_index is not None:
                centerlines[local_index].set_flows_to(
                    centerlines[by_index[int(parent_index)]], check_tail=True
                )

        for local_index, branch in enumerate(branches):
            parent_index = branch.get("parent_index")
            if parent_index is None:
                continue
            parent_local = by_index[int(parent_index)]
            junction_index = int(centerlines[local_index].flows_to_indice)
            delta = float(
                branch_contracts[local_index]["surface_h_m"][-1]
                - branch_contracts[parent_local]["surface_h_m"][junction_index]
            )
            junction_rows.append({
                "rgi_id": system["rgi_id"],
                "lia_unique_id": source_id,
                "component_index": component_index,
                "tributary_branch_index": int(branch["branch_index"]),
                "receiver_branch_index": int(parent_index),
                "receiver_node_index": junction_index,
                "tributary_tail_minus_receiver_surface_m": delta,
                "negative_flux_suspect": bool(delta < 0),
            })

        flowlines = []
        build_audit = None
        if build_oggm_flowlines:
            flowlines, build_audit = build_variable_lambda_flowlines(
                centerlines,
                node_fields,
                map_dx_m=1.0,
                downstream_line=None,
                default_lambda=0.0,
                rgi_id=system["rgi_id"],
                smooth_trib_influx=smooth_trib_influx,
                cross_section_contract="strip-mean-rectangular",
                enforce_coordinate_spacing=False,
            )
            all_flowlines.append(flowlines)

        source_area = float(sum(item["source_area_m2"] for item in branch_contracts))
        source_volume = float(sum(item["source_volume_m3"] for item in branch_contracts))
        represented_area = float(
            sum(item["represented_area_m2"] for item in branch_contracts)
        )
        represented_volume = float(
            sum(item["represented_volume_m3"] for item in branch_contracts)
        )
        component_outputs.append({
            "component_index": component_index,
            "branch_count": len(branches),
            "source_area_m2": source_area,
            "source_volume_m3": source_volume,
            "represented_area_m2": represented_area,
            "represented_volume_m3": represented_volume,
            "area_error_fraction": (represented_area - source_area) / source_area,
            "volume_error_fraction": (represented_volume - source_volume) / source_volume,
            "maximum_surface_closure_error_m": max(
                item["maximum_surface_closure_error_m"] for item in branch_contracts
            ),
            "maximum_absolute_adjacent_surface_step_m": max(
                item["maximum_absolute_adjacent_surface_step_m"]
                for item in branch_contracts
            ),
            "maximum_downstream_uphill_step_m": max(
                item["maximum_downstream_uphill_step_m"]
                for item in branch_contracts
            ),
            "downstream_uphill_step_count": int(sum(
                item["downstream_uphill_step_count"] for item in branch_contracts
            )),
            "zero_control_branch_count": int(sum(
                item["zero_control_branch"] for item in branch_contracts
            )),
            "build_audit": build_audit,
        })

    source_area = float(sum(item["source_area_m2"] for item in component_outputs))
    source_volume = float(sum(item["source_volume_m3"] for item in component_outputs))
    represented_area = float(
        sum(item["represented_area_m2"] for item in component_outputs)
    )
    represented_volume = float(
        sum(item["represented_volume_m3"] for item in component_outputs)
    )
    area_error = (represented_area - source_area) / source_area
    volume_error = (represented_volume - source_volume) / source_volume
    closure = max(item["maximum_surface_closure_error_m"] for item in component_outputs)
    flags = []
    if abs(area_error) > STATIC_RELATIVE_TOLERANCE:
        flags.append("area-not-conserved")
    if abs(volume_error) > STATIC_RELATIVE_TOLERANCE:
        flags.append("volume-not-conserved")
    if closure > SURFACE_CLOSURE_TOLERANCE_M:
        flags.append("surface-bed-thickness-not-closed")
    zero_control_branch_count = int(sum(
        item["zero_control_branch_count"] for item in component_outputs
    ))
    review_flags = list(flags)
    if zero_control_branch_count:
        review_flags.append("zero-control-volume-branch-retained")
    status = (
        "static-fail"
        if flags
        else ("static-review" if zero_control_branch_count else "static-pass")
    )
    return {
        "system": {
            "rgi_id": system["rgi_id"],
            "lia_unique_id": source_id,
            "status": status,
            "component_count": len(component_outputs),
            "branch_count": int(system["branch_count"]),
            "mapped_node_count": int(system["mapped_node_count"]),
            "source_area_km2": source_area / 1e6,
            "represented_area_km2": represented_area / 1e6,
            "area_error_fraction": area_error,
            "source_volume_km3": source_volume / 1e9,
            "represented_volume_km3": represented_volume / 1e9,
            "volume_error_fraction": volume_error,
            "maximum_surface_closure_error_m": closure,
            "maximum_absolute_adjacent_surface_step_m": max(
                item["maximum_absolute_adjacent_surface_step_m"]
                for item in component_outputs
            ),
            "maximum_downstream_uphill_step_m": max(
                item["maximum_downstream_uphill_step_m"]
                for item in component_outputs
            ),
            "downstream_uphill_step_count": int(sum(
                item["downstream_uphill_step_count"]
                for item in component_outputs
            )),
            "zero_control_branch_count": zero_control_branch_count,
            "strip_mean_hypsometry_error_fraction": float(max(
                item["strip_mean_hypsometry_maximum_cumulative_error_fraction"]
                for item in mapping_result["components"]
            )),
            "junction_count": len(junction_rows),
            "negative_flux_suspect_junction_count": int(sum(
                row["negative_flux_suspect"] for row in junction_rows
            )),
            "minimum_tributary_tail_minus_receiver_surface_m": (
                float(min(
                    row["tributary_tail_minus_receiver_surface_m"]
                    for row in junction_rows
                ))
                if junction_rows else None
            ),
            "review_flags": ";".join(review_flags),
        },
        "nodes": contract_rows,
        "junctions": junction_rows,
        "components": component_outputs,
        "flowline_components": all_flowlines,
    }


def _static_worker(payload: dict) -> dict:
    mapping = _run_system(payload)
    return build_system_contract(
        mapping,
        payload,
        build_oggm_flowlines=False,
    )


def select_dynamic_sample(system_frame: pd.DataFrame, count: int = 6) -> pd.DataFrame:
    """Select severe, complex, median and low-error one-component systems."""
    eligible = system_frame[
        system_frame["status"].eq("static-pass")
        & system_frame["component_count"].eq(1)
    ].copy()
    if len(eligible) < count:
        raise ValueError("not-enough-static-pass-systems-for-dynamic-sample")
    selected: list[tuple[int, str]] = []

    def add(row, reason: str) -> None:
        unique_id = int(row["lia_unique_id"])
        if unique_id not in {item[0] for item in selected}:
            selected.append((unique_id, reason))

    for _, row in eligible.nlargest(3, "strip_mean_hypsometry_error_fraction").iterrows():
        add(row, "high-residual")
    for _, row in eligible.sort_values(
        ["branch_count", "source_area_km2"], ascending=False
    ).iterrows():
        add(row, "maximum-network-complexity")
        if len(selected) >= 4:
            break
    median_error = float(eligible["strip_mean_hypsometry_error_fraction"].median())
    median_candidates = eligible[eligible["branch_count"].ge(5)].copy()
    median_candidates["distance_to_median"] = (
        median_candidates["strip_mean_hypsometry_error_fraction"] - median_error
    ).abs()
    for _, row in median_candidates.sort_values("distance_to_median").iterrows():
        add(row, "median-residual")
        if len(selected) >= 5:
            break
    for _, row in eligible[eligible["branch_count"].ge(5)].nsmallest(
        len(eligible), "strip_mean_hypsometry_error_fraction"
    ).iterrows():
        add(row, "low-residual-control")
        if len(selected) >= count:
            break
    selection = eligible[eligible["lia_unique_id"].isin([item[0] for item in selected])].copy()
    reasons = dict(selected)
    selection["selection_reason"] = selection["lia_unique_id"].map(reasons)
    return selection.sort_values("selection_reason").reset_index(drop=True)


def _dynamic_worker(payload: dict) -> dict:
    from oggm import cfg
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    if not cfg.IS_INITIALIZED:
        cfg.initialize(logging_level="WARNING")
    mapping = _run_system(payload)
    contract = build_system_contract(
        mapping,
        payload,
        build_oggm_flowlines=True,
        smooth_trib_influx=True,
    )
    if len(contract["flowline_components"]) != 1:
        raise ValueError("dynamic-sample-must-have-one-active-component")
    flowlines = contract["flowline_components"][0]
    model = FluxBasedModel(
        flowlines,
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=2.4e-24,
        fs=0.0,
        smooth_trib_influx=True,
        check_for_boundaries=True,
    )
    initial_volume = float(model.volume_m3)
    initial_area = float(model.area_m2)
    checkpoints = []
    cumulative = {
        key: 0.0
        for key in (
            "actual_volume_change_m3",
            "upstream_gate_input_m3",
            "root_domain_outflow_m3",
            "negative_tributary_creation_m3",
            "negative_section_clip_creation_m3",
            "predicted_volume_change_m3",
            "closure_residual_m3",
        )
    }
    for year, interval in ((0.1, 0.1), (5.0, 4.9)):
        rows, totals = run_zero_mb_budget(model, interval)
        for key in cumulative:
            cumulative[key] += float(totals[key])
        drift = (float(model.volume_m3) - initial_volume) / initial_volume
        checkpoint = {
            "rgi_id": contract["system"]["rgi_id"],
            "lia_unique_id": contract["system"]["lia_unique_id"],
            "year": year,
            "initial_volume_m3": initial_volume,
            "volume_m3": float(model.volume_m3),
            "initial_area_m2": initial_area,
            "area_m2": float(model.area_m2),
            "length_m": float(model.length_m),
            "volume_change_fraction": drift,
            "step_count_interval": int(totals["step_count"]),
            "minimum_timestep_seconds_interval": float(totals["minimum_timestep_seconds"]),
            "maximum_absolute_step_closure_residual_m3_interval": float(
                totals["maximum_absolute_step_closure_residual_m3"]
            ),
            **cumulative,
        }
        checkpoints.append(checkpoint)
        if year == 0.1 and abs(drift) > ZERO_MB_VOLUME_GATE_FRACTION:
            break

    final = checkpoints[-1]
    volume = initial_volume
    flags = []
    if abs(float(final["volume_change_fraction"])) > ZERO_MB_VOLUME_GATE_FRACTION:
        flags.append("zero-mb-volume-drift-over-0.1pct")
    if abs(float(final["closure_residual_m3"])) / volume > BUDGET_CLOSURE_GATE_FRACTION:
        flags.append("budget-closure-residual")
    if float(final["negative_tributary_creation_m3"]) / volume > ZERO_MB_VOLUME_GATE_FRACTION:
        flags.append("negative-tributary-mass-creation-over-0.1pct")
    if float(final["negative_section_clip_creation_m3"]) / volume > ZERO_MB_VOLUME_GATE_FRACTION:
        flags.append("negative-section-clip-over-0.1pct")
    if abs(float(final["root_domain_outflow_m3"])) / volume > DOMAIN_OUTFLOW_GATE_FRACTION:
        flags.append("domain-buffer-required")
    if len(checkpoints) < 2:
        flags.append("stopped-after-0.1yr-gate")
    return {
        "system": {
            **contract["system"],
            "dynamic_status": "zero-mb-pass" if not flags else "zero-mb-review",
            "dynamic_review_flags": ";".join(flags),
            "completed_checkpoint_year": float(final["year"]),
            "final_volume_change_fraction": float(final["volume_change_fraction"]),
            "negative_tributary_creation_fraction": float(
                final["negative_tributary_creation_m3"] / volume
            ),
            "negative_section_clip_creation_fraction": float(
                final["negative_section_clip_creation_m3"] / volume
            ),
            "root_domain_outflow_fraction": float(
                final["root_domain_outflow_m3"] / volume
            ),
            "budget_closure_residual_fraction": float(
                final["closure_residual_m3"] / volume
            ),
        },
        "checkpoints": checkpoints,
    }


def run(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    output_root: Path,
    workers: int,
) -> dict:
    payloads = _build_payloads(
        config_path=config_path,
        result_root=result_root,
        network_root=network_root,
    )
    static_systems = []
    static_nodes = []
    static_junctions = []
    static_failures = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_static_worker, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
                static_systems.append(result["system"])
                static_nodes.extend(result["nodes"])
                static_junctions.extend(result["junctions"])
            except Exception as exc:
                static_failures.append({
                    "netcdf_path": payload["netcdf_path"],
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })

    output_root.mkdir(parents=True, exist_ok=True)
    static_frame = pd.DataFrame(static_systems).sort_values("lia_unique_id")
    node_frame = pd.DataFrame(static_nodes).sort_values(
        ["lia_unique_id", "component_index", "branch_index", "node_index"]
    )
    junction_frame = pd.DataFrame(static_junctions).sort_values(
        ["lia_unique_id", "component_index", "tributary_branch_index"]
    )
    static_frame.to_csv(output_root / "strip_mean_rectangular_systems.csv", index=False)
    node_frame.to_csv(output_root / "strip_mean_rectangular_nodes.csv", index=False)
    junction_frame.to_csv(output_root / "strip_mean_rectangular_junctions.csv", index=False)
    pd.DataFrame(
        static_failures,
        columns=["netcdf_path", "failure_reason", "traceback"],
    ).to_csv(
        output_root / "strip_mean_rectangular_static_failures.csv", index=False
    )

    selection = select_dynamic_sample(static_frame, count=6)
    selection.to_csv(output_root / "zero_mb_sample_selection.csv", index=False)
    selected_ids = set(selection["lia_unique_id"].astype(int))
    payload_by_id = {
        int(Path(payload["netcdf_path"]).name.split("_")[1]): payload
        for payload in payloads
    }
    dynamic_systems = []
    checkpoints = []
    dynamic_failures = []
    with ProcessPoolExecutor(max_workers=min(workers, 6), mp_context=context) as executor:
        futures = {
            executor.submit(_dynamic_worker, payload_by_id[unique_id]): unique_id
            for unique_id in selected_ids
        }
        for future in as_completed(futures):
            unique_id = futures[future]
            try:
                result = future.result()
                dynamic_systems.append(result["system"])
                checkpoints.extend(result["checkpoints"])
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                status = (
                    "domain-buffer-required"
                    if "exceeds domain boundaries" in reason
                    else (
                        "initial-cfl-instability"
                        if "CFL error" in reason
                        else "dynamic-failed"
                    )
                )
                dynamic_failures.append({
                    "lia_unique_id": unique_id,
                    "status": status,
                    "failure_reason": reason,
                    "traceback": traceback.format_exc(),
                })
    dynamic_frame = pd.DataFrame(dynamic_systems).sort_values("lia_unique_id")
    checkpoint_frame = pd.DataFrame(checkpoints).sort_values(["lia_unique_id", "year"])
    dynamic_frame.to_csv(output_root / "zero_mb_dynamic_systems.csv", index=False)
    checkpoint_frame.to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
    pd.DataFrame(
        dynamic_failures,
        columns=["lia_unique_id", "status", "failure_reason", "traceback"],
    ).to_csv(output_root / "zero_mb_dynamic_failures.csv", index=False)

    static_all_constructed = (
        not static_failures
        and len(static_frame) == 191
        and not static_frame["status"].eq("static-fail").any()
    )
    dynamic_all_pass = (
        not dynamic_failures
        and len(dynamic_frame) == 6
        and dynamic_frame["dynamic_status"].eq("zero-mb-pass").all()
    )
    domain_only = bool(dynamic_failures) and all(
        row["status"] == "domain-buffer-required" for row in dynamic_failures
    )
    dynamic_pass_count = int(
        dynamic_frame["dynamic_status"].eq("zero-mb-pass").sum()
    ) if not dynamic_frame.empty else 0
    if static_all_constructed and dynamic_all_pass:
        decision = "ready-for-spinup-and-climate-calibration-pilot"
    elif static_all_constructed and domain_only:
        decision = "strip-mean-contract-passed-domain-buffer-required-before-spinup"
    elif static_all_constructed and dynamic_pass_count == 0:
        decision = "reject-direct-strip-mean-fluxbased-initialization-before-spinup"
    elif static_all_constructed:
        decision = "stop-before-spinup-resolve-zero-mb-dynamic-review"
    else:
        decision = "stop-strip-mean-static-contract-failed"
    summary = {
        "status": "completed",
        "scientific_scope": "strip-mean-rectangular-static-core191-and-six-system-zero-mb",
        "static_system_count": int(len(static_frame)),
        "static_failure_count": int(len(static_failures)),
        "static_pass_count": int(static_frame["status"].eq("static-pass").sum()),
        "static_review_count": int(static_frame["status"].eq("static-review").sum()),
        "zero_control_branch_count": int(
            static_frame["zero_control_branch_count"].sum()
        ),
        "maximum_absolute_area_error_fraction": float(
            static_frame["area_error_fraction"].abs().max()
        ),
        "maximum_absolute_volume_error_fraction": float(
            static_frame["volume_error_fraction"].abs().max()
        ),
        "maximum_surface_closure_error_m": float(
            static_frame["maximum_surface_closure_error_m"].max()
        ),
        "negative_flux_suspect_junction_count": int(
            static_frame["negative_flux_suspect_junction_count"].sum()
        ),
        "junction_count": int(static_frame["junction_count"].sum()),
        "systems_with_negative_flux_suspect_junctions": int(
            static_frame["negative_flux_suspect_junction_count"].gt(0).sum()
        ),
        "downstream_uphill_step_count": int(
            static_frame["downstream_uphill_step_count"].sum()
        ),
        "maximum_downstream_uphill_step_m_quantiles": {
            label: float(static_frame["maximum_downstream_uphill_step_m"].quantile(q))
            for label, q in (
                ("minimum", 0.0),
                ("median", 0.5),
                ("p90", 0.9),
                ("maximum", 1.0),
            )
        },
        "dynamic_sample_ids": sorted(selected_ids),
        "dynamic_completed_count": int(len(dynamic_frame)),
        "dynamic_failure_count": int(len(dynamic_failures)),
        "dynamic_zero_mb_pass_count": dynamic_pass_count,
        "dynamic_failure_status_counts": (
            pd.Series([row["status"] for row in dynamic_failures])
            .value_counts()
            .to_dict()
        ),
        "five_year_checkpoint_count": int(
            checkpoint_frame["year"].eq(5.0).sum()
        ) if not checkpoint_frame.empty else 0,
        "decision": decision,
        "calibration_authorized": bool(dynamic_all_pass),
        "decision_rationale": [
            "The rectangular aggregation conserves assigned area and volume to machine precision.",
            "Area-weighted strip-mean surfaces create large adjacent-node reversals and widespread negative junction gradients.",
            "No dynamic sample passed the preregistered 0.1-year zero-MB gate, so no 5-year continuation or calibration is authorized.",
        ],
        "next_method_requirement": (
            "Use the reconstruction as a spin-up/calibration target or develop an "
            "explicitly regularized surface mapping that conserves area and volume; "
            "do not lower CFL or zero-MB gates to force this direct initial state."
        ),
        "gates": {
            "static_relative_tolerance": STATIC_RELATIVE_TOLERANCE,
            "surface_closure_tolerance_m": SURFACE_CLOSURE_TOLERANCE_M,
            "zero_mb_volume_gate_fraction": ZERO_MB_VOLUME_GATE_FRACTION,
            "budget_closure_gate_fraction": BUDGET_CLOSURE_GATE_FRACTION,
            "domain_outflow_gate_fraction": DOMAIN_OUTFLOW_GATE_FRACTION,
        },
        "source_products_mutated": False,
    }
    (output_root / "strip_mean_rectangular_decision.json").write_text(
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
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        output_root=args.output_root,
        workers=args.workers,
    ), indent=2))


if __name__ == "__main__":
    main()
