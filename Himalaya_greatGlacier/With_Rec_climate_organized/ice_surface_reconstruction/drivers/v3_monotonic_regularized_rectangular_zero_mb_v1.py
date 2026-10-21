"""Zero-MB dynamic preflight for the Q1 regularized v3 network.

The Core-191 parent chains use SemiImplicitModel on one elevation-band
flowline.  Q1 retains the reconstructed tributary network; OGGM explicitly
rejects tributaries in SemiImplicitModel, so this structural pilot uses
FluxBasedModel and records that solver change in every result.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .flux_mass_budget import _step_budget
from .lia_dynamic_flowlines import FlowlineNodeFields, build_variable_lambda_flowlines


SCHEMA_VERSION = 1
SUPPORTED_SYSTEM_ID = 3850
VOLUME_DRIFT_GATE_FRACTION = 0.001
TERMINAL_THICKNESS_GATE_M = 10.0
CHECKPOINT_YEARS = (0.1, 5.0)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_static_preflight(preflight_root: Path, system_id: int) -> tuple[dict, pd.DataFrame]:
    summary_path = preflight_root / "static_preflight_summary.json"
    nodes_path = preflight_root / "regularized_nodes.csv"
    if not summary_path.exists() or not nodes_path.exists():
        raise FileNotFoundError("q1-v2-static-preflight-output-is-incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["system_id"]) != int(system_id):
        raise ValueError("q1-static-preflight-system-id-mismatch")
    if summary["status"] != "static-pass":
        raise ValueError("q1-static-preflight-did-not-pass")
    if int(summary.get("schema_version", -1)) != 2:
        raise ValueError("q1-zero-mb-requires-static-preflight-schema-v2")
    if bool(summary.get("source_products_mutated", True)):
        raise ValueError("q1-static-preflight-mutated-source-products")
    nodes = pd.read_csv(nodes_path)
    if set(nodes["lia_unique_id"].astype(int)) != {int(system_id)}:
        raise ValueError("q1-regularized-node-system-id-mismatch")
    if nodes["component_index"].astype(int).nunique() != 1:
        raise ValueError("q1-zero-mb-v1-requires-one-connected-component")
    return summary, nodes


def _build_flowlines(
    *,
    nodes: pd.DataFrame,
    network_path: Path,
    rgi_id: str,
) -> tuple[list, dict]:
    from oggm.core.centerlines import Centerline

    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    branches = sorted(network["branches"], key=lambda row: int(row["branch_index"]))
    centerlines = []
    fields = []
    for branch in branches:
        branch_index = int(branch["branch_index"])
        branch_nodes = nodes[
            nodes["branch_index"].astype(int).eq(branch_index)
        ].sort_values("node_index")
        coordinates = np.asarray(branch["coords_projected_m"], dtype=float)
        if len(branch_nodes) != len(coordinates):
            raise ValueError(
                f"q1-zero-mb-network-node-count-mismatch:{branch_index}"
            )
        expected = np.arange(len(coordinates), dtype=int)
        if not np.array_equal(branch_nodes["node_index"].to_numpy(dtype=int), expected):
            raise ValueError(
                f"q1-zero-mb-network-node-index-mismatch:{branch_index}"
            )
        surface = branch_nodes["regularized_surface_h_m"].to_numpy(dtype=float)
        bed = branch_nodes["regularized_bed_h_m"].to_numpy(dtype=float)
        section = branch_nodes["section_m2"].to_numpy(dtype=float)
        width = branch_nodes["surface_width_m"].to_numpy(dtype=float)
        arrays = (surface, bed, section, width)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError(f"q1-zero-mb-nonfinite-node-field:{branch_index}")
        if np.any(surface - bed < -1e-10) or np.any(section < -1e-10):
            raise ValueError(f"q1-zero-mb-negative-initial-geometry:{branch_index}")
        centerlines.append(Centerline(
            line=LineString(coordinates),
            dx=float(branch["dx_meter"]),
            map_dx=1.0,
            surface_h=surface,
            rgi_id=rgi_id,
        ))
        fields.append(FlowlineNodeFields(
            surface_h_m=surface,
            bed_h_m=bed,
            section_m2=section,
            surface_width_m=width,
        ))

    by_index = {
        int(branch["branch_index"]): index for index, branch in enumerate(branches)
    }
    for local_index, branch in enumerate(branches):
        parent = branch.get("parent_index")
        if parent is not None:
            centerlines[local_index].set_flows_to(
                centerlines[by_index[int(parent)]], check_tail=True
            )
    flowlines, audit = build_variable_lambda_flowlines(
        centerlines,
        fields,
        map_dx_m=1.0,
        downstream_line=None,
        default_lambda=0.0,
        rgi_id=rgi_id,
        smooth_trib_influx=True,
        cross_section_contract="strip-mean-rectangular",
        enforce_coordinate_spacing=False,
    )
    return flowlines, audit


def _model_state(model, *, year: float, initial_volume_m3: float) -> dict:
    root = [flowline for flowline in model.fls if flowline.flows_to is None]
    if len(root) != 1:
        raise ValueError(f"q1-zero-mb-root-count:{len(root)}")
    terminal_thickness = float(root[0].thick[-1])
    finite = bool(
        np.isfinite(model.volume_m3)
        and np.isfinite(model.area_m2)
        and np.isfinite(model.length_m)
        and all(
            np.isfinite(flowline.section).all()
            and np.isfinite(flowline.thick).all()
            for flowline in model.fls
        )
    )
    negative_section_count = int(sum(
        np.sum(np.asarray(flowline.section) < -1e-10)
        for flowline in model.fls
    ))
    volume_m3 = float(model.volume_m3)
    return {
        "year": float(year),
        "model_year": float(model.yr),
        "volume_m3": volume_m3,
        "volume_drift_fraction": (
            (volume_m3 - initial_volume_m3) / initial_volume_m3
        ),
        "area_m2": float(model.area_m2),
        "length_m": float(model.length_m),
        "root_terminal_thickness_m": terminal_thickness,
        "finite_geometry": finite,
        "negative_section_count": negative_section_count,
    }


def run(
    *,
    preflight_root: Path,
    network_root: Path,
    output_root: Path,
    system_id: int = SUPPORTED_SYSTEM_ID,
    glen_a: float = 7.375294604960076e-24,
    fs: float = 0.0,
) -> dict:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    static_summary, nodes = _load_static_preflight(preflight_root, system_id)
    rgi_id = str(static_summary["rgi_id"])
    component_index = int(nodes["component_index"].iloc[0])
    network_path = (
        network_root
        / str(system_id)
        / f"component_{component_index}"
        / "lia_centerline_network.pkl.gz"
    )
    if not network_path.exists():
        raise FileNotFoundError(f"q1-zero-mb-network-missing:{network_path}")
    flowlines, build_audit = _build_flowlines(
        nodes=nodes,
        network_path=network_path,
        rgi_id=rgi_id,
    )
    _write_json(output_root / "flowline_build_audit.json", build_audit)

    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    if not cfg.IS_INITIALIZED:
        cfg.initialize(logging_level="WARNING")
    model = FluxBasedModel(
        flowlines,
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(glen_a),
        fs=float(fs),
        smooth_trib_influx=True,
        check_for_boundaries=True,
    )
    initial_volume = float(model.volume_m3)
    initial_area = float(model.area_m2)
    static_volume = float(static_summary["regularized_represented_volume_m3"])
    static_area = float(static_summary["regularized_represented_area_m2"])
    build_volume_error = (initial_volume - static_volume) / static_volume
    build_area_error = (initial_area - static_area) / static_area

    step_rows: list[dict] = []
    checkpoints = [_model_state(model, year=0.0, initial_volume_m3=initial_volume)]
    runtime_error = None
    failure_traceback = None
    for target_year in CHECKPOINT_YEARS:
        target_time = float(target_year) * float(SEC_IN_YEAR)
        try:
            while model.t < target_time:
                row = _step_budget(model, target_time - float(model.t))
                payload = {
                    "step_index": len(step_rows),
                    "checkpoint_target_year": float(target_year),
                    **row.__dict__,
                }
                step_rows.append(payload)
            checkpoints.append(_model_state(
                model,
                year=float(target_year),
                initial_volume_m3=initial_volume,
            ))
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break

    step_frame = pd.DataFrame(step_rows)
    checkpoint_frame = pd.DataFrame(checkpoints)
    step_frame.to_csv(output_root / "zero_mb_steps.csv", index=False)
    checkpoint_frame.to_csv(output_root / "zero_mb_checkpoints.csv", index=False)

    flags = []
    if abs(build_volume_error) > 1e-12:
        flags.append("flowline-build-volume-mismatch")
    if abs(build_area_error) > 1e-12:
        flags.append("flowline-build-area-mismatch")
    if runtime_error is not None:
        lower_error = runtime_error.lower()
        if "cfl" in lower_error:
            flags.append("cfl-error")
        elif "boundar" in lower_error:
            flags.append("boundary-error")
        else:
            flags.append("numerical-error")
    if len(checkpoints) != len(CHECKPOINT_YEARS) + 1:
        flags.append("checkpoint-incomplete")
    for row in checkpoints:
        if abs(float(row["volume_drift_fraction"])) > VOLUME_DRIFT_GATE_FRACTION:
            flags.append(f"volume-drift-over-0.1pct-year-{row['year']:g}")
        if float(row["root_terminal_thickness_m"]) > TERMINAL_THICKNESS_GATE_M:
            flags.append(f"terminal-thickness-over-10m-year-{row['year']:g}")
        if not bool(row["finite_geometry"]):
            flags.append(f"nonfinite-geometry-year-{row['year']:g}")
        if int(row["negative_section_count"]):
            flags.append(f"negative-section-year-{row['year']:g}")
    flags = list(dict.fromkeys(flags))

    budget_totals = {}
    for key in (
        "actual_volume_change_m3",
        "upstream_gate_input_m3",
        "root_domain_outflow_m3",
        "negative_tributary_creation_m3",
        "negative_section_clip_creation_m3",
        "predicted_volume_change_m3",
        "closure_residual_m3",
    ):
        budget_totals[key] = (
            float(step_frame[key].sum()) if key in step_frame else 0.0
        )
    budget_totals.update({
        "step_count": int(len(step_rows)),
        "minimum_timestep_seconds": (
            float(step_frame["dt_seconds"].min())
            if not step_frame.empty else None
        ),
        "maximum_absolute_step_closure_residual_m3": (
            float(step_frame["closure_residual_m3"].abs().max())
            if not step_frame.empty else None
        ),
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "q1-monotonic-regularized-rectangular-zero-mb-v1",
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "review_flags": flags,
        "system_id": int(system_id),
        "rgi_id": rgi_id,
        "source_static_preflight": str(preflight_root.resolve()),
        "source_network": str(network_path.resolve()),
        "python_executable": sys.executable,
        "solver_change": True,
        "parent_solver": "SemiImplicitModel single elevation-band flowline",
        "q1_solver": "FluxBasedModel multi-tributary network",
        "solver_change_reason": (
            "OGGM SemiImplicitModel raises ValueError when len(flowlines)>1; "
            "Q1 retains all tributaries to preserve its structural contract"
        ),
        "mass_balance": "ScalarMassBalance(0.0)",
        "glen_a_pa_3_s_1": float(glen_a),
        "fs": float(fs),
        "smooth_trib_influx": True,
        "check_for_boundaries": True,
        "flowline_count": int(len(flowlines)),
        "initial_volume_m3": initial_volume,
        "initial_area_m2": initial_area,
        "static_to_model_volume_error_fraction": float(build_volume_error),
        "static_to_model_area_error_fraction": float(build_area_error),
        "completed_checkpoint_year": float(checkpoints[-1]["year"]),
        "checkpoint_count": int(len(checkpoints)),
        "budget_totals": budget_totals,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started),
        "gates": {
            "volume_drift_max_fraction": VOLUME_DRIFT_GATE_FRACTION,
            "root_terminal_thickness_max_m": TERMINAL_THICKNESS_GATE_M,
            "checkpoint_years": list(CHECKPOINT_YEARS),
            "cfl_error_allowed": False,
            "boundary_error_allowed": False,
            "finite_geometry_required": True,
            "negative_section_count": 0,
        },
        "source_products_mutated": False,
        "historical_chain_run": False,
        "historical_chain_authorized": False,
        "next_gate": (
            "q1-representative-static-preflights"
            if not flags else "close-q1-structural-mechanism"
        ),
    }
    _write_json(output_root / "zero_mb_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--system-id", type=int, default=SUPPORTED_SYSTEM_ID)
    parser.add_argument("--glen-a", type=float, default=7.375294604960076e-24)
    parser.add_argument("--fs", type=float, default=0.0)
    args = parser.parse_args()
    if int(args.system_id) != SUPPORTED_SYSTEM_ID:
        raise ValueError(f"q1-zero-mb-v1-supports-only-system-{SUPPORTED_SYSTEM_ID}")
    print(json.dumps(run(
        preflight_root=args.preflight_root,
        network_root=args.network_root,
        output_root=args.output_root,
        system_id=args.system_id,
        glen_a=args.glen_a,
        fs=args.fs,
    ), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
