"""Build and zero-MB test conservative LIA elevation-band flowlines."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .lia_dynamic_matrix_pilot import _configure_oggm, load_source_state, write_json
from .lia_elevation_band import (
    build_elevation_band_state,
    build_rectangular_flowline,
)


def load_main_flowline_length_targets(path: Path) -> tuple[pd.Series, str]:
    """Load one spatial main-flowline length target per LIA system."""

    frame = pd.read_csv(path)
    if "lia_unique_id" not in frame.columns:
        raise ValueError("elevation-band-length-target-id-column-missing")
    candidates = [
        column
        for column in (
            "target_main_flowline_length_m",
            "main_centerline_inside_length_m",
        )
        if column in frame.columns
    ]
    if not candidates:
        raise ValueError("elevation-band-length-target-value-column-missing")
    column = candidates[0]
    selected = frame[["lia_unique_id", column]].copy()
    if selected["lia_unique_id"].duplicated().any():
        raise ValueError("elevation-band-length-target-ids-not-unique")
    selected[column] = pd.to_numeric(selected[column], errors="raise")
    if not np.all(np.isfinite(selected[column])) or np.any(selected[column] <= 0):
        raise ValueError("elevation-band-length-target-values-invalid")
    return selected.set_index("lia_unique_id")[column].astype(float), column


def _hypsometry_maximum_cumulative_error(state, bin_size_m: float = 100.0) -> float:
    source_z = state.band_surface_h_m
    source_area = state.band_area_m2
    node_z = state.node_surface_h_m
    node_area = state.node_width_m * state.node_dx_m
    lower = np.floor(min(np.min(source_z), np.min(node_z)) / bin_size_m) * bin_size_m
    upper = np.ceil(max(np.max(source_z), np.max(node_z)) / bin_size_m) * bin_size_m
    edges = np.arange(lower, upper + bin_size_m * 0.5, bin_size_m)
    source_hist, _ = np.histogram(source_z, bins=edges, weights=source_area)
    node_hist, _ = np.histogram(node_z, bins=edges, weights=node_area)
    source_cumulative = np.cumsum(source_hist) / np.sum(source_hist)
    node_cumulative = np.cumsum(node_hist) / np.sum(node_hist)
    return float(np.max(np.abs(source_cumulative - node_cumulative)))


def _run_zero_mb(flowline, checkpoints, *, glen_a: float, fs: float):
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import ScalarMassBalance

    model = SemiImplicitModel(
        [flowline],
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(glen_a),
        fs=float(fs),
        check_for_boundaries=True,
    )
    initial_volume = float(model.volume_m3)
    initial_area = float(model.area_m2)
    rows = []
    step_rows = []
    previous_year = 0.0
    cumulative_steps = 0
    minimum_timestep = np.inf
    for checkpoint in sorted(float(value) for value in checkpoints):
        target_time = checkpoint * cfg.SEC_IN_YEAR
        while model.t < target_time:
            before = float(model.volume_m3)
            used_dt = float(model.step(target_time - model.t))
            after = float(model.volume_m3)
            cumulative_steps += 1
            minimum_timestep = min(minimum_timestep, used_dt)
            step_rows.append(
                {
                    "start_year": float(model.yr - used_dt / cfg.SEC_IN_YEAR),
                    "end_year": float(model.yr),
                    "dt_seconds": used_dt,
                    "actual_volume_change_m3": after - before,
                }
            )
        last = model.fls[-1]
        positive = np.flatnonzero(last.thick > 0)
        free_distance = (
            float((last.nx - positive[-1] - 1) * last.dx_meter)
            if len(positive)
            else float(last.nx * last.dx_meter)
        )
        rows.append(
            {
                "year": checkpoint,
                "interval_start_year": previous_year,
                "volume_m3": float(model.volume_m3),
                "area_m2": float(model.area_m2),
                "length_m": float(model.length_m),
                "volume_change_fraction_from_initial": (
                    (float(model.volume_m3) - initial_volume) / initial_volume
                ),
                "area_change_fraction_from_initial": (
                    (float(model.area_m2) - initial_area) / initial_area
                ),
                "step_count": cumulative_steps,
                "minimum_timestep_seconds": float(minimum_timestep),
                "downstream_free_distance_m": free_distance,
                "terminal_thickness_m": float(last.thick[-1]),
            }
        )
        previous_year = checkpoint
    return model, rows, step_rows


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    _configure_oggm(config)
    pilot = config["lia_elevation_band_pilot"]
    source_dir = resolve(config["output_dir"]) / str(
        config["lia_3d_system_grid_pilot_v1"]["output_subdir"]
    )
    output_dir = resolve(config["output_dir"]) / str(pilot["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    length_targets = None
    length_target_column = None
    if pilot.get("main_flowline_length_targets_path") is not None:
        length_targets, length_target_column = load_main_flowline_length_targets(
            resolve(pilot["main_flowline_length_targets_path"])
        )
    results = []
    failures = []
    for unique_id in [int(value) for value in pilot["system_ids"]]:
        started = time.perf_counter()
        try:
            candidates = sorted(
                source_dir.glob(f"lia_{unique_id}_*_system_grid_v1.nc")
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"elevation-band-system-grid-file-count:{unique_id}:{len(candidates)}"
                )
            source = load_source_state(candidates[0], pilot["bed_scenario"])
            target_length_m = None
            if length_targets is not None:
                if unique_id not in length_targets.index:
                    raise ValueError(
                        f"elevation-band-length-target-missing:{unique_id}"
                    )
                target_length_m = float(length_targets.loc[unique_id])
            state = build_elevation_band_state(
                source.surface_h_m,
                source.bed_h_m,
                source.thickness_m,
                source.valid_mask,
                cell_dx_m=abs(float(source.transform.a)),
                cell_dy_m=abs(float(source.transform.e)),
                elevation_bin_size_m=float(pilot["elevation_bin_size_m"]),
                node_dx_m=float(pilot["node_dx_m"]),
                target_length_m=target_length_m,
            )
            flowline = build_rectangular_flowline(
                state,
                rgi_id=f"LIA-{unique_id}",
                downstream_buffer_m=float(pilot["downstream_buffer_m"]),
            )
            initial_area = float(flowline.area_m2)
            initial_volume = float(flowline.volume_m3)
            model, checkpoints, step_rows = _run_zero_mb(
                flowline,
                pilot["zero_mb_checkpoints_years"],
                glen_a=float(pilot["glen_a"]),
                fs=float(pilot["fs"]),
            )
            system_dir = output_dir / str(unique_id)
            system_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(system_dir / "elevation_band_flowline_lia.pkl.gz", "wb") as stream:
                pickle.dump([flowline], stream)
            pd.DataFrame(step_rows).to_csv(
                system_dir / "semiimplicit_zero_mb_steps.csv", index=False
            )
            audit = {
                "lia_unique_id": unique_id,
                "rgi_id": source.rgi_id,
                "bed_scenario": pilot["bed_scenario"],
                "elevation_bin_size_m": float(pilot["elevation_bin_size_m"]),
                "node_dx_m": float(pilot["node_dx_m"]),
                "band_count": int(len(state.band_area_m2)),
                "ice_node_count": int(len(state.node_width_m)),
                "total_node_count_with_buffer": int(flowline.nx),
                "pseudo_flowline_length_m": float(
                    len(state.node_width_m) * state.node_dx_m
                ),
                "unscaled_pseudo_flowline_length_m": float(
                    state.unscaled_total_length_m
                ),
                "target_main_flowline_length_m": float(
                    state.target_total_length_m
                ),
                "length_target_source_column": length_target_column,
                "longitudinal_scale_factor": float(
                    state.longitudinal_scale_factor
                ),
                "initial_length_error_m": float(
                    len(state.node_width_m) * state.node_dx_m
                    - state.target_total_length_m
                ),
                "source_area_m2": state.source_area_m2,
                "initial_flowline_area_m2": initial_area,
                "area_error_fraction": (
                    (initial_area - state.source_area_m2) / state.source_area_m2
                ),
                "source_volume_m3": state.source_volume_m3,
                "initial_flowline_volume_m3": initial_volume,
                "volume_error_fraction": (
                    (initial_volume - state.source_volume_m3) / state.source_volume_m3
                ),
                "hypsometry_maximum_cumulative_area_error_fraction": (
                    _hypsometry_maximum_cumulative_error(state)
                ),
                "maximum_initial_thickness_m": float(
                    np.max(state.node_thickness_m)
                ),
                "zero_mb_checkpoints": checkpoints,
                "wall_time_seconds": float(time.perf_counter() - started),
                "source_products_mutated": False,
            }
            write_json(system_dir / "elevation_band_audit.json", audit)
            final = checkpoints[-1]
            results.append(
                {
                    "lia_unique_id": unique_id,
                    "rgi_id": source.rgi_id,
                    "status": "ok",
                    "band_count": audit["band_count"],
                    "ice_node_count": audit["ice_node_count"],
                    "pseudo_flowline_length_m": audit["pseudo_flowline_length_m"],
                    "unscaled_pseudo_flowline_length_m": audit[
                        "unscaled_pseudo_flowline_length_m"
                    ],
                    "target_main_flowline_length_m": audit[
                        "target_main_flowline_length_m"
                    ],
                    "length_target_source_column": audit[
                        "length_target_source_column"
                    ],
                    "longitudinal_scale_factor": audit[
                        "longitudinal_scale_factor"
                    ],
                    "initial_length_error_m": audit["initial_length_error_m"],
                    "area_error_fraction": audit["area_error_fraction"],
                    "volume_error_fraction": audit["volume_error_fraction"],
                    "hypsometry_maximum_cumulative_area_error_fraction": audit[
                        "hypsometry_maximum_cumulative_area_error_fraction"
                    ],
                    "final_zero_mb_year": final["year"],
                    "final_zero_mb_volume_change_fraction": final[
                        "volume_change_fraction_from_initial"
                    ],
                    "final_zero_mb_area_change_fraction": final[
                        "area_change_fraction_from_initial"
                    ],
                    "minimum_timestep_seconds": final[
                        "minimum_timestep_seconds"
                    ],
                    "downstream_free_distance_m": final[
                        "downstream_free_distance_m"
                    ],
                    "terminal_thickness_m": final["terminal_thickness_m"],
                    "wall_time_seconds": audit["wall_time_seconds"],
                }
            )
        except Exception as exc:
            failure = {
                "lia_unique_id": unique_id,
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(limit=15),
            }
            failures.append(failure)
            results.append(failure)
        pd.DataFrame(results).to_csv(
            output_dir / "elevation_band_pilot_results.csv", index=False
        )
        if failures:
            pd.DataFrame(failures).to_csv(
                output_dir / "elevation_band_pilot_failures.csv", index=False
            )
    frame = pd.DataFrame(results)
    ok = frame["status"].eq("ok")
    passed = frame[ok]
    summary = {
        "status": "completed" if bool(ok.all()) else "completed-with-failures",
        "case_count": int(len(frame)),
        "success_count": int(ok.sum()),
        "failure_count": int((~ok).sum()),
        "static_geometry_gate": bool(
            len(passed)
            and np.all(np.abs(passed["area_error_fraction"]) <= 1e-12)
            and np.all(np.abs(passed["volume_error_fraction"]) <= 1e-12)
            and np.all(
                passed["hypsometry_maximum_cumulative_area_error_fraction"]
                <= float(pilot["hypsometry_gate_fraction"])
            )
            and np.all(
                np.abs(passed["initial_length_error_m"])
                <= float(pilot["node_dx_m"]) + 1e-6
            )
        ),
        "zero_mb_gate": bool(
            len(passed)
            and np.all(
                np.abs(passed["final_zero_mb_volume_change_fraction"])
                <= float(pilot["mass_conservation_gate_fraction"])
            )
            and np.all(passed["terminal_thickness_m"] <= 10.0)
        ),
        "source_products_mutated": False,
    }
    write_json(output_dir / "elevation_band_pilot_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--maximum-years", type=float)
    parser.add_argument("--output-subdir")
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument("--use-frozen-24-sample", action="store_true")
    parser.add_argument("--node-dx", type=float)
    parser.add_argument("--downstream-buffer", type=float)
    parser.add_argument("--main-flowline-length-targets", type=Path)
    parser.add_argument("--all-target-systems", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pilot = config["lia_elevation_band_pilot"]
    if args.use_frozen_24_sample:
        sample_path = (
            resolve(config["output_dir"])
            / str(config["lia_dynamic_fixed_bed_audit"]["output_subdir"])
            / "fixed_bed_audit_sample_v03.csv"
        )
        sample = pd.read_csv(sample_path)
        pilot["system_ids"] = [
            int(value) for value in sample["lia_unique_id"]
        ]
    if args.system_id:
        pilot["system_ids"] = list(dict.fromkeys(args.system_id))
    if args.node_dx is not None:
        if args.node_dx <= 0:
            raise ValueError("elevation-band-node-dx-must-be-positive")
        pilot["node_dx_m"] = float(args.node_dx)
    if args.downstream_buffer is not None:
        if args.downstream_buffer < 0:
            raise ValueError("elevation-band-downstream-buffer-must-be-nonnegative")
        pilot["downstream_buffer_m"] = float(args.downstream_buffer)
    if args.main_flowline_length_targets is not None:
        pilot["main_flowline_length_targets_path"] = str(
            args.main_flowline_length_targets.resolve()
        )
    if args.all_target_systems:
        if pilot.get("main_flowline_length_targets_path") is None:
            raise ValueError("all-target-systems-requires-length-targets")
        targets, _ = load_main_flowline_length_targets(
            resolve(pilot["main_flowline_length_targets_path"])
        )
        pilot["system_ids"] = [int(value) for value in targets.index]
    if args.maximum_years is not None:
        maximum = float(args.maximum_years)
        pilot["zero_mb_checkpoints_years"] = [maximum]
    if args.output_subdir:
        pilot["output_subdir"] = str(args.output_subdir)
    _, summary = run(config)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
