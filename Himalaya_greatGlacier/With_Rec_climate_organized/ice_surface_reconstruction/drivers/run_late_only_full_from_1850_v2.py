"""Continue one registered late-only endpoint from a validated 1850 state."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from . import run_late_only_response_group_preflight_v1 as preflight
from .synthetic_lia_system_historical_v1 import _snapshot


PROJECT = Path(__file__).resolve().parents[4]


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _save_checkpoint(
    output: Path,
    model,
    frame: pd.DataFrame,
    year: int,
    accepted_offset: int,
    minimum_before_resume: float,
    started: float,
) -> dict[str, Any]:
    checkpoint_dir = output / "checkpoints"
    state_path = checkpoint_dir / f"state_{year}.pkl.gz"
    partial_path = output / "historical_1470_current.csv"
    _atomic_pickle(state_path, model.fls)
    _atomic_csv(partial_path, frame)
    accepted_total = accepted_offset + int(model.accepted_step_count)
    minimum_actual = min(
        float(minimum_before_resume),
        float(model.minimum_actual_timestep_seconds),
    )
    checkpoint = {
        "schema": "lia-late-only-full-checkpoint-v2",
        "year": year,
        "state": str(state_path),
        "state_sha256": preflight._sha256(state_path),
        "partial_curve": str(partial_path),
        "partial_curve_sha256": preflight._sha256(partial_path),
        "accepted_steps": accepted_total,
        "minimum_actual_timestep_seconds": minimum_actual,
        "runtime_seconds_this_process": time.perf_counter() - started,
    }
    preflight._atomic_json(checkpoint_dir / f"checkpoint_{year}.json", checkpoint)
    preflight._atomic_json(output / "checkpoint_latest.json", checkpoint)
    return checkpoint


def _formal_summary(
    config: dict[str, Any],
    frame: pd.DataFrame,
    model,
    runtime_error: str | None,
    failure_traceback: str | None,
    runtime_seconds: float,
    accepted_steps: int,
    minimum_actual_timestep_seconds: float,
    climate_audit: dict[str, Any],
) -> dict[str, Any]:
    lia_length = float(config["lia_registered_length_m"])
    rgi_length = float(config["rgi_registered_length_m"])
    separation = abs(lia_length - rgi_length)
    peak_row = frame.loc[frame["length_m"].idxmax()]
    final_row = frame.iloc[-1]
    e_lia_total = abs(float(peak_row["length_m"]) - lia_length) / lia_length
    e_lia_sep = abs(float(peak_row["length_m"]) - lia_length) / separation
    e_rgi_total = abs(float(final_row["length_m"]) - rgi_length) / rgi_length
    e_rgi_sep = abs(float(final_row["length_m"]) - rgi_length) / separation
    flags: list[str] = []
    if runtime_error:
        lower = runtime_error.lower()
        flags.append(
            "cfl-error"
            if "cfl" in lower
            else "boundary-error"
            if "boundar" in lower
            else "numerical-error"
        )
    if int(final_row["year"]) != int(config["end_year"]):
        flags.append("historical-incomplete")
    if not bool(frame["finite_geometry"].all()):
        flags.append("nonfinite-geometry")
    if int(frame["negative_section_count"].max()) > 0:
        flags.append("negative-section")
    if float(frame["root_array_end_thickness_m"].max()) > 0.0:
        flags.append("root-boundary-ice")
    if float(frame["remaining_free_downstream_m"].min()) < float(
        config["minimum_free_downstream_m"]
    ):
        flags.append("free-domain-below-registered-minimum")
    if e_lia_total > 0.1 or e_lia_sep > 0.2:
        flags.append("lia-peak-length-gate")
    if e_rgi_total > 0.1 or e_rgi_sep > 0.2:
        flags.append("rgi-2000-length-gate")
    numerical_flags = [
        flag
        for flag in flags
        if flag
        not in {
            "lia-peak-length-gate",
            "rgi-2000-length-gate",
        }
    ]
    initial_row = frame.iloc[0]
    volume_1500 = frame.loc[frame["year"].eq(1500), "volume_m3"]
    volume_2000 = frame.loc[frame["year"].eq(2000), "volume_m3"]
    return {
        "schema": "lia-late-only-full-from-1850-v2",
        "status": "historical-pass" if not flags else "historical-fail",
        "system_id": int(config["system_id"]),
        "late_delta_t_2000_c": float(config["late_delta_t_2000_c"]),
        "start_year": int(config["start_year"]),
        "resume_year": int(config["resume_year"]),
        "end_year": int(config["end_year"]),
        "initial": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in initial_row.items()
        },
        "peak": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in peak_row.items()
        },
        "final": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in final_row.items()
        },
        "e_lia_total": float(e_lia_total),
        "e_lia_separation": float(e_lia_sep),
        "e_rgi_total": float(e_rgi_total),
        "e_rgi_separation": float(e_rgi_sep),
        "length_gates": {
            "lia_total_max": 0.1,
            "lia_separation_max": 0.2,
            "rgi_total_max": 0.1,
            "rgi_separation_max": 0.2,
        },
        "volume_change_fraction_1500_2000": (
            float((volume_2000.iloc[0] - volume_1500.iloc[0]) / volume_1500.iloc[0])
            if len(volume_1500) == 1 and len(volume_2000) == 1
            else None
        ),
        "review_flags": flags,
        "numerical_flags": numerical_flags,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "accepted_steps_1850_2000": accepted_steps,
        "minimum_actual_timestep_seconds": minimum_actual_timestep_seconds,
        "runtime_seconds": runtime_seconds,
        "climate": climate_audit,
        "prefix_state_sha256": config["prefix_state_sha256"],
        "prefix_curve_sha256": config["prefix_curve_sha256"],
        "preflight_summary_sha256": config["preflight_summary_sha256"],
        "prefix_equivalence_exact": True,
        "annual_checkpoint_schema": "lia-late-only-full-checkpoint-v2",
        "resume_supported": True,
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
        "full_historical_started": True,
    }


def run(config_path: Path, resume: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    config = preflight._read_json(config_path)
    late_delta = float(config["late_delta_t_2000_c"])
    registered = {float(value) for value in config["registered_late_delta_t_2000_c"]}
    if late_delta not in registered:
        raise ValueError("unregistered-late-only-endpoint")
    controller = preflight._validate_controller(config)
    output = preflight._resolve(config["output_root"])
    prefix_root = preflight._resolve(config["prefix_root"])
    baseline = preflight._resolve(config["baseline_root"])
    prefix_state = prefix_root / "prefix_state_1850.pkl.gz"
    prefix_curve = prefix_root / "prefix_1470_1850.csv"
    preflight_summary = prefix_root / "summary.json"
    if preflight._sha256(prefix_state) != config["prefix_state_sha256"]:
        raise RuntimeError("prefix-state-hash-mismatch")
    if preflight._sha256(prefix_curve) != config["prefix_curve_sha256"]:
        raise RuntimeError("prefix-curve-hash-mismatch")
    if preflight._sha256(preflight_summary) != config["preflight_summary_sha256"]:
        raise RuntimeError("preflight-summary-hash-mismatch")
    prefix_result = preflight._read_json(preflight_summary)
    if prefix_result.get("status") != "preflight-pass":
        raise RuntimeError("registered-preflight-not-pass")

    baseline_config = preflight._read_json(baseline / "config_snapshot.json")
    baseline_summary = preflight._read_json(baseline / "historical_summary.json")
    stage2 = baseline_config["stage2"]
    start_year = int(config["start_year"])
    resume_year = int(config["resume_year"])
    end_year = int(config["end_year"])
    min_dt = float(baseline_config["minimum_timestep_seconds"])

    from oggm import cfg, utils

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])
    gdir_base = output / "synthetic_gdir"
    if not resume:
        if output.exists():
            raise FileExistsError(f"output-exists:{output}")
        output.mkdir(parents=True)
        preflight._atomic_json(output / "config_snapshot.json", config)
        shutil.copytree(baseline / "synthetic_gdir", gdir_base)
    else:
        if not output.exists():
            raise FileNotFoundError(f"resume-output-missing:{output}")
        if preflight._read_json(output / "config_snapshot.json") != config:
            raise RuntimeError("resume-config-mismatch")
    gdir = utils.GlacierDirectory(
        str(baseline_config["synthetic_rgi_id"]),
        base_dir=gdir_base,
        reset=False,
    )
    original_climate = Path(str(baseline_summary["climate"]["climate_file"]))
    climate_relative = original_climate.relative_to(baseline / "synthetic_gdir")
    copied_climate = gdir_base / climate_relative
    baseline_suffix = copied_climate.stem[len("gcm_data") :]
    tag = preflight._tag(float(config["late_delta_t_2000_c"]))
    branch_suffix = f"{baseline_suffix}_late{resume_year}_{tag}"
    branch_target = Path(gdir.get_filepath("gcm_data", filesuffix=branch_suffix))
    climate_audit_path = output / "climate_audit.json"
    if not resume:
        preflight._late_climate(
            copied_climate,
            branch_target,
            resume_year,
            end_year,
            float(config["late_delta_t_2000_c"]),
        )
        climate_audit = {
            "climate_file": str(branch_target),
            "sha256": preflight._sha256(branch_target),
            "policy": (
                f"frozen baseline through {resume_year}; linear additional "
                f"{float(config['late_delta_t_2000_c']):+.6f} K at {end_year}"
            ),
        }
        preflight._atomic_json(climate_audit_path, climate_audit)
    else:
        climate_audit = preflight._read_json(climate_audit_path)
        if preflight._sha256(branch_target) != climate_audit["sha256"]:
            raise RuntimeError("resume-climate-hash-mismatch")
    branch_mb = preflight._monthly_model(
        gdir,
        branch_suffix,
        stage2,
        start_year,
        end_year,
    )

    if resume:
        checkpoint = preflight._read_json(output / "checkpoint_latest.json")
        current_year = int(checkpoint["year"])
        checkpoint_state = Path(checkpoint["state"])
        if preflight._sha256(checkpoint_state) != checkpoint["state_sha256"]:
            raise RuntimeError("checkpoint-state-hash-mismatch")
        partial_path = Path(checkpoint["partial_curve"])
        if preflight._sha256(partial_path) != checkpoint["partial_curve_sha256"]:
            raise RuntimeError("checkpoint-curve-hash-mismatch")
        with gzip.open(checkpoint_state, "rb") as stream:
            flowlines = pickle.load(stream)
        frame = pd.read_csv(partial_path)
        accepted_offset = int(checkpoint["accepted_steps"])
        minimum_before_resume = float(checkpoint["minimum_actual_timestep_seconds"])
    else:
        current_year = resume_year
        with gzip.open(prefix_state, "rb") as stream:
            flowlines = pickle.load(stream)
        frame = pd.read_csv(prefix_curve)
        accepted_offset = 0
        minimum_before_resume = float("inf")
    model = preflight._new_model(
        flowlines,
        branch_mb,
        current_year,
        stage2,
        min_dt,
    )
    runtime_error = None
    failure_traceback = None
    try:
        for year in range(current_year + 1, end_year + 1):
            model.run_until(float(year))
            frame = pd.concat(
                [frame, pd.DataFrame([_snapshot(model, year)])],
                ignore_index=True,
            )
            checkpoint = _save_checkpoint(
                output,
                model,
                frame,
                year,
                accepted_offset,
                minimum_before_resume,
                started,
            )
            final = frame.iloc[-1]
            preflight._atomic_json(
                output / "progress.json",
                {
                    "status": "running-historical",
                    "phase": "1850-2000-continuation",
                    "late_delta_t_2000_c": float(config["late_delta_t_2000_c"]),
                    "simulated_year": year,
                    "accepted_steps": checkpoint["accepted_steps"],
                    "minimum_actual_timestep_seconds": checkpoint[
                        "minimum_actual_timestep_seconds"
                    ],
                    "volume_m3": float(final["volume_m3"]),
                    "length_m": float(final["length_m"]),
                    "remaining_free_downstream_m": float(
                        final["remaining_free_downstream_m"]
                    ),
                    "runtime_seconds_this_process": time.perf_counter() - started,
                },
            )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()

    full_curve = output / "historical_1470_2000.csv"
    _atomic_csv(full_curve, frame)
    _atomic_csv(
        output / "historical_1500_2000.csv",
        frame.loc[frame["year"].ge(1500)].reset_index(drop=True),
    )
    accepted_total = accepted_offset + int(model.accepted_step_count)
    minimum_actual = min(
        minimum_before_resume,
        float(model.minimum_actual_timestep_seconds),
    )
    summary = _formal_summary(
        config,
        frame,
        model,
        runtime_error,
        failure_traceback,
        time.perf_counter() - started,
        accepted_total,
        minimum_actual,
        climate_audit,
    )
    summary["controller"] = controller
    summary["curve_sha256"] = preflight._sha256(full_curve)
    if not summary["numerical_flags"] and int(summary["final"]["year"]) == end_year:
        state_2000 = output / "state_2000.pkl.gz"
        _atomic_pickle(state_2000, model.fls)
        summary["state_2000"] = str(state_2000)
        summary["state_2000_sha256"] = preflight._sha256(state_2000)
    else:
        summary["state_2000"] = None
        summary["state_2000_sha256"] = None
    preflight._atomic_json(output / "historical_summary_formal.json", summary)
    preflight._atomic_json(output / "historical_summary.json", summary)
    preflight._atomic_json(
        output / "progress.json",
        {
            "status": summary["status"],
            "phase": "complete",
            "late_delta_t_2000_c": float(config["late_delta_t_2000_c"]),
            "simulated_year": int(summary["final"]["year"]),
            "accepted_steps": accepted_total,
            "minimum_actual_timestep_seconds": minimum_actual,
            "runtime_seconds_this_process": summary["runtime_seconds"],
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.config.resolve(), resume=args.resume)
    except Exception as exc:
        config = preflight._read_json(args.config.resolve())
        output = preflight._resolve(config["output_root"])
        if output.exists():
            preflight._atomic_json(
                output / "historical_summary_formal.json",
                {
                    "schema": "lia-late-only-full-from-1850-v2",
                    "status": "technical-fail",
                    "runtime_error": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(),
                    "checkpoint_resume_supported": True,
                    "initial_thickness_modified": False,
                    "prohibited_source_accessed": False,
                    "full_historical_started": True,
                },
            )
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
