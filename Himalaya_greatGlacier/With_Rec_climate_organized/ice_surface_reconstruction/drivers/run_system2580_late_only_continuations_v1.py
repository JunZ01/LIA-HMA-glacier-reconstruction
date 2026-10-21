"""Continue the two registered system-2580 late-cooling branches to 2000."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .run_system2580_late_only_group_preflight_v2 import (
    PROJECT,
    VALIDATOR,
    _atomic_json,
    _monthly_model,
    _new_model,
    _read_json,
    _resolve,
    _sha256,
    _tag,
)
from .synthetic_lia_system_historical_v1 import _snapshot


def _validate_controller(config: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"controller-validator-failed:{completed.stdout}:{completed.stderr}")
    validation = json.loads(completed.stdout)
    state = _read_json(PROJECT / "LIA_CURRENT_STATE.json")
    if state.get("authoritative_route", {}).get("version") != config["expected_route_version"]:
        raise RuntimeError("unexpected-route-version")
    if int(state.get("current_focus", {}).get("system_id", -1)) != 2580:
        raise RuntimeError("unexpected-focus")
    if state.get("next_action", {}).get("launch_authorized") is not True:
        raise RuntimeError("launch-not-authorized-by-current-state")
    return {"validation": validation, "state_id": state.get("state_id")}


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _latest_checkpoint(branch_root: Path) -> Path | None:
    checkpoints = sorted(branch_root.glob("checkpoint_*.pkl.gz"))
    return checkpoints[-1] if checkpoints else None


def _score(frame: pd.DataFrame, config: dict[str, Any], runtime_error: str | None) -> dict[str, Any]:
    lia = float(config["lia_registered_length_m"])
    rgi = float(config["rgi_registered_length_m"])
    separation = abs(lia - rgi)
    peak = frame.loc[frame["length_m"].idxmax()]
    final = frame.iloc[-1]
    errors = {
        "e_lia_total": abs(float(peak["length_m"]) - lia) / lia,
        "e_lia_separation": abs(float(peak["length_m"]) - lia) / separation,
        "e_rgi_total": abs(float(final["length_m"]) - rgi) / rgi,
        "e_rgi_separation": abs(float(final["length_m"]) - rgi) / separation,
    }
    flags: list[str] = []
    if runtime_error:
        flags.append("runtime-error")
    if int(final["year"]) != int(config["end_year"]):
        flags.append("historical-incomplete")
    if not bool(frame["finite_geometry"].all()):
        flags.append("nonfinite-geometry")
    if int(frame["negative_section_count"].max()) > 0:
        flags.append("negative-section")
    if float(frame["root_array_end_thickness_m"].max()) > 1e-9:
        flags.append("root-boundary-ice")
    if float(frame["remaining_free_downstream_m"].min()) < float(config["minimum_free_downstream_m"]):
        flags.append("free-domain-below-minimum")
    if errors["e_lia_total"] > 0.1 or errors["e_lia_separation"] > 0.2:
        flags.append("lia-peak-length-gate")
    if errors["e_rgi_total"] > 0.1 or errors["e_rgi_separation"] > 0.2:
        flags.append("rgi-2000-length-gate")
    return {
        "status": "historical-pass" if not flags else "historical-fail",
        "peak": {key: value.item() if hasattr(value, "item") else value for key, value in peak.items()},
        "final": {key: value.item() if hasattr(value, "item") else value for key, value in final.items()},
        **errors,
        "review_flags": flags,
        "runtime_error": runtime_error,
    }


def _run_branch(
    config: dict[str, Any],
    output: Path,
    prefix_flowlines,
    prefix_frame: pd.DataFrame,
    baseline_config: dict[str, Any],
    baseline_summary: dict[str, Any],
    late_delta: float,
    resume: bool,
    started: float,
) -> dict[str, Any]:
    from oggm import utils

    split_year = int(config["split_year"])
    end_year = int(config["end_year"])
    tag = _tag(late_delta)
    branch_root = output / f"branch_{tag}"
    completed_summary = branch_root / "historical_summary.json"
    if resume and completed_summary.exists():
        return _read_json(completed_summary)

    source_gdir = _resolve(config["preflight_root"]) / f"branch_{tag}" / "synthetic_gdir"
    target_gdir = branch_root / "synthetic_gdir"
    checkpoint = _latest_checkpoint(branch_root) if resume and branch_root.exists() else None
    if checkpoint is None:
        if branch_root.exists():
            raise FileExistsError(f"branch-output-exists:{branch_root}")
        branch_root.mkdir(parents=True)
        shutil.copytree(source_gdir, target_gdir)
        flowlines = copy.deepcopy(prefix_flowlines)
        rows = None
        resume_year = split_year
        accepted_steps = 0
        minimum_dt = float("inf")
    else:
        with gzip.open(checkpoint, "rb") as stream:
            saved = pickle.load(stream)
        flowlines = saved["flowlines"]
        rows = saved["rows"]
        resume_year = int(saved["year"])
        accepted_steps = int(saved["accepted_steps"])
        minimum_dt = float(saved["minimum_actual_timestep_seconds"])

    stage2 = baseline_config["stage2"]
    gdir = utils.GlacierDirectory(
        str(baseline_config["synthetic_rgi_id"]), base_dir=target_gdir, reset=False
    )
    original_climate = Path(str(baseline_summary["climate"]["climate_file"]))
    baseline_root = _resolve(config["baseline_root"])
    climate_relative = original_climate.relative_to(baseline_root / "synthetic_gdir")
    baseline_suffix = (target_gdir / climate_relative).stem[len("gcm_data"):]
    branch_suffix = f"{baseline_suffix}_late{split_year}_{tag}"
    branch_climate = Path(gdir.get_filepath("gcm_data", filesuffix=branch_suffix))
    if _sha256(branch_climate) != config["branch_climate_sha256"][tag]:
        raise RuntimeError(f"branch-climate-hash-mismatch:{tag}")
    mb_model = _monthly_model(
        gdir, branch_suffix, stage2, int(config["start_year"]), end_year
    )
    model = _new_model(
        flowlines,
        mb_model,
        resume_year,
        stage2,
        float(baseline_config["minimum_timestep_seconds"]),
    )
    model.accepted_step_count = accepted_steps
    model.minimum_actual_timestep_seconds = minimum_dt
    if rows is None:
        rows = [_snapshot(model, split_year)]
    runtime_error = None
    failure_traceback = None
    try:
        for year in range(resume_year + 1, end_year + 1):
            model.run_until(float(year))
            rows.append(_snapshot(model, year))
            checkpoint_path = branch_root / f"checkpoint_{year}.pkl.gz"
            _atomic_checkpoint(
                checkpoint_path,
                {
                    "year": year,
                    "flowlines": model.fls,
                    "rows": rows,
                    "accepted_steps": model.accepted_step_count,
                    "minimum_actual_timestep_seconds": model.minimum_actual_timestep_seconds,
                },
            )
            _atomic_json(
                output / "progress.json",
                {
                    "status": "running-continuation",
                    "branch": tag,
                    "simulated_year": year,
                    "accepted_steps": model.accepted_step_count,
                    "minimum_actual_timestep_seconds": model.minimum_actual_timestep_seconds,
                    "latest_checkpoint": str(checkpoint_path),
                    "runtime_seconds": time.perf_counter() - started,
                },
            )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()

    continuation = pd.DataFrame(rows)
    full_frame = pd.concat(
        [prefix_frame, continuation.loc[continuation["year"].gt(split_year)]],
        ignore_index=True,
    )
    full_frame.to_csv(branch_root / "historical_1470_2000.csv", index=False)
    result = {
        "schema": "lia-system2580-late-only-continuation-branch-v1",
        "system_id": 2580,
        "late_delta_t_2000_c": late_delta,
        **_score(full_frame, config, runtime_error),
        "failure_traceback": failure_traceback,
        "accepted_steps_1850_2000": model.accepted_step_count,
        "minimum_actual_timestep_seconds": model.minimum_actual_timestep_seconds,
        "climate_sha256": _sha256(branch_climate),
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
    }
    if runtime_error is None and int(full_frame.iloc[-1]["year"]) == end_year:
        with gzip.open(branch_root / "state_2000.pkl.gz", "wb") as stream:
            pickle.dump(model.fls, stream, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_json(completed_summary, result)
    return result


def run(config_path: Path, resume: bool) -> dict[str, Any]:
    config = _read_json(config_path)
    output = _resolve(config["output_root"])
    if output.exists() and not resume:
        raise FileExistsError(f"output-exists:{output}")
    output.mkdir(parents=True, exist_ok=resume)
    if not resume:
        _atomic_json(output / "config_snapshot.json", config)
    started = time.perf_counter()
    controller = _validate_controller(config)
    preflight = _resolve(config["preflight_root"])
    if _sha256(preflight / "summary.json") != config["preflight_summary_sha256"]:
        raise RuntimeError("preflight-summary-hash-mismatch")
    preflight_summary = _read_json(preflight / "summary.json")
    if preflight_summary.get("status") != "preflight-pass":
        raise RuntimeError("preflight-not-pass")
    prefix_state = preflight / "prefix_state_1850.pkl.gz"
    if _sha256(prefix_state) != config["prefix_state_sha256"]:
        raise RuntimeError("prefix-state-hash-mismatch")
    with gzip.open(prefix_state, "rb") as stream:
        prefix_flowlines = pickle.load(stream)
    prefix_frame = pd.read_csv(preflight / "prefix_1470_1850.csv")
    baseline = _resolve(config["baseline_root"])
    baseline_config = _read_json(baseline / "config_snapshot.json")
    baseline_summary = _read_json(baseline / "historical_summary.json")

    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(baseline_config["stage2"]["temp_melt_c"])
    results = [
        _run_branch(
            config,
            output,
            prefix_flowlines,
            prefix_frame,
            baseline_config,
            baseline_summary,
            float(value),
            resume,
            started,
        )
        for value in config["late_delta_t_2000_c"]
    ]
    summary = {
        "schema": "lia-system2580-late-only-continuations-v1",
        "status": "complete",
        "system_id": 2580,
        "controller": controller,
        "branch_results": results,
        "historical_pass_count": sum(item["status"] == "historical-pass" for item in results),
        "reproduction_started": False,
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "progress.json",
        {"status": "complete", "simulated_year": 2000, "runtime_seconds": summary["runtime_seconds"]},
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
