"""Run the checkpointed system-5537 fixed late-cooling confirmation preflight."""

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

import pandas as pd

from .run_system2580_late_only_group_preflight_v2 import (
    PROJECT,
    VALIDATOR,
    _atomic_json,
    _branch_clean,
    _comparison,
    _late_climate,
    _monthly_model,
    _new_model,
    _read_json,
    _resolve,
    _sha256,
    _tag,
)
from .synthetic_lia_system_historical_v1 import _snapshot


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _latest_checkpoint(root: Path) -> Path | None:
    paths = sorted(root.glob("checkpoint_*.pkl.gz"))
    return paths[-1] if paths else None


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint-root-not-dict:{path}")
    return value


def _validate_controller(config: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"controller-validator-failed:{completed.stdout}:{completed.stderr}"
        )
    validation = json.loads(completed.stdout)
    state = _read_json(PROJECT / "LIA_CURRENT_STATE.json")
    if state.get("authoritative_route", {}).get("version") != config[
        "expected_route_version"
    ]:
        raise RuntimeError("unexpected-route-version")
    if int(state.get("current_focus", {}).get("system_id", -1)) != 5537:
        raise RuntimeError("unexpected-focus")
    if state.get("next_action", {}).get("launch_authorized") is not True:
        raise RuntimeError("launch-not-authorized-by-current-state")
    return {"validation": validation, "state_id": state.get("state_id")}


def _checkpoint_payload(model, rows: list[dict[str, Any]], year: int) -> dict[str, Any]:
    return {
        "year": year,
        "flowlines": model.fls,
        "rows": rows,
        "accepted_steps": model.accepted_step_count,
        "minimum_actual_timestep_seconds": model.minimum_actual_timestep_seconds,
    }


def _restore_model(saved, mb_model, stage2, minimum_timestep_seconds):
    model = _new_model(
        saved["flowlines"],
        mb_model,
        int(saved["year"]),
        stage2,
        minimum_timestep_seconds,
    )
    model.accepted_step_count = int(saved["accepted_steps"])
    model.minimum_actual_timestep_seconds = float(
        saved["minimum_actual_timestep_seconds"]
    )
    return model


def run(config_path: Path, resume: bool) -> dict[str, Any]:
    config = _read_json(config_path)
    output = _resolve(config["output_root"])
    if output.exists() and not resume:
        raise FileExistsError(f"output-exists:{output}")
    if not output.exists() and resume:
        raise FileNotFoundError(f"resume-output-missing:{output}")
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if _read_json(output / "config_snapshot.json") != config:
            raise RuntimeError("resume-config-mismatch")
    else:
        _atomic_json(output / "config_snapshot.json", config)

    started = time.perf_counter()
    controller = _validate_controller(config)
    baseline = _resolve(config["baseline_root"])
    steady = _resolve(config["steady_root"])
    baseline_config = _read_json(baseline / "config_snapshot.json")
    baseline_summary = _read_json(baseline / "historical_summary.json")
    if int(baseline_config["system_id"]) != 5537:
        raise RuntimeError("baseline-system-mismatch")
    if _sha256(steady / "lia_equilibrium_state_50y.pkl.gz") != config[
        "steady_state_sha256"
    ]:
        raise RuntimeError("steady-state-hash-mismatch")
    if _sha256(baseline / "historical_1470_2000.csv") != config[
        "baseline_curve_sha256"
    ]:
        raise RuntimeError("baseline-curve-hash-mismatch")
    if _sha256(baseline / "config_snapshot.json") != config[
        "baseline_config_sha256"
    ]:
        raise RuntimeError("baseline-config-hash-mismatch")
    if baseline_summary.get("runtime_error") is not None:
        raise RuntimeError("baseline-not-numerically-clean")

    stage2 = baseline_config["stage2"]
    start_year = int(config["start_year"])
    split_year = int(config["split_year"])
    preflight_end = int(config["preflight_end_year"])
    end_year = int(config["end_year"])
    minimum_timestep_seconds = float(baseline_config["minimum_timestep_seconds"])
    late_delta = float(config["late_delta_t_2000_c"])

    from oggm import cfg, utils

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])

    prefix_gdir_base = output / "prefix_gdir"
    if not prefix_gdir_base.exists():
        shutil.copytree(baseline / "synthetic_gdir", prefix_gdir_base)
    gdir = utils.GlacierDirectory(
        str(baseline_config["synthetic_rgi_id"]),
        base_dir=prefix_gdir_base,
        reset=False,
    )
    original_climate = Path(str(baseline_summary["climate"]["climate_file"]))
    climate_relative = original_climate.relative_to(baseline / "synthetic_gdir")
    copied_climate = prefix_gdir_base / climate_relative
    baseline_suffix = copied_climate.stem[len("gcm_data") :]
    mb_model = _monthly_model(gdir, baseline_suffix, stage2, start_year, end_year)

    prefix_checkpoints = output / "prefix_checkpoints"
    latest_prefix = _latest_checkpoint(prefix_checkpoints) if resume else None
    if latest_prefix is None:
        with gzip.open(steady / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
            flowlines = pickle.load(stream)
        prefix_model = _new_model(
            flowlines,
            mb_model,
            start_year,
            stage2,
            minimum_timestep_seconds,
        )
        prefix_rows = [_snapshot(prefix_model, start_year)]
        prefix_year = start_year
    else:
        saved = _load_checkpoint(latest_prefix)
        prefix_model = _restore_model(
            saved, mb_model, stage2, minimum_timestep_seconds
        )
        prefix_rows = saved["rows"]
        prefix_year = int(saved["year"])

    for year in range(prefix_year + 1, split_year + 1):
        prefix_model.run_until(float(year))
        prefix_rows.append(_snapshot(prefix_model, year))
        checkpoint = prefix_checkpoints / f"checkpoint_{year}.pkl.gz"
        _atomic_checkpoint(
            checkpoint, _checkpoint_payload(prefix_model, prefix_rows, year)
        )
        _atomic_json(
            output / "progress.json",
            {
                "status": "running-prefix",
                "phase": "prefix",
                "simulated_year": year,
                "accepted_steps": prefix_model.accepted_step_count,
                "minimum_actual_timestep_seconds": prefix_model.minimum_actual_timestep_seconds,
                "latest_checkpoint": str(checkpoint),
                "runtime_seconds_this_process": time.perf_counter() - started,
            },
        )

    prefix_frame = pd.DataFrame(prefix_rows)
    prefix_csv = output / "prefix_1470_1850.csv"
    prefix_frame.to_csv(prefix_csv, index=False)
    persisted_prefix = pd.read_csv(prefix_csv)
    baseline_frame = pd.read_csv(baseline / "historical_1470_2000.csv")
    comparison = _comparison(persisted_prefix, baseline_frame)
    comparison["comparison_basis"] = "persisted-csv-to-persisted-csv"
    _atomic_json(output / "prefix_equivalence.json", comparison)
    prefix_state = output / "prefix_state_1850.pkl.gz"
    if not prefix_state.exists():
        with gzip.open(prefix_state, "wb") as stream:
            pickle.dump(prefix_model.fls, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if not comparison["exact"]:
        raise RuntimeError("persisted-baseline-prefix-not-exact")

    tag = _tag(late_delta)
    branch_root = output / f"branch_{tag}"
    branch_gdir_base = branch_root / "synthetic_gdir"
    if not branch_gdir_base.exists():
        branch_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prefix_gdir_base, branch_gdir_base)
    branch_gdir = utils.GlacierDirectory(
        str(baseline_config["synthetic_rgi_id"]),
        base_dir=branch_gdir_base,
        reset=False,
    )
    branch_source = branch_gdir_base / climate_relative
    branch_suffix = f"{baseline_suffix}_late{split_year}_{tag}"
    branch_target = Path(
        branch_gdir.get_filepath("gcm_data", filesuffix=branch_suffix)
    )
    if not branch_target.exists():
        _late_climate(
            branch_source, branch_target, split_year, end_year, late_delta
        )
    branch_mb = _monthly_model(
        branch_gdir, branch_suffix, stage2, start_year, end_year
    )

    branch_checkpoints = branch_root / "checkpoints"
    latest_branch = _latest_checkpoint(branch_checkpoints) if resume else None
    if latest_branch is None:
        branch_model = _new_model(
            prefix_model.fls,
            branch_mb,
            split_year,
            stage2,
            minimum_timestep_seconds,
        )
        branch_rows = [_snapshot(branch_model, split_year)]
        branch_year = split_year
    else:
        saved = _load_checkpoint(latest_branch)
        branch_model = _restore_model(
            saved, branch_mb, stage2, minimum_timestep_seconds
        )
        branch_rows = saved["rows"]
        branch_year = int(saved["year"])

    runtime_error = None
    failure_traceback = None
    try:
        for year in range(branch_year + 1, preflight_end + 1):
            branch_model.run_until(float(year))
            branch_rows.append(_snapshot(branch_model, year))
            checkpoint = branch_checkpoints / f"checkpoint_{year}.pkl.gz"
            _atomic_checkpoint(
                checkpoint, _checkpoint_payload(branch_model, branch_rows, year)
            )
            _atomic_json(
                output / "progress.json",
                {
                    "status": "running-preflight",
                    "phase": f"branch-{tag}",
                    "simulated_year": year,
                    "accepted_steps": branch_model.accepted_step_count,
                    "minimum_actual_timestep_seconds": branch_model.minimum_actual_timestep_seconds,
                    "latest_checkpoint": str(checkpoint),
                    "runtime_seconds_this_process": time.perf_counter() - started,
                },
            )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()

    branch_frame = pd.DataFrame(branch_rows)
    branch_frame.to_csv(branch_root / "preflight_1850_1860.csv", index=False)
    clean, flags = _branch_clean(branch_frame, preflight_end, runtime_error)
    baseline_1860 = baseline_frame.loc[
        baseline_frame["year"].eq(preflight_end)
    ].iloc[0]
    final = branch_frame.iloc[-1]
    volume_delta = float(final["volume_m3"] - baseline_1860["volume_m3"])
    positive_response = volume_delta > 0.0
    if not positive_response:
        flags.append("nonpositive-cooling-volume-response")
    branch_result = {
        "late_delta_t_2000_c": late_delta,
        "status": "preflight-pass" if clean and positive_response else "preflight-fail",
        "flags": flags,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "final": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in final.items()
        },
        "volume_delta_vs_baseline_1860_m3": volume_delta,
        "length_delta_vs_baseline_1860_m": float(
            final["length_m"] - baseline_1860["length_m"]
        ),
        "positive_cooling_response": positive_response,
        "accepted_steps": branch_model.accepted_step_count,
        "minimum_actual_timestep_seconds": branch_model.minimum_actual_timestep_seconds,
        "climate_sha256": _sha256(branch_target),
    }
    _atomic_json(branch_root / "preflight_summary.json", branch_result)

    passed = comparison["exact"] and branch_result["status"] == "preflight-pass"
    summary = {
        "schema": "lia-system5537-late-only-confirmation-preflight-v1",
        "status": "preflight-pass" if passed else "preflight-fail",
        "system_id": 5537,
        "controller": controller,
        "prefix_equivalence_exact": comparison["exact"],
        "prefix_equivalence_basis": comparison["comparison_basis"],
        "prefix_state": str(prefix_state),
        "prefix_state_sha256": _sha256(prefix_state),
        "prefix_accepted_steps": prefix_model.accepted_step_count,
        "prefix_minimum_actual_timestep_seconds": prefix_model.minimum_actual_timestep_seconds,
        "branch_result": branch_result,
        "full_historical_started": False,
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
        "runtime_seconds_this_process": time.perf_counter() - started,
        "resume_used": resume,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "progress.json",
        {
            "status": summary["status"],
            "phase": "complete",
            "simulated_year": preflight_end,
            "runtime_seconds_this_process": summary["runtime_seconds_this_process"],
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.config.resolve(), args.resume)
    except Exception as exc:
        config = _read_json(args.config.resolve())
        output = _resolve(config["output_root"])
        if output.exists():
            _atomic_json(
                output / "summary.json",
                {
                    "schema": "lia-system5537-late-only-confirmation-preflight-v1",
                    "status": "technical-fail",
                    "runtime_error": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(),
                    "full_historical_started": False,
                    "initial_thickness_modified": False,
                    "prohibited_source_accessed": False,
                },
            )
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
