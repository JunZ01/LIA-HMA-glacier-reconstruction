"""Checkpointed shared-prefix and finite late-only response preflight.

The driver is a generic successor to the system-2580 preflight driver.  It
reuses its tested model/climate helpers, adds resumable five-year prefix
checkpoints, and supports either warming or cooling response order.  It never
continues a branch beyond the configured short preflight end year.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
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


def _config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _atomic_pickle_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


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
    route = state.get("authoritative_route", {})
    focus = state.get("current_focus", {})
    if route.get("version") != config["expected_route_version"]:
        raise RuntimeError(f"unexpected-route-version:{route.get('version')}")
    if int(focus.get("system_id", -1)) != int(config["system_id"]):
        raise RuntimeError(f"unexpected-focus:{focus.get('system_id')}")
    if state.get("next_action", {}).get("launch_authorized") is not True:
        raise RuntimeError("launch-not-authorized-by-current-state")
    return {"validation": validation, "state_id": state.get("state_id")}


def _latest_prefix_checkpoint(directory: Path, fingerprint: str) -> dict[str, Any] | None:
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("year_*.pkl.gz"), reverse=True):
        with gzip.open(path, "rb") as stream:
            checkpoint = pickle.load(stream)
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise RuntimeError(f"prefix-checkpoint-fingerprint-mismatch:{path}")
        checkpoint["path"] = str(path)
        return checkpoint
    return None


def _response_order(branches: list[dict[str, Any]], expected_sign: str) -> bool:
    ordered = sorted(branches, key=lambda item: abs(float(item["late_delta_t_2000_c"])))
    volumes = [float(item["volume_delta_vs_baseline_preflight_end_m3"]) for item in ordered]
    if expected_sign == "negative":
        return bool(all(value <= 0.0 for value in volumes) and all(b <= a for a, b in zip(volumes, volumes[1:])))
    if expected_sign == "positive":
        return bool(all(value >= 0.0 for value in volumes) and all(b >= a for a, b in zip(volumes, volumes[1:])))
    raise ValueError(f"unsupported-expected-volume-response-sign:{expected_sign}")


def run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    output = _resolve(config["output_root"])
    fingerprint = _config_fingerprint(config)
    resumed = False
    if output.exists():
        snapshot = output / "config_snapshot.json"
        if not bool(config.get("resume_allowed", False)) or not snapshot.is_file():
            raise FileExistsError(f"output-exists:{output}")
        if _config_fingerprint(_read_json(snapshot)) != fingerprint:
            raise RuntimeError("existing-output-config-fingerprint-mismatch")
        terminal = output / "summary.json"
        if terminal.is_file() and _read_json(terminal).get("status") in {
            "preflight-pass",
            "preflight-fail",
        }:
            raise FileExistsError(f"terminal-output-exists:{output}")
        resumed = True
    else:
        output.mkdir(parents=True)
        _atomic_json(output / "config_snapshot.json", config)

    started = time.perf_counter()
    controller = _validate_controller(config)
    baseline = _resolve(config["baseline_root"])
    steady = _resolve(config["steady_root"])
    baseline_config = _read_json(baseline / "config_snapshot.json")
    baseline_summary = _read_json(baseline / "historical_summary.json")
    stage2 = baseline_config["stage2"]
    start_year = int(config["start_year"])
    split_year = int(config["split_year"])
    preflight_end = int(config["preflight_end_year"])
    end_year = int(config["end_year"])
    min_dt = float(baseline_config["minimum_timestep_seconds"])
    if _sha256(steady / "lia_equilibrium_state_50y.pkl.gz") != config["steady_state_sha256"]:
        raise RuntimeError("steady-state-hash-mismatch")
    if _sha256(baseline / "historical_1470_2000.csv") != config["baseline_curve_sha256"]:
        raise RuntimeError("baseline-curve-hash-mismatch")
    if baseline_summary.get("runtime_error") is not None:
        raise RuntimeError("baseline-not-numerically-clean")

    from oggm import cfg, utils

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])
    prefix_gdir_base = output / "prefix_gdir"
    if not prefix_gdir_base.exists():
        shutil.copytree(baseline / "synthetic_gdir", prefix_gdir_base)
    gdir = utils.GlacierDirectory(
        str(baseline_config["synthetic_rgi_id"]), base_dir=prefix_gdir_base, reset=False
    )
    original_climate = Path(str(baseline_summary["climate"]["climate_file"]))
    climate_relative = original_climate.relative_to(baseline / "synthetic_gdir")
    copied_climate = prefix_gdir_base / climate_relative
    baseline_suffix = copied_climate.stem[len("gcm_data"):]
    mb_model = _monthly_model(gdir, baseline_suffix, stage2, start_year, end_year)

    checkpoint_directory = output / "prefix_checkpoints"
    checkpoint = _latest_prefix_checkpoint(checkpoint_directory, fingerprint)
    if checkpoint is None:
        with gzip.open(steady / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
            flowlines = pickle.load(stream)
        current_year = start_year
        model = _new_model(flowlines, mb_model, start_year, stage2, min_dt)
        rows = [_snapshot(model, start_year)]
        cumulative_steps = 0
        cumulative_min_dt = float("inf")
    else:
        current_year = int(checkpoint["year"])
        rows = checkpoint["rows"]
        model = _new_model(checkpoint["flowlines"], mb_model, current_year, stage2, min_dt)
        cumulative_steps = int(checkpoint["accepted_steps"])
        cumulative_min_dt = float(checkpoint["minimum_actual_timestep_seconds"])
        resumed = True

    for year in range(current_year + 1, split_year + 1):
        model.run_until(float(year))
        rows.append(_snapshot(model, year))
        accepted_steps = cumulative_steps + model.accepted_step_count
        minimum_actual_dt = min(cumulative_min_dt, model.minimum_actual_timestep_seconds)
        _atomic_json(
            output / "progress.json",
            {
                "status": "running-prefix",
                "phase": "prefix",
                "simulated_year": year,
                "accepted_steps": accepted_steps,
                "minimum_actual_timestep_seconds": minimum_actual_dt,
                "runtime_seconds_this_process": time.perf_counter() - started,
                "resumed": resumed,
                "latest_complete_checkpoint_year": year if year % int(config["checkpoint_interval_years"]) == 0 else None,
            },
        )
        if year % int(config["checkpoint_interval_years"]) == 0 or year == split_year:
            _atomic_pickle_gzip(
                checkpoint_directory / f"year_{year:04d}.pkl.gz",
                {
                    "config_fingerprint": fingerprint,
                    "year": year,
                    "flowlines": model.fls,
                    "rows": rows,
                    "accepted_steps": accepted_steps,
                    "minimum_actual_timestep_seconds": minimum_actual_dt,
                },
            )

    prefix_steps = cumulative_steps + model.accepted_step_count
    prefix_min_dt = min(cumulative_min_dt, model.minimum_actual_timestep_seconds)
    prefix_frame = pd.DataFrame(rows)
    prefix_csv = output / f"prefix_{start_year}_{split_year}.csv"
    prefix_frame.to_csv(prefix_csv, index=False)
    prefix_frame_persisted = pd.read_csv(prefix_csv)
    baseline_frame = pd.read_csv(baseline / "historical_1470_2000.csv")
    comparison = _comparison(prefix_frame_persisted, baseline_frame)
    comparison["comparison_basis"] = "persisted-csv-to-persisted-csv"
    comparison["in_memory_vs_persisted_reference_diagnostic"] = _comparison(prefix_frame, baseline_frame)
    _atomic_json(output / "prefix_equivalence.json", comparison)
    prefix_state = output / f"prefix_state_{split_year}.pkl.gz"
    _atomic_pickle_gzip(prefix_state, model.fls)
    if not comparison["exact"]:
        raise RuntimeError("persisted-baseline-prefix-not-exact")

    baseline_end = baseline_frame.loc[baseline_frame["year"].eq(preflight_end)].iloc[0]
    branch_results: list[dict[str, Any]] = []
    for late_delta in [float(value) for value in config["late_delta_t_2000_c"]]:
        tag = _tag(late_delta)
        branch_root = output / f"branch_{tag}"
        branch_gdir_base = branch_root / "synthetic_gdir"
        shutil.copytree(prefix_gdir_base, branch_gdir_base)
        branch_gdir = utils.GlacierDirectory(
            str(baseline_config["synthetic_rgi_id"]), base_dir=branch_gdir_base, reset=False
        )
        branch_source = branch_gdir_base / climate_relative
        branch_suffix = f"{baseline_suffix}_late{split_year}_{tag}"
        branch_target = Path(branch_gdir.get_filepath("gcm_data", filesuffix=branch_suffix))
        _late_climate(branch_source, branch_target, split_year, end_year, late_delta)
        branch_mb = _monthly_model(branch_gdir, branch_suffix, stage2, start_year, end_year)
        branch_model = _new_model(model.fls, branch_mb, split_year, stage2, min_dt)
        branch_rows = [_snapshot(branch_model, split_year)]
        runtime_error = None
        failure_traceback = None
        try:
            for year in range(split_year + 1, preflight_end + 1):
                branch_model.run_until(float(year))
                branch_rows.append(_snapshot(branch_model, year))
                _atomic_json(
                    output / "progress.json",
                    {
                        "status": "running-preflight",
                        "phase": f"branch-{tag}",
                        "simulated_year": year,
                        "accepted_steps": branch_model.accepted_step_count,
                        "minimum_actual_timestep_seconds": branch_model.minimum_actual_timestep_seconds,
                        "runtime_seconds_this_process": time.perf_counter() - started,
                    },
                )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
        frame = pd.DataFrame(branch_rows)
        frame.to_csv(branch_root / f"preflight_{split_year}_{preflight_end}.csv", index=False)
        clean, flags = _branch_clean(frame, preflight_end, runtime_error)
        final = frame.iloc[-1]
        result = {
            "late_delta_t_2000_c": late_delta,
            "status": "preflight-pass" if clean else "preflight-fail",
            "flags": flags,
            "runtime_error": runtime_error,
            "failure_traceback": failure_traceback,
            "final": {key: (value.item() if hasattr(value, "item") else value) for key, value in final.items()},
            "volume_delta_vs_baseline_preflight_end_m3": float(final["volume_m3"] - baseline_end["volume_m3"]),
            "length_delta_vs_baseline_preflight_end_m": float(final["length_m"] - baseline_end["length_m"]),
            "accepted_steps": branch_model.accepted_step_count,
            "minimum_actual_timestep_seconds": branch_model.minimum_actual_timestep_seconds,
            "climate_sha256": _sha256(branch_target),
        }
        _atomic_json(branch_root / "preflight_summary.json", result)
        branch_results.append(result)

    response_order_pass = _response_order(branch_results, str(config["expected_volume_response_sign"]))
    all_clean = all(item["status"] == "preflight-pass" for item in branch_results)
    summary = {
        "schema": "lia-late-only-response-group-preflight-v1",
        "status": "preflight-pass" if all_clean and response_order_pass else "preflight-fail",
        "system_id": int(config["system_id"]),
        "controller": controller,
        "config_fingerprint": fingerprint,
        "resumed": resumed,
        "prefix_equivalence_exact": comparison["exact"],
        "prefix_equivalence_basis": comparison["comparison_basis"],
        "prefix_state": str(prefix_state),
        "prefix_state_sha256": _sha256(prefix_state),
        "prefix_accepted_steps": prefix_steps,
        "prefix_minimum_actual_timestep_seconds": prefix_min_dt,
        "branch_results": branch_results,
        "expected_volume_response_sign": config["expected_volume_response_sign"],
        "response_order_pass": response_order_pass,
        "full_historical_started": False,
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
        "runtime_seconds_this_process": time.perf_counter() - started,
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
    args = parser.parse_args()
    try:
        result = run(args.config.resolve())
    except Exception as exc:
        config = _read_json(args.config.resolve())
        output = _resolve(config["output_root"])
        if output.exists():
            _atomic_json(
                output / "summary.json",
                {
                    "schema": "lia-late-only-response-group-preflight-v1",
                    "status": "technical-fail",
                    "system_id": int(config["system_id"]),
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
