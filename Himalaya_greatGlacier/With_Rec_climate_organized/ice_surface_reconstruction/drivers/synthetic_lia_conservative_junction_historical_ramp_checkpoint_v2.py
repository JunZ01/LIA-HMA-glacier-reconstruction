"""Checkpointed conservative historical ramp from a passed equilibrium state."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from ..paths import resolve
from . import run_late_only_response_group_preflight_v1 as control
from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel
from .synthetic_lia_conservative_junction_historical_ramp_v1 import _ramp_climate
from .synthetic_lia_system_historical_v1 import _snapshot


class InstrumentedModel(ConservativeOneWayFluxBasedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted_step_count = 0
        self.minimum_actual_timestep_seconds = float("inf")

    def step(self, dt):
        used = float(super().step(dt))
        self.accepted_step_count += 1
        self.minimum_actual_timestep_seconds = min(
            self.minimum_actual_timestep_seconds, used
        )
        return used


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _atomic_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _atomic_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _controller(config, resume):
    completed = subprocess.run(
        [sys.executable, str(control.VALIDATOR)],
        cwd=control.PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"controller-validator-failed:{completed.stdout}:{completed.stderr}"
        )
    state = control._read_json(control.PROJECT / "LIA_CURRENT_STATE.json")
    route = state["authoritative_route"]
    focus = state["current_focus"]
    if int(focus["system_id"]) != int(config["system_id"]):
        raise RuntimeError(f"unexpected-focus:{focus['system_id']}")
    if state["next_action"].get("launch_authorized") is not True:
        raise RuntimeError("launch-not-authorized-by-current-state")
    if not resume and route["version"] != config["expected_route_version"]:
        raise RuntimeError(f"unexpected-route-version:{route['version']}")
    return {"state_id": state["state_id"], "validator": json.loads(completed.stdout)}


def _new_model(flowlines, mb_model, year, config):
    stage2 = config["stage2"]
    return InstrumentedModel(
        copy.deepcopy(flowlines),
        mb_model=mb_model,
        y0=float(year),
        glen_a=float(stage2["glen_a"]),
        fs=float(stage2["fs"]),
        smooth_trib_influx=True,
        check_for_boundaries=True,
        min_dt=float(config["minimum_timestep_seconds"]),
    )


def _checkpoint(config, output, model, frame, year, offset, prior_min, started, cpu):
    directory = output / "checkpoints"
    state_path = directory / f"state_{year}.pkl.gz"
    curve_path = output / "historical_1470_current.csv"
    _atomic_pickle(state_path, model.fls)
    _atomic_csv(curve_path, frame)
    item = {
        "schema": "lia-conservative-ramp-checkpoint-v2",
        "system_id": int(config["system_id"]),
        "config_fingerprint": control._config_fingerprint(config),
        "forward_delta_t_end_c": float(config["forward_delta_t_end_c"]),
        "year": int(year),
        "state": str(state_path),
        "state_sha256": control._sha256(state_path),
        "partial_curve": str(curve_path),
        "partial_curve_sha256": control._sha256(curve_path),
        "accepted_steps": offset + int(model.accepted_step_count),
        "rejected_steps": 0,
        "minimum_actual_timestep_seconds": _finite(
            min(prior_min, model.minimum_actual_timestep_seconds)
        ),
        "cumulative_prevented_negative_tributary_m3": float(
            model.cumulative_prevented_negative_tributary_m3
        ),
        "runtime_seconds_this_process": time.perf_counter() - started,
        "cpu_seconds_this_process": time.process_time() - cpu,
    }
    control._atomic_json(directory / f"checkpoint_{year}.json", item)
    control._atomic_json(output / "checkpoint_latest.json", item)
    return item


def _score(config, frame, model, error, trace, runtime, cpu, steps, min_dt, climate):
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
    flags = []
    if error:
        lower = error.lower()
        flags.append(
            "cfl-error"
            if "cfl" in lower
            else "boundary-error"
            if "boundar" in lower
            else "numerical-error"
        )
    if int(final["year"]) != int(config["end_year"]):
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
    if (
        errors["e_lia_total"] > float(config["lia_total_error_max"])
        or errors["e_lia_separation"] > float(config["lia_separation_error_max"])
    ):
        flags.append("lia-peak-length-gate")
    if (
        errors["e_rgi_total"] > float(config["rgi_total_error_max"])
        or errors["e_rgi_separation"] > float(config["rgi_separation_error_max"])
    ):
        flags.append("rgi-2000-length-gate")
    numerical = [
        flag
        for flag in flags
        if flag not in {"lia-peak-length-gate", "rgi-2000-length-gate"}
    ]
    convert = lambda row: {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }
    volume_1500 = frame.loc[frame["year"].eq(1500), "volume_m3"]
    volume_2000 = frame.loc[frame["year"].eq(2000), "volume_m3"]
    return {
        "schema": "lia-conservative-ramp-historical-v2",
        "status": "historical-pass" if not flags else "historical-fail",
        "system_id": int(config["system_id"]),
        "config_fingerprint": control._config_fingerprint(config),
        "start_year": int(config["start_year"]),
        "end_year": int(config["end_year"]),
        "forward_delta_t_start_c": float(config["stage2"]["delta_t_c"]),
        "forward_delta_t_end_c": float(config["forward_delta_t_end_c"]),
        "minimum_timestep_seconds": float(config["minimum_timestep_seconds"]),
        "initial": convert(frame.iloc[0]),
        "peak": convert(peak),
        "final": convert(final),
        **errors,
        "volume_change_fraction_1500_2000": (
            float((volume_2000.iloc[0] - volume_1500.iloc[0]) / volume_1500.iloc[0])
            if len(volume_1500) == 1 and len(volume_2000) == 1
            else None
        ),
        "cumulative_prevented_negative_tributary_m3": float(
            model.cumulative_prevented_negative_tributary_m3
        ),
        "review_flags": flags,
        "numerical_flags": numerical,
        "runtime_error": error,
        "failure_traceback": trace,
        "runtime_seconds": float(runtime),
        "cpu_seconds": float(cpu),
        "accepted_steps": int(steps),
        "rejected_steps": 0,
        "minimum_actual_timestep_seconds": _finite(min_dt),
        "climate": climate,
        "annual_checkpoint_schema": "lia-conservative-ramp-checkpoint-v2",
        "resume_supported": True,
        "independent_reproduction_required": True,
        "accepted_count_effect": 0,
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
    }


def run(config_path, resume=False):
    started, cpu = time.perf_counter(), time.process_time()
    config = control._read_json(config_path)
    if config.get("full_historical_authorized") is not True:
        raise RuntimeError("full-historical-not-authorized-in-config")
    if config.get("initial_thickness_modification_allowed") is not False:
        raise RuntimeError("initial-thickness-modification-policy-violation")
    if config.get("prohibited_source_accessed") is not False:
        raise RuntimeError("prohibited-source-policy-violation")
    controller = _controller(config, resume)
    inversion = resolve(config["inversion_root"])
    steady = resolve(config["steady_root"])
    output = resolve(config["output_root"])
    if control._read_json(steady / "steady_50y_summary.json")["status"] != "steady-50y-pass":
        raise RuntimeError("registered-equilibrium-not-pass")

    from oggm import cfg, utils
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    gdir_base = output / "synthetic_gdir"
    if resume:
        if control._read_json(output / "config_snapshot.json") != config:
            raise RuntimeError("resume-config-mismatch")
        saved = control._read_json(output / "checkpoint_latest.json")
        if saved["config_fingerprint"] != control._config_fingerprint(config):
            raise RuntimeError("checkpoint-config-fingerprint-mismatch")
        state_path, curve_path = Path(saved["state"]), Path(saved["partial_curve"])
        if control._sha256(state_path) != saved["state_sha256"]:
            raise RuntimeError("checkpoint-state-hash-mismatch")
        if control._sha256(curve_path) != saved["partial_curve_sha256"]:
            raise RuntimeError("checkpoint-curve-hash-mismatch")
        with gzip.open(state_path, "rb") as stream:
            flowlines = pickle.load(stream)
        frame = pd.read_csv(curve_path)
        year, offset = int(saved["year"]), int(saved["accepted_steps"])
        prior_min = float(saved.get("minimum_actual_timestep_seconds") or float("inf"))
        prior_prevented = float(
            saved.get("cumulative_prevented_negative_tributary_m3", 0.0)
        )
        climate = control._read_json(output / "climate_audit.json")
        suffix = climate["climate_filesuffix"]
        if control._sha256(Path(climate["climate_file"])) != climate["sha256"]:
            raise RuntimeError("resume-climate-hash-mismatch")
    else:
        if output.exists():
            raise FileExistsError(f"output-exists:{output}")
        output.mkdir(parents=True)
        control._atomic_json(output / "config_snapshot.json", config)
        shutil.copytree(inversion / "synthetic_gdir", gdir_base)
        gdir = utils.GlacierDirectory(
            str(config["synthetic_rgi_id"]), base_dir=gdir_base, reset=False
        )
        pixels = pd.read_csv(resolve(config["climate_pixels"]))
        rows = pixels[
            pixels["rgi_id"].astype(str).eq(str(config["main_rgi_id"]))
            & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
        ]
        if len(rows) != 1:
            raise ValueError(f"climate-pixel-row-count:{len(rows)}")
        suffix, climate = _ramp_climate(gdir, config, rows.iloc[0])
        climate["climate_filesuffix"] = suffix
        control._atomic_json(output / "climate_audit.json", climate)
        with gzip.open(steady / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
            flowlines = pickle.load(stream)
        year, offset, prior_min, prior_prevented = (
            int(config["start_year"]),
            0,
            float("inf"),
            0.0,
        )
        frame = None

    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]), base_dir=gdir_base, reset=False
    )
    stage2 = config["stage2"]
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=int(config["start_year"]),
        ye=int(config["end_year"]),
        repeat=False,
        check_calib_params=False,
    )
    model = _new_model(flowlines, mb_model, year, config)
    model.cumulative_prevented_negative_tributary_m3 = prior_prevented
    if frame is None:
        frame = pd.DataFrame([_snapshot(model, year)])
    error = trace = None
    try:
        for target in range(year + 1, int(config["end_year"]) + 1):
            model.run_until(float(target))
            frame = pd.concat(
                [frame, pd.DataFrame([_snapshot(model, target)])], ignore_index=True
            )
            saved = _checkpoint(
                config, output, model, frame, target, offset, prior_min, started, cpu
            )
            final = frame.iloc[-1]
            control._atomic_json(
                output / "progress.json",
                {
                    "status": "running-historical",
                    "system_id": int(config["system_id"]),
                    "forward_delta_t_end_c": float(config["forward_delta_t_end_c"]),
                    "simulated_year": target,
                    "accepted_steps": saved["accepted_steps"],
                    "rejected_steps": 0,
                    "minimum_actual_timestep_seconds": saved[
                        "minimum_actual_timestep_seconds"
                    ],
                    "runtime_seconds_this_process": time.perf_counter() - started,
                    "cpu_seconds_this_process": time.process_time() - cpu,
                    "length_m": float(final["length_m"]),
                    "volume_m3": float(final["volume_m3"]),
                    "remaining_free_downstream_m": float(
                        final["remaining_free_downstream_m"]
                    ),
                    "latest_complete_checkpoint_year": target,
                },
            )
    except Exception as exc:
        error, trace = f"{type(exc).__name__}: {exc}", traceback.format_exc()

    _atomic_csv(output / "historical_1470_2000.csv", frame)
    _atomic_csv(
        output / "historical_1500_2000.csv",
        frame.loc[frame["year"].ge(1500)].reset_index(drop=True),
    )
    steps = offset + int(model.accepted_step_count)
    min_dt = min(prior_min, model.minimum_actual_timestep_seconds)
    summary = _score(
        config,
        frame,
        model,
        error,
        trace,
        time.perf_counter() - started,
        time.process_time() - cpu,
        steps,
        min_dt,
        climate,
    )
    summary["controller"] = controller
    summary["curve_sha256"] = control._sha256(output / "historical_1470_2000.csv")
    if summary["status"] == "historical-pass":
        state_path = output / "state_2000.pkl.gz"
        _atomic_pickle(state_path, model.fls)
        summary["state_2000"] = str(state_path)
        summary["state_2000_sha256"] = control._sha256(state_path)
    else:
        summary["state_2000"] = summary["state_2000_sha256"] = None
    control._atomic_json(output / "historical_summary_formal.json", summary)
    control._atomic_json(output / "historical_summary.json", summary)
    control._atomic_json(
        output / "progress.json",
        {
            "status": summary["status"],
            "system_id": int(config["system_id"]),
            "forward_delta_t_end_c": float(config["forward_delta_t_end_c"]),
            "simulated_year": int(summary["final"]["year"]),
            "accepted_steps": steps,
            "rejected_steps": 0,
            "minimum_actual_timestep_seconds": _finite(min_dt),
            "runtime_seconds_this_process": summary["runtime_seconds"],
            "cpu_seconds_this_process": summary["cpu_seconds"],
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.config.resolve(), resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
