"""Materialize a shared system-2580 prefix and two late-only preflights.

The 1470-1850 segment must exactly reproduce the registered gamma=0.25
baseline.  Only after that gate passes are two fixed post-1850 cooling ramps
run to 1860.  This driver never continues either branch to 2000.
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

import numpy as np
import pandas as pd
import xarray as xr

from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel
from .synthetic_lia_system_historical_v1 import _snapshot


PROJECT = Path(__file__).resolve().parents[4]
VALIDATOR = (
    PROJECT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json-root-not-object:{path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _tag(value: float) -> str:
    return f"{value:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")


class InstrumentedConservativeModel(ConservativeOneWayFluxBasedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted_step_count = 0
        self.minimum_actual_timestep_seconds = float("inf")

    def step(self, dt):
        used = float(super().step(dt))
        self.accepted_step_count += 1
        self.minimum_actual_timestep_seconds = min(self.minimum_actual_timestep_seconds, used)
        return used


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
    if int(focus.get("system_id", -1)) != 2580:
        raise RuntimeError(f"unexpected-focus:{focus.get('system_id')}")
    if state.get("next_action", {}).get("launch_authorized") is not True:
        raise RuntimeError("launch-not-authorized-by-current-state")
    return {"validation": validation, "state_id": state.get("state_id")}


def _monthly_model(gdir, suffix: str, stage2: dict[str, Any], start: int, end: int):
    from oggm.core.massbalance import MonthlyTIModel

    return MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start,
        ye=end,
        repeat=False,
        check_calib_params=False,
    )


def _new_model(flowlines, mb_model, year: int, stage2: dict[str, Any], min_dt: float):
    return InstrumentedConservativeModel(
        copy.deepcopy(flowlines),
        mb_model=mb_model,
        y0=float(year),
        glen_a=float(stage2["glen_a"]),
        fs=float(stage2["fs"]),
        smooth_trib_influx=True,
        check_for_boundaries=True,
        min_dt=min_dt,
    )


def _comparison(prefix: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "year",
        "volume_m3",
        "area_m2",
        "length_m",
        "root_last_active_index",
        "root_last_active_thickness_m",
        "root_array_end_thickness_m",
        "remaining_free_downstream_m",
        "finite_geometry",
        "negative_section_count",
    ]
    reference = reference.loc[reference["year"].isin(prefix["year"]), columns].reset_index(drop=True)
    prefix = prefix[columns].reset_index(drop=True)
    metrics: dict[str, Any] = {
        "row_count_equal": len(prefix) == len(reference),
        "columns": {},
    }
    exact = metrics["row_count_equal"]
    for column in columns:
        left = prefix[column].to_numpy()
        right = reference[column].to_numpy()
        equal = bool(left.shape == right.shape and np.array_equal(left, right, equal_nan=True))
        max_abs = None
        if left.shape == right.shape and np.issubdtype(left.dtype, np.number):
            max_abs = float(np.nanmax(np.abs(left.astype(float) - right.astype(float))))
        metrics["columns"][column] = {"exact": equal, "max_abs_difference": max_abs}
        exact = exact and equal
    metrics["exact"] = bool(exact)
    return metrics


def _late_climate(source: Path, target: Path, split_year: int, end_year: int, late_delta: float) -> None:
    with xr.open_dataset(source, use_cftime=True) as source_ds:
        dataset = source_ds.load()
    decimal_year = np.asarray(
        [date.year + (date.month - 1) / 12.0 for date in dataset["time"].values],
        dtype=float,
    )
    fraction = np.clip(
        (decimal_year - float(split_year)) / float(end_year - split_year), 0.0, 1.0
    )
    attrs = dict(dataset["temp"].attrs)
    dataset["temp"] = dataset["temp"] + xr.DataArray(
        late_delta * fraction,
        dims=("time",),
        coords={"time": dataset["time"]},
    )
    dataset["temp"].attrs = attrs
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs["late_only_temperature_correction"] = (
        f"zero through {split_year}; linear to {late_delta:+.6f} K at {end_year}"
    )
    dataset.to_netcdf(target)
    dataset.close()


def _branch_clean(frame: pd.DataFrame, final_year: int, runtime_error: str | None) -> tuple[bool, list[str]]:
    flags: list[str] = []
    if runtime_error:
        flags.append("runtime-error")
    if int(frame.iloc[-1]["year"]) != final_year:
        flags.append("preflight-incomplete")
    if not bool(frame["finite_geometry"].all()):
        flags.append("nonfinite-geometry")
    if int(frame["negative_section_count"].max()) > 0:
        flags.append("negative-section")
    if float(frame["root_array_end_thickness_m"].max()) > 1e-9:
        flags.append("root-boundary-ice")
    if float(frame["remaining_free_downstream_m"].min()) < 5000.0:
        flags.append("free-domain-below-5000m")
    return not flags, flags


def run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    output = _resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(f"output-exists:{output}")
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
    shutil.copytree(baseline / "synthetic_gdir", prefix_gdir_base)
    gdir = utils.GlacierDirectory(
        str(baseline_config["synthetic_rgi_id"]), base_dir=prefix_gdir_base, reset=False
    )
    original_climate = Path(str(baseline_summary["climate"]["climate_file"]))
    climate_relative = original_climate.relative_to(baseline / "synthetic_gdir")
    copied_climate = prefix_gdir_base / climate_relative
    baseline_suffix = copied_climate.stem[len("gcm_data"):]
    mb_model = _monthly_model(gdir, baseline_suffix, stage2, start_year, end_year)
    with gzip.open(steady / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
        flowlines = pickle.load(stream)
    model = _new_model(flowlines, mb_model, start_year, stage2, min_dt)
    rows = [_snapshot(model, start_year)]
    for year in range(start_year + 1, split_year + 1):
        model.run_until(float(year))
        rows.append(_snapshot(model, year))
        if year % 5 == 0 or year == split_year:
            _atomic_json(
                output / "progress.json",
                {
                    "status": "running-prefix",
                    "phase": "prefix",
                    "simulated_year": year,
                    "accepted_steps": model.accepted_step_count,
                    "minimum_actual_timestep_seconds": model.minimum_actual_timestep_seconds,
                    "runtime_seconds": time.perf_counter() - started,
                },
            )
    prefix_frame = pd.DataFrame(rows)
    prefix_frame.to_csv(output / "prefix_1470_1850.csv", index=False)
    baseline_frame = pd.read_csv(baseline / "historical_1470_2000.csv")
    comparison = _comparison(prefix_frame, baseline_frame)
    _atomic_json(output / "prefix_equivalence.json", comparison)
    if not comparison["exact"]:
        raise RuntimeError("baseline-prefix-not-exact")
    prefix_state = output / "prefix_state_1850.pkl.gz"
    with gzip.open(prefix_state, "wb") as stream:
        pickle.dump(model.fls, stream, protocol=pickle.HIGHEST_PROTOCOL)

    baseline_1860 = baseline_frame.loc[baseline_frame["year"].eq(preflight_end)].iloc[0]
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
                        "runtime_seconds": time.perf_counter() - started,
                    },
                )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
        frame = pd.DataFrame(branch_rows)
        frame.to_csv(branch_root / "preflight_1850_1860.csv", index=False)
        clean, flags = _branch_clean(frame, preflight_end, runtime_error)
        final = frame.iloc[-1]
        result = {
            "late_delta_t_2000_c": late_delta,
            "status": "preflight-pass" if clean else "preflight-fail",
            "flags": flags,
            "runtime_error": runtime_error,
            "failure_traceback": failure_traceback,
            "final": {key: (value.item() if hasattr(value, "item") else value) for key, value in final.items()},
            "volume_delta_vs_baseline_1860_m3": float(final["volume_m3"] - baseline_1860["volume_m3"]),
            "length_delta_vs_baseline_1860_m": float(final["length_m"] - baseline_1860["length_m"]),
            "accepted_steps": branch_model.accepted_step_count,
            "minimum_actual_timestep_seconds": branch_model.minimum_actual_timestep_seconds,
            "climate_sha256": _sha256(branch_target),
        }
        _atomic_json(branch_root / "preflight_summary.json", result)
        branch_results.append(result)

    ordered = sorted(branch_results, key=lambda item: item["late_delta_t_2000_c"], reverse=True)
    response_order_pass = bool(
        all(item["volume_delta_vs_baseline_1860_m3"] >= 0.0 for item in ordered)
        and ordered[1]["volume_delta_vs_baseline_1860_m3"]
        >= ordered[0]["volume_delta_vs_baseline_1860_m3"]
    )
    all_clean = all(item["status"] == "preflight-pass" for item in branch_results)
    summary = {
        "schema": "lia-system2580-late-only-group-preflight-v1",
        "status": "preflight-pass" if all_clean and response_order_pass else "preflight-fail",
        "system_id": 2580,
        "controller": controller,
        "prefix_equivalence_exact": comparison["exact"],
        "prefix_state": str(prefix_state),
        "prefix_state_sha256": _sha256(prefix_state),
        "prefix_accepted_steps": model.accepted_step_count,
        "prefix_minimum_actual_timestep_seconds": model.minimum_actual_timestep_seconds,
        "branch_results": branch_results,
        "response_order_pass": response_order_pass,
        "full_historical_started": False,
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "progress.json",
        {
            "status": summary["status"],
            "phase": "complete",
            "simulated_year": preflight_end,
            "runtime_seconds": summary["runtime_seconds"],
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
                    "schema": "lia-system2580-late-only-group-preflight-v1",
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
