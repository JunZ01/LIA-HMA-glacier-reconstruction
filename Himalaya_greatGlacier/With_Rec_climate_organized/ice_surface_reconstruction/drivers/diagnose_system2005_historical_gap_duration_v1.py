"""Replay the frozen 1470--2000 ramp through 1478 and persist gap duration."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from . import diagnose_system2005_first_internal_gap_v1 as event_tools
from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from . import network_topology_adaptive_split_system2005_zero_mb_v4 as solver
from .synthetic_lia_conservative_junction_historical_ramp_v1 import _ramp_climate


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _record(flowlines, event_nodes, previous_sections, step, year, phase, tolerance):
    rows = []
    event_tools._record_nodes(
        rows, flowlines, sorted(event_nodes), event_nodes, previous_sections,
        step, year, phase, tolerance,
    )
    for row in rows:
        row["year"] = year
    return rows


def run(config: dict, output: Path):
    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH, SEC_IN_YEAR
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    steady_root = Path(config["steady_root"])
    steady = json.loads((steady_root / "steady_50y_summary_formal.json").read_text(encoding="utf-8"))
    if steady.get("status") != "steady-50y-pass" or not all(steady.get("checks", {}).values()):
        raise RuntimeError("registered-formal-steady-not-pass")
    state_path = steady_root / "lia_equilibrium_state_50y.pkl.gz"

    target_gdir_base = output / "synthetic_gdir"
    shutil.copytree(Path(config["inversion_root"]) / "synthetic_gdir", target_gdir_base)
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=target_gdir_base, reset=False)
    pixels = pd.read_csv(Path(config["climate_pixels"]))
    selected = pixels[
        pixels["rgi_id"].astype(str).eq(str(config["main_rgi_id"]))
        & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
    ]
    if len(selected) != 1:
        raise RuntimeError(f"climate-pixel-row-count:{len(selected)}")
    suffix, climate_audit = _ramp_climate(gdir, config, selected.iloc[0])
    _write_json(output / "climate_audit.json", climate_audit)
    stage2 = config["stage2"]
    start_year = int(config["start_year"])
    ramp_end_year = int(config["end_year"])
    stop_year = int(config["diagnostic_stop_year"])
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year, ye=stop_year, repeat=False, check_calib_params=False,
    )
    with gzip.open(state_path, "rb") as stream:
        flowlines = pickle.load(stream)
    coefficients = base.Coefficients(float(stage2["glen_a"]), float(stage2["fs"]))
    initial_volume = base.total_volume(flowlines)
    closure_limit = max(1e-6, initial_volume * 1e-12)
    tolerance = float(config["active_thickness_tolerance_m"])
    target = float(stop_year - start_year) * float(SEC_IN_YEAR)
    maximum_dt = float(config["maximum_timestep_seconds"])
    minimum_dt = float(config["minimum_timestep_seconds"])
    elapsed = 0.0
    step_count = 0
    current_event = None
    events = []
    trace = []
    max_gap_count = 0
    max_closure = 0.0
    max_kkt = 0.0
    next_year = start_year + 1
    started = time.perf_counter()
    solver.conservative_boundary_step.config = config

    while elapsed < target - 1e-6:
        month_index = int(np.floor((elapsed + 1e-9) / float(SEC_IN_MONTH)))
        dt = min(maximum_dt, (month_index + 1) * float(SEC_IN_MONTH) - elapsed, target - elapsed)
        if dt < minimum_dt - 1e-9:
            raise RuntimeError(f"timestep-below-floor:{dt}")
        previous_sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
        before = base.total_volume(flowlines)
        model_year = start_year + elapsed / float(SEC_IN_YEAR)
        accepted_mb, _, _ = climate.connected_ice_climate_source(flowlines, mb_model, model_year, dt)
        step = solver.conservative_boundary_step(flowlines, coefficients, dt, closure_limit, 0.0)
        after = base.total_volume(flowlines)
        closure = after - before - accepted_mb
        if abs(closure) > closure_limit:
            raise RuntimeError(f"historical-step-budget-closure:{closure}:{closure_limit}")
        elapsed += dt
        step_count += 1
        max_closure = max(max_closure, abs(closure), abs(float(step["actual_closure_m3"])))
        max_kkt = max(max_kkt, float(step["maximum_absolute_KKT_stationarity"]))
        support = topology.support_state(flowlines, tolerance)
        gap_count = int(support["internal_gap_branch_count"])
        max_gap_count = max(max_gap_count, gap_count)
        absolute_year = start_year + elapsed / float(SEC_IN_YEAR)

        if gap_count and current_event is None:
            nodes = {
                int(branch): event_tools._gap_nodes(flowlines[int(branch)], tolerance)
                for branch in support["internal_gap_branches"]
            }
            current_event = {
                "event_index": len(events) + 1,
                "start_step": step_count,
                "start_year": absolute_year,
                "start_seconds": elapsed,
                "maximum_gap_branch_count": gap_count,
                "branches": sorted(nodes),
                "nodes": nodes,
            }
            trace.extend(_record(flowlines, nodes, previous_sections, step_count, absolute_year, "onset", tolerance))
        elif gap_count and current_event is not None:
            current_event["maximum_gap_branch_count"] = max(current_event["maximum_gap_branch_count"], gap_count)
            for branch in support["internal_gap_branches"]:
                branch = int(branch)
                values = event_tools._gap_nodes(flowlines[branch], tolerance)
                current_event["nodes"].setdefault(branch, [])
                current_event["nodes"][branch] = sorted(set(current_event["nodes"][branch] + values))
                if branch not in current_event["branches"]:
                    current_event["branches"].append(branch)
        elif not gap_count and current_event is not None:
            current_event["end_step"] = step_count
            current_event["end_year"] = absolute_year
            current_event["duration_steps"] = step_count - current_event["start_step"]
            current_event["duration_seconds"] = elapsed - current_event.pop("start_seconds")
            trace.extend(_record(flowlines, current_event["nodes"], previous_sections, step_count, absolute_year, "closed", tolerance))
            events.append(current_event)
            current_event = None

        if absolute_year + 1e-10 >= next_year:
            _write_json(output / "progress.json", {
                "schema": "lia-191-system2005-historical-gap-duration-progress-v1",
                "year": absolute_year,
                "step_count": step_count,
                "completed_gap_event_count": len(events),
                "gap_event_open": current_event is not None,
                "gap_event_start_year": None if current_event is None else current_event["start_year"],
                "open_event_duration_days": None if current_event is None else (elapsed - current_event["start_seconds"]) / 86400.0,
                "maximum_internal_gap_branch_count": max_gap_count,
                "runtime_seconds": time.perf_counter() - started,
            })
            next_year += 1

    if current_event is not None:
        trace.extend(_record(flowlines, current_event["nodes"], previous_sections, step_count, stop_year, "open-at-stop", tolerance))
        current_event["end_step"] = None
        current_event["end_year"] = None
        current_event["duration_steps"] = step_count - current_event["start_step"]
        current_event["duration_seconds"] = elapsed - current_event.pop("start_seconds")
        current_event["open_at_diagnostic_stop"] = True
        events.append(current_event)

    summary = {
        "schema": "lia-191-system2005-historical-gap-duration-v1",
        "status": "diagnostic-complete",
        "system_id": 2005,
        "forward_delta_t_end_c": float(config["forward_delta_t_end_c"]),
        "climate_ramp_start_year": start_year,
        "climate_ramp_end_year": ramp_end_year,
        "diagnostic_stop_year": stop_year,
        "step_count": step_count,
        "event_count": len(events),
        "events": events,
        "maximum_internal_gap_branch_count": max_gap_count,
        "maximum_absolute_step_closure_m3": max_closure,
        "maximum_absolute_KKT_stationarity": max_kkt,
        "closure_limit_m3": closure_limit,
        "scientific_state_modified": False,
        "historical_completion_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    return summary, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config_snapshot.json", config)
    try:
        summary, trace = run(config, output)
        error = None
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-historical-gap-duration-v1",
            "status": "technical-fail",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        trace, error = [], exc
    _write_json(output / "summary.json", summary)
    if trace:
        with (output / "event_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader(); writer.writerows(trace)
    print(json.dumps(summary, allow_nan=False))
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
