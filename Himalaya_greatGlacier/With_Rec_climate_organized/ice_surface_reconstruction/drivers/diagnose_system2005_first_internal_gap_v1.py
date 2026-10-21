"""Replay the registered cold steady run until its first internal support gap.

This is a read-only scientific diagnostic of the dynamic state: it does not
alter the solver, mass-balance forcing, time step, thresholds, or flowlines.
It records the exact nodes immediately before, during, and after the first
gap and stops shortly after the gap closes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import time
import traceback

import numpy as np

from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from . import network_topology_adaptive_split_system2005_zero_mb_v4 as solver


ZERO_PASS = "zero-mb-5y-pass-topology-adaptive-split"


def _gap_nodes(flowline, tolerance):
    thick = np.asarray(flowline.thick, dtype=float)
    active = np.flatnonzero(thick > tolerance)
    if len(active) < 2:
        return []
    first, last = int(active[0]), int(active[-1])
    return [i for i in range(first, last + 1) if thick[i] <= tolerance]


def _record_nodes(rows, flowlines, branches, event_nodes, previous_sections,
                  step_count, elapsed_seconds, phase, tolerance):
    from oggm.cfg import SEC_IN_YEAR

    for branch in branches:
        fl = flowlines[branch]
        current_section = np.asarray(fl.section, dtype=float)
        nodes = set(event_nodes.get(branch, []))
        if not nodes:
            nodes.update(_gap_nodes(fl, tolerance))
        expanded = set()
        for node in nodes:
            expanded.update(range(max(0, node - 2), min(len(fl.thick), node + 3)))
        for node in sorted(expanded):
            rows.append({
                "phase": phase,
                "step": step_count,
                "year": elapsed_seconds / float(SEC_IN_YEAR),
                "branch": branch,
                "node": node,
                "is_gap_node": node in nodes,
                "previous_section_m2": float(previous_sections[branch][node]),
                "section_m2": float(current_section[node]),
                "thickness_m": float(np.asarray(fl.thick, dtype=float)[node]),
                "surface_m": float(np.asarray(fl.surface_h, dtype=float)[node]),
                "bed_m": float(np.asarray(fl.bed_h, dtype=float)[node]),
                "width_m": float(np.asarray(fl.widths_m, dtype=float)[node]),
            })


def run(config):
    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH, SEC_IN_YEAR
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    stage2 = config["stage2"]
    cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])

    zero_path = base.resolve(config["formal_zero_mb_summary"])
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    if zero.get("status") != ZERO_PASS or float(zero.get("years", -1)) != 5.0:
        raise RuntimeError("registered-five-year-zero-gate-not-pass")
    if not all(zero.get("checks", {}).values()):
        raise RuntimeError("registered-five-year-zero-checks-not-all-pass")

    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=base.resolve(config["inversion_root"]) / "synthetic_gdir",
        reset=False,
    )
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=config["climate_filesuffix"],
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=int(config["spinup_window"][0]),
        ye=int(config["spinup_window"][1]),
        repeat=True,
        check_calib_params=False,
    )
    flowlines = base.load_flowlines(base.resolve(config["model_flowlines"]))
    coefficients = base.Coefficients(float(stage2["glen_a"]), float(stage2["fs"]))
    initial_volume = base.total_volume(flowlines)
    closure_limit = max(1e-6, initial_volume * 1e-12)
    tolerance = float(config["active_thickness_tolerance_m"])
    target = float(config["maximum_years"]) * float(SEC_IN_YEAR)
    maximum_dt = float(config["maximum_timestep_seconds"])
    elapsed = 0.0
    step_count = 0
    started = time.perf_counter()
    event_started = False
    event_closed = False
    event_start_step = None
    event_start_seconds = None
    event_end_step = None
    event_end_seconds = None
    event_branches = []
    event_nodes = {}
    maximum_gap_count = 0
    trace_rows = []
    post_close_steps = 0
    solver.conservative_boundary_step.config = config

    while elapsed < target - 1e-6:
        month_index = int(np.floor((elapsed + 1e-9) / float(SEC_IN_MONTH)))
        next_month = (month_index + 1) * float(SEC_IN_MONTH)
        dt = min(maximum_dt, next_month - elapsed, target - elapsed)
        if dt < float(config["minimum_timestep_seconds"]) - 1e-9:
            raise RuntimeError(f"timestep-below-floor:{dt}")
        previous_sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
        before = base.total_volume(flowlines)
        accepted_mb, _, _ = climate.connected_ice_climate_source(
            flowlines, mb_model, elapsed / float(SEC_IN_YEAR), dt
        )
        step = solver.conservative_boundary_step(
            flowlines, coefficients, dt, closure_limit, 0.0
        )
        after = base.total_volume(flowlines)
        budget_closure = after - before - accepted_mb
        if abs(budget_closure) > closure_limit:
            raise RuntimeError(f"climate-step-budget-closure:{budget_closure}:{closure_limit}")
        elapsed += dt
        step_count += 1
        support = topology.support_state(flowlines, tolerance)
        gap_count = int(support["internal_gap_branch_count"])
        maximum_gap_count = max(maximum_gap_count, gap_count)

        if gap_count and not event_started:
            event_started = True
            event_start_step = step_count
            event_start_seconds = elapsed
            event_branches = [int(v) for v in support["internal_gap_branches"]]
            event_nodes = {branch: _gap_nodes(flowlines[branch], tolerance)
                           for branch in event_branches}
            _record_nodes(
                trace_rows, flowlines, event_branches, event_nodes,
                previous_sections, step_count, elapsed, "onset", tolerance,
            )
        elif event_started and not event_closed and gap_count:
            _record_nodes(
                trace_rows, flowlines, event_branches, event_nodes,
                previous_sections, step_count, elapsed, "persisting", tolerance,
            )
        elif event_started and not event_closed and not gap_count:
            event_closed = True
            event_end_step = step_count
            event_end_seconds = elapsed
            _record_nodes(
                trace_rows, flowlines, event_branches, event_nodes,
                previous_sections, step_count, elapsed, "closed", tolerance,
            )
        elif event_closed:
            post_close_steps += 1
            _record_nodes(
                trace_rows, flowlines, event_branches, event_nodes,
                previous_sections, step_count, elapsed, "post-close", tolerance,
            )
            if post_close_steps >= int(config["post_close_steps"]):
                break

    if not event_started:
        status = "no-gap-observed-through-maximum-years"
    elif not event_closed:
        status = "gap-started-not-closed-before-limit"
    else:
        status = "first-gap-captured-and-closed"
    duration_seconds = None
    if event_start_seconds is not None and event_end_seconds is not None:
        duration_seconds = event_end_seconds - event_start_seconds
    return {
        "schema": "lia-191-system2005-first-internal-gap-diagnostic-v1",
        "status": status,
        "system_id": 2005,
        "scientific_state_modified": False,
        "solver_or_forcing_modified": False,
        "maximum_years": float(config["maximum_years"]),
        "elapsed_years": elapsed / float(SEC_IN_YEAR),
        "step_count": step_count,
        "maximum_timestep_seconds": maximum_dt,
        "active_thickness_tolerance_m": tolerance,
        "event_start_step": event_start_step,
        "event_start_year": None if event_start_seconds is None else event_start_seconds / float(SEC_IN_YEAR),
        "event_end_step": event_end_step,
        "event_end_year": None if event_end_seconds is None else event_end_seconds / float(SEC_IN_YEAR),
        "event_duration_seconds": duration_seconds,
        "event_duration_steps": None if event_start_step is None or event_end_step is None else event_end_step - event_start_step,
        "event_branches": event_branches,
        "event_nodes": event_nodes,
        "maximum_internal_gap_branch_count": maximum_gap_count,
        "initial_volume_m3": initial_volume,
        "final_volume_m3_at_stop": base.total_volume(flowlines),
        "runtime_seconds": time.perf_counter() - started,
        "formal_zero_mb_summary_sha256": base.sha256(zero_path),
        "source_model_flowlines_sha256": base.sha256(base.resolve(config["model_flowlines"])),
        "historical_run": False,
    }, trace_rows, flowlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = base.resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    try:
        summary, rows, flowlines = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-first-internal-gap-diagnostic-v1",
            "status": "technical-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "historical_run": False,
        }
        rows, flowlines = [], None
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if rows:
        with (output / "event_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if flowlines is not None and summary.get("event_start_step") is not None:
        with gzip.open(output / "flowlines_after_event.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary))
    if summary.get("status") not in {
        "first-gap-captured-and-closed", "gap-started-not-closed-before-limit",
        "no-gap-observed-through-maximum-years",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
