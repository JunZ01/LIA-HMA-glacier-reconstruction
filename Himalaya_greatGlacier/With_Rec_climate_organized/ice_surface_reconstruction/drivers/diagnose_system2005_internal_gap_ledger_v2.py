"""Full 50-year cold steady replay with a non-mutating internal-gap ledger."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import time
import traceback

import numpy as np

from . import diagnose_system2005_first_internal_gap_v1 as event_tools
from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from . import network_topology_adaptive_split_system2005_zero_mb_v4 as solver


ZERO_PASS = "zero-mb-5y-pass-topology-adaptive-split"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _node_snapshot(flowlines, event_nodes, previous_sections, step_count,
                   elapsed_seconds, phase, tolerance):
    rows = []
    event_tools._record_nodes(
        rows, flowlines, sorted(event_nodes), event_nodes, previous_sections,
        step_count, elapsed_seconds, phase, tolerance,
    )
    return rows


def run(config, output):
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
    initial_zero = [np.asarray(fl.section, dtype=float) == 0.0 for fl in flowlines]
    initial_support = topology.support_state(
        flowlines, float(config["active_thickness_tolerance_m"])
    )
    coefficients = base.Coefficients(float(stage2["glen_a"]), float(stage2["fs"]))
    initial_volume = base.total_volume(flowlines)
    initial_area = climate.ice_area(flowlines)
    closure_limit = max(1e-6, initial_volume * 1e-12)
    tolerance = float(config["active_thickness_tolerance_m"])
    registered = float(config["registered_lia_length_m"])
    target = float(config["years"]) * float(SEC_IN_YEAR)
    maximum_dt = float(config["maximum_timestep_seconds"])
    elapsed = 0.0
    step_count = 0
    cumulative_mb = 0.0
    cumulative_overmelt = 0.0
    max_closure = 0.0
    max_kkt = 0.0
    max_active_constraints = 0
    maximum_gap_count = 0
    events = []
    current_event = None
    trace_rows = []
    annual_rows = []
    next_record_year = 0
    started = time.perf_counter()
    solver.conservative_boundary_step.config = config

    while elapsed < target - 1e-6:
        month_index = int(np.floor((elapsed + 1e-9) / float(SEC_IN_MONTH)))
        next_month = (month_index + 1) * float(SEC_IN_MONTH)
        dt = min(maximum_dt, next_month - elapsed, target - elapsed)
        if dt < float(config["minimum_timestep_seconds"]) - 1e-9:
            raise RuntimeError(f"timestep-below-floor:{dt}")
        previous_sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
        before = base.total_volume(flowlines)
        accepted_mb, prevented, _ = climate.connected_ice_climate_source(
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
        cumulative_mb += accepted_mb
        cumulative_overmelt += prevented
        max_closure = max(
            max_closure, abs(budget_closure), abs(float(step["actual_closure_m3"]))
        )
        max_kkt = max(max_kkt, float(step["maximum_absolute_KKT_stationarity"]))
        max_active_constraints = max(
            max_active_constraints, int(step["active_constraint_count"])
        )
        support = topology.support_state(flowlines, tolerance)
        gap_count = int(support["internal_gap_branch_count"])
        maximum_gap_count = max(maximum_gap_count, gap_count)

        if gap_count and current_event is None:
            nodes = {
                int(branch): event_tools._gap_nodes(flowlines[int(branch)], tolerance)
                for branch in support["internal_gap_branches"]
            }
            current_event = {
                "event_index": len(events) + 1,
                "start_step": step_count,
                "start_year": elapsed / float(SEC_IN_YEAR),
                "start_seconds": elapsed,
                "maximum_gap_branch_count": gap_count,
                "branches": sorted(nodes),
                "nodes": nodes,
                "minimum_gap_node_thickness_m": min(
                    float(np.asarray(flowlines[b].thick)[n])
                    for b, values in nodes.items() for n in values
                ),
            }
            trace_rows.extend(_node_snapshot(
                flowlines, nodes, previous_sections, step_count, elapsed,
                "onset", tolerance,
            ))
        elif gap_count and current_event is not None:
            current_event["maximum_gap_branch_count"] = max(
                current_event["maximum_gap_branch_count"], gap_count
            )
            for branch in support["internal_gap_branches"]:
                branch = int(branch)
                values = event_tools._gap_nodes(flowlines[branch], tolerance)
                current_event["nodes"].setdefault(branch, [])
                current_event["nodes"][branch] = sorted(set(
                    current_event["nodes"][branch] + values
                ))
                if branch not in current_event["branches"]:
                    current_event["branches"].append(branch)
                for node in values:
                    current_event["minimum_gap_node_thickness_m"] = min(
                        current_event["minimum_gap_node_thickness_m"],
                        float(np.asarray(flowlines[branch].thick)[node]),
                    )
        elif not gap_count and current_event is not None:
            current_event["end_step"] = step_count
            current_event["end_year"] = elapsed / float(SEC_IN_YEAR)
            current_event["duration_steps"] = step_count - current_event["start_step"]
            current_event["duration_seconds"] = elapsed - current_event.pop("start_seconds")
            current_event["branches"] = sorted(current_event["branches"])
            trace_rows.extend(_node_snapshot(
                flowlines, current_event["nodes"], previous_sections,
                step_count, elapsed, "closed", tolerance,
            ))
            events.append(current_event)
            current_event = None

        year = elapsed / float(SEC_IN_YEAR)
        if year + 1e-10 >= next_record_year or elapsed >= target - 1e-6:
            root = climate.root_metrics(flowlines, registered)
            annual_rows.append({
                "year": year,
                "volume_m3": after,
                "area_m2": climate.ice_area(flowlines),
                **root,
                "disconnected_tributary_count": support["disconnected_tributary_count"],
                "internal_gap_branch_count": gap_count,
                "completed_gap_event_count": len(events),
                "gap_event_open": current_event is not None,
                "cumulative_accepted_mb_m3": cumulative_mb,
            })
            next_record_year = int(np.floor(year)) + 1
            _write_json(output / "progress.json", {
                "schema": "lia-191-system2005-internal-gap-ledger-progress-v2",
                "system_id": 2005,
                "year": year,
                "step_count": step_count,
                "completed_gap_event_count": len(events),
                "gap_event_open": current_event is not None,
                "maximum_internal_gap_branch_count": maximum_gap_count,
                "runtime_seconds": time.perf_counter() - started,
            })

    if current_event is not None:
        current_event["end_step"] = None
        current_event["end_year"] = None
        current_event["duration_steps"] = step_count - current_event["start_step"]
        current_event["duration_seconds"] = elapsed - current_event.pop("start_seconds")
        current_event["open_at_final"] = True
        events.append(current_event)

    final_volume = base.total_volume(flowlines)
    final_area = climate.ice_area(flowlines)
    final_support = topology.support_state(flowlines, tolerance)
    newly_activated = sum(
        int(np.count_nonzero(mask & (np.asarray(fl.section) > 0.0)))
        for mask, fl in zip(initial_zero, flowlines)
    )
    root = climate.root_metrics(flowlines, registered)
    volume_change = (final_volume - initial_volume) / initial_volume
    total_budget = final_volume - initial_volume - cumulative_mb
    core_checks = {
        "completed_target": abs(elapsed - target) <= 1e-5,
        "per_step_closure": max_closure <= closure_limit,
        "KKT_stationarity": max_kkt <= 1e-5,
        "minimum_timestep_floor": True,
        "finite_nonnegative": all(
            np.isfinite(fl.section).all() and np.isfinite(fl.thick).all()
            and np.all(np.asarray(fl.section) >= 0.0)
            and np.all(np.asarray(fl.thick) >= 0.0)
            for fl in flowlines
        ),
        "artificial_positive_support_absent": newly_activated == 0,
        "terminal_thickness": root["terminal_thickness_m"] <= 10.0,
        "volume_change": abs(volume_change) <= float(config["gates"]["volume_change_fraction_max"]),
        "root_length": root["length_relative_error"] <= float(config["gates"]["lia_length_relative_error_max"]),
        "free_downstream": root["free_downstream_m"] >= float(config["gates"]["minimum_free_downstream_m"]),
    }
    summary = {
        "schema": "lia-191-system2005-internal-gap-ledger-v2",
        "status": "full-50y-ledger-complete-core-gates-pass" if all(core_checks.values()) else "full-50y-ledger-complete-core-gates-fail",
        "formal_steady_adjudicated": False,
        "system_id": 2005,
        "scientific_state_modified": False,
        "solver_or_forcing_modified": False,
        "years": float(config["years"]),
        "step_count": step_count,
        "maximum_timestep_seconds": maximum_dt,
        "event_count": len(events),
        "events": events,
        "maximum_internal_gap_branch_count": maximum_gap_count,
        "maximum_event_duration_seconds": max((e["duration_seconds"] for e in events), default=0.0),
        "maximum_event_duration_steps": max((e["duration_steps"] for e in events), default=0),
        "initial_support_state": initial_support,
        "final_support_state": final_support,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_change_fraction": volume_change,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "area_change_fraction": (final_area - initial_area) / initial_area,
        "cumulative_accepted_mb_m3": cumulative_mb,
        "cumulative_prevented_overmelt_m3": cumulative_overmelt,
        "total_budget_closure_m3_diagnostic_only": total_budget,
        "maximum_absolute_step_closure_m3": max_closure,
        "maximum_absolute_KKT_stationarity": max_kkt,
        "maximum_active_constraint_count": max_active_constraints,
        "closure_limit_m3": closure_limit,
        "newly_activated_initial_zero_node_count": newly_activated,
        "root": root,
        "core_checks_excluding_gap_policy": core_checks,
        "runtime_seconds": time.perf_counter() - started,
        "formal_zero_mb_summary_sha256": base.sha256(zero_path),
        "source_model_flowlines_sha256": base.sha256(base.resolve(config["model_flowlines"])),
        "historical_run": False,
    }
    return summary, annual_rows, trace_rows, flowlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config_snapshot.json", config)
    try:
        summary, annual_rows, trace_rows, flowlines = run(config, output)
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-internal-gap-ledger-v2",
            "status": "technical-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "historical_run": False,
        }
        annual_rows, trace_rows, flowlines = [], [], None
    _write_json(output / "summary.json", summary)
    if annual_rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(annual_rows[0]))
            writer.writeheader(); writer.writerows(annual_rows)
    if trace_rows:
        with (output / "event_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
            writer.writeheader(); writer.writerows(trace_rows)
    if flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y_unadjudicated.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary))
    if summary.get("status") == "technical-fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
