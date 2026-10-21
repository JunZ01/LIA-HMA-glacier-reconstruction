"""Repeated-climate screen for the system-2005 topology-adaptive network solver."""

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
    initial_zero = [np.asarray(fl.section) == 0.0 for fl in flowlines]
    initial_support = topology.support_state(flowlines, float(config["active_thickness_tolerance_m"]))
    coefficients = base.Coefficients(float(stage2["glen_a"]), float(stage2["fs"]))
    initial_volume = base.total_volume(flowlines)
    initial_area = climate.ice_area(flowlines)
    closure_limit = max(1e-6, initial_volume * 1e-12)
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
    max_gaps = initial_support["internal_gap_branch_count"]
    rows = []
    next_record_year = 0
    started = time.perf_counter()
    solver.conservative_boundary_step.config = config

    while elapsed < target - 1e-6:
        month_index = int(np.floor((elapsed + 1e-9) / float(SEC_IN_MONTH)))
        next_month = (month_index + 1) * float(SEC_IN_MONTH)
        dt = min(maximum_dt, next_month - elapsed, target - elapsed)
        if dt < float(config["minimum_timestep_seconds"]) - 1e-9:
            raise RuntimeError(f"timestep-below-floor:{dt}")
        before = base.total_volume(flowlines)
        accepted_mb, prevented, _ = climate.connected_ice_climate_source(
            flowlines, mb_model, elapsed / float(SEC_IN_YEAR), dt
        )
        step = solver.conservative_boundary_step(flowlines, coefficients, dt, closure_limit, 0.0)
        after = base.total_volume(flowlines)
        budget_closure = after - before - accepted_mb
        if abs(budget_closure) > closure_limit:
            raise RuntimeError(f"climate-step-budget-closure:{budget_closure}:{closure_limit}")
        elapsed += dt
        step_count += 1
        cumulative_mb += accepted_mb
        cumulative_overmelt += prevented
        max_closure = max(max_closure, abs(budget_closure), abs(float(step["actual_closure_m3"])))
        max_kkt = max(max_kkt, float(step["maximum_absolute_KKT_stationarity"]))
        max_active_constraints = max(max_active_constraints, int(step["active_constraint_count"]))
        support = topology.support_state(flowlines, float(config["active_thickness_tolerance_m"]))
        max_gaps = max(max_gaps, int(support["internal_gap_branch_count"]))
        year = elapsed / float(SEC_IN_YEAR)
        if year + 1e-10 >= next_record_year or elapsed >= target - 1e-6:
            rows.append({
                "year": year,
                "volume_m3": after,
                "area_m2": climate.ice_area(flowlines),
                **climate.root_metrics(flowlines, registered),
                "disconnected_tributary_count": support["disconnected_tributary_count"],
                "internal_gap_branch_count": support["internal_gap_branch_count"],
                "cumulative_accepted_mb_m3": cumulative_mb,
            })
            next_record_year = int(np.floor(year)) + 1

    final_volume = base.total_volume(flowlines)
    final_area = climate.ice_area(flowlines)
    final_support = topology.support_state(flowlines, float(config["active_thickness_tolerance_m"]))
    newly_activated = sum(
        int(np.count_nonzero(mask & (np.asarray(fl.section) > 0.0)))
        for mask, fl in zip(initial_zero, flowlines)
    )
    root = climate.root_metrics(flowlines, registered)
    volume_change = (final_volume - initial_volume) / initial_volume
    total_budget = final_volume - initial_volume - cumulative_mb
    numerical_checks = {
        "completed_target": abs(elapsed - target) <= 1e-5,
        "per_step_closure": max_closure <= closure_limit,
        "total_budget_closure": abs(total_budget) <= closure_limit,
        "KKT_stationarity": max_kkt <= 1e-5,
        "minimum_timestep_floor": True,
        "finite_nonnegative": all(
            np.isfinite(fl.section).all() and np.isfinite(fl.thick).all()
            and np.all(np.asarray(fl.section) >= 0.0) and np.all(np.asarray(fl.thick) >= 0.0)
            for fl in flowlines
        ),
        "internal_active_gaps_absent": max_gaps == 0,
        "artificial_positive_support_absent": newly_activated == 0,
        "terminal_thickness": root["terminal_thickness_m"] <= 10.0,
    }
    gates = config["gates"]
    steady_checks = {
        **numerical_checks,
        "volume_change": abs(volume_change) <= float(gates["volume_change_fraction_max"]),
        "root_length": root["length_relative_error"] <= float(gates["lia_length_relative_error_max"]),
        "free_downstream": root["free_downstream_m"] >= float(gates["minimum_free_downstream_m"]),
    }
    years = float(config["years"])
    if years == 50.0:
        status = "steady-50y-pass" if all(steady_checks.values()) else "steady-50y-fail"
    else:
        status = "climate-screen-pass" if all(numerical_checks.values()) else "climate-screen-fail"
    summary = {
        "schema": "lia-191-network-topology-adaptive-split-system2005-climate-v1",
        "status": status,
        "system_id": 2005,
        "mechanism": "mass-conserving-dynamic-active-component-split",
        "years": years,
        "maximum_timestep_seconds": maximum_dt,
        "minimum_timestep_seconds": float(config["minimum_timestep_seconds"]),
        "step_count": step_count,
        "spinup_window": config["spinup_window"],
        "stage2": stage2,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_change_fraction": volume_change,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "area_change_fraction": (final_area - initial_area) / initial_area,
        "cumulative_accepted_mb_m3": cumulative_mb,
        "cumulative_prevented_overmelt_m3": cumulative_overmelt,
        "total_budget_closure_m3": total_budget,
        "maximum_absolute_step_closure_m3": max_closure,
        "maximum_absolute_KKT_stationarity": max_kkt,
        "maximum_active_constraint_count": max_active_constraints,
        "closure_limit_m3": closure_limit,
        "maximum_internal_active_gap_count": max_gaps,
        "newly_activated_initial_zero_node_count": newly_activated,
        "initial_support_state": initial_support,
        "final_support_state": final_support,
        "root": root,
        "numerical_checks": numerical_checks,
        "steady_checks": steady_checks,
        "runtime_seconds": time.perf_counter() - started,
        "posthoc_section_clip_used": False,
        "artificial_positive_support_used": False,
        "formal_zero_mb_summary_sha256": base.sha256(zero_path),
        "historical_run": False,
    }
    return summary, rows, flowlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        summary, rows, flowlines = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-topology-adaptive-split-system2005-climate-v1",
            "status": "climate-technical-or-scientific-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "historical_run": False,
        }
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if summary.get("status") == "steady-50y-pass" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary))
    if summary.get("status") not in {"climate-screen-pass", "steady-50y-pass"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
