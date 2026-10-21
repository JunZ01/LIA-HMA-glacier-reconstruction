"""Repeated-climate integration with connected-ice MB and conservative front transport."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_connected_front_step_v2 import connected_front_step_v2


PROJECT = Path(__file__).resolve().parents[4]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def ice_area(flowlines) -> float:
    return float(sum(np.sum(np.asarray(fl.widths_m)[np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M]) * fl.dx_meter for fl in flowlines))


def root_metrics(flowlines, registered_length: float) -> dict:
    root = next(fl for fl in flowlines if fl.flows_to is None)
    active = np.flatnonzero(np.asarray(root.thick) > base.ACTIVE_TOLERANCE_M)
    length = float((active[-1] + 1) * root.dx_meter) if active.size else 0.0
    free = float((root.nx - active[-1] - 1) * root.dx_meter) if active.size else float(root.nx * root.dx_meter)
    return {"length_m": length, "length_relative_error": abs(length - registered_length) / registered_length, "free_downstream_m": free, "terminal_thickness_m": float(root.thick[-1])}


def gap_count(flowlines) -> int:
    count = 0
    for fl in flowlines:
        active = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active.size:
            count += int(np.count_nonzero(np.diff(active) > 1))
    return count


def connected_ice_climate_source(flowlines, mb_model, model_year: float, dt_seconds: float) -> tuple[float, float, int]:
    accepted_volume = 0.0
    prevented_overmelt = 0.0
    remote_zero_nodes = 0
    for branch, fl in enumerate(flowlines):
        section_before = np.asarray(fl.section, dtype=float).copy()
        active = np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M
        remote_zero_nodes += int(np.count_nonzero(~active))
        mb = np.asarray(
            mb_model.get_monthly_mb(np.asarray(fl.surface_h), year=model_year, fl_id=branch, fls=flowlines),
            dtype=float,
        )
        requested = section_before.copy()
        requested[active] += float(dt_seconds) * mb[active] * np.asarray(fl.widths_m)[active]
        prevented_overmelt += float(np.sum(np.maximum(-requested[active], 0.0)) * fl.dx_meter)
        after = section_before.copy()
        after[active] = np.maximum(requested[active], 0.0)
        if np.any(after[~active] != section_before[~active]):
            raise RuntimeError("remote-zero-node-received-local-mb")
        accepted_volume += float(np.sum(after - section_before) * fl.dx_meter)
        try:
            with np.errstate(invalid="raise", over="raise"):
                fl.section = after
        except FloatingPointError as exc:
            raise RuntimeError(f"climate-section-setter-floating-error:{branch}:{exc}") from exc
    return accepted_volume, prevented_overmelt, remote_zero_nodes


def run(config: dict):
    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH, SEC_IN_YEAR
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    zero = json.loads(resolve(config["formal_zero_mb_summary"]).read_text(encoding="utf-8"))
    if zero.get("status") != "zero-mb-pass" or zero.get("system_id") != 2005:
        raise RuntimeError("connected-front-five-year-zero-gate-not-pass")
    if zero.get("maximum_internal_active_gap_count") != 0 or zero.get("posthoc_section_clipping") is not False:
        raise RuntimeError("connected-front-five-year-intermediate-gate-not-pass")
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=resolve(config["inversion_root"]) / "synthetic_gdir", reset=False)
    stage2 = config["stage2"]
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
    flowlines = base.load_flowlines(resolve(config["model_flowlines"]))
    coefficients = base.Coefficients(float(stage2["glen_a"]), float(stage2["fs"]))
    initial_volume = base.total_volume(flowlines)
    initial_area = ice_area(flowlines)
    closure_limit = max(1e-6, initial_volume * 1e-12)
    registered = float(config["registered_lia_length_m"])
    target = float(config["years"]) * float(SEC_IN_YEAR)
    max_dt = float(config["maximum_timestep_seconds"])
    elapsed = 0.0
    step_count = 0
    max_closure = 0.0
    max_node_residual = 0.0
    max_projection_conservation = 0.0
    max_gaps = 0
    max_limited = 0
    max_remote_zero = 0
    cumulative_mb = 0.0
    cumulative_overmelt = 0.0
    rows = []
    started = time.perf_counter()
    next_record_year = 0
    while elapsed < target - 1e-6:
        month_index = int(np.floor((elapsed + 1e-9) / float(SEC_IN_MONTH)))
        next_month = (month_index + 1) * float(SEC_IN_MONTH)
        dt = min(max_dt, next_month - elapsed, target - elapsed)
        if dt < 30.0 - 1e-9:
            raise RuntimeError(f"timestep-below-30s:{dt}")
        before = base.total_volume(flowlines)
        accepted_mb, prevented, remote_zero = connected_ice_climate_source(flowlines, mb_model, elapsed / float(SEC_IN_YEAR), dt)
        step = connected_front_step_v2(flowlines, coefficients, dt, closure_limit)
        after = base.total_volume(flowlines)
        budget_closure = after - before - accepted_mb
        if abs(budget_closure) > closure_limit:
            raise RuntimeError(f"climate-step-budget-closure:{budget_closure}:{closure_limit}")
        elapsed += dt
        step_count += 1
        cumulative_mb += accepted_mb
        cumulative_overmelt += prevented
        max_remote_zero = max(max_remote_zero, remote_zero)
        max_closure = max(max_closure, abs(budget_closure), abs(step["actual_closure_m3"]))
        max_node_residual = max(max_node_residual, step["maximum_absolute_linear_residual_m3"])
        max_projection_conservation = max(max_projection_conservation, abs(step["projection_global_conservation_m3"]))
        max_gaps = max(max_gaps, gap_count(flowlines), int(step["internal_active_gap_count"]))
        max_limited = max(max_limited, int(step["limited_edge_count"]))
        current_year = elapsed / float(SEC_IN_YEAR)
        if current_year + 1e-10 >= next_record_year or elapsed >= target - 1e-6:
            rows.append({"year": current_year, "volume_m3": after, "area_m2": ice_area(flowlines), **root_metrics(flowlines, registered), "internal_active_gap_count": gap_count(flowlines), "limited_edge_count": int(step["limited_edge_count"]), "cumulative_accepted_mb_m3": cumulative_mb, "cumulative_prevented_overmelt_m3": cumulative_overmelt})
            next_record_year = int(np.floor(current_year)) + 1
    final_volume = base.total_volume(flowlines)
    final_area = ice_area(flowlines)
    dvol = (final_volume - initial_volume) / initial_volume
    darea = (final_area - initial_area) / initial_area
    budget_total = final_volume - initial_volume - cumulative_mb
    rm = root_metrics(flowlines, registered)
    gates = config["gates"]
    checks = {
        "completed_target": abs(elapsed - target) <= 1e-5,
        "volume_change": abs(dvol) <= float(gates["volume_change_fraction_max"]),
        "root_length": rm["length_relative_error"] <= float(gates["lia_length_relative_error_max"]),
        "free_downstream": rm["free_downstream_m"] >= float(gates["minimum_free_downstream_m"]),
        "terminal_thickness": rm["terminal_thickness_m"] <= 10.0,
        "internal_gaps": max_gaps == 0,
        "per_step_closure": max_closure <= closure_limit,
        "total_budget_closure": abs(budget_total) <= closure_limit,
        "projection_conservation": max_projection_conservation <= closure_limit,
        "finite_nonnegative": all(np.isfinite(fl.section).all() and np.isfinite(fl.thick).all() and np.all(fl.section >= 0) and np.all(fl.thick >= 0) for fl in flowlines),
    }
    if float(config["years"]) == 50.0:
        status = "steady-50y-pass" if all(checks.values()) else "steady-50y-fail"
    else:
        status = "climate-screen-pass" if all(value for key, value in checks.items() if key != "volume_change") else "climate-screen-fail"
    summary = {
        "schema": "lia-191-network-connected-front-system2005-climate-v2",
        "status": status,
        "system_id": 2005,
        "mechanism": "network-capacity-connected-front-exact-zero-v2",
        "mass_balance_domain": "currently-connected-active-ice-only; one dry downstream front control per branch",
        "years": float(config["years"]),
        "maximum_timestep_seconds": max_dt,
        "step_count": step_count,
        "spinup_window": config["spinup_window"],
        "stage2": stage2,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_change_fraction": dvol,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "area_change_fraction": darea,
        "cumulative_accepted_mb_m3": cumulative_mb,
        "cumulative_prevented_overmelt_m3": cumulative_overmelt,
        "maximum_remote_zero_node_count": max_remote_zero,
        "total_budget_closure_m3": budget_total,
        "maximum_absolute_step_closure_m3": max_closure,
        "maximum_absolute_node_residual_m3": max_node_residual,
        "maximum_absolute_projection_conservation_m3": max_projection_conservation,
        "closure_limit_m3": closure_limit,
        "maximum_internal_active_gap_count": max_gaps,
        "maximum_limited_edge_count": max_limited,
        "root": rm,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "posthoc_section_clip_used": False,
        "remote_zero_ice_local_mb_used": False,
        "historical_run": False,
    }
    return summary, rows, flowlines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        summary, rows, flowlines = run(config)
    except Exception as exc:
        summary = {"schema": "lia-191-network-connected-front-system2005-climate-v2", "status": "climate-technical-or-scientific-fail", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "historical_run": False}
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if summary["status"] == "steady-50y-pass" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] in {"climate-technical-or-scientific-fail", "climate-screen-fail", "steady-50y-fail"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
