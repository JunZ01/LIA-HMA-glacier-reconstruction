"""Repeated-climate integration for the system-2005 capacity solver."""

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
from .network_capacity_constrained_system2005_step_v2 import capacity_one_step_v2


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


def internal_gap_count(flowlines) -> int:
    total = 0
    for fl in flowlines:
        active = np.flatnonzero(np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M)
        if active.size:
            total += int(np.count_nonzero(np.diff(active) > 1))
    return total


def climate_source_step(flowlines, mb_model, model_year: float, dt_seconds: float) -> tuple[float, float]:
    accepted_volume = 0.0
    prevented_overmelt = 0.0
    for branch, fl in enumerate(flowlines):
        section_before = np.asarray(fl.section, dtype=float).copy()
        widths = np.asarray(fl.widths_m, dtype=float)
        mb = np.asarray(mb_model.get_monthly_mb(np.asarray(fl.surface_h), year=model_year, fl_id=branch, fls=flowlines), dtype=float)
        effective_width = np.where((mb > 0.0) & (widths == 0.0), 10.0, widths)
        requested = section_before + float(dt_seconds) * mb * effective_width
        prevented_overmelt += float(np.sum(np.maximum(-requested, 0.0)) * fl.dx_meter)
        section_after = np.maximum(requested, 0.0)
        accepted_volume += float(np.sum(section_after - section_before) * fl.dx_meter)
        fl.section = section_after
    return accepted_volume, prevented_overmelt


def run(config: dict) -> tuple[dict, list[dict], object]:
    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH, SEC_IN_YEAR
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    zero = json.loads(resolve(config["formal_zero_mb_summary"]).read_text(encoding="utf-8"))
    if zero["status"] != "zero-mb-pass" or zero["system_id"] != 2005:
        raise RuntimeError("formal-zero-mb-gate-not-pass")
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=resolve(config["inversion_root"]) / "synthetic_gdir", reset=False)
    stage2 = config["stage2"]
    mb_model = MonthlyTIModel(gdir, filename="gcm_data", input_filesuffix=config["climate_filesuffix"], melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]), temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]), ys=int(config["spinup_window"][0]), ye=int(config["spinup_window"][1]), repeat=True, check_calib_params=False)
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
    max_gaps = 0
    max_limited = 0
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
        accepted_mb, prevented = climate_source_step(flowlines, mb_model, elapsed / float(SEC_IN_YEAR), dt)
        step = capacity_one_step_v2(flowlines, coefficients, dt, closure_limit)
        after = base.total_volume(flowlines)
        budget_closure = after - before - accepted_mb
        if abs(budget_closure) > closure_limit:
            raise RuntimeError(f"climate-step-budget-closure:{budget_closure}:{closure_limit}")
        elapsed += dt
        step_count += 1
        cumulative_mb += accepted_mb
        cumulative_overmelt += prevented
        max_closure = max(max_closure, abs(budget_closure), abs(step["actual_closure_m3"]))
        max_node_residual = max(max_node_residual, step["maximum_absolute_linear_residual_m3"])
        max_gaps = max(max_gaps, internal_gap_count(flowlines))
        max_limited = max(max_limited, int(step["limited_edge_count"]))
        current_year = elapsed / float(SEC_IN_YEAR)
        if current_year + 1e-10 >= next_record_year or elapsed >= target - 1e-6:
            rm = root_metrics(flowlines, registered)
            rows.append({"year": current_year, "volume_m3": after, "area_m2": ice_area(flowlines), **rm, "internal_active_gap_count": internal_gap_count(flowlines), "limited_edge_count": int(step["limited_edge_count"]), "cumulative_accepted_mb_m3": cumulative_mb, "cumulative_prevented_overmelt_m3": cumulative_overmelt})
            next_record_year = int(np.floor(current_year)) + 1
    final_volume = base.total_volume(flowlines)
    final_area = ice_area(flowlines)
    dvol = (final_volume - initial_volume) / initial_volume
    darea = (final_area - initial_area) / initial_area
    budget_total = final_volume - initial_volume - cumulative_mb
    rm = root_metrics(flowlines, registered)
    gates = config["gates"]
    checks = {"completed_target": abs(elapsed - target) <= 1e-5, "volume_change": abs(dvol) <= float(gates["volume_change_fraction_max"]), "root_length": rm["length_relative_error"] <= float(gates["lia_length_relative_error_max"]), "free_downstream": rm["free_downstream_m"] >= float(gates["minimum_free_downstream_m"]), "terminal_thickness": rm["terminal_thickness_m"] <= 10.0, "internal_gaps": max_gaps == 0, "per_step_closure": max_closure <= closure_limit, "total_budget_closure": abs(budget_total) <= closure_limit, "finite_nonnegative": all(np.isfinite(fl.section).all() and np.isfinite(fl.thick).all() and np.all(fl.section >= -1e-8) and np.all(fl.thick >= -1e-10) for fl in flowlines)}
    status = "steady-50y-pass" if float(config["years"]) == 50.0 and all(checks.values()) else ("climate-screen-pass" if all(value for key, value in checks.items() if key != "volume_change") else "climate-screen-fail")
    summary = {"status": status, "system_id": 2005, "mechanism": "network-dag-capacity-constrained-least-change-flux-v1", "years": float(config["years"]), "maximum_timestep_seconds": max_dt, "step_count": step_count, "spinup_window": config["spinup_window"], "stage2": stage2, "initial_volume_m3": initial_volume, "final_volume_m3": final_volume, "volume_change_fraction": dvol, "initial_area_m2": initial_area, "final_area_m2": final_area, "area_change_fraction": darea, "cumulative_accepted_mb_m3": cumulative_mb, "cumulative_prevented_overmelt_m3": cumulative_overmelt, "total_budget_closure_m3": budget_total, "maximum_absolute_step_closure_m3": max_closure, "maximum_absolute_node_residual_m3": max_node_residual, "closure_limit_m3": closure_limit, "maximum_internal_active_gap_count": max_gaps, "maximum_limited_edge_count": max_limited, "root": rm, "checks": checks, "runtime_seconds": time.perf_counter() - started, "posthoc_section_clip_used": False, "historical_run": False}
    return summary, rows, flowlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        summary, rows, flowlines = run(config)
    except Exception as exc:
        summary = {"status": "climate-technical-or-scientific-fail", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "historical_run": False}
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    if summary["status"] == "steady-50y-pass" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] in {"climate-technical-or-scientific-fail", "climate-screen-fail"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
