"""Checkpointable 1470--2000 system-2005 historical endpoint with a gap ledger.

The physical update is exactly the registered topology-adaptive split step used
by the passing 50-year steady state.  This driver changes only the prescribed
historical temperature ramp and records, but never repairs, support events.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import stat
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..paths import resolve
from . import diagnose_system2005_first_internal_gap_v1 as event_tools
from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from . import network_topology_adaptive_split_system2005_zero_mb_v9 as solver
from . import synthetic_lia_conservative_junction_historical_ramp_v1 as ramp


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CHECKPOINT_SCHEMA = "lia-191-system2005-historical-checkpoint-v2"


def _checkpoint_contract(config: dict, steady_state_sha256: str) -> dict:
    keys = (
        "system_id", "main_rgi_id", "synthetic_rgi_id", "inversion_root",
        "steady_root", "tributary_separation_policy", "climate_pixels", "climate_id", "reconstruction_climate",
        "start_year", "end_year", "forward_delta_t_end_c",
        "maximum_timestep_seconds", "minimum_timestep_seconds",
        "section_solver_interior_floor_m2", "active_set_tolerance",
        "boundary_canonicalization_tolerance_m2", "active_thickness_tolerance_m",
        "lia_registered_length_m", "rgi_registered_length_m", "stage2", "formal_gates",
    )
    return {
        "schema": "lia-191-system2005-historical-checkpoint-contract-v2",
        "solver": "network-topology-adaptive-split-system2005-zero-mb-v9",
        "steady_state_sha256": steady_state_sha256,
        "config": {key: config[key] for key in keys},
    }


def _contract_sha256(contract: dict) -> str:
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_checkpoint(output: Path, year: int, contract_sha256: str, state: dict) -> Path:
    checkpoint_root = output / "checkpoints" / f"year_{int(year):04d}"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    state_path = checkpoint_root / "state.pkl.gz"
    state_tmp = checkpoint_root / "state.pkl.gz.tmp"
    with gzip.open(state_tmp, "wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(state_tmp, state_path)
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint-complete",
        "system_id": 2005,
        "year": int(year),
        "step_count": int(state["step_count"]),
        "elapsed_seconds": float(state["elapsed"]),
        "contract_sha256": contract_sha256,
        "state_file": state_path.name,
        "state_sha256": _sha256(state_path),
    }
    manifest_path = checkpoint_root / "manifest.json"
    manifest_tmp = checkpoint_root / "manifest.json.tmp"
    _write_json(manifest_tmp, manifest)
    os.replace(manifest_tmp, manifest_path)
    return manifest_path


def _load_checkpoint(manifest_path: Path, contract_sha256: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("status") != "checkpoint-complete":
        raise RuntimeError("resume-checkpoint-manifest-invalid")
    if manifest.get("contract_sha256") != contract_sha256:
        raise RuntimeError("resume-checkpoint-contract-mismatch")
    state_path = manifest_path.parent / str(manifest["state_file"])
    if _sha256(state_path) != manifest.get("state_sha256"):
        raise RuntimeError("resume-checkpoint-state-hash-mismatch")
    with gzip.open(state_path, "rb") as stream:
        state = pickle.load(stream)
    if int(state["step_count"]) != int(manifest["step_count"]):
        raise RuntimeError("resume-checkpoint-step-mismatch")
    return manifest, state

def _isolated_ramp_climate(gdir, config, pixel):
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    suffix, audit = ramp._write_case_climate(
        gdir,
        Path(config["reconstruction_climate"]),
        pixel,
        "off",
        start_year=start_year,
        end_year=end_year,
    )
    start_bias = float(config["stage2"]["delta_t_c"])
    end_bias = float(config["forward_delta_t_end_c"])
    ramp_suffix = f"{suffix}_bias_ramp_{ramp._tag(start_bias)}_to_{ramp._tag(end_bias)}"
    source_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    target_path = Path(gdir.get_filepath("gcm_data", filesuffix=ramp_suffix))
    if target_path.exists():
        target_path.chmod(target_path.stat().st_mode | stat.S_IWRITE | stat.S_IREAD)
        target_path.unlink()
    with xr.open_dataset(source_path, use_cftime=True) as source:
        dataset = source.load()
    decimal_year = np.asarray(
        [date.year + (date.month - 1) / 12.0 for date in dataset["time"].values],
        dtype=float,
    )
    fraction = np.clip(
        (decimal_year - float(start_year)) / float(end_year - start_year), 0.0, 1.0
    )
    attrs = dict(dataset["temp"].attrs)
    dataset["temp"] = dataset["temp"] + xr.DataArray(
        (end_bias - start_bias) * fraction,
        dims=("time",),
        coords={"time": dataset["time"]},
    )
    dataset["temp"].attrs = attrs
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs["forward_temperature_bias_policy"] = (
        f"linear effective bias from {start_bias:+.6f} K at {start_year} "
        f"to {end_bias:+.6f} K at {end_year}"
    )
    dataset.to_netcdf(target_path)
    dataset.close()
    return ramp_suffix, {
        **audit,
        "base_climate_file": str(source_path),
        "climate_file": str(target_path),
        "sha256": _sha256(target_path),
        "temperature_bias_policy": "linear-ramp-in-forward-climate-file",
        "effective_delta_t_start_c": start_bias,
        "effective_delta_t_end_c": end_bias,
        "additional_warming_2000_c": end_bias - start_bias,
        "isolated_clone_existing_ramp_cache_rebuilt": True,
    }

def _snapshot(flowlines, year: int, registered_lia_length_m: float, support: dict) -> dict:
    root = climate.root_metrics(flowlines, registered_lia_length_m)
    return {
        "year": int(year),
        "volume_m3": base.total_volume(flowlines),
        "area_m2": climate.ice_area(flowlines),
        **root,
        "finite_geometry": bool(all(
            np.isfinite(fl.section).all() and np.isfinite(fl.thick).all()
            for fl in flowlines
        )),
        "negative_section_count": int(sum(
            np.count_nonzero(np.asarray(fl.section) < -1e-10) for fl in flowlines
        )),
        "disconnected_tributary_count": int(support["disconnected_tributary_count"]),
        "internal_gap_branch_count": int(support["internal_gap_branch_count"]),
    }


def _node_snapshot(flowlines, event_nodes, previous_sections, step_count,
                   model_year, phase, tolerance):
    rows = []
    event_tools._record_nodes(
        rows, flowlines, sorted(event_nodes), event_nodes, previous_sections,
        step_count, model_year, phase, tolerance,
    )
    for row in rows:
        row["year"] = model_year
    return rows


def run(config: dict, output: Path):
    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH, SEC_IN_YEAR
    from oggm.core.massbalance import MonthlyTIModel

    steady_root = resolve(config["steady_root"])
    steady_summary_path = steady_root / "steady_50y_summary_formal.json"
    steady = json.loads(steady_summary_path.read_text(encoding="utf-8"))
    if steady.get("status") != "steady-50y-pass" or not all(steady.get("checks", {}).values()):
        raise RuntimeError("registered-formal-steady-not-pass")
    state_path = steady_root / "lia_equilibrium_state_50y.pkl.gz"
    expected_state_hash = str(steady["evidence"]["formal_state_sha256"]).lower()
    if _sha256(state_path).lower() != expected_state_hash:
        raise RuntimeError("registered-formal-steady-state-hash-mismatch")
    checkpoint_contract = _checkpoint_contract(config, expected_state_hash)
    checkpoint_contract_sha256 = _contract_sha256(checkpoint_contract)

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    target_gdir_base = output / "synthetic_gdir"
    shutil.copytree(resolve(config["inversion_root"]) / "synthetic_gdir", target_gdir_base)
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=target_gdir_base, reset=False)
    pixels = pd.read_csv(resolve(config["climate_pixels"]))
    selected = pixels[
        pixels["rgi_id"].astype(str).eq(str(config["main_rgi_id"]))
        & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
    ]
    if len(selected) != 1:
        raise RuntimeError(f"climate-pixel-row-count:{len(selected)}")
    suffix, climate_audit = _isolated_ramp_climate(gdir, config, selected.iloc[0])
    _write_json(output / "climate_audit.json", climate_audit)
    stage2 = config["stage2"]
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year,
        ye=end_year,
        repeat=False,
        check_calib_params=False,
    )
    coefficients = base.Coefficients(float(stage2["glen_a"]), float(stage2["fs"]))
    tolerance = float(config["active_thickness_tolerance_m"])
    registered_lia = float(config["lia_registered_length_m"])
    target_seconds = float(end_year - start_year) * float(SEC_IN_YEAR)
    maximum_dt = float(config["maximum_timestep_seconds"])
    minimum_dt = float(config["minimum_timestep_seconds"])
    resume_manifest_path = config.get("resume_checkpoint_manifest")
    resumed_from_checkpoint = resume_manifest_path is not None
    resume_manifest = None
    prior_runtime_seconds = 0.0
    if resumed_from_checkpoint:
        resume_manifest, checkpoint_state = _load_checkpoint(
            resolve(resume_manifest_path), checkpoint_contract_sha256
        )
        flowlines = checkpoint_state["flowlines"]
        closure_limit = float(checkpoint_state["closure_limit"])
        elapsed = float(checkpoint_state["elapsed"])
        step_count = int(checkpoint_state["step_count"])
        cumulative_mb = float(checkpoint_state["cumulative_mb"])
        cumulative_overmelt = float(checkpoint_state["cumulative_overmelt"])
        max_closure = float(checkpoint_state["max_closure"])
        max_kkt = float(checkpoint_state["max_kkt"])
        max_active_constraints = int(checkpoint_state["max_active_constraints"])
        maximum_gap_count = int(checkpoint_state["maximum_gap_count"])
        events = checkpoint_state["events"]
        current_event = checkpoint_state["current_event"]
        trace_rows = checkpoint_state["trace_rows"]
        annual_rows = checkpoint_state["annual_rows"]
        next_record_year = int(checkpoint_state["next_record_year"])
        prior_runtime_seconds = float(checkpoint_state["runtime_seconds_completed"])
        expected_elapsed = float(resume_manifest["year"] - start_year) * float(SEC_IN_YEAR)
        if abs(elapsed - expected_elapsed) > 1e-6:
            raise RuntimeError("resume-checkpoint-year-elapsed-mismatch")
    else:
        with gzip.open(state_path, "rb") as stream:
            flowlines = pickle.load(stream)
        initial_volume = base.total_volume(flowlines)
        closure_limit = max(1e-6, initial_volume * 1e-12)
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
        support = topology.support_state(flowlines, tolerance)
        annual_rows = [_snapshot(flowlines, start_year, registered_lia, support)]
        next_record_year = start_year + 1
    support = topology.support_state(flowlines, tolerance)
    started = time.perf_counter()
    solver.reset_structure_cache()
    solver.conservative_boundary_step.config = config

    while elapsed < target_seconds - 1e-6:
        month_index = int(np.floor((elapsed + 1e-9) / float(SEC_IN_MONTH)))
        next_month = (month_index + 1) * float(SEC_IN_MONTH)
        dt = min(maximum_dt, next_month - elapsed, target_seconds - elapsed)
        if dt < minimum_dt - 1e-9:
            raise RuntimeError(f"timestep-below-floor:{dt}:{minimum_dt}")
        previous_sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
        before = base.total_volume(flowlines)
        model_year = start_year + elapsed / float(SEC_IN_YEAR)
        accepted_mb, prevented, _ = climate.connected_ice_climate_source(
            flowlines, mb_model, model_year, dt
        )
        step = solver.conservative_boundary_step(
            flowlines, coefficients, dt, closure_limit, 0.0
        )
        after = base.total_volume(flowlines)
        budget_closure = after - before - accepted_mb
        if abs(budget_closure) > closure_limit:
            raise RuntimeError(f"historical-step-budget-closure:{budget_closure}:{closure_limit}")
        elapsed += dt
        step_count += 1
        cumulative_mb += accepted_mb
        cumulative_overmelt += prevented
        max_closure = max(max_closure, abs(budget_closure), abs(float(step["actual_closure_m3"])))
        max_kkt = max(max_kkt, float(step["maximum_absolute_KKT_stationarity"]))
        max_active_constraints = max(max_active_constraints, int(step["active_constraint_count"]))
        support = topology.support_state(flowlines, tolerance)
        gap_count = int(support["internal_gap_branch_count"])
        maximum_gap_count = max(maximum_gap_count, gap_count)
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
            trace_rows.extend(_node_snapshot(
                flowlines, nodes, previous_sections, step_count, absolute_year,
                "onset", tolerance,
            ))
        elif gap_count and current_event is not None:
            current_event["maximum_gap_branch_count"] = max(
                int(current_event["maximum_gap_branch_count"]), gap_count
            )
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
            current_event["duration_steps"] = step_count - int(current_event["start_step"])
            current_event["duration_seconds"] = elapsed - float(current_event.pop("start_seconds"))
            current_event["branches"] = sorted(current_event["branches"])
            trace_rows.extend(_node_snapshot(
                flowlines, current_event["nodes"], previous_sections,
                step_count, absolute_year, "closed", tolerance,
            ))
            events.append(current_event)
            current_event = None

        if absolute_year + 1e-10 >= next_record_year or elapsed >= target_seconds - 1e-6:
            checkpoint_year = next_record_year
            annual_rows.append(_snapshot(flowlines, checkpoint_year, registered_lia, support))
            next_record_year += 1
            runtime_seconds_completed = prior_runtime_seconds + time.perf_counter() - started
            _write_json(output / "progress.json", {
                "schema": "lia-191-system2005-topology-adaptive-historical-progress-v3-checkpoint",
                "system_id": 2005,
                "year": absolute_year,
                "step_count": step_count,
                "completed_gap_event_count": len(events),
                "gap_event_open": current_event is not None,
                "maximum_internal_gap_branch_count": maximum_gap_count,
                "runtime_seconds": runtime_seconds_completed,
                "checkpoint_contract_sha256": checkpoint_contract_sha256,
            })
            checkpoint_state = {
                "schema": CHECKPOINT_SCHEMA,
                "flowlines": flowlines,
                "closure_limit": closure_limit,
                "elapsed": elapsed,
                "step_count": step_count,
                "cumulative_mb": cumulative_mb,
                "cumulative_overmelt": cumulative_overmelt,
                "max_closure": max_closure,
                "max_kkt": max_kkt,
                "max_active_constraints": max_active_constraints,
                "maximum_gap_count": maximum_gap_count,
                "events": events,
                "current_event": current_event,
                "trace_rows": trace_rows,
                "annual_rows": annual_rows,
                "next_record_year": next_record_year,
                "runtime_seconds_completed": runtime_seconds_completed,
            }
            checkpoint_manifest = _write_checkpoint(
                output, checkpoint_year, checkpoint_contract_sha256, checkpoint_state
            )
            if int(config.get("stop_after_checkpoint_year", -1)) == int(checkpoint_year):
                raise RuntimeError(
                    f"intentional-stop-after-checkpoint:{checkpoint_year}:{checkpoint_manifest}"
                )

    if current_event is not None:
        current_event["end_step"] = None
        current_event["end_year"] = None
        current_event["duration_steps"] = step_count - int(current_event["start_step"])
        current_event["duration_seconds"] = elapsed - float(current_event.pop("start_seconds"))
        current_event["open_at_final"] = True
        events.append(current_event)

    frame = pd.DataFrame(annual_rows)
    lia_length = float(config["lia_registered_length_m"])
    rgi_length = float(config["rgi_registered_length_m"])
    separation = abs(lia_length - rgi_length)
    peak_row = frame.loc[frame["length_m"].idxmax()]
    final = annual_rows[-1]
    e_lia_total = abs(float(peak_row["length_m"]) - lia_length) / lia_length
    e_lia_sep = abs(float(peak_row["length_m"]) - lia_length) / separation
    e_rgi_total = abs(float(final["length_m"]) - rgi_length) / rgi_length
    e_rgi_sep = abs(float(final["length_m"]) - rgi_length) / separation
    numerical_checks = {
        "completed_target": abs(elapsed - target_seconds) <= 1e-5,
        "per_step_closure": max_closure <= closure_limit,
        "KKT_stationarity": max_kkt <= 1e-5,
        "finite_nonnegative": bool(frame["finite_geometry"].all()) and int(frame["negative_section_count"].max()) == 0,
        "root_boundary_ice_absent": max(float(row["terminal_thickness_m"]) for row in annual_rows) <= 10.0,
        "observation_domain_overflow_absent": min(float(row["free_downstream_m"]) for row in annual_rows) >= 0.0,
    }
    length_checks = {
        "lia_total": e_lia_total <= float(config["formal_gates"]["total_length_error_max"]),
        "lia_separation": e_lia_sep <= float(config["formal_gates"]["separation_length_error_max"]),
        "rgi_total": e_rgi_total <= float(config["formal_gates"]["total_length_error_max"]),
        "rgi_separation": e_rgi_sep <= float(config["formal_gates"]["separation_length_error_max"]),
    }
    volume_1500_2000 = None
    if frame["year"].eq(1500).any() and frame["year"].eq(2000).any():
        volume_1500 = float(frame.loc[frame["year"].eq(1500), "volume_m3"].iloc[0])
        volume_2000 = float(frame.loc[frame["year"].eq(2000), "volume_m3"].iloc[0])
        volume_1500_2000 = (volume_2000 - volume_1500) / volume_1500
    return {
        "schema": "lia-191-system2005-topology-adaptive-historical-v3-checkpoint",
        "status": "historical-complete-unadjudicated-gap-ledger",
        "system_id": 2005,
        "start_year": start_year,
        "end_year": end_year,
        "forward_delta_t_start_c": float(stage2["delta_t_c"]),
        "forward_delta_t_end_c": float(config["forward_delta_t_end_c"]),
        "step_count": step_count,
        "maximum_timestep_seconds": maximum_dt,
        "minimum_timestep_seconds": minimum_dt,
        "initial": annual_rows[0],
        "peak": {key: peak_row[key].item() if hasattr(peak_row[key], "item") else peak_row[key] for key in peak_row.index},
        "final": final,
        "e_lia_total": e_lia_total,
        "e_lia_separation": e_lia_sep,
        "e_rgi_total": e_rgi_total,
        "e_rgi_separation": e_rgi_sep,
        "length_checks": length_checks,
        "numerical_checks": numerical_checks,
        "event_count": len(events),
        "events": events,
        "maximum_internal_gap_branch_count": maximum_gap_count,
        "final_internal_gap_branch_count": int(topology.support_state(flowlines, tolerance)["internal_gap_branch_count"]),
        "maximum_absolute_step_closure_m3": max_closure,
        "maximum_absolute_KKT_stationarity": max_kkt,
        "maximum_active_constraint_count": max_active_constraints,
        "closure_limit_m3": closure_limit,
        "cumulative_accepted_mb_m3": cumulative_mb,
        "cumulative_prevented_overmelt_m3": cumulative_overmelt,
        "volume_change_fraction_start_end": (
            (float(final["volume_m3"]) - float(annual_rows[0]["volume_m3"]))
            / float(annual_rows[0]["volume_m3"])
        ),
        "volume_change_fraction_1500_2000": volume_1500_2000,
        "scientific_state_modified": False,
        "solver_or_forcing_modified_beyond_registered_temperature_ramp": False,
        "formal_historical_adjudicated": False,
        "climate": climate_audit,
        "runtime_seconds": prior_runtime_seconds + time.perf_counter() - started,
        "steady_summary_sha256": _sha256(steady_summary_path),
        "steady_state_sha256": expected_state_hash,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resume_checkpoint_manifest": None if resume_manifest is None else str(resolve(resume_manifest_path)),
        "realized_reverse_tolerance_m": solver.REALIZED_REVERSE_TOLERANCE_M,
    }, annual_rows, trace_rows, flowlines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config_snapshot.json", config)
    try:
        summary, rows, trace, flowlines = run(config, output)
        runtime_error = None
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-topology-adaptive-historical-v3-checkpoint",
            "status": "technical-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "formal_historical_adjudicated": False,
        }
        rows, trace, flowlines, runtime_error = [], [], None, exc
    _write_json(output / "historical_summary_unadjudicated.json", summary)
    if rows:
        with (output / "historical_1470_2000.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    if trace:
        with (output / "event_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader(); writer.writerows(trace)
    if flowlines is not None:
        with gzip.open(output / "state_2000_unadjudicated.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary, allow_nan=False))
    if runtime_error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

