"""Checkpointed 50-year source-mask MonthlyTI steady test for system 2005."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from netCDF4 import Dataset

from .run_system2005_masked_climate_1y_preflight_v1 import split_step
from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge
from .torch_planform_2d_system2005_adaptive_relaxation_60s_v1 import ERROR_GATE
from .torch_planform_2d_system2005_adaptive_relaxation_600s_v1 import (
    _area_metrics,
    _with_relative_area_change,
)
from .torch_planform_2d_system2005_steady50_monthly_ti_formal_v1 import (
    _control_lengths,
    _free_domain_metrics,
    _path_metrics,
)


TARGET_YEARS = 50
INITIAL_DT_SECONDS = 0.03
MINIMUM_DT_SECONDS = 0.001
MAXIMUM_DT_SECONDS = 86_400.0
MAXIMUM_CONSECUTIVE_REJECTIONS = 20
MAXIMUM_ACCEPTED_STEPS = 500_000
GROWTH_ERROR_THRESHOLD = ERROR_GATE / 4.0
VOLUME_CHANGE_GATE = 0.02
LENGTH_RELATIVE_ERROR_GATE = 0.10
FREE_DOMAIN_GATE_M = 5_000.0
LENGTH_ICE_THRESHOLD_M = 10.0
CUMULATIVE_BUDGET_RELATIVE_GATE = 1.0e-10
CHECKPOINT_SCHEMA = "lia-191-system2005-masked-steady50-checkpoint-v1"
GENERIC_STEADY_SCHEMA = "lia-191-generic-planform2d-gpu-steady50-config-v1"


def is_generic_steady_config(config: dict[str, Any]) -> bool:
    return config.get("schema") == GENERIC_STEADY_SCHEMA


def climate_month_contract(config: dict[str, Any], package_month_count: int) -> tuple[int, bool]:
    if not is_generic_steady_config(config):
        if package_month_count != TARGET_YEARS * 12:
            raise RuntimeError("climate-month-count-not-600")
        return package_month_count, False
    period_months = int(config.get("parent_climate_period_months", 0))
    repeat = bool(config.get("repeat_parent_climate", False))
    if period_months <= 0 or package_month_count != period_months:
        raise RuntimeError("parent-climate-period-mismatch")
    if not repeat:
        raise RuntimeError("parent-climate-repeat-not-authorized")
    return period_months, True


def source_month_index(global_month: int, period_months: int, repeat: bool) -> int:
    return global_month % period_months if repeat else min(global_month, period_months - 1)


def expected_checkpoint_years(target_years: int) -> set[str]:
    return {str(year) for year in range(1, target_years + 1)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "system_id",
        "target_years",
        "geology",
        "geology_sha256",
        "static_summary",
        "static_summary_sha256",
        "climate_package",
        "climate_package_sha256",
        "observation_path",
        "observation_path_sha256",
        "domain_variable",
        "glen_a",
        "driver_sha256",
        "expected_route",
        "expected_route_sha256",
    )
    payload = {key: config[key] for key in keys}
    if is_generic_steady_config(config):
        payload["parent_climate_start_year"] = int(config["parent_climate_start_year"])
        payload["parent_climate_end_year"] = int(config["parent_climate_end_year"])
        payload["parent_climate_period_months"] = int(config["parent_climate_period_months"])
        payload["repeat_parent_climate"] = bool(config["repeat_parent_climate"])
    payload["solver"] = {
        "operator": "source-half/masked-frozen-diffusion/source-half",
        "initial_dt_seconds": INITIAL_DT_SECONDS,
        "minimum_dt_seconds": MINIMUM_DT_SECONDS,
        "maximum_dt_seconds": MAXIMUM_DT_SECONDS,
        "error_gate_relative_l2": ERROR_GATE,
        "monthly_boundaries_exact": True,
        "annual_atomic_checkpoints": True,
    }
    payload["gates"] = {
        "volume_change_relative": VOLUME_CHANGE_GATE,
        "registered_length_relative_error": LENGTH_RELATIVE_ERROR_GATE,
        "free_domain_m": FREE_DOMAIN_GATE_M,
        "outside_domain_zero": True,
        "cumulative_budget_relative": CUMULATIVE_BUDGET_RELATIVE_GATE,
    }
    return payload


def contract_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(json_bytes(contract_payload(config))).hexdigest().upper()


def write_checkpoint(
    checkpoint_root: Path,
    *,
    year: int,
    state: np.ndarray,
    elapsed_seconds: float,
    proposed_dt_seconds: float,
    counters: dict[str, Any],
    checkpoint_metrics: dict[str, Any],
    contract_hash: str,
) -> Path:
    directory = checkpoint_root / f"year_{year:03d}"
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=False)
    state_path = directory / "state.npz"
    temporary_state = directory / "state.tmp.npz"
    np.savez_compressed(
        temporary_state,
        thickness_m=np.asarray(state, dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        proposed_dt_seconds=np.asarray(proposed_dt_seconds, dtype=np.float64),
        counters_json=np.asarray(json.dumps(counters, sort_keys=True, allow_nan=False)),
        checkpoint_metrics_json=np.asarray(
            json.dumps(checkpoint_metrics, sort_keys=True, allow_nan=False)
        ),
        contract_sha256=np.asarray(contract_hash),
        schema=np.asarray(CHECKPOINT_SCHEMA),
    )
    os.replace(temporary_state, state_path)
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint-complete",
        "year": year,
        "elapsed_seconds": elapsed_seconds,
        "contract_sha256": contract_hash,
        "state_file": "state.npz",
        "state_sha256": sha256(state_path),
    }
    manifest_path = directory / "manifest.json"
    atomic_bytes(manifest_path, json_bytes(manifest))
    atomic_bytes(
        checkpoint_root / "latest.json",
        json_bytes({"status": "checkpoint-complete", "manifest": str(manifest_path.resolve())}),
    )
    return manifest_path


def load_checkpoint(manifest_path: Path, expected_contract_hash: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "checkpoint-complete" or manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("checkpoint-manifest-incomplete")
    if manifest.get("contract_sha256") != expected_contract_hash:
        raise RuntimeError("checkpoint-contract-hash-mismatch")
    state_path = manifest_path.parent / manifest["state_file"]
    if sha256(state_path) != manifest["state_sha256"]:
        raise RuntimeError("checkpoint-state-hash-mismatch")
    with np.load(state_path, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != CHECKPOINT_SCHEMA:
            raise RuntimeError("checkpoint-state-schema-mismatch")
        if str(archive["contract_sha256"].item()) != expected_contract_hash:
            raise RuntimeError("checkpoint-state-contract-mismatch")
        result = {
            "state": np.asarray(archive["thickness_m"], dtype=np.float64),
            "elapsed_seconds": float(archive["elapsed_seconds"].item()),
            "proposed_dt_seconds": float(archive["proposed_dt_seconds"].item()),
            "counters": json.loads(str(archive["counters_json"].item())),
            "checkpoint_metrics": json.loads(str(archive["checkpoint_metrics_json"].item())),
            "manifest": manifest,
        }
    if not math.isclose(
        result["elapsed_seconds"], float(manifest["elapsed_seconds"]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("checkpoint-elapsed-mismatch")
    return result


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project_root"])
    paths = {
        name: root / config[name]
        for name in ("geology", "static_summary", "climate_package", "observation_path", "expected_route")
    }
    for name, path in paths.items():
        expected = config[f"{name}_sha256"]
        if sha256(path) != expected.upper():
            raise RuntimeError(f"{name}-hash-mismatch")
    static = json.loads(paths["static_summary"].read_text(encoding="utf-8"))
    with Dataset(paths["geology"]) as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        mask = np.asarray(dataset.variables[config["domain_variable"]][:], dtype=float) > 0.5
        x = np.asarray(dataset.variables["x"][:], dtype=np.float64)
        y = np.asarray(dataset.variables["y"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    with paths["observation_path"].open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    path_xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    controls = _control_lengths(path_xy)
    registered_length = float(static["registered_lia_length_m"])
    if not np.isfinite(bed).all() or not np.isfinite(thickness).all() or np.any(thickness < 0.0):
        raise RuntimeError("invalid-initial-grid")
    if np.any((thickness > 0.0) & ~mask):
        raise RuntimeError("initial-ice-outside-domain")
    return {
        "root": root,
        "paths": paths,
        "bed": bed,
        "initial": thickness,
        "mask": mask,
        "x": x,
        "y": y,
        "dx": dx,
        "path_xy": path_xy,
        "controls": controls,
        "registered_length": registered_length,
    }


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    generic = is_generic_steady_config(config)
    if config["target_years"] != TARGET_YEARS or (not generic and config["system_id"] != 2005):
        raise RuntimeError("target-contract-mismatch")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("runtime-version-mismatch")
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")
    loaded = load_inputs(config)
    output = loaded["root"] / config["output_root"]
    contract_hash = contract_sha256(config)
    checkpoint_root = output / "checkpoints"
    resumed = resume_manifest is not None
    if resumed:
        if not output.is_dir() or not checkpoint_root.is_dir():
            raise RuntimeError("resume-output-missing")
        restored = load_checkpoint(resume_manifest.resolve(), contract_hash)
        state_np = restored["state"]
        elapsed = restored["elapsed_seconds"]
        proposed_dt = restored["proposed_dt_seconds"]
        counters = restored["counters"]
        checkpoint_metrics = restored["checkpoint_metrics"]
        start_year = int(restored["manifest"]["year"])
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=False)
        checkpoint_root.mkdir()
        atomic_bytes(output / "config_snapshot.json", json_bytes(config))
        atomic_bytes(output / "checkpoint_contract.json", json_bytes(contract_payload(config)))
        state_np = loaded["initial"].copy()
        elapsed = 0.0
        proposed_dt = INITIAL_DT_SECONDS
        counters = {
            "accepted_steps": 0,
            "rejected_steps": 0,
            "solver_calls": 0,
            "projection_calls_accepted_path": 0,
            "cumulative_source_m3": 0.0,
            "cumulative_prevented_overmelt_m3": 0.0,
            "maximum_step_closure_m3": 0.0,
            "maximum_budget_residual_m3": 0.0,
            "maximum_error_relative_l2": 0.0,
            "minimum_accepted_dt_seconds": None,
            "maximum_accepted_dt_seconds": 0.0,
        }
        checkpoint_metrics = {}
        start_year = 0

    if state_np.shape != loaded["initial"].shape or np.any(state_np[~loaded["mask"]] != 0.0):
        raise RuntimeError("resume-state-domain-mismatch")
    bridge = TorchMonthlyTIBridge(loaded["paths"]["climate_package"], device="cuda:0")
    climate_period_months, repeat_parent_climate = climate_month_contract(config, bridge.month_count)
    bed = torch.as_tensor(loaded["bed"], dtype=torch.float64, device="cuda:0")
    state = torch.as_tensor(state_np, dtype=torch.float64, device="cuda:0")
    mask = torch.as_tensor(loaded["mask"], dtype=torch.bool, device="cuda:0")
    initial = torch.as_tensor(loaded["initial"], dtype=torch.float64, device="cuda:0")
    cell_area = loaded["dx"] ** 2
    initial_volume = float(torch.sum(initial).item()) * cell_area
    per_step_tolerance = max(1e-6, initial_volume * 1e-12)
    cumulative_tolerance = max(1e-6, initial_volume * CUMULATIVE_BUDGET_RELATIVE_GATE)
    initial_area = _area_metrics(initial, cell_area)
    gamma = 2.0 * float(config["glen_a"]) * (900.0 * 9.81) ** 3 / 5.0
    target_seconds = TARGET_YEARS * bridge.sec_in_year
    history_path = output / f"step_history_part_{start_year:03d}_{int(time.time())}.csv"
    fieldnames = [
        "trial", "accepted", "elapsed_before_years", "month", "dt_seconds",
        "error_relative_l2", "step_closure_m3", "budget_residual_m3", "source_volume_m3",
    ]
    wall_started = time.perf_counter()
    consecutive_rejections = 0
    outside_maximum = 0.0
    with history_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        trial = 0
        while elapsed < target_seconds and counters["accepted_steps"] < MAXIMUM_ACCEPTED_STEPS:
            year = int(math.floor(elapsed / bridge.sec_in_year + 1e-12))
            next_year = (year + 1) * bridge.sec_in_year
            global_month = int(math.floor(elapsed / bridge.sec_in_month + 1e-12))
            month = source_month_index(global_month, climate_period_months, repeat_parent_climate)
            next_month = (global_month + 1) * bridge.sec_in_month
            dt = min(proposed_dt, next_year - elapsed, next_month - elapsed, target_seconds - elapsed)
            if dt <= 0.0:
                raise RuntimeError("nonpositive-adaptive-dt")
            full, fd = split_step(
                state, bed, mask, bridge, month=month, dx=loaded["dx"], gamma=gamma,
                dt=dt, cell_area=cell_area,
            )
            half, hd = split_step(
                state, bed, mask, bridge, month=month, dx=loaded["dx"], gamma=gamma,
                dt=0.5 * dt, cell_area=cell_area,
            )
            half2, h2d = split_step(
                half, bed, mask, bridge, month=month, dx=loaded["dx"], gamma=gamma,
                dt=0.5 * dt, cell_area=cell_area,
            )
            torch.cuda.synchronize()
            counters["solver_calls"] += 6
            delta = half2 - state
            denominator = max(float(torch.linalg.vector_norm(delta).item()), 1e-30)
            error = float(torch.linalg.vector_norm(half2 - full).item()) / denominator
            closure = max(abs(float(hd["diffusion_closure_m3"])), abs(float(h2d["diffusion_closure_m3"])))
            source_volume = float(hd["source_volume_m3"]) + float(h2d["source_volume_m3"])
            budget_residual = float(torch.sum(delta).item()) * cell_area - source_volume
            numerical = all(bool(item["numerical_pass"]) for item in (fd, hd, h2d))
            numerical &= closure <= per_step_tolerance and abs(budget_residual) <= per_step_tolerance
            accepted = numerical and error <= ERROR_GATE
            trial += 1
            writer.writerow({
                "trial": trial, "accepted": accepted, "elapsed_before_years": elapsed / bridge.sec_in_year,
                "month": month + 1, "dt_seconds": dt, "error_relative_l2": error,
                "step_closure_m3": closure, "budget_residual_m3": budget_residual,
                "source_volume_m3": source_volume,
            })
            if accepted:
                state = half2
                elapsed += dt
                if abs(elapsed - next_month) <= 1e-7:
                    elapsed = next_month
                if abs(elapsed - next_year) <= 1e-7:
                    elapsed = next_year
                counters["accepted_steps"] += 1
                counters["cumulative_source_m3"] += source_volume
                counters["projection_calls_accepted_path"] += int(hd["projection_applied"]) + int(h2d["projection_applied"])
                counters["maximum_step_closure_m3"] = max(counters["maximum_step_closure_m3"], closure)
                counters["maximum_budget_residual_m3"] = max(counters["maximum_budget_residual_m3"], abs(budget_residual))
                counters["maximum_error_relative_l2"] = max(counters["maximum_error_relative_l2"], error)
                current_min = counters["minimum_accepted_dt_seconds"]
                counters["minimum_accepted_dt_seconds"] = dt if current_min is None else min(current_min, dt)
                counters["maximum_accepted_dt_seconds"] = max(counters["maximum_accepted_dt_seconds"], dt)
                consecutive_rejections = 0
                proposed_dt = min(MAXIMUM_DT_SECONDS, 2.0 * dt) if error <= GROWTH_ERROR_THRESHOLD else dt
                outside_maximum = max(outside_maximum, float(torch.max(torch.abs(state[~mask])).item()))
                if elapsed == next_year:
                    completed_year = int(round(elapsed / bridge.sec_in_year))
                    state_year = state.detach().cpu().numpy()
                    volume = float(torch.sum(state).item()) * cell_area
                    path = _path_metrics(
                        state_year, loaded["x"], loaded["y"], loaded["path_xy"],
                        loaded["controls"], loaded["registered_length"],
                    )
                    domain = _free_domain_metrics(state_year, loaded["dx"])
                    checkpoint_metrics[str(completed_year)] = {
                        "elapsed_years": float(completed_year),
                        "volume_relative_change": (volume - initial_volume) / initial_volume,
                        "cumulative_source_m3": counters["cumulative_source_m3"],
                        "cumulative_budget_residual_m3": volume - initial_volume - counters["cumulative_source_m3"],
                        "active_area_m2": float(torch.count_nonzero(state > bridge.active_thickness_tolerance_m).item()) * cell_area,
                        "root_path_metrics": path,
                        "free_domain_metrics": domain,
                    }
                    manifest = write_checkpoint(
                        checkpoint_root, year=completed_year, state=state_year,
                        elapsed_seconds=elapsed, proposed_dt_seconds=proposed_dt,
                        counters=counters, checkpoint_metrics=checkpoint_metrics,
                        contract_hash=contract_hash,
                    )
                    checkpoint_metrics[str(completed_year)]["manifest"] = str(manifest)
                    atomic_bytes(output / "progress.json", json_bytes({
                        "status": "running" if completed_year < TARGET_YEARS else "finishing",
                        "completed_years": completed_year,
                        "accepted_steps": counters["accepted_steps"],
                        "rejected_steps": counters["rejected_steps"],
                        "volume_relative_change": checkpoint_metrics[str(completed_year)]["volume_relative_change"],
                        "outside_domain_maximum_thickness_m": outside_maximum,
                    }))
                    stream.flush()
                    print(json.dumps({"year": completed_year, "volume_change": checkpoint_metrics[str(completed_year)]["volume_relative_change"]}), flush=True)
            else:
                counters["rejected_steps"] += 1
                consecutive_rejections += 1
                proposed_dt = 0.5 * dt
                if proposed_dt < MINIMUM_DT_SECONDS or consecutive_rejections > MAXIMUM_CONSECUTIVE_REJECTIONS:
                    break

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    final_np = state.detach().cpu().numpy()
    final_volume = float(torch.sum(state).item()) * cell_area
    final_volume_change = (final_volume - initial_volume) / initial_volume
    final_path = _path_metrics(
        final_np, loaded["x"], loaded["y"], loaded["path_xy"], loaded["controls"], loaded["registered_length"]
    )
    final_domain = _free_domain_metrics(final_np, loaded["dx"])
    cumulative_budget = final_volume - initial_volume - counters["cumulative_source_m3"]
    completed = math.isclose(elapsed, target_seconds, rel_tol=0.0, abs_tol=1e-7)
    final_state_path = output / "final_state_50y.npz"
    final_state_hash = None
    if completed:
        temporary = output / "final_state_50y.tmp.npz"
        np.savez_compressed(temporary, thickness_m=final_np, domain_mask=loaded["mask"])
        os.replace(temporary, final_state_path)
        final_state_hash = sha256(final_state_path)
    current_hashes = {name: sha256(path) for name, path in loaded["paths"].items()}
    checks = {
        "target_50y_completed": completed,
        "all_annual_checkpoints_recorded": set(checkpoint_metrics)
        == expected_checkpoint_years(TARGET_YEARS),
        "steady_volume_change_pass": abs(final_volume_change) <= VOLUME_CHANGE_GATE,
        "registered_length_error_pass": final_path["registered_length_relative_error"] <= LENGTH_RELATIVE_ERROR_GATE,
        "free_domain_pass": final_domain["minimum_active_cell_center_to_grid_edge_m"] >= FREE_DOMAIN_GATE_M,
        "mass_budget_closure": abs(cumulative_budget) <= cumulative_tolerance,
        "maximum_step_closure": counters["maximum_step_closure_m3"] <= per_step_tolerance,
        "accepted_error_gate_respected": counters["maximum_error_relative_l2"] <= ERROR_GATE,
        "finite_nonnegative": bool(torch.isfinite(state).all()) and int(torch.count_nonzero(state < 0).item()) == 0,
        "outside_domain_zero": outside_maximum == 0.0 and int(torch.count_nonzero(state[~mask]).item()) == 0,
        "source_hashes_unchanged": current_hashes == {name: config[f"{name}_sha256"].upper() for name in current_hashes},
        "runtime_exact": torch.__version__ == "2.11.0+cu128" and torch.version.cuda == "12.8",
        "final_state_saved": final_state_hash is not None,
    }
    status = "masked-steady50-pass" if all(checks.values()) else "masked-steady50-fail-stop"
    summary = {
        "schema": (
            "lia-191-generic-planform2d-gpu-steady50-result-v1"
            if generic
            else "lia-191-system2005-masked-steady50-checkpointed-v1"
        ),
        "status": status,
        "system_id": int(config["system_id"]),
        "target_years": TARGET_YEARS,
        "completed_years": elapsed / bridge.sec_in_year,
        "resumed_from_checkpoint": resumed,
        "resume_manifest": str(resume_manifest) if resume_manifest else None,
        "checkpoint_contract_sha256": contract_hash,
        "runtime_seconds_this_process": wall_seconds,
        "accepted_steps": counters["accepted_steps"],
        "rejected_steps": counters["rejected_steps"],
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "final_volume_relative_change": final_volume_change,
        "cumulative_climate_source_m3": counters["cumulative_source_m3"],
        "cumulative_mass_budget_residual_m3": cumulative_budget,
        "outside_domain_maximum_thickness_m": outside_maximum,
        "initial_area_metrics": initial_area,
        "final_area_metrics": _with_relative_area_change(_area_metrics(state, cell_area), initial_area),
        "final_root_path_metrics": final_path,
        "final_free_domain_metrics": final_domain,
        "checkpoint_metrics": checkpoint_metrics,
        "checks": checks,
        "counters": counters,
        "final_state_sha256": final_state_hash,
        "history_part": str(history_path),
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "historical_run": False,
        "parent_climate_repeated": repeat_parent_climate,
        "parent_climate_period_months": climate_period_months,
    }
    atomic_bytes(output / "summary.json", json_bytes(summary))
    atomic_bytes(output / "progress.json", json_bytes({"status": status, "completed_years": summary["completed_years"]}))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
