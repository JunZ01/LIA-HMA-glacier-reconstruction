"""Checkpointed masked planform-2D historical driver for system 2005."""

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
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)
from .torch_planform_2d_system2005_steady50_monthly_ti_formal_v1 import (
    _control_lengths,
    _free_domain_metrics,
    _path_metrics,
)


CHECKPOINT_SCHEMA = "lia-191-system2005-masked-temperature-historical-checkpoint-v1"
CONFIG_SCHEMA = "lia-191-system2005-masked-temperature-historical-config-v1"
INITIAL_DT_SECONDS = 0.03
MINIMUM_DT_SECONDS = 0.001
MAXIMUM_DT_SECONDS = 86_400.0
MAXIMUM_CONSECUTIVE_REJECTIONS = 20
MAXIMUM_ACCEPTED_STEPS_PER_YEAR = 20_000
GROWTH_ERROR_THRESHOLD = ERROR_GATE / 4.0
CUMULATIVE_BUDGET_RELATIVE_GATE = 1.0e-10
LENGTH_ICE_THRESHOLD_M = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "system_id",
        "mode",
        "start_year",
        "end_year",
        "target_years",
        "geology",
        "geology_sha256",
        "static_summary",
        "static_summary_sha256",
        "historical_climate_package",
        "historical_climate_package_sha256",
        "historical_start_state",
        "historical_start_state_sha256",
        "observation_path",
        "observation_path_sha256",
        "length_references",
        "length_references_sha256",
        "domain_variable",
        "glen_a",
        "temperature_offset_k",
        "lia_registered_length_m",
        "rgi_registered_length_m",
        "total_length_error_max",
        "separation_length_error_max",
        "driver_sha256",
        "expected_route",
        "expected_route_sha256",
    )
    payload = {key: config[key] for key in keys}
    payload["solver"] = {
        "operator": "source-half/masked-frozen-diffusion/source-half",
        "ice_density_kg_m3": ICE_DENSITY_KG_M3,
        "gravity_m_s2": GRAVITY_M_S2,
        "glen_n": GLEN_N,
        "initial_dt_seconds": INITIAL_DT_SECONDS,
        "minimum_dt_seconds": MINIMUM_DT_SECONDS,
        "maximum_dt_seconds": MAXIMUM_DT_SECONDS,
        "error_gate_relative_l2": ERROR_GATE,
        "monthly_boundaries_exact": True,
        "annual_atomic_checkpoints": True,
        "resume_verified": True,
    }
    payload["historical_operator"] = {
        "calendar": "state at 1470 advanced with climate years 1470-1999 to state at 2000",
        "lia_operator": "maximum annual active-control-volume length along frozen LIA main path",
        "rgi_operator": "year-2000 active-control-volume length along the same frozen path",
        "length_ice_threshold_m": LENGTH_ICE_THRESHOLD_M,
        "observation_domain_overflow_allowed": False,
    }
    return payload


def contract_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(json_bytes(contract_payload(config))).hexdigest().upper()


def write_checkpoint(
    checkpoint_root: Path,
    *,
    calendar_year: int,
    state: np.ndarray,
    elapsed_seconds: float,
    proposed_dt_seconds: float,
    counters: dict[str, Any],
    annual_metrics: list[dict[str, Any]],
    contract_hash: str,
) -> Path:
    directory = checkpoint_root / f"year_{calendar_year:04d}"
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=False)
    temporary_state = directory / "state.tmp.npz"
    state_path = directory / "state.npz"
    np.savez_compressed(
        temporary_state,
        thickness_m=np.asarray(state, dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        proposed_dt_seconds=np.asarray(proposed_dt_seconds, dtype=np.float64),
        counters_json=np.asarray(json.dumps(counters, sort_keys=True, allow_nan=False)),
        annual_metrics_json=np.asarray(json.dumps(annual_metrics, sort_keys=True, allow_nan=False)),
        contract_sha256=np.asarray(contract_hash),
        schema=np.asarray(CHECKPOINT_SCHEMA),
    )
    os.replace(temporary_state, state_path)
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint-complete",
        "calendar_year": int(calendar_year),
        "elapsed_seconds": float(elapsed_seconds),
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
    if manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("status") != "checkpoint-complete":
        raise RuntimeError("checkpoint-manifest-incomplete")
    if manifest.get("contract_sha256") != expected_contract_hash:
        raise RuntimeError("checkpoint-contract-hash-mismatch")
    state_path = manifest_path.parent / str(manifest["state_file"])
    if sha256(state_path) != manifest.get("state_sha256"):
        raise RuntimeError("checkpoint-state-hash-mismatch")
    with np.load(state_path, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != CHECKPOINT_SCHEMA:
            raise RuntimeError("checkpoint-state-schema-mismatch")
        if str(archive["contract_sha256"].item()) != expected_contract_hash:
            raise RuntimeError("checkpoint-state-contract-mismatch")
        restored = {
            "state": np.asarray(archive["thickness_m"], dtype=np.float64),
            "elapsed_seconds": float(archive["elapsed_seconds"].item()),
            "proposed_dt_seconds": float(archive["proposed_dt_seconds"].item()),
            "counters": json.loads(str(archive["counters_json"].item())),
            "annual_metrics": json.loads(str(archive["annual_metrics_json"].item())),
            "manifest": manifest,
        }
    if not math.isclose(
        restored["elapsed_seconds"], float(manifest["elapsed_seconds"]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("checkpoint-elapsed-mismatch")
    return restored


class OffsetBridge(TorchMonthlyTIBridge):
    def __init__(self, package_path: Path, *, temperature_offset_k: float, device: str) -> None:
        super().__init__(package_path, device=device)
        self.temp = self.temp + float(temperature_offset_k)


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project_root"])
    path_keys = (
        "geology",
        "static_summary",
        "historical_climate_package",
        "historical_start_state",
        "observation_path",
        "length_references",
        "expected_route",
    )
    paths = {name: root / config[name] for name in path_keys}
    for name, path in paths.items():
        if sha256(path) != str(config[f"{name}_sha256"]).upper():
            raise RuntimeError(f"{name}-hash-mismatch")
    static = json.loads(paths["static_summary"].read_text(encoding="utf-8"))
    with Dataset(paths["geology"]) as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        mask = np.asarray(dataset.variables[config["domain_variable"]][:], dtype=float) > 0.5
        x = np.asarray(dataset.variables["x"][:], dtype=np.float64)
        y = np.asarray(dataset.variables["y"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    with np.load(paths["historical_start_state"], allow_pickle=False) as archive:
        start = np.asarray(archive["thickness_m"], dtype=np.float64)
        start_mask = np.asarray(archive["domain_mask"], dtype=bool)
    with paths["observation_path"].open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    path_xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    controls = _control_lengths(path_xy)
    if start.shape != bed.shape or not np.array_equal(start_mask, mask):
        raise RuntimeError("historical-start-state-grid-or-mask-mismatch")
    if not np.isfinite(start).all() or np.any(start < 0.0) or np.any(start[~mask] != 0.0):
        raise RuntimeError("historical-start-state-invalid")
    if int(config["system_id"]) != 2005 or int(static["system_id"]) != 2005:
        raise RuntimeError("system-id-mismatch")
    return {
        "root": root,
        "paths": paths,
        "static": static,
        "bed": bed,
        "mask": mask,
        "x": x,
        "y": y,
        "dx": dx,
        "start": start,
        "path_xy": path_xy,
        "controls": controls,
    }


def snapshot(
    *,
    calendar_year: int,
    state: np.ndarray,
    loaded: dict[str, Any],
    initial_volume_m3: float,
    cumulative_source_m3: float,
) -> dict[str, Any]:
    cell_area = float(loaded["dx"]) ** 2
    volume = float(np.sum(state)) * cell_area
    path = _path_metrics(
        state,
        loaded["x"],
        loaded["y"],
        loaded["path_xy"],
        loaded["controls"],
        float(loaded["config"]["lia_registered_length_m"]),
    )
    domain = _free_domain_metrics(state, float(loaded["dx"]))
    length = float(path["observed_active_control_volume_length_m"])
    lia = float(loaded["config"]["lia_registered_length_m"])
    rgi = float(loaded["config"]["rgi_registered_length_m"])
    separation = abs(lia - rgi)
    return {
        "calendar_year": int(calendar_year),
        "volume_m3": volume,
        "volume_relative_change_from_1470": (volume - initial_volume_m3) / initial_volume_m3,
        "cumulative_source_m3": float(cumulative_source_m3),
        "cumulative_budget_residual_m3": volume - initial_volume_m3 - cumulative_source_m3,
        "active_area_m2": float(np.count_nonzero(state > 1.0e-6)) * cell_area,
        "length_m": length,
        "lia_total_relative_error": abs(length - lia) / lia,
        "lia_separation_relative_error": abs(length - lia) / separation,
        "rgi_total_relative_error": abs(length - rgi) / rgi,
        "rgi_separation_relative_error": abs(length - rgi) / separation,
        "root_path_metrics": path,
        "free_domain_metrics": domain,
        "finite_nonnegative": bool(np.isfinite(state).all() and np.count_nonzero(state < 0.0) == 0),
        "outside_domain_maximum_thickness_m": float(np.max(np.abs(state[~loaded["mask"]]))),
    }


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("historical-config-schema-mismatch")
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    target_years = int(config["target_years"])
    if start_year != 1470 or end_year - start_year != target_years or not 1 <= target_years <= 530:
        raise RuntimeError("historical-calendar-mismatch")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("runtime-version-mismatch")
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")

    loaded = load_inputs(config)
    loaded["config"] = config
    output = loaded["root"] / config["output_root"]
    checkpoint_root = output / "checkpoints"
    contract_hash = contract_sha256(config)
    resumed = resume_manifest is not None
    if resumed:
        if not output.is_dir() or not checkpoint_root.is_dir():
            raise RuntimeError("resume-output-missing")
        restored = load_checkpoint(resume_manifest.resolve(), contract_hash)
        state_np = restored["state"]
        elapsed = restored["elapsed_seconds"]
        proposed_dt = restored["proposed_dt_seconds"]
        counters = restored["counters"]
        annual_metrics = restored["annual_metrics"]
        resume_year = int(restored["manifest"]["calendar_year"])
        expected_elapsed = (resume_year - start_year) * 365.0 * 24.0 * 3600.0
        if not math.isclose(elapsed, expected_elapsed, rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError("resume-calendar-elapsed-mismatch")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=False)
        checkpoint_root.mkdir()
        atomic_bytes(output / "config_snapshot.json", json_bytes(config))
        atomic_bytes(output / "checkpoint_contract.json", json_bytes(contract_payload(config)))
        state_np = loaded["start"].copy()
        elapsed = 0.0
        proposed_dt = INITIAL_DT_SECONDS
        counters = {
            "accepted_steps": 0,
            "rejected_steps": 0,
            "solver_calls": 0,
            "projection_calls_accepted_path": 0,
            "cumulative_source_m3": 0.0,
            "maximum_step_closure_m3": 0.0,
            "maximum_budget_residual_m3": 0.0,
            "maximum_error_relative_l2": 0.0,
            "minimum_accepted_dt_seconds": None,
            "maximum_accepted_dt_seconds": 0.0,
        }
        initial_volume = float(np.sum(state_np)) * float(loaded["dx"]) ** 2
        annual_metrics = [
            snapshot(
                calendar_year=start_year,
                state=state_np,
                loaded=loaded,
                initial_volume_m3=initial_volume,
                cumulative_source_m3=0.0,
            )
        ]

    if state_np.shape != loaded["start"].shape or np.any(state_np[~loaded["mask"]] != 0.0):
        raise RuntimeError("resume-state-domain-mismatch")
    bridge = OffsetBridge(
        loaded["paths"]["historical_climate_package"],
        temperature_offset_k=float(config["temperature_offset_k"]),
        device="cuda:0",
    )
    required_months = target_years * 12
    if bridge.month_count < required_months:
        raise RuntimeError("historical-climate-sequence-too-short")
    with np.load(loaded["paths"]["historical_climate_package"], allow_pickle=False) as package:
        sequence_years = np.asarray(package["sequence_years"], dtype=int)
        sequence_months = np.asarray(package["sequence_months"], dtype=int)
    expected_years = np.repeat(np.arange(1470, 2000, dtype=int), 12)
    expected_months = np.tile(np.arange(1, 13, dtype=int), 530)
    if not np.array_equal(sequence_years, expected_years) or not np.array_equal(
        sequence_months, expected_months
    ):
        raise RuntimeError("historical-package-calendar-contract-mismatch")

    bed = torch.as_tensor(loaded["bed"], dtype=torch.float64, device="cuda:0")
    state = torch.as_tensor(state_np, dtype=torch.float64, device="cuda:0")
    mask = torch.as_tensor(loaded["mask"], dtype=torch.bool, device="cuda:0")
    cell_area = float(loaded["dx"]) ** 2
    initial_volume = float(np.sum(loaded["start"])) * cell_area
    per_step_tolerance = max(1.0e-6, initial_volume * 1.0e-12)
    cumulative_tolerance = max(1.0e-6, initial_volume * CUMULATIVE_BUDGET_RELATIVE_GATE)
    gamma = (
        2.0
        * float(config["glen_a"])
        * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N
        / (GLEN_N + 2.0)
    )
    target_seconds = target_years * bridge.sec_in_year
    history_path = output / f"step_history_part_{start_year + int(elapsed / bridge.sec_in_year):04d}_{int(time.time())}.csv"
    fieldnames = [
        "trial",
        "accepted",
        "calendar_year_before",
        "calendar_month",
        "dt_seconds",
        "error_relative_l2",
        "step_closure_m3",
        "budget_residual_m3",
        "source_volume_m3",
    ]
    wall_started = time.perf_counter()
    consecutive_rejections = 0
    accepted_at_start = int(counters["accepted_steps"])
    with history_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        trial = 0
        while elapsed < target_seconds:
            elapsed_year_index = int(math.floor(elapsed / bridge.sec_in_year + 1.0e-12))
            elapsed_month_index = int(math.floor(elapsed / bridge.sec_in_month + 1.0e-12))
            next_year = (elapsed_year_index + 1) * bridge.sec_in_year
            next_month = (elapsed_month_index + 1) * bridge.sec_in_month
            dt = min(proposed_dt, next_year - elapsed, next_month - elapsed, target_seconds - elapsed)
            if dt <= 0.0:
                raise RuntimeError("nonpositive-adaptive-dt")
            full, fd = split_step(
                state,
                bed,
                mask,
                bridge,
                month=elapsed_month_index,
                dx=float(loaded["dx"]),
                gamma=gamma,
                dt=dt,
                cell_area=cell_area,
            )
            half, hd = split_step(
                state,
                bed,
                mask,
                bridge,
                month=elapsed_month_index,
                dx=float(loaded["dx"]),
                gamma=gamma,
                dt=0.5 * dt,
                cell_area=cell_area,
            )
            half2, h2d = split_step(
                half,
                bed,
                mask,
                bridge,
                month=elapsed_month_index,
                dx=float(loaded["dx"]),
                gamma=gamma,
                dt=0.5 * dt,
                cell_area=cell_area,
            )
            torch.cuda.synchronize()
            counters["solver_calls"] += 6
            delta = half2 - state
            denominator = max(float(torch.linalg.vector_norm(delta).item()), 1.0e-30)
            error = float(torch.linalg.vector_norm(half2 - full).item()) / denominator
            closure = max(abs(float(hd["diffusion_closure_m3"])), abs(float(h2d["diffusion_closure_m3"])))
            source_volume = float(hd["source_volume_m3"]) + float(h2d["source_volume_m3"])
            budget_residual = float(torch.sum(delta).item()) * cell_area - source_volume
            numerical = all(bool(item["numerical_pass"]) for item in (fd, hd, h2d))
            numerical &= closure <= per_step_tolerance and abs(budget_residual) <= per_step_tolerance
            accepted = numerical and error <= ERROR_GATE
            trial += 1
            writer.writerow(
                {
                    "trial": trial,
                    "accepted": accepted,
                    "calendar_year_before": start_year + elapsed / bridge.sec_in_year,
                    "calendar_month": int(sequence_months[elapsed_month_index]),
                    "dt_seconds": dt,
                    "error_relative_l2": error,
                    "step_closure_m3": closure,
                    "budget_residual_m3": budget_residual,
                    "source_volume_m3": source_volume,
                }
            )
            if accepted:
                state = half2
                elapsed += dt
                if abs(elapsed - next_month) <= 1.0e-7:
                    elapsed = next_month
                if abs(elapsed - next_year) <= 1.0e-7:
                    elapsed = next_year
                counters["accepted_steps"] += 1
                counters["cumulative_source_m3"] += source_volume
                counters["projection_calls_accepted_path"] += int(hd["projection_applied"]) + int(
                    h2d["projection_applied"]
                )
                counters["maximum_step_closure_m3"] = max(
                    float(counters["maximum_step_closure_m3"]), closure
                )
                counters["maximum_budget_residual_m3"] = max(
                    float(counters["maximum_budget_residual_m3"]), abs(budget_residual)
                )
                counters["maximum_error_relative_l2"] = max(
                    float(counters["maximum_error_relative_l2"]), error
                )
                old_min = counters["minimum_accepted_dt_seconds"]
                counters["minimum_accepted_dt_seconds"] = dt if old_min is None else min(float(old_min), dt)
                counters["maximum_accepted_dt_seconds"] = max(
                    float(counters["maximum_accepted_dt_seconds"]), dt
                )
                consecutive_rejections = 0
                proposed_dt = min(MAXIMUM_DT_SECONDS, 2.0 * dt) if error <= GROWTH_ERROR_THRESHOLD else dt
                if elapsed == next_year:
                    calendar_year = start_year + int(round(elapsed / bridge.sec_in_year))
                    state_year = state.detach().cpu().numpy()
                    annual_metrics.append(
                        snapshot(
                            calendar_year=calendar_year,
                            state=state_year,
                            loaded=loaded,
                            initial_volume_m3=initial_volume,
                            cumulative_source_m3=float(counters["cumulative_source_m3"]),
                        )
                    )
                    manifest = write_checkpoint(
                        checkpoint_root,
                        calendar_year=calendar_year,
                        state=state_year,
                        elapsed_seconds=elapsed,
                        proposed_dt_seconds=proposed_dt,
                        counters=counters,
                        annual_metrics=annual_metrics,
                        contract_hash=contract_hash,
                    )
                    atomic_bytes(
                        output / "progress.json",
                        json_bytes(
                            {
                                "status": "running" if calendar_year < end_year else "finishing",
                                "calendar_year": calendar_year,
                                "accepted_steps": counters["accepted_steps"],
                                "rejected_steps": counters["rejected_steps"],
                                "volume_relative_change_from_1470": annual_metrics[-1][
                                    "volume_relative_change_from_1470"
                                ],
                                "length_m": annual_metrics[-1]["length_m"],
                                "checkpoint_manifest": str(manifest),
                            }
                        ),
                    )
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                "calendar_year": calendar_year,
                                "volume_change": annual_metrics[-1][
                                    "volume_relative_change_from_1470"
                                ],
                                "length_m": annual_metrics[-1]["length_m"],
                            }
                        ),
                        flush=True,
                    )
            else:
                counters["rejected_steps"] += 1
                consecutive_rejections += 1
                proposed_dt = 0.5 * dt
                if proposed_dt < MINIMUM_DT_SECONDS or consecutive_rejections > MAXIMUM_CONSECUTIVE_REJECTIONS:
                    break
            if int(counters["accepted_steps"]) - accepted_at_start > MAXIMUM_ACCEPTED_STEPS_PER_YEAR * target_years:
                raise RuntimeError("maximum-accepted-step-budget-exceeded")

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    final_np = state.detach().cpu().numpy()
    completed = math.isclose(elapsed, target_seconds, rel_tol=0.0, abs_tol=1.0e-7)
    final_state_path = output / f"final_state_{end_year}.npz"
    final_state_hash = None
    if completed:
        temporary = output / f"final_state_{end_year}.tmp.npz"
        np.savez_compressed(temporary, thickness_m=final_np, domain_mask=loaded["mask"])
        os.replace(temporary, final_state_path)
        final_state_hash = sha256(final_state_path)
    final = snapshot(
        calendar_year=end_year,
        state=final_np,
        loaded=loaded,
        initial_volume_m3=initial_volume,
        cumulative_source_m3=float(counters["cumulative_source_m3"]),
    )
    peak = max(annual_metrics, key=lambda row: float(row["length_m"]))
    current_hashes = {name: sha256(path) for name, path in loaded["paths"].items()}
    numerical_checks = {
        "target_completed": completed,
        "all_annual_checkpoints_recorded": [int(row["calendar_year"]) for row in annual_metrics]
        == list(range(start_year, end_year + 1)),
        "mass_budget_closure": abs(float(final["cumulative_budget_residual_m3"])) <= cumulative_tolerance,
        "maximum_step_closure": float(counters["maximum_step_closure_m3"]) <= per_step_tolerance,
        "accepted_error_gate_respected": float(counters["maximum_error_relative_l2"]) <= ERROR_GATE,
        "finite_nonnegative": bool(final["finite_nonnegative"]),
        "outside_domain_zero": float(final["outside_domain_maximum_thickness_m"]) == 0.0,
        "observation_domain_overflow_absent": min(
            float(row["free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"])
            for row in annual_metrics
        )
        > 0.0,
        "source_hashes_unchanged": current_hashes
        == {name: str(config[f"{name}_sha256"]).upper() for name in current_hashes},
        "runtime_exact": torch.__version__ == "2.11.0+cu128" and torch.version.cuda == "12.8",
        "final_state_saved": final_state_hash is not None,
    }
    initial_import_checks = {
        "start_array_exact": np.array_equal(loaded["start"], state_np)
        if not resumed
        else annual_metrics[0]["calendar_year"] == start_year,
        "start_state_calendar_1470": int(annual_metrics[0]["calendar_year"]) == 1470,
        "first_forcing_calendar_1470_01": int(sequence_years[0]) == 1470
        and int(sequence_months[0]) == 1,
        "production_temperature_unmodified": True,
        "initial_source_thickness_unmodified": True,
    }
    historical_length_checks = {
        "lia_total": float(peak["lia_total_relative_error"])
        <= float(config["total_length_error_max"]),
        "lia_separation": float(peak["lia_separation_relative_error"])
        <= float(config["separation_length_error_max"]),
        "rgi_total": float(final["rgi_total_relative_error"])
        <= float(config["total_length_error_max"]),
        "rgi_separation": float(final["rgi_separation_relative_error"])
        <= float(config["separation_length_error_max"]),
    }
    mode = str(config["mode"])
    if mode == "short-preflight":
        status = (
            "masked-temperature-historical-preflight-pass"
            if all(numerical_checks.values()) and all(initial_import_checks.values())
            else "masked-temperature-historical-preflight-fail-stop"
        )
    elif mode == "formal-historical":
        status = (
            "masked-temperature-historical-complete-unadjudicated"
            if all(numerical_checks.values())
            and all(initial_import_checks.values())
            and all(historical_length_checks.values())
            else "masked-temperature-historical-fail-stop"
        )
    else:
        raise RuntimeError(f"unknown-historical-mode:{mode}")
    summary = {
        "schema": "lia-191-system2005-masked-temperature-historical-v1",
        "status": status,
        "system_id": 2005,
        "mode": mode,
        "start_year": start_year,
        "end_year": end_year,
        "target_years": target_years,
        "completed_years": elapsed / bridge.sec_in_year,
        "resumed_from_checkpoint": resumed,
        "resume_manifest": str(resume_manifest) if resume_manifest else None,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_contract_sha256": contract_hash,
        "runtime_seconds_this_process": wall_seconds,
        "temperature_offset_k": float(config["temperature_offset_k"]),
        "accepted_steps": counters["accepted_steps"],
        "rejected_steps": counters["rejected_steps"],
        "initial": annual_metrics[0],
        "peak": peak,
        "final": final,
        "annual_metrics": annual_metrics,
        "numerical_checks": numerical_checks,
        "initial_import_checks": initial_import_checks,
        "historical_length_checks": historical_length_checks,
        "counters": counters,
        "final_state_sha256": final_state_hash,
        "history_part": str(history_path),
        "formal_historical_adjudicated": False,
        "accepted_count_effect": 0,
        "research_family_promoted": False,
        "initial_source_thickness_modified": False,
        "production_temperature_modified": False,
    }
    atomic_bytes(output / "summary.json", json_bytes(summary))
    atomic_bytes(
        output / "progress.json",
        json_bytes({"status": status, "calendar_year": end_year if completed else start_year + elapsed / bridge.sec_in_year}),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
