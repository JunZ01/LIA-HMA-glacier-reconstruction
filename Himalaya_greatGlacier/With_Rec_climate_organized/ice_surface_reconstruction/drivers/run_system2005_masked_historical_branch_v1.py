"""Checkpointed system-2005 masked 2D historical branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import run_system2005_masked_steady50_checkpointed_v1 as engine
from . import run_system2005_masked_steady50_checkpointed_v2 as atomic_driver
from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)


CHECKPOINT_SCHEMA = "lia-191-system2005-masked-historical-checkpoint-v1"


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def historical_checkpoint_writer(
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
    config = historical_checkpoint_writer.config
    calendar_year = int(config["start_year"]) + int(year)
    directory = checkpoint_root / f"year_{calendar_year:04d}"
    temporary = checkpoint_root / f"year_{calendar_year:04d}.tmp"
    if directory.exists() or temporary.exists():
        raise FileExistsError(directory)
    temporary.mkdir(parents=False)
    state_path = temporary / "state.npz"
    np.savez_compressed(
        state_path,
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
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint-complete",
        "system_id": 2005,
        "calendar_year": calendar_year,
        "completed_years": int(year),
        "year": int(year),
        "elapsed_seconds": elapsed_seconds,
        "contract_sha256": contract_hash,
        "state_file": "state.npz",
        "state_sha256": engine.sha256(state_path),
        "thickness_array_sha256": array_sha256(state),
    }
    (temporary / "manifest.json").write_bytes(engine.json_bytes(manifest))
    os.replace(temporary, directory)
    manifest_path = directory / "manifest.json"
    engine.atomic_bytes(
        checkpoint_root / "latest.json",
        engine.json_bytes(
            {
                "status": "checkpoint-complete",
                "calendar_year": calendar_year,
                "completed_years": int(year),
                "manifest": str(manifest_path.resolve()),
            }
        ),
    )
    return manifest_path


historical_checkpoint_writer.config = {}


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema"] != "lia-191-system2005-masked-historical-branch-config-v1":
        raise RuntimeError("historical-config-schema-mismatch")
    if int(config["system_id"]) != 2005:
        raise RuntimeError("wrong-system")
    if not bool(config.get("launch_authorized", False)):
        raise RuntimeError("historical-branch-launch-not-authorized")
    root = Path(config["project_root"])
    launch_route = root / config["launch_route"]
    if engine.sha256(launch_route) != str(config["launch_route_sha256"]).upper():
        raise RuntimeError("launch-route-hash-mismatch")
    route = json.loads(launch_route.read_text(encoding="utf-8"))
    if not route.get("launch_authorized"):
        raise RuntimeError("route-does-not-authorize-launch")

    start_year = int(config["start_year"])
    target_end_year = int(config["target_end_year"])
    target_years = target_end_year - start_year
    if target_years <= 0 or int(config["target_years"]) != target_years:
        raise RuntimeError("historical-target-years-mismatch")
    if target_end_year > int(config["historical_end_year"]):
        raise RuntimeError("historical-target-after-registered-end")

    start_state_path = root / config["historical_start_state"]
    if engine.sha256(start_state_path) != str(config["historical_start_state_sha256"]).upper():
        raise RuntimeError("historical-start-state-file-hash-mismatch")
    with np.load(start_state_path, allow_pickle=False) as archive:
        start_thickness = np.asarray(archive["thickness_m"], dtype=np.float64)
        start_mask = np.asarray(archive["domain_mask"], dtype=bool)
    if array_sha256(start_thickness) != str(config["historical_start_thickness_array_sha256"]).upper():
        raise RuntimeError("historical-start-thickness-array-hash-mismatch")
    if array_sha256(start_mask) != str(config["historical_start_domain_mask_array_sha256"]).upper():
        raise RuntimeError("historical-start-mask-array-hash-mismatch")

    offset_start = float(config["temperature_offset_1470_k"])
    offset_end = float(config["temperature_offset_2000_k"])
    registered_end = int(config["historical_end_year"])

    class HistoricalBridge(TorchMonthlyTIBridge):
        def __init__(self, package_path: Path, *, device: str = "cuda:0") -> None:
            super().__init__(package_path, device=device)
            with np.load(package_path, allow_pickle=False) as package:
                years = np.asarray(package["years"], dtype=np.float64)
                months = np.asarray(package["months"], dtype=np.float64)
            decimal_years = years + (months - 1.0) / 12.0
            fraction = np.clip(
                (decimal_years - start_year) / float(registered_end - start_year), 0.0, 1.0
            )
            offsets = offset_start + (offset_end - offset_start) * fraction
            self.temp = self.temp + torch.as_tensor(offsets, dtype=torch.float64, device=device)

    original_load_inputs = engine.load_inputs
    original_contract_payload = engine.contract_payload
    original_split_step = engine.split_step
    original_target_years = engine.TARGET_YEARS
    original_maximum_steps = engine.MAXIMUM_ACCEPTED_STEPS
    original_schema = engine.CHECKPOINT_SCHEMA
    original_bridge = engine.TorchMonthlyTIBridge
    original_writer = engine.write_checkpoint

    def historical_load_inputs(value: dict[str, Any]) -> dict[str, Any]:
        loaded = original_load_inputs(value)
        if start_thickness.shape != loaded["initial"].shape:
            raise RuntimeError("historical-start-state-shape-mismatch")
        if not np.array_equal(start_mask, loaded["mask"]):
            raise RuntimeError("historical-start-domain-mask-mismatch")
        if np.any(start_thickness < 0.0) or np.any(start_thickness[~start_mask] != 0.0):
            raise RuntimeError("historical-start-state-domain-invalid")
        loaded["initial"] = start_thickness.copy()
        return loaded

    def historical_contract_payload(value: dict[str, Any]) -> dict[str, Any]:
        payload = original_contract_payload(value)
        payload["historical"] = {
            "start_year": start_year,
            "target_end_year": target_end_year,
            "registered_end_year": registered_end,
            "historical_start_state": value["historical_start_state"],
            "historical_start_state_sha256": value["historical_start_state_sha256"],
            "historical_start_thickness_array_sha256": value[
                "historical_start_thickness_array_sha256"
            ],
            "temperature_offset_1470_k": offset_start,
            "temperature_offset_2000_k": offset_end,
            "temperature_policy": "linear-in-calendar-time in-memory research overlay",
            "lia_registered_length_m": value["lia_registered_length_m"],
            "rgi_registered_length_m": value["rgi_registered_length_m"],
            "total_length_error_max": value["formal_gates"]["total_length_error_max"],
            "separation_length_error_max": value["formal_gates"][
                "separation_length_error_max"
            ],
        }
        return payload

    gamma = (
        2.0
        * float(config["glen_a"])
        * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N
        / (GLEN_N + 2.0)
    )

    def frozen_split_step(*args: Any, **kwargs: Any):
        kwargs["gamma"] = gamma
        return original_split_step(*args, **kwargs)

    historical_checkpoint_writer.config = config
    engine.load_inputs = historical_load_inputs
    engine.contract_payload = historical_contract_payload
    engine.split_step = frozen_split_step
    engine.TARGET_YEARS = target_years
    engine.MAXIMUM_ACCEPTED_STEPS = int(config["maximum_accepted_steps"])
    engine.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    engine.TorchMonthlyTIBridge = HistoricalBridge
    engine.write_checkpoint = historical_checkpoint_writer
    try:
        result = engine.run(config_path, resume_manifest)
    finally:
        engine.load_inputs = original_load_inputs
        engine.contract_payload = original_contract_payload
        engine.split_step = original_split_step
        engine.TARGET_YEARS = original_target_years
        engine.MAXIMUM_ACCEPTED_STEPS = original_maximum_steps
        engine.CHECKPOINT_SCHEMA = original_schema
        engine.TorchMonthlyTIBridge = original_bridge
        engine.write_checkpoint = original_writer

    initial_loaded = original_load_inputs(config)
    initial_path = engine._path_metrics(
        start_thickness,
        initial_loaded["x"],
        initial_loaded["y"],
        initial_loaded["path_xy"],
        initial_loaded["controls"],
        float(config["lia_registered_length_m"]),
    )
    annual_lengths = [{"calendar_year": start_year, "length_m": initial_path[
        "observed_active_control_volume_length_m"
    ]}]
    for key in sorted(result["checkpoint_metrics"], key=int):
        annual_lengths.append(
            {
                "calendar_year": start_year + int(key),
                "length_m": result["checkpoint_metrics"][key]["root_path_metrics"][
                    "observed_active_control_volume_length_m"
                ],
            }
        )
    peak = max(annual_lengths, key=lambda row: row["length_m"])
    final_length = float(result["final_root_path_metrics"]["observed_active_control_volume_length_m"])
    lia = float(config["lia_registered_length_m"])
    rgi = float(config["rgi_registered_length_m"])
    separation = abs(lia - rgi)
    scores = {
        "peak_calendar_year": int(peak["calendar_year"]),
        "peak_length_m": float(peak["length_m"]),
        "final_calendar_year": target_end_year,
        "final_length_m": final_length,
        "lia_total_error_fraction": abs(float(peak["length_m"]) - lia) / lia,
        "lia_separation_error_fraction": abs(float(peak["length_m"]) - lia) / separation,
        "rgi_total_error_fraction": abs(final_length - rgi) / rgi,
        "rgi_separation_error_fraction": abs(final_length - rgi) / separation,
    }
    numerical_keys = (
        "target_50y_completed",
        "all_annual_checkpoints_recorded",
        "free_domain_pass",
        "mass_budget_closure",
        "maximum_step_closure",
        "accepted_error_gate_respected",
        "finite_nonnegative",
        "outside_domain_zero",
        "source_hashes_unchanged",
        "runtime_exact",
        "final_state_saved",
    )
    numerical_checks = {key: bool(result["checks"][key]) for key in numerical_keys}
    numerical_checks["historical_start_state_exact"] = True
    numerical_checks["historical_calendar_package"] = True
    status = (
        "historical-preflight-branch-pass"
        if all(numerical_checks.values())
        else "historical-preflight-branch-technical-fail"
    )
    result.update(
        {
            "schema": "lia-191-system2005-masked-historical-branch-v1",
            "status": status,
            "run_role": config["run_role"],
            "start_year": start_year,
            "target_end_year": target_end_year,
            "actual_calendar_year": start_year + int(round(result["completed_years"])),
            "temperature_offset_1470_k": offset_start,
            "temperature_offset_2000_k": offset_end,
            "historical_scores_diagnostic_until_full_run": scores,
            "annual_lengths": annual_lengths,
            "historical_numerical_checks": numerical_checks,
            "initial_thickness_modified": False,
            "production_temperature_modified": False,
            "historical_run": True,
            "frozen_constants": {
                "ice_density_kg_m3": ICE_DENSITY_KG_M3,
                "gravity_m_s2": GRAVITY_M_S2,
                "glen_n": GLEN_N,
                "gamma": gamma,
            },
        }
    )
    output = root / config["output_root"]
    engine.atomic_bytes(output / "historical_summary_formal.json", engine.json_bytes(result))
    engine.atomic_bytes(
        output / "progress.json",
        engine.json_bytes(
            {
                "status": status,
                "actual_calendar_year": result["actual_calendar_year"],
                "completed_years": result["completed_years"],
                "accepted_steps": result["accepted_steps"],
                "rejected_steps": result["rejected_steps"],
            }
        ),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
