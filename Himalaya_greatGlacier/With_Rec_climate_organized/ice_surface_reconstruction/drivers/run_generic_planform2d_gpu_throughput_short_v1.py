"""Run one registered short planform-2D GPU throughput trial.

The trial uses the existing masked frozen-diffusivity split operator and
MonthlyTI package. It writes and restores one atomic midpoint checkpoint,
then reports numerical health and throughput. It is not a zero-MB, steady, or
historical production run.
"""

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
from .torch_planform_2d_system2005_adaptive_relaxation_60s_v1 import (
    ERROR_GATE,
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)


SCHEMA = "lia-191-generic-planform2d-gpu-throughput-short-config-v1"
CHECKPOINT_SCHEMA = "lia-191-generic-planform2d-gpu-throughput-checkpoint-v1"
INITIAL_DT_SECONDS = 0.03
MINIMUM_DT_SECONDS = 0.001
MAXIMUM_DT_SECONDS = 86_400.0
MAXIMUM_CONSECUTIVE_REJECTIONS = 20


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


def atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "system_id",
        "run_role",
        "target_years",
        "geology",
        "geology_sha256",
        "climate_package",
        "climate_package_sha256",
        "glen_a",
        "launch_route",
        "launch_route_sha256",
        "driver_sha256",
    )
    payload = {key: config[key] for key in keys}
    payload["solver"] = {
        "operator": "source-half/masked-frozen-diffusion/source-half",
        "initial_dt_seconds": INITIAL_DT_SECONDS,
        "minimum_dt_seconds": MINIMUM_DT_SECONDS,
        "maximum_dt_seconds": MAXIMUM_DT_SECONDS,
        "error_gate_relative_l2": ERROR_GATE,
        "midpoint_atomic_checkpoint_restore": True,
    }
    payload["gates"] = {
        "finite_nonnegative": True,
        "outside_domain_zero": True,
        "per_step_and_cumulative_mass_relative": 1.0e-10,
        "source_hashes_unchanged": True,
    }
    return payload


def contract_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(json_bytes(contract_payload(config))).hexdigest().upper()


def write_and_restore_checkpoint(
    root: Path,
    *,
    state: np.ndarray,
    elapsed_seconds: float,
    proposed_dt_seconds: float,
    accepted_steps: int,
    rejected_steps: int,
    contract_hash: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    final = root / "checkpoint_midpoint"
    temporary = root / "checkpoint_midpoint.tmp"
    if final.exists() or temporary.exists():
        raise FileExistsError(final)
    temporary.mkdir()
    state_path = temporary / "state.npz"
    np.savez_compressed(
        state_path,
        thickness_m=np.asarray(state, dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        proposed_dt_seconds=np.asarray(proposed_dt_seconds, dtype=np.float64),
        accepted_steps=np.asarray(accepted_steps, dtype=np.int64),
        rejected_steps=np.asarray(rejected_steps, dtype=np.int64),
        contract_sha256=np.asarray(contract_hash),
        schema=np.asarray(CHECKPOINT_SCHEMA),
    )
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint-complete",
        "system_id": None,
        "elapsed_seconds": elapsed_seconds,
        "contract_sha256": contract_hash,
        "state_file": "state.npz",
        "state_sha256": sha256(state_path),
    }
    (temporary / "manifest.json").write_bytes(json_bytes(manifest))
    os.replace(temporary, final)

    saved_manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    saved_state_path = final / str(saved_manifest["state_file"])
    if sha256(saved_state_path) != saved_manifest["state_sha256"]:
        raise RuntimeError("checkpoint-state-hash-mismatch")
    if saved_manifest["contract_sha256"] != contract_hash:
        raise RuntimeError("checkpoint-contract-mismatch")
    with np.load(saved_state_path, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != CHECKPOINT_SCHEMA:
            raise RuntimeError("checkpoint-schema-mismatch")
        if str(archive["contract_sha256"].item()) != contract_hash:
            raise RuntimeError("checkpoint-state-contract-mismatch")
        restored = np.asarray(archive["thickness_m"], dtype=np.float64)
        metadata = {
            "elapsed_seconds": float(archive["elapsed_seconds"].item()),
            "proposed_dt_seconds": float(archive["proposed_dt_seconds"].item()),
            "accepted_steps": int(archive["accepted_steps"].item()),
            "rejected_steps": int(archive["rejected_steps"].item()),
            "manifest": str((final / "manifest.json").resolve()),
            "manifest_sha256": sha256(final / "manifest.json"),
            "state_sha256": sha256(saved_state_path),
        }
    if not np.array_equal(restored, state):
        raise RuntimeError("checkpoint-restore-array-mismatch")
    if not math.isclose(metadata["elapsed_seconds"], elapsed_seconds, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("checkpoint-restore-elapsed-mismatch")
    return restored, metadata


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise RuntimeError("throughput-config-schema-mismatch")
    if not bool(config.get("launch_authorized", False)):
        raise RuntimeError("throughput-launch-not-authorized")
    target_years = float(config["target_years"])
    if not 0.0 < target_years <= 1.0:
        raise RuntimeError("throughput-target-years-out-of-range")
    root = Path(config["project_root"])
    paths = {
        name: root / config[name] for name in ("geology", "climate_package", "launch_route")
    }
    for name, path in paths.items():
        if sha256(path) != str(config[f"{name}_sha256"]).upper():
            raise RuntimeError(f"{name}-hash-mismatch")
    route = json.loads(paths["launch_route"].read_text(encoding="utf-8"))
    if not route.get("launch_authorized"):
        raise RuntimeError("route-does-not-authorize-launch")
    if int(config["system_id"]) not in [int(value) for value in route["authorized_systems"]]:
        raise RuntimeError("system-not-authorized-by-route")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("runtime-version-mismatch")
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")

    output = root / config["output_root"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    atomic_bytes(output / "config_snapshot.json", json_bytes(config))
    contract_hash = contract_sha256(config)
    atomic_bytes(output / "checkpoint_contract.json", json_bytes(contract_payload(config)))

    with Dataset(paths["geology"]) as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        initial_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        mask_np = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))
    if not np.isfinite(bed_np).all() or not np.isfinite(initial_np).all():
        raise RuntimeError("initial-grid-not-finite")
    if np.any(initial_np < 0.0) or np.any(initial_np[~mask_np] != 0.0):
        raise RuntimeError("initial-grid-domain-invalid")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    state = torch.as_tensor(initial_np, dtype=torch.float64, device="cuda:0")
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device="cuda:0")
    bridge = TorchMonthlyTIBridge(paths["climate_package"], device="cuda:0")
    target_seconds = target_years * bridge.sec_in_year
    checkpoint_seconds = 0.5 * target_seconds
    cell_area = dx**2
    initial_volume = float(torch.sum(state).item()) * cell_area
    tolerance = max(1.0e-6, initial_volume * 1.0e-10)
    gamma = (
        2.0
        * float(config["glen_a"])
        * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N
        / (GLEN_N + 2.0)
    )

    elapsed = 0.0
    proposed = INITIAL_DT_SECONDS
    accepted = 0
    rejected = 0
    consecutive_rejections = 0
    cumulative_source = 0.0
    maximum_closure = 0.0
    maximum_error = 0.0
    outside_maximum = 0.0
    minimum_accepted_dt = None
    checkpoint_metadata = None
    history_path = output / "step_history.csv"
    fieldnames = [
        "trial",
        "accepted",
        "elapsed_before_years",
        "month",
        "dt_seconds",
        "error_relative_l2",
        "step_closure_m3",
        "budget_residual_m3",
        "source_volume_m3",
    ]
    wall_started = time.perf_counter()
    with history_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        trial = 0
        while elapsed < target_seconds:
            month = min(
                int(math.floor(elapsed / bridge.sec_in_month + 1.0e-12)),
                bridge.month_count - 1,
            )
            next_month = (month + 1) * bridge.sec_in_month
            boundaries = [proposed, next_month - elapsed, target_seconds - elapsed]
            if checkpoint_metadata is None:
                boundaries.append(checkpoint_seconds - elapsed)
            dt = min(value for value in boundaries if value > 1.0e-10)
            full, fd = split_step(
                state,
                bed,
                mask,
                bridge,
                month=month,
                dx=dx,
                gamma=gamma,
                dt=dt,
                cell_area=cell_area,
            )
            half, hd = split_step(
                state,
                bed,
                mask,
                bridge,
                month=month,
                dx=dx,
                gamma=gamma,
                dt=0.5 * dt,
                cell_area=cell_area,
            )
            half2, h2d = split_step(
                half,
                bed,
                mask,
                bridge,
                month=month,
                dx=dx,
                gamma=gamma,
                dt=0.5 * dt,
                cell_area=cell_area,
            )
            torch.cuda.synchronize()
            delta = half2 - state
            denominator = max(float(torch.linalg.vector_norm(delta).item()), 1.0e-30)
            error = float(torch.linalg.vector_norm(half2 - full).item()) / denominator
            closure = max(
                abs(float(hd["diffusion_closure_m3"])),
                abs(float(h2d["diffusion_closure_m3"])),
            )
            source_volume = float(hd["source_volume_m3"]) + float(h2d["source_volume_m3"])
            budget_residual = float(torch.sum(delta).item()) * cell_area - source_volume
            numerical = all(bool(item["numerical_pass"]) for item in (fd, hd, h2d))
            numerical &= closure <= tolerance and abs(budget_residual) <= tolerance
            step_accepted = numerical and error <= ERROR_GATE
            trial += 1
            writer.writerow(
                {
                    "trial": trial,
                    "accepted": step_accepted,
                    "elapsed_before_years": elapsed / bridge.sec_in_year,
                    "month": month + 1,
                    "dt_seconds": dt,
                    "error_relative_l2": error,
                    "step_closure_m3": closure,
                    "budget_residual_m3": budget_residual,
                    "source_volume_m3": source_volume,
                }
            )
            if step_accepted:
                state = half2
                elapsed += dt
                accepted += 1
                consecutive_rejections = 0
                cumulative_source += source_volume
                maximum_closure = max(maximum_closure, closure, abs(budget_residual))
                maximum_error = max(maximum_error, error)
                outside_maximum = max(
                    outside_maximum, float(torch.max(torch.abs(state[~mask])).item())
                )
                minimum_accepted_dt = (
                    dt if minimum_accepted_dt is None else min(minimum_accepted_dt, dt)
                )
                proposed = min(MAXIMUM_DT_SECONDS, 2.0 * dt) if error <= ERROR_GATE / 4.0 else dt
                if checkpoint_metadata is None and math.isclose(
                    elapsed, checkpoint_seconds, rel_tol=0.0, abs_tol=1.0e-7
                ):
                    state_np = state.detach().cpu().numpy()
                    restored, checkpoint_metadata = write_and_restore_checkpoint(
                        output,
                        state=state_np,
                        elapsed_seconds=elapsed,
                        proposed_dt_seconds=proposed,
                        accepted_steps=accepted,
                        rejected_steps=rejected,
                        contract_hash=contract_hash,
                    )
                    state = torch.as_tensor(restored, dtype=torch.float64, device="cuda:0")
                    atomic_bytes(
                        output / "progress.json",
                        json_bytes(
                            {
                                "status": "midpoint-checkpoint-restored",
                                "system_id": int(config["system_id"]),
                                "completed_years": elapsed / bridge.sec_in_year,
                                "accepted_steps": accepted,
                                "rejected_steps": rejected,
                            }
                        ),
                    )
            else:
                rejected += 1
                consecutive_rejections += 1
                proposed = 0.5 * dt
                if proposed < MINIMUM_DT_SECONDS:
                    raise RuntimeError("minimum-adaptive-dt-exhausted")
                if consecutive_rejections > MAXIMUM_CONSECUTIVE_REJECTIONS:
                    raise RuntimeError("consecutive-rejection-limit-exhausted")

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    final_np = state.detach().cpu().numpy()
    final_volume = float(torch.sum(state).item()) * cell_area
    cumulative_budget = final_volume - initial_volume - cumulative_source
    final_state = output / "final_state_short.npz"
    np.savez_compressed(final_state, thickness_m=final_np, domain_mask=mask_np)
    current_hashes = {name: sha256(path) for name, path in paths.items()}
    completed = math.isclose(elapsed, target_seconds, rel_tol=0.0, abs_tol=1.0e-7)
    checks = {
        "target_completed": completed,
        "finite_nonnegative": bool(torch.isfinite(state).all())
        and int(torch.count_nonzero(state < 0).item()) == 0,
        "outside_domain_zero": outside_maximum == 0.0
        and int(torch.count_nonzero(state[~mask]).item()) == 0,
        "mass_budget_closure": abs(cumulative_budget) <= tolerance,
        "maximum_step_closure": maximum_closure <= tolerance,
        "accepted_error_gate_respected": maximum_error <= ERROR_GATE,
        "checkpoint_written_and_restored_exactly": checkpoint_metadata is not None,
        "source_hashes_unchanged": current_hashes
        == {name: str(config[f"{name}_sha256"]).upper() for name in paths},
        "runtime_exact": torch.__version__ == "2.11.0+cu128"
        and torch.version.cuda == "12.8",
        "initial_thickness_unchanged": not bool(
            config.get("initial_thickness_modification_allowed", False)
        ),
    }
    status = "gpu-throughput-short-pass" if all(checks.values()) else "gpu-throughput-short-fail"
    summary = {
        "schema": "lia-191-generic-planform2d-gpu-throughput-short-result-v1",
        "status": status,
        "system_id": int(config["system_id"]),
        "run_role": str(config["run_role"]),
        "target_years": target_years,
        "completed_years": elapsed / bridge.sec_in_year,
        "wall_seconds": wall_seconds,
        "simulated_years_per_hour": (elapsed / bridge.sec_in_year) * 3600.0 / wall_seconds,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "minimum_accepted_dt_seconds": minimum_accepted_dt,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "cumulative_climate_source_m3": cumulative_source,
        "cumulative_mass_budget_residual_m3": cumulative_budget,
        "maximum_step_or_budget_closure_m3": maximum_closure,
        "maximum_accepted_error_relative_l2": maximum_error,
        "outside_domain_maximum_thickness_m": outside_maximum,
        "checkpoint_restore": checkpoint_metadata,
        "peak_torch_memory_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024.0**2,
        "peak_torch_memory_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024.0**2,
        "checks": checks,
        "source_hashes": current_hashes,
        "checkpoint_contract_sha256": contract_hash,
        "final_state_sha256": sha256(final_state),
        "initial_thickness_modified": False,
        "physics_modified": False,
        "gates_modified": False,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    atomic_bytes(output / "summary.json", json_bytes(summary))
    atomic_bytes(
        output / "progress.json",
        json_bytes(
            {
                "status": status,
                "system_id": int(config["system_id"]),
                "completed_years": summary["completed_years"],
                "accepted_steps": accepted,
                "rejected_steps": rejected,
            }
        ),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
