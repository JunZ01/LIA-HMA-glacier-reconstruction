"""Run the registered generic planform-2D GPU qualification preflight.

Each system starts twice from the immutable source thickness. The first phase
is a 0.1-year zero-MB masked/no-flux test; the second is a one-year MonthlyTI
response test. Both phases use the same split operator and perform an exact
midpoint checkpoint round trip. This driver never runs steady or historical
dynamics.
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

from .generic_planform2d_adapter_contract_v1 import array_sha256
from .run_system2005_masked_climate_1y_preflight_v1 import split_step
from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge
from .torch_planform_2d_system2005_adaptive_relaxation_60s_v1 import (
    ERROR_GATE,
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)


SCHEMA = "lia-191-generic-planform2d-gpu-qualification-preflight-config-v1"
CHECKPOINT_SCHEMA = "lia-191-generic-planform2d-gpu-qualification-checkpoint-v1"
ZERO_MB_YEARS = 0.1
CLIMATE_RESPONSE_YEARS = 1.0
ZERO_MB_MASS_RELATIVE_TOLERANCE = 1.0e-12
CLIMATE_MASS_RELATIVE_TOLERANCE = 1.0e-10
INITIAL_DT_SECONDS = 0.03
MINIMUM_DT_SECONDS = 0.001
MAXIMUM_DT_SECONDS = 86_400.0
MAXIMUM_CONSECUTIVE_REJECTIONS = 20
STAGE2_KEYS = (
    "melt_f",
    "prcp_fac",
    "glen_a",
    "fs",
    "temp_melt_c",
    "base_temp_bias_c",
    "delta_t_c",
)


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
        "geology",
        "geology_sha256",
        "source_thickness_sha256",
        "geology_initial_thickness_sha256",
        "climate_package",
        "climate_package_sha256",
        "static_summary",
        "static_summary_sha256",
        "upstream_evidence",
        "upstream_evidence_sha256",
        "upstream_config_snapshot",
        "upstream_config_snapshot_sha256",
        "upstream_status",
        "source_shape",
        "adapter_shape",
        "stage2_frozen",
        "launch_route",
        "launch_route_sha256",
        "driver_sha256",
    )
    payload = {key: config[key] for key in keys}
    payload["phases"] = {
        "zero_mb_years": ZERO_MB_YEARS,
        "climate_response_years": CLIMATE_RESPONSE_YEARS,
        "restart_each_phase_from_source_thickness": True,
        "midpoint_checkpoint_bitwise_restore": True,
    }
    payload["solver"] = {
        "operator": "source-half/masked-frozen-diffusion/source-half",
        "initial_dt_seconds": INITIAL_DT_SECONDS,
        "minimum_dt_seconds": MINIMUM_DT_SECONDS,
        "maximum_dt_seconds": MAXIMUM_DT_SECONDS,
        "error_gate_relative_l2": ERROR_GATE,
    }
    payload["gates"] = {
        "zero_mb_mass_relative": ZERO_MB_MASS_RELATIVE_TOLERANCE,
        "climate_mass_relative": CLIMATE_MASS_RELATIVE_TOLERANCE,
        "finite_nonnegative": True,
        "outside_domain_zero": True,
        "climate_source_nonzero": True,
        "source_hashes_unchanged": True,
    }
    return payload


def contract_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(json_bytes(contract_payload(config))).hexdigest().upper()


class ZeroMassBalanceBridge:
    """Identity source compatible with the shared split-step core."""

    def apply_connected_ice_source(
        self,
        thickness_m: torch.Tensor,
        bed_m: torch.Tensor,
        *,
        month_index: int,
        dt_seconds: float,
    ) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
        del bed_m, month_index, dt_seconds
        return thickness_m.clone(), {
            "active_cell_count": int(torch.count_nonzero(thickness_m > 0.0).item()),
            "remote_zero_cell_count": int(torch.count_nonzero(thickness_m <= 0.0).item()),
            "remote_zero_unchanged": True,
            "prevented_overmelt_sum_m": 0.0,
            "accepted_thickness_change_sum_m": 0.0,
            "state_finite": bool(torch.isfinite(thickness_m).all()),
            "negative_cell_count": int(torch.count_nonzero(thickness_m < 0.0).item()),
        }


def write_and_restore_checkpoint(
    root: Path,
    *,
    name: str,
    state: np.ndarray,
    elapsed_seconds: float,
    proposed_dt_seconds: float,
    accepted_steps: int,
    rejected_steps: int,
    contract_hash: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    final = root / name
    temporary = root / f"{name}.tmp"
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


def validate_frozen_inputs(
    config: dict[str, Any], paths: dict[str, Path], initial_np: np.ndarray
) -> dict[str, bool]:
    static = json.loads(paths["static_summary"].read_text(encoding="utf-8"))
    upstream = json.loads(paths["upstream_evidence"].read_text(encoding="utf-8"))
    upstream_config = json.loads(
        paths["upstream_config_snapshot"].read_text(encoding="utf-8")
    )
    expected_stage2 = config["stage2_frozen"]
    static_stage2 = static["stage2_frozen"]
    upstream_stage2 = upstream_config["stage2"]
    checks = {
        "static_system_id_matches": int(static["system_id"]) == int(config["system_id"]),
        "upstream_system_id_matches": int(upstream["system_id"]) == int(config["system_id"]),
        "upstream_status_matches": str(upstream["status"]) == str(config["upstream_status"]),
        "static_source_shape_matches": list(static["source_shape"])
        == list(config["source_shape"]),
        "static_adapter_shape_matches": list(static["padded_shape"])
        == list(config["adapter_shape"]),
        "geology_adapter_shape_matches": list(initial_np.shape)
        == list(config["adapter_shape"]),
        "geology_initial_thickness_hash_matches": array_sha256(initial_np)
        == str(config["geology_initial_thickness_sha256"]).upper(),
        "static_source_thickness_hash_matches": str(
            static["adapter_contract"]["source_thickness_sha256"]
        ).upper()
        == str(config["source_thickness_sha256"]).upper(),
        "static_reports_no_thickness_mutation": not bool(
            static["adapter_contract"]["initial_thickness_modified"]
        ),
    }
    for key in STAGE2_KEYS:
        checks[f"static_stage2_{key}_matches"] = static_stage2[key] == expected_stage2[key]
        checks[f"upstream_stage2_{key}_matches"] = upstream_stage2[key] == expected_stage2[key]
    return checks


def run_phase(
    *,
    output: Path,
    phase_name: str,
    initial: torch.Tensor,
    bed: torch.Tensor,
    mask: torch.Tensor,
    bridge: Any,
    sec_in_year: float,
    sec_in_month: float,
    target_years: float,
    mass_relative_tolerance: float,
    dx: float,
    gamma: float,
    contract_hash: str,
    climate_phase: bool,
) -> dict[str, Any]:
    state = initial.clone()
    target_seconds = target_years * sec_in_year
    checkpoint_seconds = 0.5 * target_seconds
    cell_area = dx**2
    initial_volume = float(torch.sum(state).item()) * cell_area
    tolerance = max(1.0e-6, initial_volume * mass_relative_tolerance)
    elapsed = 0.0
    proposed = INITIAL_DT_SECONDS
    accepted = rejected = consecutive_rejections = 0
    cumulative_source = maximum_closure = maximum_error = outside_maximum = 0.0
    minimum_accepted_dt = None
    checkpoint_metadata = None
    history_path = output / f"{phase_name}_step_history.csv"
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
    started = time.perf_counter()
    with history_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        trial = 0
        while elapsed < target_seconds:
            month = int(math.floor(elapsed / sec_in_month + 1.0e-12)) if climate_phase else 0
            boundaries = [proposed, target_seconds - elapsed]
            if climate_phase:
                boundaries.append((month + 1) * sec_in_month - elapsed)
            if checkpoint_metadata is None:
                boundaries.append(checkpoint_seconds - elapsed)
            dt = min(value for value in boundaries if value > 1.0e-10)
            full, fd = split_step(
                state, bed, mask, bridge, month=month, dx=dx, gamma=gamma, dt=dt,
                cell_area=cell_area,
            )
            half, hd = split_step(
                state, bed, mask, bridge, month=month, dx=dx, gamma=gamma, dt=0.5 * dt,
                cell_area=cell_area,
            )
            half2, h2d = split_step(
                half, bed, mask, bridge, month=month, dx=dx, gamma=gamma, dt=0.5 * dt,
                cell_area=cell_area,
            )
            torch.cuda.synchronize()
            delta = half2 - state
            denominator = max(float(torch.linalg.vector_norm(delta).item()), 1.0e-30)
            error = float(torch.linalg.vector_norm(half2 - full).item()) / denominator
            closure = max(
                abs(float(fd["diffusion_closure_m3"])),
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
                    "elapsed_before_years": elapsed / sec_in_year,
                    "month": month + 1 if climate_phase else 0,
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
                    restored, checkpoint_metadata = write_and_restore_checkpoint(
                        output,
                        name=f"{phase_name}_checkpoint_midpoint",
                        state=state.detach().cpu().numpy(),
                        elapsed_seconds=elapsed,
                        proposed_dt_seconds=proposed,
                        accepted_steps=accepted,
                        rejected_steps=rejected,
                        contract_hash=contract_hash,
                    )
                    state = torch.as_tensor(restored, dtype=torch.float64, device="cuda:0")
            else:
                rejected += 1
                consecutive_rejections += 1
                proposed = 0.5 * dt
                if proposed < MINIMUM_DT_SECONDS:
                    raise RuntimeError(f"{phase_name}-minimum-adaptive-dt-exhausted")
                if consecutive_rejections > MAXIMUM_CONSECUTIVE_REJECTIONS:
                    raise RuntimeError(f"{phase_name}-consecutive-rejection-limit-exhausted")

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    final_np = state.detach().cpu().numpy()
    final_volume = float(torch.sum(state).item()) * cell_area
    cumulative_budget = final_volume - initial_volume - cumulative_source
    final_path = output / f"{phase_name}_final_state.npz"
    np.savez_compressed(final_path, thickness_m=final_np, domain_mask=mask.detach().cpu().numpy())
    forcing_tolerance = max(1.0e-6, initial_volume * 1.0e-12)
    checks = {
        "target_completed": math.isclose(elapsed, target_seconds, rel_tol=0.0, abs_tol=1.0e-7),
        "finite_nonnegative": bool(torch.isfinite(state).all())
        and int(torch.count_nonzero(state < 0).item()) == 0,
        "outside_domain_zero": outside_maximum == 0.0
        and int(torch.count_nonzero((state > 0.0) & ~mask).item()) == 0,
        "mass_budget_closure": abs(cumulative_budget) <= tolerance,
        "maximum_step_closure": maximum_closure <= tolerance,
        "accepted_error_gate_respected": maximum_error <= ERROR_GATE,
        "checkpoint_written_and_restored_bitwise": checkpoint_metadata is not None,
        "climate_source_nonzero_or_zero_mb_exact": (
            abs(cumulative_source) > forcing_tolerance if climate_phase else cumulative_source == 0.0
        ),
    }
    return {
        "phase": phase_name,
        "status": "pass" if all(checks.values()) else "fail-stop",
        "target_years": target_years,
        "completed_years": elapsed / sec_in_year,
        "wall_seconds": wall_seconds,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "minimum_accepted_dt_seconds": minimum_accepted_dt,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "cumulative_source_m3": cumulative_source,
        "cumulative_mass_budget_residual_m3": cumulative_budget,
        "mass_tolerance_m3": tolerance,
        "maximum_step_or_budget_closure_m3": maximum_closure,
        "maximum_accepted_error_relative_l2": maximum_error,
        "outside_domain_maximum_thickness_m": outside_maximum,
        "checkpoint_restore": checkpoint_metadata,
        "checks": checks,
        "final_state_sha256": sha256(final_path),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise RuntimeError("qualification-config-schema-mismatch")
    if config.get("run_role") != "scientific-qualification-preflight":
        raise RuntimeError("qualification-run-role-mismatch")
    if not bool(config.get("launch_authorized", False)):
        raise RuntimeError("qualification-launch-not-authorized")
    if sha256(Path(__file__)) != str(config["driver_sha256"]).upper():
        raise RuntimeError("driver-hash-mismatch")
    root = Path(config["project_root"])
    path_keys = (
        "geology",
        "climate_package",
        "static_summary",
        "upstream_evidence",
        "upstream_config_snapshot",
        "launch_route",
    )
    paths = {name: root / config[name] for name in path_keys}
    for name, path in paths.items():
        if sha256(path) != str(config[f"{name}_sha256"]).upper():
            raise RuntimeError(f"{name}-hash-mismatch")
    route = json.loads(paths["launch_route"].read_text(encoding="utf-8"))
    if not route.get("launch_authorized"):
        raise RuntimeError("route-does-not-authorize-launch")
    if int(config["system_id"]) not in [int(value) for value in route["authorized_systems"]]:
        raise RuntimeError("system-not-authorized-by-route")
    if route.get("authorized_run_role") != "scientific-qualification-preflight":
        raise RuntimeError("route-run-role-mismatch")
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
    atomic_bytes(output / "qualification_contract.json", json_bytes(contract_payload(config)))

    with Dataset(paths["geology"]) as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        initial_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        mask_np = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))
    if not np.isfinite(bed_np).all() or not np.isfinite(initial_np).all():
        raise RuntimeError("initial-grid-not-finite")
    if np.any(initial_np < 0.0) or np.any(initial_np[~mask_np] != 0.0):
        raise RuntimeError("initial-grid-domain-invalid")
    frozen_checks = validate_frozen_inputs(config, paths, initial_np)
    if not all(frozen_checks.values()):
        raise RuntimeError("frozen-input-contract-failed")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    initial = torch.as_tensor(initial_np, dtype=torch.float64, device="cuda:0")
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device="cuda:0")
    climate_bridge = TorchMonthlyTIBridge(paths["climate_package"], device="cuda:0")
    zero_bridge = ZeroMassBalanceBridge()
    gamma = (
        2.0
        * float(config["stage2_frozen"]["glen_a"])
        * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N
        / (GLEN_N + 2.0)
    )

    zero_result = run_phase(
        output=output,
        phase_name="zero_mb_0p1y",
        initial=initial,
        bed=bed,
        mask=mask,
        bridge=zero_bridge,
        sec_in_year=climate_bridge.sec_in_year,
        sec_in_month=climate_bridge.sec_in_month,
        target_years=ZERO_MB_YEARS,
        mass_relative_tolerance=ZERO_MB_MASS_RELATIVE_TOLERANCE,
        dx=dx,
        gamma=gamma,
        contract_hash=contract_hash,
        climate_phase=False,
    )
    if zero_result["status"] != "pass":
        climate_result = None
    else:
        climate_result = run_phase(
            output=output,
            phase_name="climate_response_1y",
            initial=initial,
            bed=bed,
            mask=mask,
            bridge=climate_bridge,
            sec_in_year=climate_bridge.sec_in_year,
            sec_in_month=climate_bridge.sec_in_month,
            target_years=CLIMATE_RESPONSE_YEARS,
            mass_relative_tolerance=CLIMATE_MASS_RELATIVE_TOLERANCE,
            dx=dx,
            gamma=gamma,
            contract_hash=contract_hash,
            climate_phase=True,
        )

    source_hashes_after = {name: sha256(path) for name, path in paths.items()}
    source_hashes_unchanged = all(
        source_hashes_after[name] == str(config[f"{name}_sha256"]).upper()
        for name in path_keys
    )
    with Dataset(paths["geology"]) as dataset:
        source_thickness_after = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
    adapter_thickness_unchanged = (
        array_sha256(source_thickness_after)
        == str(config["geology_initial_thickness_sha256"]).upper()
    )
    overall_checks = {
        "frozen_input_contract": all(frozen_checks.values()),
        "zero_mb_phase_pass": zero_result["status"] == "pass",
        "climate_response_phase_pass": climate_result is not None
        and climate_result["status"] == "pass",
        "source_hashes_unchanged": source_hashes_unchanged,
        "adapter_thickness_unchanged": adapter_thickness_unchanged,
        "runtime_exact": torch.__version__ == "2.11.0+cu128" and torch.version.cuda == "12.8",
    }
    summary = {
        "schema": "lia-191-generic-planform2d-gpu-qualification-preflight-result-v1",
        "status": "qualification-preflight-pass" if all(overall_checks.values()) else "qualification-preflight-fail-stop",
        "system_id": int(config["system_id"]),
        "run_role": config["run_role"],
        "upstream_status": config["upstream_status"],
        "stage2_frozen": config["stage2_frozen"],
        "source_shape": config["source_shape"],
        "adapter_shape": config["adapter_shape"],
        "source_thickness_sha256": config["source_thickness_sha256"],
        "geology_initial_thickness_sha256": config["geology_initial_thickness_sha256"],
        "zero_mb": zero_result,
        "climate_response": climate_result,
        "frozen_input_checks": frozen_checks,
        "overall_checks": overall_checks,
        "peak_torch_memory_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
        "peak_torch_memory_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
        "source_hashes_after": source_hashes_after,
        "qualification_contract_sha256": contract_hash,
        "initial_thickness_modified": False,
        "physics_modified": False,
        "gates_modified": False,
        "zero_mb_run": True,
        "short_climate_response_run": True,
        "steady_run": False,
        "historical_run": False,
        "accepted_count_effect": 0,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "next_action": "formal paired adjudication before any steady-stage launch",
    }
    atomic_bytes(output / "summary.json", json_bytes(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
