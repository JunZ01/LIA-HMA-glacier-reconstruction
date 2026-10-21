"""CUDA equivalence, checkpoint, and throughput runner for the approved v2 preflight."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import torch
import numpy as np


DRIVERS = Path(__file__).resolve().parent
sys.path.insert(0, str(DRIVERS))
from lia_1d_gpu_preflight_guard_v2 import OUTPUT_ROOT, PROJECT_ROOT, sha256, validate_launch_authorization
from torch_flowline_1d_batched_semiimplicit_v1 import BatchedFlowlineState, SEC_IN_YEAR


STAGES = ("after_step", "after_month", "after_year", "at_checkpoint", "uninterrupted_year10", "resumed_year10")
TOLERANCES = {
    "length_absolute_m": 0.0,
    "area_relative": 1e-9,
    "volume_relative": 1e-9,
    "thickness_max_absolute_m": 1e-7,
    "thickness_relative_l2": 1e-10,
    "climatic_mass_balance_relative": 1e-10,
    "minimum_actual_timestep_seconds_absolute": 1e-9,
}


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def tensor_dict_to_python(diagnostics: dict, lane_count: int) -> list[dict]:
    rows = [dict() for _ in range(lane_count)]
    for key, value in diagnostics.items():
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu().numpy()
            for lane in range(lane_count):
                rows[lane][key] = float(data[lane])
        else:
            for lane in range(lane_count):
                rows[lane][key] = int(value) if isinstance(value, int) else float(value)
    return rows


def create_state(arrays: dict, indices: np.ndarray, device: torch.device, *,
                 thick_override=None, time_seconds=0.0, accepted_steps=0,
                 rejected_steps=0, minimum_dt=float("inf"), cumulative_mb=None,
                 cumulative_residual=None) -> BatchedFlowlineState:
    def t(name, dtype=torch.float64):
        return torch.as_tensor(arrays[name][indices], dtype=dtype, device=device)
    thick = t("thick") if thick_override is None else torch.as_tensor(thick_override, dtype=torch.float64, device=device)
    return BatchedFlowlineState(
        bed_h=t("bed_h"), widths_m=t("widths_m"), thick=thick,
        valid_mask=t("valid_mask", torch.bool), nx=t("nx", torch.int64),
        dx_meter=t("dx_meter"), glen_a=t("glen_a"), melt_f=t("melt_f"),
        prcp_fac=t("prcp_fac"), temp_bias=t("temp_bias"), temp_melt=t("temp_melt"),
        climate_temp=t("climate_temp"), climate_prcp=t("climate_prcp"),
        climate_gradient=t("climate_gradient"), climate_ref_hgt=t("climate_ref_hgt"),
        climate_years=t("climate_years", torch.int64), y0=1836.0,
        time_seconds=float(time_seconds), accepted_steps=int(accepted_steps),
        rejected_steps=int(rejected_steps), minimum_actual_timestep_seconds=float(minimum_dt),
        cumulative_climatic_mass_balance_m3=(None if cumulative_mb is None else torch.as_tensor(cumulative_mb, dtype=torch.float64, device=device)),
        cumulative_mass_budget_residual_m3=(None if cumulative_residual is None else torch.as_tensor(cumulative_residual, dtype=torch.float64, device=device)),
    )


class NvidiaSampler:
    def __init__(self):
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                )
                parts = [part.strip() for part in completed.stdout.strip().split(",")]
                self.rows.append({"wall_time": time.time(), "gpu_timestamp": parts[0],
                                  "utilization_percent": float(parts[1]),
                                  "memory_used_mib": float(parts[2]), "memory_total_mib": float(parts[3])})
            except Exception as exc:
                self.rows.append({"wall_time": time.time(), "sampling_error": repr(exc)})
            self.stop_event.wait(1.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=5)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0)


def compare_stage(stage: str, gpu_rows: list[dict], gpu_thick: np.ndarray,
                  cpu_records: list[dict], cpu_thick: np.ndarray,
                  valid_mask: np.ndarray, initial_volume: np.ndarray) -> dict:
    lanes = []
    all_pass = True
    for lane, (gpu, cpu) in enumerate(zip(gpu_rows, cpu_records)):
        cpu_diag = cpu[stage]
        mask = valid_mask[lane]
        difference = gpu_thick[lane, mask] - cpu_thick[lane, mask]
        max_abs = float(np.max(np.abs(difference)))
        rel_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(cpu_thick[lane, mask]), 1.0))
        residual_limit = max(1e-3, 1e-10 * float(initial_volume[lane]))
        errors = {
            "length_absolute_m": abs(gpu["length_m"] - cpu_diag["length_m"]),
            "area_relative": relative_error(gpu["area_m2"], cpu_diag["area_m2"]),
            "volume_relative": relative_error(gpu["volume_m3"], cpu_diag["volume_m3"]),
            "thickness_max_absolute_m": max_abs,
            "thickness_relative_l2": rel_l2,
            "climatic_mass_balance_relative": relative_error(gpu["cumulative_climatic_mass_balance_m3"], cpu_diag["cumulative_climatic_mass_balance_m3"]),
            "mass_budget_residual_delta_m3": abs(gpu["cumulative_mass_budget_residual_m3"] - cpu_diag["cumulative_mass_budget_residual_m3"]),
            "minimum_actual_timestep_seconds_absolute": abs(gpu["minimum_actual_timestep_seconds"] - cpu_diag["minimum_actual_timestep_seconds"]),
        }
        finite = bool(np.all(np.isfinite(gpu_thick[lane, mask])))
        nonnegative = bool(np.all(gpu_thick[lane, mask] >= 0.0))
        padding_zero = bool(np.all(gpu_thick[lane, ~mask] == 0.0))
        passed = (
            errors["length_absolute_m"] <= TOLERANCES["length_absolute_m"]
            and errors["area_relative"] <= TOLERANCES["area_relative"]
            and errors["volume_relative"] <= TOLERANCES["volume_relative"]
            and errors["thickness_max_absolute_m"] <= TOLERANCES["thickness_max_absolute_m"]
            and errors["thickness_relative_l2"] <= TOLERANCES["thickness_relative_l2"]
            and errors["climatic_mass_balance_relative"] <= TOLERANCES["climatic_mass_balance_relative"]
            and errors["mass_budget_residual_delta_m3"] <= residual_limit
            and errors["minimum_actual_timestep_seconds_absolute"] <= TOLERANCES["minimum_actual_timestep_seconds_absolute"]
            and int(gpu["accepted_steps"]) == int(cpu_diag["accepted_steps"])
            and int(gpu["rejected_steps"]) == int(cpu_diag["rejected_steps"]) == 0
            and finite and nonnegative and padding_zero
        )
        all_pass &= passed
        lanes.append({"lane": lane, "cell": cpu["cell"], "pass": passed,
                      "errors": errors, "mass_budget_residual_limit_m3": residual_limit,
                      "accepted_steps_exact": int(gpu["accepted_steps"]) == int(cpu_diag["accepted_steps"]),
                      "rejected_steps_exact_zero": int(gpu["rejected_steps"]) == int(cpu_diag["rejected_steps"]) == 0,
                      "finite": finite, "nonnegative": nonnegative, "padding_zero": padding_zero})
    return {"stage": stage, "pass": all_pass, "lanes": lanes}


def run_timed_configuration(arrays: dict, batch_size: int, device: torch.device) -> dict:
    indices_all = np.arange(24, dtype=np.int64)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_cpu = time.process_time()
    torch.cuda.synchronize()
    started = time.perf_counter()
    failed_lanes = []
    completed_years = 0.0
    for start in range(0, 24, batch_size):
        indices = indices_all[start:start + batch_size]
        try:
            state = create_state(arrays, indices, device)
            state.run_until(1846.0)
            completed_years += 10.0 * len(indices)
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            if len(indices) == 1:
                failed_lanes.append({"lane": int(indices[0]), "error": repr(exc)})
            else:
                for index in indices:
                    try:
                        isolated = create_state(arrays, np.array([index]), device)
                        isolated.run_until(1846.0)
                        completed_years += 10.0
                    except Exception as isolated_exc:
                        failed_lanes.append({"lane": int(index), "error": repr(isolated_exc)})
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    return {
        "wall_seconds": wall,
        "cpu_process_seconds": time.process_time() - started_cpu,
        "completed_model_years": completed_years,
        "completed_model_years_per_wall_hour": completed_years * 3600.0 / wall,
        "completed_cells_per_wall_hour": (completed_years / 10.0) * 3600.0 / wall,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "failed_lanes": failed_lanes,
    }


def main() -> int:
    authorization = validate_launch_authorization(output_must_be_absent=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    gpu_root = OUTPUT_ROOT / "gpu"
    if gpu_root.exists():
        raise FileExistsError("GPU output root already exists")
    gpu_root.mkdir(parents=True)
    with np.load(authorization["npz_path"]) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    cpu_summary_path = OUTPUT_ROOT / "cpu/summary.json"
    cpu_states_path = OUTPUT_ROOT / "cpu/oracle_states.npz"
    with cpu_summary_path.open("r", encoding="utf-8") as handle:
        cpu_summary = json.load(handle)
    if sha256(cpu_states_path) != cpu_summary["oracle_states_sha256"]:
        raise RuntimeError("CPU oracle states hash mismatch")
    with np.load(cpu_states_path) as bundle:
        cpu_states = {name: bundle[name] for name in bundle.files}
    cpu_records = cpu_summary["records"]
    indices = np.arange(24, dtype=np.int64)
    state = create_state(arrays, indices, device)
    gpu_stage_rows = {}
    gpu_stage_thickness = {}
    state.step(108000.0)
    gpu_stage_rows["after_step"] = tensor_dict_to_python(state.diagnostics(), 24)
    gpu_stage_thickness["after_step"] = state.thick.detach().cpu().numpy().copy()
    state.run_until(1836.0 + 1.0 / 12.0)
    gpu_stage_rows["after_month"] = tensor_dict_to_python(state.diagnostics(), 24)
    gpu_stage_thickness["after_month"] = state.thick.detach().cpu().numpy().copy()
    state.run_until(1837.0)
    gpu_stage_rows["after_year"] = tensor_dict_to_python(state.diagnostics(), 24)
    gpu_stage_thickness["after_year"] = state.thick.detach().cpu().numpy().copy()
    stage_b = [compare_stage(stage, gpu_stage_rows[stage], gpu_stage_thickness[stage],
                             cpu_records, cpu_states[f"{stage}_thick"], arrays["valid_mask"],
                             np.array([record["initial_volume_m3"] for record in cpu_records]))
               for stage in ("after_step", "after_month", "after_year")]
    if not all(item["pass"] for item in stage_b):
        summary = {"schema": "lia-1d-gpu-batch-preflight-v2", "status": "stage-b-equivalence-fail",
                   "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
                   "device": torch.cuda.get_device_name(0), "stage_b": stage_b,
                   "stage_c_run": False, "full_191_gpu_matrix_run": False,
                   "planform_2d_or_igm_run": False}
        write_json_atomic(gpu_root / "summary.json", summary)
        print(json.dumps({"status": summary["status"]}, indent=2))
        return 2

    state.run_until(1841.0)
    gpu_stage_rows["at_checkpoint"] = tensor_dict_to_python(state.diagnostics(), 24)
    gpu_stage_thickness["at_checkpoint"] = state.thick.detach().cpu().numpy().copy()
    checkpoint = gpu_root / "batch-24/checkpoints/year-0005/state.npz"
    diagnostics = state.diagnostics()
    write_npz_atomic(
        checkpoint,
        thick=state.thick.detach().cpu().numpy(), nx=arrays["nx"], valid_mask=arrays["valid_mask"],
        model_time_seconds=np.array([state.time_seconds], dtype=np.float64),
        accepted_steps=np.array([state.accepted_steps], dtype=np.int64),
        rejected_steps=np.array([state.rejected_steps], dtype=np.int64),
        minimum_actual_timestep_seconds=np.array([state.minimum_actual_timestep_seconds], dtype=np.float64),
        cumulative_climatic_mass_balance_m3=diagnostics["cumulative_climatic_mass_balance_m3"].cpu().numpy(),
        cumulative_mass_budget_residual_m3=diagnostics["cumulative_mass_budget_residual_m3"].cpu().numpy(),
    )
    checkpoint_manifest = {
        "schema": "lia-1d-gpu-preflight-gpu-checkpoint-v1",
        "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
        "state_sha256": sha256(checkpoint), "year": 1841.0,
        "bridge_sha256": authorization["fixture"]["npz_sha256"],
    }
    write_json_atomic(checkpoint.parent / "manifest.json", checkpoint_manifest)
    with np.load(checkpoint) as saved:
        saved_arrays = {name: saved[name] for name in saved.files}
    restore_bitwise = bool(np.array_equal(saved_arrays["thick"], state.thick.detach().cpu().numpy()))
    state.run_until(1846.0)
    gpu_stage_rows["uninterrupted_year10"] = tensor_dict_to_python(state.diagnostics(), 24)
    gpu_stage_thickness["uninterrupted_year10"] = state.thick.detach().cpu().numpy().copy()
    resumed = create_state(
        arrays, indices, device, thick_override=saved_arrays["thick"],
        time_seconds=float(saved_arrays["model_time_seconds"][0]),
        accepted_steps=int(saved_arrays["accepted_steps"][0]), rejected_steps=int(saved_arrays["rejected_steps"][0]),
        minimum_dt=float(saved_arrays["minimum_actual_timestep_seconds"][0]),
        cumulative_mb=saved_arrays["cumulative_climatic_mass_balance_m3"],
        cumulative_residual=saved_arrays["cumulative_mass_budget_residual_m3"],
    )
    resumed_restore_bitwise = bool(torch.equal(resumed.thick.cpu(), torch.as_tensor(saved_arrays["thick"])))
    resumed.run_until(1846.0)
    gpu_stage_rows["resumed_year10"] = tensor_dict_to_python(resumed.diagnostics(), 24)
    gpu_stage_thickness["resumed_year10"] = resumed.thick.detach().cpu().numpy().copy()
    stage_c_equivalence = [compare_stage(stage, gpu_stage_rows[stage], gpu_stage_thickness[stage],
                                         cpu_records, cpu_states[f"{stage}_thick"], arrays["valid_mask"],
                                         np.array([record["initial_volume_m3"] for record in cpu_records]))
                           for stage in ("at_checkpoint", "uninterrupted_year10", "resumed_year10")]
    same_backend_resume_max_abs = float(np.max(np.abs(
        gpu_stage_thickness["uninterrupted_year10"] - gpu_stage_thickness["resumed_year10"]
    )))
    checkpoint_pass = restore_bitwise and resumed_restore_bitwise and same_backend_resume_max_abs == 0.0
    equivalence_pass = all(item["pass"] for item in stage_b + stage_c_equivalence)
    if not equivalence_pass or not checkpoint_pass:
        summary = {"schema": "lia-1d-gpu-batch-preflight-v2", "status": "stage-c-equivalence-or-checkpoint-fail",
                   "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
                   "device": torch.cuda.get_device_name(0), "stage_b": stage_b,
                   "stage_c_equivalence": stage_c_equivalence, "checkpoint_restore_bitwise": restore_bitwise,
                   "resumed_payload_bitwise": resumed_restore_bitwise,
                   "same_backend_resume_max_abs_thickness_m": same_backend_resume_max_abs,
                   "throughput_run": False, "full_191_gpu_matrix_run": False,
                   "planform_2d_or_igm_run": False}
        write_json_atomic(gpu_root / "summary.json", summary)
        print(json.dumps({"status": summary["status"]}, indent=2))
        return 3

    throughput = []
    oom_seen = False
    for batch_size in (24, 16, 8, 4, 2, 1):
        if oom_seen:
            torch.cuda.empty_cache()
        try:
            run_timed_configuration(arrays, batch_size, device)
            repeats = []
            with NvidiaSampler() as sampler:
                for _ in range(3):
                    repeats.append(run_timed_configuration(arrays, batch_size, device))
            rates = [item["completed_model_years_per_wall_hour"] for item in repeats]
            variation = (max(rates) - min(rates)) / max(float(np.median(rates)), 1.0)
            telemetry_path = OUTPUT_ROOT / f"telemetry/gpu-batch-{batch_size}.json"
            write_json_atomic(telemetry_path, {"schema": "lia-1d-gpu-telemetry-v1", "samples": sampler.rows})
            throughput.append({"batch_size": batch_size, "status": "pass", "repeats": repeats,
                               "median_model_years_per_wall_hour": float(np.median(rates)),
                               "range_model_years_per_wall_hour": [min(rates), max(rates)],
                               "relative_run_to_run_range": variation,
                               "telemetry_path": str(telemetry_path.relative_to(PROJECT_ROOT)),
                               "telemetry_sha256": sha256(telemetry_path)})
        except torch.cuda.OutOfMemoryError as exc:
            oom_seen = True
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            throughput.append({"batch_size": batch_size, "status": "oom", "error": repr(exc)})
            if batch_size == 1:
                break
    passing = [item for item in throughput if item["status"] == "pass"]
    best = max(passing, key=lambda item: item["median_model_years_per_wall_hour"]) if passing else None
    speedup = (best["median_model_years_per_wall_hour"] / cpu_summary["completed_model_years_per_wall_hour"]) if best else 0.0
    stability_pass = bool(passing) and all(item["relative_run_to_run_range"] <= 0.05 for item in passing)
    no_failures = all(not repeat["failed_lanes"] for item in passing for repeat in item["repeats"])
    promotion_gate = equivalence_pass and checkpoint_pass and speedup >= 1.5 and stability_pass and no_failures and not oom_seen
    summary = {
        "schema": "lia-1d-gpu-batch-preflight-v2", "status": "complete",
        "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
        "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda, "precision": "float64", "cell_count": 24,
        "stage_b": stage_b, "stage_c_equivalence": stage_c_equivalence,
        "checkpoint_restore_bitwise": restore_bitwise,
        "resumed_payload_bitwise": resumed_restore_bitwise,
        "same_backend_resume_max_abs_thickness_m": same_backend_resume_max_abs,
        "throughput": throughput,
        "cpu_oracle_model_years_per_wall_hour": cpu_summary["completed_model_years_per_wall_hour"],
        "best_gpu_batch_size": None if best is None else best["batch_size"],
        "best_gpu_model_years_per_wall_hour": None if best is None else best["median_model_years_per_wall_hour"],
        "gpu_to_cpu_speedup": speedup, "promotion_gate_pass": promotion_gate,
        "promotion_effect": "proposal-only; no full-191 launch",
        "full_191_gpu_matrix_run": False, "planform_2d_or_igm_run": False,
    }
    write_json_atomic(gpu_root / "summary.json", summary)
    print(json.dumps({"status": "complete", "promotion_gate_pass": promotion_gate,
                      "best_batch_size": summary["best_gpu_batch_size"], "speedup": speedup}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
