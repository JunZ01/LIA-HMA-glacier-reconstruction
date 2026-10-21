"""Minimal 24-lane GPU Stage-B equivalence run with durable stage evidence."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import glob as python_glob
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from lia_1d_gpu_stage_b_v7_guard import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    THREAD_ENVIRONMENT,
    sha256,
    validate_v7_launch,
)


SS4_PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
TORCH_SITE = PROJECT_ROOT / ".runtime/torch-cu128-v1/site-packages"
TORCH_LIB = TORCH_SITE / "torch/lib"
WINDOWS = Path(os.environ.get("SystemRoot", "C:/Windows"))
EXPECTED_PATH = (TORCH_LIB, WINDOWS / "System32", WINDOWS)
TORCH_INIT = TORCH_SITE / "torch/__init__.py"
IOMP_RUNTIME = TORCH_LIB / "libiomp5md.dll"
IOMP_STUBS = TORCH_LIB / "libiompstubs5md.dll"
TORCH_INIT_SHA256 = "9805FE98AF34A65B1AB758C7909CC7B9906F40777257E513968545B19E319AA1"
IOMP_RUNTIME_SHA256 = "EB14CF741B10A6CA176ACDED284BFF7B069E421F49927E28FEF9685C6F62ABF3"
IOMP_STUBS_SHA256 = "3707FAE5BAA5C77DA4A355D9977CF132251310DB89923BF944B1DA6DEC702922"
TOLERANCES = {
    "length_absolute_m": 0.0,
    "area_relative": 1e-9,
    "volume_relative": 1e-9,
    "thickness_max_absolute_m": 1e-7,
    "thickness_relative_l2": 1e-10,
    "climatic_mass_balance_relative": 1e-10,
    "minimum_actual_timestep_seconds_absolute": 1e-9,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def verify_child_environment() -> None:
    failures = []
    if os.environ.get("LIA_1D_GPU_V7_STAGE_B_CHILD") != "1":
        failures.append("missing v7 child marker")
    if normalized(sys.executable) != normalized(SS4_PYTHON):
        failures.append(f"wrong Python executable: {sys.executable}")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        failures.append("KMP_DUPLICATE_LIB_OK is forbidden")
    actual_path = tuple(normalized(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item)
    expected_path = tuple(normalized(item) for item in EXPECTED_PATH)
    if actual_path != expected_path:
        failures.append(f"PATH differs from the frozen single-runtime PATH: {actual_path}")
    if normalized(os.environ.get("PYTHONPATH", "")) != normalized(TORCH_SITE):
        failures.append("PYTHONPATH is not exactly the project CUDA torch site-packages")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            failures.append(f"{name} must equal 1")
    if failures:
        raise RuntimeError("v7 child environment failed: " + "; ".join(failures))


def loaded_iomp_modules() -> list[dict]:
    """Return loaded Intel OpenMP module paths without importing psutil."""
    if os.name != "nt":
        return []
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    capacity = 2048
    modules = (wintypes.HMODULE * capacity)()
    enum_modules = psapi.EnumProcessModules
    enum_modules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    enum_modules.restype = wintypes.BOOL
    if not enum_modules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
        raise ctypes.WinError()
    get_module_name = psapi.GetModuleFileNameExW
    get_module_name.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    get_module_name.restype = wintypes.DWORD
    count = min(needed.value // ctypes.sizeof(wintypes.HMODULE), capacity)
    rows = []
    for index in range(count):
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_module_name(process, modules[index], buffer, len(buffer))
        if not length:
            continue
        path = Path(buffer.value)
        if "iomp" in path.name.lower():
            rows.append({
                "path": str(path),
                "name": path.name,
                "sha256": sha256(path) if path.is_file() else None,
            })
    return sorted(rows, key=lambda row: normalized(row["path"]))


def import_torch_with_stubs_filtered():
    """Import Torch while filtering only the frozen unneeded stubs DLL."""
    if sha256(TORCH_INIT) != TORCH_INIT_SHA256:
        raise RuntimeError("project torch/__init__.py hash mismatch")
    if sha256(IOMP_RUNTIME) != IOMP_RUNTIME_SHA256:
        raise RuntimeError("project libiomp5md.dll hash mismatch")
    if sha256(IOMP_STUBS) != IOMP_STUBS_SHA256 or IOMP_STUBS.stat().st_size != 43880:
        raise RuntimeError("project libiompstubs5md.dll identity mismatch")

    original_glob = python_glob.glob
    target_pattern = normalized(TORCH_LIB / "*.dll")
    filter_report = {"target_glob_calls": 0, "removed": []}

    def torch_import_glob(pathname, *args, **kwargs):
        matches = original_glob(pathname, *args, **kwargs)
        if normalized(pathname) != target_pattern:
            return matches
        filter_report["target_glob_calls"] += 1
        kept = []
        for match in matches:
            candidate = Path(match)
            if candidate.name.lower() == "libiompstubs5md.dll":
                if normalized(candidate) != normalized(IOMP_STUBS):
                    raise RuntimeError(f"unexpected OpenMP stubs path: {candidate}")
                if sha256(candidate) != IOMP_STUBS_SHA256 or candidate.stat().st_size != 43880:
                    raise RuntimeError("OpenMP stubs hash or size changed during torch import")
                filter_report["removed"].append({
                    "path": str(candidate),
                    "sha256": IOMP_STUBS_SHA256,
                    "size_bytes": candidate.stat().st_size,
                })
                continue
            kept.append(match)
        return kept

    python_glob.glob = torch_import_glob
    try:
        torch = importlib.import_module("torch")
    finally:
        python_glob.glob = original_glob
    if python_glob.glob is not original_glob:
        raise RuntimeError("glob.glob was not restored after torch import")
    if filter_report["target_glob_calls"] < 1 or len(filter_report["removed"]) != 1:
        raise RuntimeError(f"torch DLL glob stubs filter did not execute exactly once: {filter_report}")

    mapped_iomp = loaded_iomp_modules()
    if len(mapped_iomp) != 1:
        raise RuntimeError(f"expected exactly one mapped Intel OpenMP module, found {mapped_iomp}")
    if normalized(mapped_iomp[0]["path"]) != normalized(IOMP_RUNTIME):
        raise RuntimeError(f"mapped Intel OpenMP module is not the project runtime: {mapped_iomp}")
    if mapped_iomp[0]["sha256"] != IOMP_RUNTIME_SHA256:
        raise RuntimeError("mapped project Intel OpenMP runtime hash mismatch")
    return torch, filter_report, mapped_iomp


def write_json_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.replace(path)


def write_npz_new(path: Path, np, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.replace(path)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0)


def tensor_diagnostics_to_rows(torch, diagnostics: dict, lane_count: int) -> list[dict]:
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


def create_state(torch, BatchedFlowlineState, arrays: dict, indices, device):
    def tensor(name, dtype=torch.float64):
        return torch.as_tensor(arrays[name][indices], dtype=dtype, device=device)
    return BatchedFlowlineState(
        bed_h=tensor("bed_h"),
        widths_m=tensor("widths_m"),
        thick=tensor("thick"),
        valid_mask=tensor("valid_mask", torch.bool),
        nx=tensor("nx", torch.int64),
        dx_meter=tensor("dx_meter"),
        glen_a=tensor("glen_a"),
        melt_f=tensor("melt_f"),
        prcp_fac=tensor("prcp_fac"),
        temp_bias=tensor("temp_bias"),
        temp_melt=tensor("temp_melt"),
        climate_temp=tensor("climate_temp"),
        climate_prcp=tensor("climate_prcp"),
        climate_gradient=tensor("climate_gradient"),
        climate_ref_hgt=tensor("climate_ref_hgt"),
        climate_years=tensor("climate_years", torch.int64),
        y0=1836.0,
    )


def compare_stage(np, stage: str, gpu_rows: list[dict], gpu_thick, cpu_records: list[dict],
                  cpu_thick, valid_mask, initial_volume) -> dict:
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
            "climatic_mass_balance_relative": relative_error(
                gpu["cumulative_climatic_mass_balance_m3"],
                cpu_diag["cumulative_climatic_mass_balance_m3"],
            ),
            "mass_budget_residual_delta_m3": abs(
                gpu["cumulative_mass_budget_residual_m3"]
                - cpu_diag["cumulative_mass_budget_residual_m3"]
            ),
            "minimum_actual_timestep_seconds_absolute": abs(
                gpu["minimum_actual_timestep_seconds"]
                - cpu_diag["minimum_actual_timestep_seconds"]
            ),
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
            and errors["minimum_actual_timestep_seconds_absolute"]
            <= TOLERANCES["minimum_actual_timestep_seconds_absolute"]
            and int(gpu["accepted_steps"]) == int(cpu_diag["accepted_steps"])
            and int(gpu["rejected_steps"]) == int(cpu_diag["rejected_steps"]) == 0
            and finite and nonnegative and padding_zero
        )
        all_pass &= passed
        lanes.append({
            "lane": lane,
            "cell": cpu["cell"],
            "pass": passed,
            "errors": errors,
            "mass_budget_residual_limit_m3": residual_limit,
            "accepted_steps_exact": int(gpu["accepted_steps"]) == int(cpu_diag["accepted_steps"]),
            "rejected_steps_exact_zero": int(gpu["rejected_steps"]) == int(cpu_diag["rejected_steps"]) == 0,
            "finite": finite,
            "nonnegative": nonnegative,
            "padding_zero": padding_zero,
        })
    return {"stage": stage, "pass": all_pass, "lanes": lanes}


def main() -> int:
    verify_child_environment()
    authorization = validate_v7_launch(output_must_be_absent=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    events = OUTPUT_ROOT / "events"
    started = time.perf_counter()
    current_stage = "guard-passed"

    def event(number: int, stage: str, status: str, **extra) -> None:
        payload = {
            "schema": "lia-1d-gpu-v7-stage-event-v1",
            "transaction_id": "lia-1d-gpu-batch-preflight-v7-stage-b",
            "sequence": number,
            "stage": stage,
            "status": status,
            "timestamp_utc": utc_now(),
            "wall_seconds_since_start": time.perf_counter() - started,
            "process_id": os.getpid(),
            "python_executable": sys.executable,
            "loaded_iomp_modules": loaded_iomp_modules(),
            **extra,
        }
        write_json_new(events / f"{number:03d}-{stage}.json", payload)

    try:
        event(
            0,
            "guard-passed-before-numerical-imports",
            "pass",
            controller_state_sha256=authorization["state_sha256"],
            transaction_sha256=authorization["transaction_sha256"],
            cpu12_current_focus=authorization["state"]["current_focus"]["system_id"],
            cpu12_next_action=authorization["state"]["next_action"],
            cpu_thread_environment={name: os.environ[name] for name in THREAD_ENVIRONMENT},
            kmp_duplicate_lib_ok=False,
            torch_init_sha256=sha256(TORCH_INIT),
            iomp_runtime_sha256=sha256(IOMP_RUNTIME),
            iomp_stubs_sha256=sha256(IOMP_STUBS),
        )

        current_stage = "numerical-imports"
        torch, filter_report, mapped_iomp = import_torch_with_stubs_filtered()
        import numpy as np
        from torch_flowline_1d_batched_semiimplicit_v1 import BatchedFlowlineState
        event(
            10,
            "numerical-imports-with-stubs-filter-restored",
            "pass",
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            torch_dll_glob_filter=filter_report,
            glob_restored=True,
            unique_mapped_iomp=mapped_iomp,
            torch_runtime_files_modified=False,
        )

        current_stage = "runtime-configuration"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("PyTorch CPU thread limits are not both one")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda:0")
        probe = torch.ones(8, dtype=torch.float64, device=device).sum()
        torch.cuda.synchronize()
        if float(probe.cpu()) != 8.0:
            raise RuntimeError("CUDA float64 probe mismatch")
        event(
            20,
            "runtime-configuration",
            "pass",
            device=torch.cuda.get_device_name(0),
            cuda_runtime=torch.version.cuda,
            precision="float64",
            torch_cpu_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            cuda_float64_sum=float(probe.cpu()),
        )

        current_stage = "inputs-loaded"
        with np.load(authorization["fixture_path"]) as bundle:
            arrays = {name: bundle[name] for name in bundle.files}
        with authorization["cpu_summary_path"].open("r", encoding="utf-8") as handle:
            cpu_summary = json.load(handle)
        with np.load(authorization["cpu_states_path"]) as bundle:
            cpu_states = {name: bundle[name] for name in bundle.files}
        if cpu_summary.get("status") != "complete" or cpu_summary.get("cell_count") != 24:
            raise RuntimeError("CPU oracle is not the frozen complete 24-cell result")
        if len(cpu_summary.get("records", [])) != 24:
            raise RuntimeError("CPU oracle record count is not 24")
        event(
            30,
            "inputs-loaded",
            "pass",
            cell_count=24,
            fixture_sha256=sha256(authorization["fixture_path"]),
            cpu_summary_sha256=sha256(authorization["cpu_summary_path"]),
            cpu_states_sha256=sha256(authorization["cpu_states_path"]),
        )

        current_stage = "state-created"
        indices = np.arange(24, dtype=np.int64)
        state = create_state(torch, BatchedFlowlineState, arrays, indices, device)
        torch.cuda.synchronize()
        event(
            40,
            "state-created",
            "pass",
            state_shape=list(state.thick.shape),
            state_dtype=str(state.thick.dtype),
            state_device=str(state.thick.device),
        )

        current_stage = "annual-mass-balance"
        annual_mb = state.annual_mass_balance()
        torch.cuda.synchronize()
        event(
            50,
            "annual-mass-balance",
            "pass",
            finite=bool(torch.all(torch.isfinite(annual_mb)).item()),
            sum=float(annual_mb.sum().cpu()),
        )

        comparisons = []
        cpu_records = cpu_summary["records"]
        initial_volume = np.array([record["initial_volume_m3"] for record in cpu_records])
        schedule = (
            ("after_step", 108000.0, 100, "one-step"),
            ("after_month", 1836.0 + 1.0 / 12.0, 200, "one-month"),
            ("after_year", 1837.0, 300, "one-year"),
        )
        for stage, target, number, label in schedule:
            current_stage = f"{label}-dynamics"
            if stage == "after_step":
                actual_dt = state.step(target)
                advance = {"requested_dt_seconds": target, "actual_dt_seconds": actual_dt}
            else:
                state.run_until(target)
                advance = {"target_year": target}
            torch.cuda.synchronize()
            diagnostics = tensor_diagnostics_to_rows(torch, state.diagnostics(), 24)
            thickness = state.thick.detach().cpu().numpy().copy()
            thickness_path = OUTPUT_ROOT / "states" / f"{stage}.npz"
            write_npz_new(thickness_path, np, thick=thickness)
            event(
                number,
                f"{label}-gpu-complete-before-comparison",
                "pass",
                model_year=state.year,
                accepted_steps=state.accepted_steps,
                rejected_steps=state.rejected_steps,
                minimum_actual_timestep_seconds=state.minimum_actual_timestep_seconds,
                thickness_path=str(thickness_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                thickness_sha256=sha256(thickness_path),
                **advance,
            )

            current_stage = f"{label}-comparison"
            comparison = compare_stage(
                np,
                stage,
                diagnostics,
                thickness,
                cpu_records,
                cpu_states[f"{stage}_thick"],
                arrays["valid_mask"],
                initial_volume,
            )
            comparisons.append(comparison)
            event(number + 10, f"{label}-cpu-gpu-comparison", "pass" if comparison["pass"] else "fail", comparison=comparison)
            if not comparison["pass"]:
                summary = {
                    "schema": "lia-1d-gpu-batch-preflight-v7-stage-b-result-v1",
                    "status": f"{label}-equivalence-fail",
                    "transaction_id": "lia-1d-gpu-batch-preflight-v7-stage-b",
                    "cell_count": 24,
                    "comparisons": comparisons,
                    "checkpoint_run": False,
                    "throughput_run": False,
                    "full_191_gpu_matrix_run": False,
                    "planform_2d_or_igm_run": False,
                }
                write_json_new(OUTPUT_ROOT / "summary.json", summary)
                return 2

        current_stage = "stage-b-summary"
        summary = {
            "schema": "lia-1d-gpu-batch-preflight-v7-stage-b-result-v1",
            "status": "stage-b-equivalence-pass",
            "transaction_id": "lia-1d-gpu-batch-preflight-v7-stage-b",
            "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
            "cell_count": 24,
            "precision": "float64",
            "fixed_dt_seconds": 108000.0,
            "comparisons": comparisons,
            "checkpoint_run": False,
            "throughput_run": False,
            "full_191_gpu_matrix_run": False,
            "planform_2d_or_igm_run": False,
            "cpu12_modified_or_controlled": False,
            "wall_seconds": time.perf_counter() - started,
        }
        write_json_new(OUTPUT_ROOT / "summary.json", summary)
        event(400, "stage-b-terminal", "pass", summary_sha256=sha256(OUTPUT_ROOT / "summary.json"))
        print(json.dumps({"status": summary["status"], "output": str(OUTPUT_ROOT)}, indent=2))
        return 0
    except BaseException as exc:
        failure = {
            "schema": "lia-1d-gpu-batch-preflight-v7-stage-b-technical-failure-v1",
            "status": "technical-failure",
            "failed_stage": current_stage,
            "timestamp_utc": utc_now(),
            "process_id": os.getpid(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "loaded_iomp_modules": loaded_iomp_modules(),
            "checkpoint_run": False,
            "throughput_run": False,
            "full_191_gpu_matrix_run": False,
            "planform_2d_or_igm_run": False,
            "cpu12_modified_or_controlled": False,
        }
        failure_path = OUTPUT_ROOT / "technical_failure.json"
        if not failure_path.exists():
            write_json_new(failure_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
