"""Fixed-24 GPU-only state generation at one step, one month, and one year."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import glob as python_glob
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from lia_1d_gpu_only_v8_guard import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    THREAD_ENVIRONMENT,
    sha256,
    validate_v8_launch,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def verify_child_environment() -> None:
    failures = []
    if os.environ.get("LIA_1D_GPU_ONLY_V8_CHILD") != "1":
        failures.append("missing v8 child marker")
    if normalized(sys.executable) != normalized(SS4_PYTHON):
        failures.append(f"wrong Python executable: {sys.executable}")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        failures.append("KMP_DUPLICATE_LIB_OK is forbidden")
    actual_path = tuple(normalized(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item)
    expected_path = tuple(normalized(item) for item in EXPECTED_PATH)
    if actual_path != expected_path:
        failures.append(f"PATH differs from frozen single-runtime PATH: {actual_path}")
    if normalized(os.environ.get("PYTHONPATH", "")) != normalized(TORCH_SITE):
        failures.append("PYTHONPATH is not exactly the project CUDA Torch site-packages")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            failures.append(f"{name} must equal 1")
    if failures:
        raise RuntimeError("v8 child environment failed: " + "; ".join(failures))


def loaded_iomp_modules() -> list[dict]:
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
                    raise RuntimeError("OpenMP stubs hash or size changed during Torch import")
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
        raise RuntimeError("glob.glob was not restored after Torch import")
    if filter_report["target_glob_calls"] < 1 or len(filter_report["removed"]) != 1:
        raise RuntimeError(f"Torch DLL stubs filter did not execute exactly once: {filter_report}")

    mapped_iomp = loaded_iomp_modules()
    if len(mapped_iomp) != 1:
        raise RuntimeError(f"expected exactly one mapped Intel OpenMP module, found {mapped_iomp}")
    if normalized(mapped_iomp[0]["path"]) != normalized(IOMP_RUNTIME):
        raise RuntimeError(f"mapped Intel OpenMP module is not project runtime: {mapped_iomp}")
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


def create_state(torch, BatchedFlowlineState, arrays: dict, device):
    indices = slice(0, 24)

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


def tensor_diagnostics_to_rows(torch, diagnostics: dict, lane_count: int) -> list[dict]:
    rows = [dict() for _ in range(lane_count)]
    for key, value in diagnostics.items():
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu().tolist()
            for lane in range(lane_count):
                rows[lane][key] = float(data[lane])
        else:
            scalar = int(value) if isinstance(value, int) else float(value)
            for lane in range(lane_count):
                rows[lane][key] = scalar
    return rows


def main() -> int:
    verify_child_environment()
    authorization = validate_v8_launch(output_must_be_absent=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    events = OUTPUT_ROOT / "events"
    started = time.perf_counter()
    current_stage = "guard-passed"

    def event(number: int, stage: str, status: str, **extra) -> None:
        payload = {
            "schema": "lia-1d-gpu-only-v8-stage-event-v1",
            "transaction_id": "lia-1d-gpu-only-v8-fixed24-stage-states",
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
            cpu_state_preserved=True,
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
            numpy_linalg_imported_or_called=False,
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
        event(
            30,
            "fixed-inputs-loaded",
            "pass",
            cell_count=24,
            fixture_sha256=sha256(authorization["fixture_path"]),
            cpu_oracle_loaded=False,
            cpu_comparison_enabled=False,
            numpy_linalg_imported_or_called=False,
        )

        current_stage = "state-created"
        state = create_state(torch, BatchedFlowlineState, arrays, device)
        torch.cuda.synchronize()
        event(
            40,
            "gpu-state-created-from-original-fixed-initial-state",
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
            "gpu-annual-mass-balance",
            "pass",
            finite=bool(torch.all(torch.isfinite(annual_mb)).item()),
            sum=float(annual_mb.sum().cpu()),
        )

        stages = []
        schedule = (
            ("after_step", 108000.0, 100, "one-step"),
            ("after_month", 1836.0 + 1.0 / 12.0, 200, "one-month"),
            ("after_year", 1837.0, 300, "one-year"),
        )
        for stage, target, number, label in schedule:
            current_stage = f"{label}-gpu-dynamics"
            if stage == "after_step":
                actual_dt = state.step(target)
                advance = {"requested_dt_seconds": target, "actual_dt_seconds": actual_dt}
            else:
                state.run_until(target)
                advance = {"target_year": target}
            torch.cuda.synchronize()

            diagnostics = state.diagnostics()
            finite = bool(torch.all(torch.isfinite(state.thick[state.valid_mask])).item())
            nonnegative = bool(torch.all(state.thick[state.valid_mask] >= 0.0).item())
            padding_zero = bool(torch.all(state.thick[~state.valid_mask] == 0.0).item())
            if not finite or not nonnegative or not padding_zero:
                raise RuntimeError(
                    f"{label} GPU state validity failed: "
                    f"finite={finite}, nonnegative={nonnegative}, padding_zero={padding_zero}"
                )
            diagnostic_rows = tensor_diagnostics_to_rows(torch, diagnostics, 24)
            thickness = state.thick.detach().cpu().numpy().copy()
            state_path = OUTPUT_ROOT / "states" / f"{stage}.npz"
            diagnostics_path = OUTPUT_ROOT / "diagnostics" / f"{stage}.json"
            write_npz_new(state_path, np, thick=thickness)
            write_json_new(
                diagnostics_path,
                {
                    "schema": "lia-1d-gpu-only-v8-stage-diagnostics-v1",
                    "stage": stage,
                    "cell_count": 24,
                    "model_year": state.year,
                    "accepted_steps": state.accepted_steps,
                    "rejected_steps": state.rejected_steps,
                    "minimum_actual_timestep_seconds": state.minimum_actual_timestep_seconds,
                    "finite": finite,
                    "nonnegative": nonnegative,
                    "padding_zero": padding_zero,
                    "lanes": diagnostic_rows,
                    "cpu_oracle_loaded": False,
                    "cpu_comparison_run": False,
                    "numpy_linalg_imported_or_called": False,
                },
            )
            stage_record = {
                "stage": stage,
                "model_year": state.year,
                "accepted_steps": state.accepted_steps,
                "rejected_steps": state.rejected_steps,
                "minimum_actual_timestep_seconds": state.minimum_actual_timestep_seconds,
                "state_path": str(state_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "state_sha256": sha256(state_path),
                "diagnostics_path": str(diagnostics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "diagnostics_sha256": sha256(diagnostics_path),
                "finite": finite,
                "nonnegative": nonnegative,
                "padding_zero": padding_zero,
                **advance,
            }
            stages.append(stage_record)
            event(number, f"{label}-gpu-state-and-diagnostics-saved", "pass", **stage_record)

        current_stage = "gpu-only-summary"
        summary = {
            "schema": "lia-1d-gpu-only-v8-fixed24-result-v1",
            "status": "gpu-only-stage-states-complete",
            "transaction_id": "lia-1d-gpu-only-v8-fixed24-stage-states",
            "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
            "cell_count": 24,
            "precision": "float64",
            "fixed_dt_seconds": 108000.0,
            "continuous_trajectory_from_original_fixed_initial_state": True,
            "stages": stages,
            "cpu_oracle_loaded": False,
            "cpu_comparison_run": False,
            "numpy_linalg_imported_or_called": False,
            "equivalence_claimed": False,
            "checkpoint_run": False,
            "throughput_run": False,
            "full_191_gpu_matrix_run": False,
            "planform_2d_or_igm_run": False,
            "cpu12_or_cpu8_modified_or_controlled": False,
            "wall_seconds": time.perf_counter() - started,
        }
        write_json_new(OUTPUT_ROOT / "summary.json", summary)
        event(400, "gpu-only-terminal", "pass", summary_sha256=sha256(OUTPUT_ROOT / "summary.json"))
        print(json.dumps({"status": summary["status"], "output": str(OUTPUT_ROOT)}, indent=2))
        return 0
    except BaseException as exc:
        failure = {
            "schema": "lia-1d-gpu-only-v8-technical-failure-v1",
            "status": "technical-failure",
            "failed_stage": current_stage,
            "timestamp_utc": utc_now(),
            "process_id": os.getpid(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "loaded_iomp_modules": loaded_iomp_modules(),
            "cpu_oracle_loaded": False,
            "cpu_comparison_run": False,
            "numpy_linalg_imported_or_called": False,
            "checkpoint_run": False,
            "throughput_run": False,
            "full_191_gpu_matrix_run": False,
            "planform_2d_or_igm_run": False,
            "cpu12_or_cpu8_modified_or_controlled": False,
        }
        failure_path = OUTPUT_ROOT / "technical_failure.json"
        if not failure_path.exists():
            write_json_new(failure_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
