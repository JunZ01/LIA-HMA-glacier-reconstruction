"""Run the independent 191-system, 1337-cell GPU spin-up production batch."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import glob as python_glob
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from lia_191_gpu_spinup_v10_239_guard import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    THREAD_ENVIRONMENT,
    sha256,
    validate_launch,
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

BLOCK_YEARS = 300
MAX_YEARS = 3000
CYCLE_YEARS = 30
VOLUME_CYCLE_TOL = 0.003
VOLUME_90_TOL = 0.009
LENGTH_GRID_TOL = 1.0
BOUNDARY_MARGIN_M = 5000.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def verify_child_environment() -> None:
    failures = []
    if os.environ.get("LIA_191_GPU_SPINUP_V1_CHILD") != "1":
        failures.append("missing GPU spin-up child marker")
    if normalized(sys.executable) != normalized(SS4_PYTHON):
        failures.append(f"wrong Python executable: {sys.executable}")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        failures.append("KMP_DUPLICATE_LIB_OK is forbidden")
    actual_path = tuple(normalized(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item)
    expected_path = tuple(normalized(item) for item in EXPECTED_PATH)
    if actual_path != expected_path:
        failures.append("PATH differs from the isolated CUDA runtime")
    if normalized(os.environ.get("PYTHONPATH", "")) != normalized(TORCH_SITE):
        failures.append("PYTHONPATH is not the project CUDA Torch site-packages")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            failures.append(f"{name} must equal 1")
    if failures:
        raise RuntimeError("GPU child environment failed: " + "; ".join(failures))


def loaded_iomp_modules() -> list[dict]:
    if os.name != "nt":
        return []
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    modules = (wintypes.HMODULE * 2048)()
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
        raise ctypes.WinError()
    rows = []
    count = min(needed.value // ctypes.sizeof(wintypes.HMODULE), 2048)
    for index in range(count):
        buffer = ctypes.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(process, modules[index], buffer, len(buffer)):
            continue
        path = Path(buffer.value)
        if "iomp" in path.name.lower():
            rows.append({"path": str(path), "name": path.name, "sha256": sha256(path)})
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
    report = {"target_glob_calls": 0, "removed": []}

    def filtered(pathname, *args, **kwargs):
        matches = original_glob(pathname, *args, **kwargs)
        if normalized(pathname) != target_pattern:
            return matches
        report["target_glob_calls"] += 1
        kept = []
        for match in matches:
            candidate = Path(match)
            if candidate.name.lower() == "libiompstubs5md.dll":
                report["removed"].append(str(candidate))
            else:
                kept.append(match)
        return kept

    python_glob.glob = filtered
    try:
        torch = importlib.import_module("torch")
    finally:
        python_glob.glob = original_glob
    if report["target_glob_calls"] < 1 or len(report["removed"]) != 1:
        raise RuntimeError(f"Torch OpenMP stubs filter mismatch: {report}")
    mapped = loaded_iomp_modules()
    if len(mapped) != 1 or normalized(mapped[0]["path"]) != normalized(IOMP_RUNTIME):
        raise RuntimeError(f"unexpected mapped OpenMP runtimes: {mapped}")
    return torch, report, mapped


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    atomic_json(path, payload)


def write_npz_new(path: Path, np, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def create_state(torch, State, arrays, cell_indices, device, checkpoint=None):
    cells = arrays["cell_system_index"][cell_indices]

    def tensor(value, dtype=torch.float64):
        return torch.as_tensor(value, dtype=dtype, device=device)

    count = len(cell_indices)
    state = State(
        bed_h=tensor(arrays["system_bed_h"][cells]),
        widths_m=tensor(arrays["system_widths_m"][cells]),
        thick=tensor(arrays["system_thick"][cells] if checkpoint is None else checkpoint["thick"]),
        valid_mask=tensor(arrays["system_valid_mask"][cells], torch.bool),
        nx=tensor(arrays["system_nx"][cells], torch.int64),
        dx_meter=tensor(arrays["system_dx_meter"][cells]),
        glen_a=tensor(arrays["cell_glen_a"][cell_indices]),
        melt_f=tensor(arrays["cell_melt_f"][cell_indices]),
        prcp_fac=tensor(arrays["cell_prcp_fac"][cell_indices]),
        temp_bias=tensor(arrays["cell_temp_bias"][cell_indices]),
        temp_melt=tensor(arrays["cell_temp_melt"][cell_indices]),
        climate_temp=tensor(arrays["system_climate_temp"][cells]),
        climate_prcp=tensor(arrays["system_climate_prcp"][cells]),
        climate_gradient=tensor(arrays["system_climate_gradient"][cells]),
        climate_ref_hgt=tensor(arrays["system_climate_ref_hgt"][cells]),
        climate_years=tensor(np_broadcast(arrays["climate_years"], count), torch.int64),
        y0=0.0,
        time_seconds=0.0 if checkpoint is None else float(checkpoint["current_year"]) * 31_536_000.0,
        accepted_steps=0 if checkpoint is None else int(checkpoint["accepted_steps"]),
        rejected_steps=0 if checkpoint is None else int(checkpoint["rejected_steps"]),
        minimum_actual_timestep_seconds=float("inf") if checkpoint is None else float(checkpoint["minimum_actual_timestep_seconds"]),
        cumulative_climatic_mass_balance_m3=(
            None if checkpoint is None else tensor(checkpoint["cumulative_climatic_mass_balance_m3"])
        ),
        cumulative_mass_budget_residual_m3=(
            None if checkpoint is None else tensor(checkpoint["cumulative_mass_budget_residual_m3"])
        ),
    )
    return state


def np_broadcast(years, count):
    import numpy as np
    return np.broadcast_to(years, (count, len(years))).copy()


def snapshot(state, torch):
    diagnostics = state.diagnostics()
    return {
        "volume_m3": diagnostics["volume_m3"].detach().cpu().numpy(),
        "area_m2": diagnostics["area_m2"].detach().cpu().numpy(),
        "length_m": diagnostics["length_m"].detach().cpu().numpy(),
        "accepted_steps": int(diagnostics["accepted_steps"]),
        "rejected_steps": int(diagnostics["rejected_steps"]),
        "minimum_actual_timestep_seconds": float(diagnostics["minimum_actual_timestep_seconds"]),
    }


def convergence(history, year, positions, dx, np):
    years = [year - 90, year - 60, year - 30, year]
    volume = [history[item]["volume_m3"][positions] for item in years]
    length = [history[item]["length_m"][positions] for item in years]
    tiny = np.finfo(np.float64).tiny

    def relative(left, right):
        return np.abs(left - right) / np.maximum(np.maximum(np.abs(left), np.abs(right)), tiny)

    volume_cycles = np.stack([relative(volume[i], volume[i + 1]) for i in range(3)], axis=1)
    length_cycles = np.stack([np.abs(length[i + 1] - length[i]) / dx for i in range(3)], axis=1)
    volume_90 = relative(volume[0], volume[-1])
    length_90 = np.abs(length[-1] - length[0]) / dx
    passed = (
        np.all(volume_cycles <= VOLUME_CYCLE_TOL, axis=1)
        & (volume_90 <= VOLUME_90_TOL)
        & np.all(length_cycles <= LENGTH_GRID_TOL, axis=1)
        & (length_90 <= LENGTH_GRID_TOL)
    )
    return {
        "years": years,
        "volume_relative_cycle_differences": volume_cycles,
        "volume_relative_90y_drift": volume_90,
        "length_grid_cycle_differences": length_cycles,
        "length_grid_90y_difference": length_90,
        "passed": passed,
    }


def lane_metrics(metrics, index):
    return {
        "years": metrics["years"],
        "volume_relative_cycle_differences": metrics["volume_relative_cycle_differences"][index].tolist(),
        "volume_relative_90y_drift": float(metrics["volume_relative_90y_drift"][index]),
        "length_grid_cycle_differences": metrics["length_grid_cycle_differences"][index].tolist(),
        "length_grid_90y_difference": float(metrics["length_grid_90y_difference"][index]),
        "passed": bool(metrics["passed"][index]),
    }


def subset_metrics(metrics, indices):
    return {
        "years": metrics["years"],
        "volume_relative_cycle_differences": metrics["volume_relative_cycle_differences"][indices],
        "volume_relative_90y_drift": metrics["volume_relative_90y_drift"][indices],
        "length_grid_cycle_differences": metrics["length_grid_cycle_differences"][indices],
        "length_grid_90y_difference": metrics["length_grid_90y_difference"][indices],
        "passed": metrics["passed"][indices],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    verify_child_environment()
    authorization = validate_launch(output_must_be_absent=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stage = "guard-passed"
    try:
        torch, filter_report, mapped_iomp = import_torch_with_stubs_filtered()
        import numpy as np
        from torch_flowline_1d_batched_spinup_repeat_v1 import (
            BatchedFlowlineStateSpinupRepeat,
            select_lanes,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda:0")
        torch.ones(8, dtype=torch.float64, device=device).sum().cpu()
        with np.load(authorization["fixture_path"]) as bundle:
            arrays = {name: bundle[name] for name in bundle.files}
        cells = authorization["fixture"]["cells"]
        total = len(cells)
        if total != 1337 or args.batch_size <= 0:
            raise RuntimeError("invalid fixture count or batch size")
        write_json_new(OUTPUT_ROOT / "run_manifest.json", {
            "schema": "lia-191-gpu-spinup-run-manifest-v1",
            "status": "running",
            "started_at_utc": utc_now(),
            "process_id": os.getpid(),
            "transaction_sha256": authorization["transaction_sha256"],
            "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
            "fixture_sha256": sha256(authorization["fixture_path"]),
            "system_count": 191,
            "cell_count": total,
            "batch_size": args.batch_size,
            "batch_count": math.ceil(total / args.batch_size),
            "precision": "float64",
            "device": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "torch_version": torch.__version__,
            "torch_import_filter": filter_report,
            "mapped_iomp": mapped_iomp,
            "scientific_contract": {
                "climate": "ensemble_rec_ref-1441-1470-repeat",
                "block_years": BLOCK_YEARS,
                "maximum_years": MAX_YEARS,
                "cycle_years": CYCLE_YEARS,
                "volume_cycle_tolerance": VOLUME_CYCLE_TOL,
                "volume_90y_tolerance": VOLUME_90_TOL,
                "length_grid_tolerance": LENGTH_GRID_TOL,
                "boundary_margin_m": BOUNDARY_MARGIN_M,
            },
            "cpu_mainline_modified_or_controlled": False,
            "cpu_comparison_run": False,
        })

        completed = 0
        converged = 0
        nonconverged = 0
        scientific_domain_failures = 0

        def update_status(**extra):
            atomic_json(OUTPUT_ROOT / "status.json", {
                "schema": "lia-191-gpu-spinup-status-v1",
                "updated_at_utc": utc_now(),
                "status": "running",
                "process_id": os.getpid(),
                "completed": completed,
                "converged": converged,
                "max_duration_nonconverged": nonconverged,
                "scientific_domain_failures": scientific_domain_failures,
                "pending": total - completed,
                "technical_failures": 0,
                "wall_seconds": time.perf_counter() - started,
                **extra,
            })

        def save_terminal(
            state,
            local_index,
            cell_index,
            status,
            year,
            boundary,
            confirmation,
            diag,
            thick_cpu,
        ):
            nonlocal completed, converged, nonconverged, scientific_domain_failures
            cell = cells[int(cell_index)]
            member = OUTPUT_ROOT / "m" / cell["member_key"]
            member.mkdir(parents=True, exist_ok=False)
            n = int(state.nx[local_index].item())
            thick = thick_cpu[local_index, :n].copy()
            state_path = member / "state.npz"
            write_npz_new(
                state_path,
                np,
                thick=thick,
                current_year=np.asarray([year], dtype=np.int64),
                cumulative_climatic_mass_balance_m3=np.asarray([
                    float(state.cumulative_climatic_mass_balance_m3[local_index].item())
                ]),
                cumulative_mass_budget_residual_m3=np.asarray([
                    float(state.cumulative_mass_budget_residual_m3[local_index].item())
                ]),
            )
            result = {
                "schema": "lia-191-gpu-spinup-member-result-v1",
                "terminal": True,
                "status": status,
                "technical_failure": False,
                "scientific_nonconvergence": status == "max-duration-nonconverged",
                "cell": cell,
                "current_year": int(year),
                "diagnostics": {
                    "volume_m3": float(diag["volume_m3"][local_index]),
                    "area_m2": float(diag["area_m2"][local_index]),
                    "length_m": float(diag["length_m"][local_index]),
                    "accepted_steps": diag["accepted_steps"],
                    "rejected_steps": diag["rejected_steps"],
                    "minimum_actual_timestep_seconds": diag["minimum_actual_timestep_seconds"],
                    "minimum_thickness_m": float(np.min(thick)),
                    "maximum_thickness_m": float(np.max(thick)),
                },
                "boundary_convergence": boundary,
                "rolling_30y_confirmation": confirmation,
                "state_path": str(state_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "state_sha256": sha256(state_path),
                "cpu_comparison_run": False,
                "cpu_mainline_modified_or_controlled": False,
            }
            write_json_new(member / "result.json", result)
            completed += 1
            if status == "converged":
                converged += 1
            elif status == "max-duration-nonconverged":
                nonconverged += 1
            else:
                scientific_domain_failures += 1

        batch_count = math.ceil(total / args.batch_size)
        for batch_number, batch_start in enumerate(range(0, total, args.batch_size)):
            batch_indices = np.arange(batch_start, min(batch_start + args.batch_size, total), dtype=np.int64)
            batch_size = len(batch_indices)
            active_cells = batch_indices.copy()
            active_positions = np.arange(batch_size, dtype=np.int64)
            state = create_state(torch, BatchedFlowlineStateSpinupRepeat, arrays, active_cells, device)
            history = {}

            def record(year):
                diag = snapshot(state, torch)
                full = {
                    "volume_m3": np.full(batch_size, np.nan, dtype=np.float64),
                    "area_m2": np.full(batch_size, np.nan, dtype=np.float64),
                    "length_m": np.full(batch_size, np.nan, dtype=np.float64),
                }
                for name in full:
                    full[name][active_positions] = diag[name]
                history[int(year)] = full

            record(0)
            current_year = 0
            update_status(
                batch_number=batch_number,
                batch_count=batch_count,
                batch_cells=batch_size,
                batch_active=len(active_cells),
                model_year=current_year,
                latest_checkpoint=None,
            )
            while len(active_cells) and current_year < MAX_YEARS:
                boundary_year = min(((current_year // BLOCK_YEARS) + 1) * BLOCK_YEARS, MAX_YEARS)
                next_cycle = ((current_year // CYCLE_YEARS) + 1) * CYCLE_YEARS
                for year in range(next_cycle, boundary_year + 1, CYCLE_YEARS):
                    stage = f"batch-{batch_number}-year-{year}"
                    state.run_until(float(year))
                    torch.cuda.synchronize()
                    record(year)
                current_year = boundary_year
                dx = arrays["system_dx_meter"][arrays["cell_system_index"][active_cells]]
                metrics = convergence(history, current_year, active_positions, dx, np)
                thick_cpu = state.thick.detach().cpu().numpy()
                valid_cpu = state.valid_mask.detach().cpu().numpy()
                diag = snapshot(state, torch)
                domain_failure = np.zeros(len(active_cells), dtype=np.bool_)
                boundary_margin = np.zeros(len(active_cells), dtype=np.float64)
                for local in range(len(active_cells)):
                    valid_thick = thick_cpu[local, valid_cpu[local]]
                    positive = np.flatnonzero(valid_thick > 0.0)
                    last = int(positive[-1]) if len(positive) else -1
                    free_nodes = len(valid_thick) if last < 0 else len(valid_thick) - 1 - last
                    boundary_margin[local] = free_nodes * dx[local]
                    finite = np.isfinite(valid_thick).all()
                    positive_state = (
                        diag["volume_m3"][local] > 0.0
                        and diag["area_m2"][local] > 0.0
                        and diag["length_m"][local] > 0.0
                        and np.min(valid_thick) >= 0.0
                    )
                    domain_failure[local] = (
                        not finite or not positive_state or boundary_margin[local] < BOUNDARY_MARGIN_M
                    )

                for local in np.flatnonzero(domain_failure):
                    save_terminal(
                        state,
                        int(local),
                        int(active_cells[local]),
                        "scientific-domain-failure",
                        current_year,
                        lane_metrics(metrics, int(local)),
                        None,
                        diag,
                        thick_cpu,
                    )
                domain_survivors = np.flatnonzero(~domain_failure)
                if not len(domain_survivors):
                    active_cells = active_cells[domain_survivors]
                    active_positions = active_positions[domain_survivors]
                    update_status(
                        batch_number=batch_number,
                        batch_count=batch_count,
                        batch_cells=batch_size,
                        batch_active=0,
                        model_year=current_year,
                        latest_checkpoint=None,
                        accepted_steps=diag["accepted_steps"],
                        rejected_steps=diag["rejected_steps"],
                        minimum_actual_timestep_seconds=diag["minimum_actual_timestep_seconds"],
                    )
                    break
                if len(domain_survivors) != len(active_cells):
                    survivor_t = torch.as_tensor(
                        domain_survivors, dtype=torch.int64, device=device
                    )
                    state = select_lanes(state, survivor_t)
                    active_cells = active_cells[domain_survivors]
                    active_positions = active_positions[domain_survivors]
                    dx = dx[domain_survivors]
                    metrics = subset_metrics(metrics, domain_survivors)

                terminal_mask = np.zeros(len(active_cells), dtype=np.bool_)
                terminal_status = np.full(len(active_cells), "", dtype=object)
                confirmation_metrics = None
                confirmation_pass = np.zeros(len(active_cells), dtype=np.bool_)
                if current_year == MAX_YEARS:
                    terminal_mask[:] = True
                    terminal_status[:] = "max-duration-nonconverged"
                else:
                    state.run_until(float(current_year + CYCLE_YEARS))
                    torch.cuda.synchronize()
                    record(current_year + CYCLE_YEARS)
                    confirmation_metrics = convergence(
                        history, current_year + CYCLE_YEARS, active_positions, dx, np
                    )
                    confirmation_pass = metrics["passed"] & confirmation_metrics["passed"]
                    terminal_mask[confirmation_pass] = True
                    terminal_status[confirmation_pass] = "converged"
                    current_year += CYCLE_YEARS

                terminal_diag = snapshot(state, torch)
                terminal_thick = state.thick.detach().cpu().numpy()
                for local in np.flatnonzero(terminal_mask):
                    save_terminal(
                        state,
                        int(local),
                        int(active_cells[local]),
                        str(terminal_status[local]),
                        current_year,
                        lane_metrics(metrics, int(local)),
                        None if confirmation_metrics is None else lane_metrics(confirmation_metrics, int(local)),
                        terminal_diag,
                        terminal_thick,
                    )

                keep = np.flatnonzero(~terminal_mask)
                if len(keep):
                    keep_t = torch.as_tensor(keep, dtype=torch.int64, device=device)
                    state = select_lanes(state, keep_t)
                    active_cells = active_cells[keep]
                    active_positions = active_positions[keep]
                    checkpoint = OUTPUT_ROOT / "b" / f"b{batch_number:03d}" / "checkpoints" / f"year-{current_year:04d}.npz"
                    write_npz_new(
                        checkpoint,
                        np,
                        active_cells=active_cells,
                        active_positions=active_positions,
                        thick=state.thick.detach().cpu().numpy(),
                        current_year=np.asarray([current_year], dtype=np.int64),
                        accepted_steps=np.asarray([state.accepted_steps], dtype=np.int64),
                        rejected_steps=np.asarray([state.rejected_steps], dtype=np.int64),
                        minimum_actual_timestep_seconds=np.asarray([state.minimum_actual_timestep_seconds]),
                        cumulative_climatic_mass_balance_m3=state.cumulative_climatic_mass_balance_m3.detach().cpu().numpy(),
                        cumulative_mass_budget_residual_m3=state.cumulative_mass_budget_residual_m3.detach().cpu().numpy(),
                    )
                    latest_checkpoint = str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/")
                else:
                    active_cells = active_cells[keep]
                    active_positions = active_positions[keep]
                    latest_checkpoint = None
                update_status(
                    batch_number=batch_number,
                    batch_count=batch_count,
                    batch_cells=batch_size,
                    batch_active=len(active_cells),
                    model_year=current_year,
                    latest_checkpoint=latest_checkpoint,
                    accepted_steps=state.accepted_steps if len(active_cells) else diag["accepted_steps"],
                    rejected_steps=state.rejected_steps if len(active_cells) else diag["rejected_steps"],
                    minimum_actual_timestep_seconds=(
                        state.minimum_actual_timestep_seconds if len(active_cells)
                        else diag["minimum_actual_timestep_seconds"]
                    ),
                )
            write_json_new(OUTPUT_ROOT / "b" / f"b{batch_number:03d}" / "summary.json", {
                "schema": "lia-191-gpu-spinup-batch-summary-v1",
                "status": "complete",
                "batch_number": batch_number,
                "cell_start": int(batch_start),
                "cell_count": batch_size,
                "completed_total": completed,
                "timestamp_utc": utc_now(),
            })
            del state
            torch.cuda.empty_cache()

        summary = {
            "schema": "lia-191-gpu-spinup-summary-v1",
            "status": "completed",
            "completed": completed,
            "converged": converged,
            "max_duration_nonconverged": nonconverged,
            "scientific_domain_failures": scientific_domain_failures,
            "technical_failures": 0,
            "system_count": 191,
            "cell_count": total,
            "wall_seconds": time.perf_counter() - started,
            "finished_at_utc": utc_now(),
            "cpu_comparison_run": False,
            "cpu_mainline_modified_or_controlled": False,
        }
        write_json_new(OUTPUT_ROOT / "summary.json", summary)
        atomic_json(OUTPUT_ROOT / "status.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except BaseException as exc:
        failure = {
            "schema": "lia-191-gpu-spinup-technical-failure-v1",
            "status": "technical-failure",
            "failed_stage": stage,
            "timestamp_utc": utc_now(),
            "process_id": os.getpid(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "loaded_iomp_modules": loaded_iomp_modules(),
            "cpu_comparison_run": False,
            "cpu_mainline_modified_or_controlled": False,
        }
        atomic_json(OUTPUT_ROOT / "technical_failure.json", failure)
        atomic_json(OUTPUT_ROOT / "status.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
