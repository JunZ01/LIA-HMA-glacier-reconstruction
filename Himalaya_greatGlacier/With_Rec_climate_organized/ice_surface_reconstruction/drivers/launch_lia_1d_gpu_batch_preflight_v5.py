"""Pre-Python-process launcher for the v5 one-dimensional GPU preflight.

For the GPU phase the first process sets the project CUDA torch DLL and Python
package directories in a child environment, then spawns a fresh SS4 Python
process.  The child binds the frozen numerical drivers to v5.  The unsafe
KMP_DUPLICATE_LIB_OK workaround is prohibited.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


DRIVERS = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(DRIVERS))
import lia_1d_gpu_preflight_guard_v2 as frozen_guard


TRANSACTION_PATH = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_transaction_v5/transaction.json"
FIXTURE_MANIFEST_PATH = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1/manifest.json"
OUTPUT_ROOT = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_v5"
CPU_DRIVER = DRIVERS / "run_lia_1d_gpu_preflight_cpu_oracle_v2.py"
GPU_DRIVER = DRIVERS / "run_lia_1d_gpu_batch_preflight_v2.py"
BACKEND = DRIVERS / "torch_flowline_1d_batched_semiimplicit_v1.py"
CUDA_TORCH_SITE = PROJECT_ROOT / ".runtime/torch-cu128-v1/site-packages"
CUDA_TORCH_LIB = CUDA_TORCH_SITE / "torch/lib"
CHILD_MARKER = "LIA_1D_GPU_V5_CHILD"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepend_unique(value: str, current: str) -> str:
    wanted = os.path.normcase(os.path.normpath(value))
    parts = [part for part in current.split(os.pathsep) if part]
    parts = [part for part in parts if os.path.normcase(os.path.normpath(part)) != wanted]
    return os.pathsep.join([value, *parts])


def child_environment() -> dict[str, str]:
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        raise RuntimeError("v5 forbids KMP_DUPLICATE_LIB_OK")
    if not CUDA_TORCH_LIB.is_dir() or not CUDA_TORCH_SITE.is_dir():
        raise RuntimeError("missing project-bound CUDA PyTorch runtime")
    env = os.environ.copy()
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env["PATH"] = prepend_unique(str(CUDA_TORCH_LIB), env.get("PATH", ""))
    env["PYTHONPATH"] = prepend_unique(str(CUDA_TORCH_SITE), env.get("PYTHONPATH", ""))
    env[CHILD_MARKER] = "1"
    return env


def verify_child_environment() -> None:
    if os.environ.get(CHILD_MARKER) != "1":
        raise RuntimeError("GPU numerical work must run in the v5 child process")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        raise RuntimeError("v5 forbids KMP_DUPLICATE_LIB_OK")
    path_first = os.environ.get("PATH", "").split(os.pathsep)[0]
    python_first = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    if os.path.normcase(os.path.normpath(path_first)) != os.path.normcase(os.path.normpath(str(CUDA_TORCH_LIB))):
        raise RuntimeError("project CUDA torch/lib is not first in child PATH")
    if os.path.normcase(os.path.normpath(python_first)) != os.path.normcase(os.path.normpath(str(CUDA_TORCH_SITE))):
        raise RuntimeError("project CUDA site-packages is not first in child PYTHONPATH")


def run_environment_check() -> int:
    verify_child_environment()
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    vector = torch.ones(8, dtype=torch.float64, device="cuda:0")
    result = (vector * vector).sum()
    torch.cuda.synchronize()
    payload = {
        "status": "pass-no-model-work",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": True,
        "float64_gpu_sum": float(result.cpu()),
        "numpy_sum": float(np.arange(8, dtype=np.float64).sum()),
        "kmp_duplicate_lib_ok": False,
    }
    print(json.dumps(payload, indent=2))
    return 0


def validate_v5_launch(*, output_must_be_absent: bool = False) -> dict:
    state_path = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = load_json(state_path)
    transaction = load_json(TRANSACTION_PATH)
    fixture = load_json(FIXTURE_MANIFEST_PATH)
    failures = []
    if transaction.get("transaction_id") != "lia-1d-gpu-batch-preflight-v5":
        failures.append("wrong v5 transaction")
    if state.get("next_action", {}).get("launch_authorized") is not True:
        failures.append("controller launch_authorized is not true")
    if state.get("next_action", {}).get("transaction_id") != "lia-1d-gpu-batch-preflight-v5":
        failures.append("controller next action is not v5")
    focus = state.get("current_focus", {})
    if focus.get("system_id") != "all191-oggm-flowline-1d-cpu12":
        failures.append("CPU12 production current focus is not preserved")
    if focus.get("production_authorized") is not False or focus.get("production_launched") is not False:
        failures.append("CPU12 production must remain unauthorized and unlaunched")
    branch = state.get("research_branches", {}).get("lia_1d_gpu_batch_preflight_v5", {})
    if branch.get("launch_authorized") is not True or branch.get("cell_count") != 24:
        failures.append("v5 research branch is not authorized for exactly 24 cells")
    if branch.get("full_191_gpu_matrix_authorized") is not False:
        failures.append("full-191 GPU prohibition missing")
    if branch.get("planform_2d_or_igm_allowed") is not False:
        failures.append("2-D/IGM prohibition missing")
    route_item = state.get("authoritative_route", {})
    route_path = PROJECT_ROOT / route_item.get("path", "")
    if not route_path.exists() or frozen_guard.sha256(route_path) != route_item.get("sha256", "").upper():
        failures.append("route hash mismatch")
    expected = transaction.get("unchanged_implementation", {})
    if frozen_guard.sha256(CPU_DRIVER) != expected.get("cpu_oracle_driver_sha256"):
        failures.append("CPU driver hash mismatch")
    if frozen_guard.sha256(GPU_DRIVER) != expected.get("gpu_runner_sha256"):
        failures.append("GPU driver hash mismatch")
    if frozen_guard.sha256(BACKEND) != expected.get("backend_sha256"):
        failures.append("backend hash mismatch")
    npz_path = PROJECT_ROOT / fixture.get("npz_path", "")
    if not npz_path.exists() or frozen_guard.sha256(npz_path) != expected.get("static_bridge_sha256"):
        failures.append("bridge hash mismatch")
    if fixture.get("cell_count") != 24:
        failures.append("bridge is not fixed to 24 cells")
    if output_must_be_absent and OUTPUT_ROOT.exists():
        failures.append("v5 output root already exists")
    if failures:
        raise RuntimeError("v5 launch guard failed: " + "; ".join(failures))
    return {
        "state": state,
        "transaction": transaction,
        "fixture": fixture,
        "route_path": route_path,
        "route_sha256": frozen_guard.sha256(route_path),
        "state_sha256": frozen_guard.sha256(state_path),
        "npz_path": npz_path,
    }


def finalize_summary(phase: str) -> None:
    summary_path = OUTPUT_ROOT / ("cpu/summary.json" if phase == "cpu" else "gpu/summary.json")
    if not summary_path.exists():
        return
    payload = load_json(summary_path)
    payload["transaction_id"] = "lia-1d-gpu-batch-preflight-v5"
    payload["administrative_launcher"] = str(Path(__file__).resolve().relative_to(PROJECT_ROOT))
    payload["numerical_driver_reused_from"] = "lia-1d-gpu-batch-preflight-v2"
    payload["gpu_process_policy"] = "PATH/PYTHONPATH fixed before child Python startup; KMP_DUPLICATE_LIB_OK forbidden"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=summary_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(summary_path)


def run_bound_phase(phase: str) -> int:
    if phase == "gpu":
        verify_child_environment()
    frozen_guard.TRANSACTION_PATH = TRANSACTION_PATH
    frozen_guard.FIXTURE_MANIFEST_PATH = FIXTURE_MANIFEST_PATH
    frozen_guard.OUTPUT_ROOT = OUTPUT_ROOT
    frozen_guard.validate_launch_authorization = validate_v5_launch
    target = CPU_DRIVER if phase == "cpu" else GPU_DRIVER
    code = 0
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        code = int(exc.code or 0)
    finally:
        finalize_summary(phase)
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cpu", "gpu"))
    parser.add_argument("--environment-check-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "gpu" and os.environ.get(CHILD_MARKER) != "1":
        command = [sys.executable, str(Path(__file__).resolve()), "gpu"]
        if args.environment_check_only:
            command.append("--environment-check-only")
        return subprocess.run(command, env=child_environment(), check=False).returncode
    if args.environment_check_only:
        if args.phase != "gpu":
            raise RuntimeError("environment check applies only to the GPU child")
        return run_environment_check()
    return run_bound_phase(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
