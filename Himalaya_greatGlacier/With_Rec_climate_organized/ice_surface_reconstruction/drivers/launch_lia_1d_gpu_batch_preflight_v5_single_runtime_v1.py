"""GPU-only v5 restart launcher with one OpenMP-capable library search path.

The outer process may be any Python.  The numerical child is always the fixed
SS4 interpreter.  Its PATH is reduced to the project CUDA torch DLL directory
and Windows system directories before Python starts.  The passed v5 CPU oracle
is reused and never rerun.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DRIVERS = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(DRIVERS))
import launch_lia_1d_gpu_batch_preflight_v5 as v5
import launch_lia_1d_gpu_batch_preflight_v5_guard_retry_v1 as guard_retry


SS4_PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
WINDOWS_DIR = Path(os.environ.get("SystemRoot", "C:/Windows"))
ALLOWED_PATH = [v5.CUDA_TORCH_LIB, WINDOWS_DIR / "System32", WINDOWS_DIR]
SINGLE_RUNTIME_MARKER = "LIA_1D_GPU_V5_SINGLE_RUNTIME_CHILD"


def normalized(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def child_environment() -> dict[str, str]:
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        raise RuntimeError("single-runtime GPU restart forbids KMP_DUPLICATE_LIB_OK")
    if not SS4_PYTHON.is_file() or not v5.CUDA_TORCH_LIB.is_dir() or not v5.CUDA_TORCH_SITE.is_dir():
        raise RuntimeError("required SS4 Python or project CUDA PyTorch runtime is missing")
    env = os.environ.copy()
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env["PATH"] = os.pathsep.join(str(path) for path in ALLOWED_PATH)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = v5.prepend_unique(str(v5.CUDA_TORCH_SITE), existing_pythonpath)
    env[v5.CHILD_MARKER] = "1"
    env[SINGLE_RUNTIME_MARKER] = "1"
    return env


def verify_child_environment() -> None:
    if os.environ.get(SINGLE_RUNTIME_MARKER) != "1":
        raise RuntimeError("GPU numerical work must run in the single-runtime child")
    if normalized(sys.executable) != normalized(SS4_PYTHON):
        raise RuntimeError(f"wrong GPU Python: {sys.executable}")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        raise RuntimeError("single-runtime GPU restart forbids KMP_DUPLICATE_LIB_OK")
    actual_path = [normalized(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    expected_path = [normalized(item) for item in ALLOWED_PATH]
    if actual_path != expected_path:
        raise RuntimeError(f"GPU child PATH is not the registered single-runtime path: {actual_path}")
    pythonpath_first = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    if normalized(pythonpath_first) != normalized(v5.CUDA_TORCH_SITE):
        raise RuntimeError("project CUDA site-packages is not first in PYTHONPATH")


def environment_check() -> int:
    verify_child_environment()
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    vector = torch.ones(8, dtype=torch.float64, device="cuda:0")
    value = (vector * vector).sum()
    torch.cuda.synchronize()
    print(json.dumps({
        "status": "pass-no-model-work",
        "python": sys.executable,
        "path": os.environ.get("PATH", ""),
        "pythonpath_first": os.environ.get("PYTHONPATH", "").split(os.pathsep)[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": True,
        "float64_gpu_sum": float(value.cpu()),
        "kmp_duplicate_lib_ok": False,
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("gpu",))
    parser.add_argument("--environment-check-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get(SINGLE_RUNTIME_MARKER) != "1":
        command = [str(SS4_PYTHON), str(Path(__file__).resolve()), "gpu"]
        if args.environment_check_only:
            command.append("--environment-check-only")
        return subprocess.run(command, env=child_environment(), check=False).returncode
    if args.environment_check_only:
        return environment_check()
    verify_child_environment()
    return guard_retry.run_phase("gpu")


if __name__ == "__main__":
    raise SystemExit(main())
