"""Fresh-process launcher for the independent fixed-24 GPU-only v9 run."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DRIVERS = Path(__file__).resolve().parent
SS4_PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
TORCH_SITE = PROJECT_ROOT / ".runtime/torch-cu128-v1/site-packages"
TORCH_LIB = TORCH_SITE / "torch/lib"
WINDOWS = Path(os.environ.get("SystemRoot", "C:/Windows"))
RUNNER = DRIVERS / "run_lia_1d_gpu_only_v9.py"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def child_environment() -> dict[str, str]:
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        raise RuntimeError("v9 forbids KMP_DUPLICATE_LIB_OK")
    if not SS4_PYTHON.is_file() or not TORCH_SITE.is_dir() or not TORCH_LIB.is_dir():
        raise RuntimeError("registered SS4 Python or project CUDA PyTorch runtime is missing")
    env = os.environ.copy()
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env.pop("PYTHONHOME", None)
    env["PATH"] = os.pathsep.join((str(TORCH_LIB), str(WINDOWS / "System32"), str(WINDOWS)))
    env["PYTHONPATH"] = str(TORCH_SITE)
    for name in THREAD_ENVIRONMENT:
        env[name] = "1"
    env["LIA_1D_GPU_ONLY_V9_CHILD"] = "1"
    return env


def main() -> int:
    command = [str(SS4_PYTHON), str(RUNNER)]
    return subprocess.run(command, env=child_environment(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
