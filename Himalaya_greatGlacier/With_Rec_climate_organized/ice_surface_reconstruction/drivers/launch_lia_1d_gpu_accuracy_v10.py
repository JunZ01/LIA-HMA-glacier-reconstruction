from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
SITE = ROOT / ".runtime/torch-cu128-v1/site-packages"
LIB = SITE / "torch/lib"
WINDOWS = Path(os.environ.get("SystemRoot", "C:/Windows"))
RUNNER = Path(__file__).with_name("run_lia_1d_gpu_accuracy_v10.py")
THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def main() -> int:
    environment = os.environ.copy()
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    environment.pop("PYTHONHOME", None)
    environment["PATH"] = os.pathsep.join(
        map(str, (LIB, WINDOWS / "System32", WINDOWS))
    )
    environment["PYTHONPATH"] = str(SITE)
    environment["LIA_1D_GPU_ACCURACY_V10_CHILD"] = "1"
    for name in THREAD_NAMES:
        environment[name] = "1"
    return subprocess.run(
        [str(PYTHON), str(RUNNER)], env=environment, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
