"""Run synthetic CPU tensor tests in the GPU runtime without opening CUDA."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    from launch_lia_191_gpu_spinup_v10_239 import child_environment, SS4_PYTHON
    if os.environ.get("LIA_GPU_MATRIX_TEST_CHILD") != "1":
        env = child_environment()
        env["LIA_GPU_MATRIX_TEST_CHILD"] = "1"
        return subprocess.run([str(SS4_PYTHON), "-B", str(Path(__file__))], env=env).returncode
    from run_lia_191_gpu_spinup_v10_239 import verify_child_environment, import_torch_with_stubs_filtered
    verify_child_environment()
    torch, _, _ = import_torch_with_stubs_filtered()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"),
                                              pattern="test_torch_flowline_1d_matrix_v2.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
