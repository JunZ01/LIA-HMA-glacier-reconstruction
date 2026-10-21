"""No-model verification of the exact v7 Torch import filter."""

from __future__ import annotations

import json
import os

from run_lia_1d_gpu_stage_b_v7 import (
    IOMP_RUNTIME_SHA256,
    IOMP_STUBS_SHA256,
    TORCH_INIT_SHA256,
    import_torch_with_stubs_filtered,
    verify_child_environment,
)


def main() -> int:
    verify_child_environment()
    torch, filter_report, mapped_iomp = import_torch_with_stubs_filtered()
    import numpy as np

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    value = torch.ones(8, dtype=torch.float64, device="cuda:0").sum()
    torch.cuda.synchronize()
    payload = {
        "schema": "lia-1d-gpu-v7-filtered-import-no-model-check-v1",
        "status": "pass-no-model-work",
        "process_id": os.getpid(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "cuda_float64_sum": float(value.cpu()),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_init_sha256": TORCH_INIT_SHA256,
        "filtered_stubs_sha256": IOMP_STUBS_SHA256,
        "glob_restored": True,
        "mapped_iomp_modules": mapped_iomp,
        "required_unique_runtime_sha256": IOMP_RUNTIME_SHA256,
        "filter_report": filter_report,
        "kmp_duplicate_lib_ok": False,
        "model_state_created": False,
        "model_step_run": False,
    }
    if payload["cuda_float64_sum"] != 8.0:
        raise RuntimeError("CUDA float64 sum mismatch")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
