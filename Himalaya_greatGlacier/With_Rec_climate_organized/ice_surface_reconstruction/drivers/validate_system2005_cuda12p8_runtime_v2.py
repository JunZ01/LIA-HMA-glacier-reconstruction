"""Corrected CUDA 12.8 runtime validation with ABI-aware extension discovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import torch


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RUNTIME = ROOT / ".runtime/torch-cu128-v1"
SITE = RUNTIME / "site-packages"
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_cuda12p8_runtime_validation_v2"
V1 = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_cuda12p8_runtime_validation_v1/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    extensions = sorted((SITE / "torch").glob("_C*.pyd"))
    if len(extensions) != 1:
        raise RuntimeError(f"torch-extension-count:{len(extensions)}")
    torch_path = Path(torch.__file__).resolve()
    tensor = torch.arange(1024, device="cuda", dtype=torch.float64)
    result = float(torch.sum(tensor * tensor).item())
    torch.cuda.synchronize()
    runtime_bytes = sum(path.stat().st_size for path in SITE.rglob("*") if path.is_file())
    checks = {
        "torch_loaded_from_e_drive_runtime": str(torch_path).lower().startswith(str(SITE.resolve()).lower()),
        "torch_version_exact": torch.__version__ == "2.11.0+cu128",
        "cuda_runtime_exact": torch.version.cuda == "12.8",
        "cuda_available": torch.cuda.is_available(),
        "gpu_capability_sm120": tuple(torch.cuda.get_device_capability(0)) == (12, 0),
        "gpu_kernel_result_exact": result == 357389824.0,
        "torch_abi_extension_unique": len(extensions) == 1,
        "netcdf4_available": bool(netCDF4.__version__),
        "numpy_available": bool(np.__version__),
        "pip_cache_absent": not (RUNTIME / "pip-cache").exists(),
        "temporary_directory_absent": not (RUNTIME / "temp").exists(),
    }
    summary = {
        "schema": "lia-191-system2005-cuda12p8-runtime-validation-v2",
        "status": "cuda12p8-runtime-pass" if all(checks.values()) else "cuda12p8-runtime-fail-stop",
        "supersedes": "v1 manifest-only interruption caused by fixed _C.pyd filename assumption",
        "v1_summary_sha256": sha256(V1),
        "runtime_root": str(RUNTIME.relative_to(ROOT)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_file": str(torch_path),
        "torch_extension": str(extensions[0]),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "netcdf4_version": netCDF4.__version__,
        "numpy_version": np.__version__,
        "runtime_size_bytes": runtime_bytes,
        "runtime_size_gib": runtime_bytes / 1024**3,
        "checks": checks,
        "production_files_modified": False,
    }
    OUT.mkdir(parents=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "driver": str(DRIVER.relative_to(ROOT)),
        "driver_sha256": sha256(DRIVER),
        "torch_init_sha256": sha256(torch_path),
        "torch_extension_sha256": sha256(extensions[0]),
        "v1_summary_sha256": sha256(V1),
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
