"""Verify that IGM 3.2 and the system-2005 geology execute on the WSL GPU.

Run this file with the isolated WSL Python environment.  It does not advance
glacier time.  The test imports IGM, loads the prepared NetCDF, executes
representative TensorFlow stencil/reduction operations on GPU 0, and records
device, timing, and mass-closure diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import igm
import netCDF4
import numpy as np
import tensorflow as tf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(geology_path: Path, static_summary_path: Path, output: Path) -> dict:
    static = json.loads(static_summary_path.read_text(encoding="utf-8"))
    if static.get("status") != "static-pass":
        raise RuntimeError("static-preflight-not-passed")
    if _sha256(geology_path) != static.get("geology_nc_sha256"):
        raise RuntimeError("geology-hash-mismatch")

    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not physical_gpus:
        raise RuntimeError("tensorflow-gpu-not-visible")

    with netCDF4.Dataset(geology_path) as dataset:
        thickness = np.asarray(dataset.variables["thk"][:], dtype=np.float32)
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float32)
        surface = np.asarray(dataset.variables["usurf"][:], dtype=np.float32)
        x = np.asarray(dataset.variables["x"][:], dtype=np.float32)
    dx = float(x[1] - x[0])

    with tf.device("/GPU:0"):
        thk = tf.constant(thickness)
        topg = tf.constant(bed)
        usurf = tf.constant(surface)
        kernel = tf.constant(
            [[[[0.0]], [[1.0]], [[0.0]]], [[[1.0]], [[-4.0]], [[1.0]]], [[[0.0]], [[1.0]], [[0.0]]]],
            dtype=tf.float32,
        )
        start_first = time.perf_counter()
        laplacian = tf.nn.conv2d(thk[None, :, :, None], kernel, strides=1, padding="SAME")
        first_checksum = tf.reduce_sum(tf.abs(laplacian))
        closure = tf.reduce_max(tf.abs(usurf - topg - thk))
        volume = tf.reduce_sum(thk) * tf.cast(dx * dx, tf.float32)
        _ = (first_checksum.numpy(), closure.numpy(), volume.numpy())
        first_seconds = time.perf_counter() - start_first

        start_warm = time.perf_counter()
        for _ in range(20):
            laplacian = tf.nn.conv2d(thk[None, :, :, None], kernel, strides=1, padding="SAME")
            warm_checksum = tf.reduce_sum(tf.abs(laplacian))
        _ = warm_checksum.numpy()
        warm_seconds = time.perf_counter() - start_warm

    gpu_details = tf.config.experimental.get_device_details(physical_gpus[0])
    measured_volume = float(volume.numpy())
    expected_volume = float(static["model_volume_m3"])
    volume_error = abs(measured_volume - expected_volume) / expected_volume
    closure_m = float(closure.numpy())
    passed = bool(
        "GPU" in thk.device.upper()
        and np.isfinite(measured_volume)
        and volume_error <= 2.0e-6
        and closure_m <= 2.0e-3
    )
    summary = {
        "schema": "lia-191-igm-planform2d-system2005-gpu-smoke-v1",
        "status": "gpu-smoke-pass" if passed else "gpu-smoke-fail",
        "system_id": 2005,
        "dynamics_run": False,
        "igm_version": getattr(igm, "__version__", "unknown"),
        "tensorflow_version": tf.__version__,
        "physical_gpu_count": len(physical_gpus),
        "gpu_name": gpu_details.get("device_name", "unknown"),
        "tensor_device": thk.device,
        "grid_shape": list(thickness.shape),
        "dx_m": dx,
        "expected_volume_m3": expected_volume,
        "gpu_reduction_volume_m3": measured_volume,
        "gpu_reduction_volume_relative_error": volume_error,
        "surface_bed_thickness_closure_m": closure_m,
        "first_gpu_stencil_seconds": first_seconds,
        "twenty_warm_gpu_stencils_seconds": warm_seconds,
        "geology_nc": str(geology_path),
        "geology_nc_sha256": _sha256(geology_path),
        "next_authorized_stage": "igm-native-load-and-one-dynamics-step" if passed else None,
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.geology, args.static_summary, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
