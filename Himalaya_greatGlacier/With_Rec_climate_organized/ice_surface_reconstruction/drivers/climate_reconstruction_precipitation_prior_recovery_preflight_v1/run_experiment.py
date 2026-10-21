"""Launch and evaluate the MATLAB R2024b precipitation-prior recovery preflight."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MATLAB_EXE = Path(r"E:\Program Files\MATLAB\R2024b\bin\matlab.exe")
PHYDA_ROOT = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
CESM_ROOT = Path(r"D:\VM\gcm_7\cesm")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_root() -> Path:
    archive_folder = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper_folder = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive_folder / paper_folder / "work1" / "compared_result_all"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }


def scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name]).reshape(-1)[0])


def main() -> int:
    root = project_root()
    driver_dir = Path(__file__).resolve().parent
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_prior_recovery_preflight_v1"
    )
    output.mkdir(parents=True, exist_ok=True)
    archive = archive_root()
    paths = {
        "matlab_loader": PHYDA_ROOT / "load_S_720360.m",
        "model_years": PHYDA_ROOT / "modelyrs.m",
        "monthly_to_annual": PHYDA_ROOT / "mon2ann.m",
        "observed_climatology_loader": PHYDA_ROOT / "load_obsclim.m",
        "cru_cl_precipitation": PHYDA_ROOT / "input-recon" / "cru_cl_1.1.6190.pre.dat",
        "cesm_grid_source": PHYDA_ROOT / "input-recon" / "b.e11.BLMTRC5CN.f19_g16.010.cam.h0.TREFHT.085001-184912.nc",
        "cesm_precc": CESM_ROOT / "cesm_PRECC_merged.nc",
        "cesm_precl": CESM_ROOT / "cesm_PRECL_merged.nc",
        "compact_prior": archive / "python_data" / "prior_fields.mat",
        "archived_config": archive / "fixed_proxy_experiments_69" / "pr" / "experiment_config.mat",
        "archived_indices": archive / "fixed_proxy_experiments_69" / "pr" / "indices" / "all_Xb_indices.mat",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if not MATLAB_EXE.exists():
        raise FileNotFoundError(f"missing-matlab-r2024b:{MATLAB_EXE}")

    before = manifest(paths)
    matlab_output = output / "precipitation_prior_336x848_mm_day.mat"
    stdout_path = output / "matlab_r2024b_stdout.log"
    stderr_path = output / "matlab_r2024b_stderr.log"
    prefdir = output / "matlab_prefdir"
    prefdir.mkdir(exist_ok=True)
    launch = driver_dir / "launch.m"
    command = [str(MATLAB_EXE), "-batch", f"run('{launch.as_posix()}')"]
    environment = os.environ.copy()
    environment["MATLAB_PREFDIR"] = str(prefdir)
    completed = subprocess.run(
        command,
        cwd=driver_dir,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"matlab-r2024b-failed:{completed.returncode}")
    if not matlab_output.exists():
        raise FileNotFoundError(f"missing-matlab-output:{matlab_output}")

    with h5py.File(matlab_output, "r") as handle:
        prior = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float).T
        years = np.asarray(handle["member_years"], dtype=int).reshape(-1)
        lat = np.asarray(handle["lat"], dtype=float).reshape(-1)
        lon = np.asarray(handle["lon"], dtype=float).reshape(-1)
        metrics = {
            "repeat_max_abs_mm_day": scalar(handle, "repeat_max_abs_mm_day"),
            "repeat_rmse_mm_day": scalar(handle, "repeat_rmse_mm_day"),
            "repeat_changed_count": int(scalar(handle, "repeat_changed_count")),
            "compact_overlap_max_abs_mm_day": scalar(handle, "compact_overlap_max_abs_mm_day"),
            "compact_overlap_rmse_mm_day": scalar(handle, "compact_overlap_rmse_mm_day"),
            "compact_overlap_exact_fraction": scalar(handle, "compact_overlap_exact_fraction"),
            "compact_overlap_changed_count": int(scalar(handle, "compact_overlap_changed_count")),
            "compact_zero_mask_mismatch_count": int(scalar(handle, "compact_zero_mask_mismatch_count")),
            "compact_nonzero_max_abs_mm_day": scalar(handle, "compact_nonzero_max_abs_mm_day"),
            "compact_nonzero_rmse_mm_day": scalar(handle, "compact_nonzero_rmse_mm_day"),
            "compact_nonzero_changed_count": int(scalar(handle, "compact_nonzero_changed_count")),
            "compact_nonzero_count": int(scalar(handle, "compact_nonzero_count")),
            "unit_ratio_median": scalar(handle, "unit_ratio_median"),
            "unit_ratio_p01": scalar(handle, "unit_ratio_p01"),
            "unit_ratio_p99": scalar(handle, "unit_ratio_p99"),
            "prior_min_mm_day": scalar(handle, "prior_min_mm_day"),
            "prior_mean_mm_day": scalar(handle, "prior_mean_mm_day"),
            "prior_median_mm_day": scalar(handle, "prior_median_mm_day"),
            "prior_max_mm_day": scalar(handle, "prior_max_mm_day"),
            "prior_zero_count": int(scalar(handle, "prior_zero_count")),
        }

    shape_gate = prior.shape == (336, 848) and years.tolist() == list(range(1001, 1849))
    coordinate_gate = lat.size == 14 and lon.size == 24 and lat.size * lon.size == 336
    repeat_gate = (
        metrics["repeat_max_abs_mm_day"] == 0
        and metrics["repeat_rmse_mm_day"] == 0
        and metrics["repeat_changed_count"] == 0
    )
    unit_gate = (
        abs(metrics["unit_ratio_median"] - 30.0) <= 1e-10
        and metrics["compact_zero_mask_mismatch_count"] == 0
    )
    finite_gate = bool(np.isfinite(prior).all())
    prior_ready = bool(shape_gate and coordinate_gate and repeat_gate and unit_gate and finite_gate)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    axes[0].hist(prior.reshape(-1), bins=90, color="#357a68", alpha=0.84)
    axes[0].set_xlim(0, np.percentile(prior, 99.5))
    axes[0].set_xlabel("Annual precipitation (mm/day)")
    axes[0].set_ylabel("State-member count")
    axes[0].set_title("Recovered 336 x 848 prior")

    regional = prior.mean(axis=0)
    axes[1].plot(years, regional, color="#2b5f8a", lw=1.2)
    axes[1].set_xlabel("CESM member year")
    axes[1].set_ylabel("Unweighted HMA mean (mm/day)")
    axes[1].set_title("Recovered prior chronology")
    axes[1].grid(alpha=0.2)

    spatial = prior.mean(axis=1).reshape(lon.size, lat.size).T
    image = axes[2].pcolormesh(lon, lat, spatial, shading="auto", cmap="YlGnBu")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].set_title("Member-mean precipitation")
    figure.colorbar(image, ax=axes[2], label="mm/day")
    figure.savefig(output / "figure_precipitation_prior_preflight.png", dpi=220)
    plt.close(figure)

    after = manifest(paths)
    source_hashes_unchanged = before == after
    status = "pass" if prior_ready and source_hashes_unchanged else "fail"
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_prior_recovery_preflight_v1",
        "date": "2026-08-13",
        "matlab_release_required": "R2024b",
        "matlab_executable": str(MATLAB_EXE),
        "matlab_returncode": completed.returncode,
        "no_oggm": True,
        "no_enkf": True,
        "prior_shape": list(prior.shape),
        "member_year_range": [int(years[0]), int(years[-1])],
        "state_grid": {"lat_count": int(lat.size), "lon_count": int(lon.size)},
        "metrics": metrics,
        "gates": {
            "shape_gate": shape_gate,
            "coordinate_gate": coordinate_gate,
            "repeat_gate": repeat_gate,
            "unit_gate": unit_gate,
            "finite_gate": finite_gate,
            "source_hashes_unchanged": source_hashes_unchanged,
        },
        "status": status,
        "prior_ready_for_single_experiment_replay": prior_ready,
        "interpretation_boundary": (
            "Passing this preflight freezes a reproducible current-loader precipitation prior. "
            "It does not prove bitwise identity with the missing May 17 loader snapshot or validate "
            "the precipitation reconstruction."
        ),
    }
    (output / "input_manifest_before.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8"
    )
    (output / "input_manifest_after.json").write_text(
        json.dumps(after, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# Precipitation Prior Recovery Preflight Result",
        "",
        f"Status: **{status}**",
        "",
        f"- MATLAB: R2024b at `{MATLAB_EXE}`",
        f"- Prior shape: `{prior.shape[0]} x {prior.shape[1]}`",
        f"- Member years: `{years[0]}-{years[-1]}`",
        f"- Same-seed repeat maximum difference: `{metrics['repeat_max_abs_mm_day']:.12g} mm/day`",
        f"- Compact export median unit ratio: `{metrics['unit_ratio_median']:.12g}`",
        f"- Compact export exact-overlap fraction: `{metrics['compact_overlap_exact_fraction']:.6f}`",
        f"- Compact zero-mask mismatch count: `{metrics['compact_zero_mask_mismatch_count']}`",
        f"- Source hashes unchanged: `{source_hashes_unchanged}`",
        "",
        "## Decision",
        "",
        (
            "The recovered prior may enter one archived EnKF replay experiment."
            if prior_ready
            else "Stop before EnKF replay and diagnose the failed prior gate."
        ),
        "",
        summary["interpretation_boundary"],
        "",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if status != "pass":
        raise RuntimeError("precipitation-prior-preflight-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

