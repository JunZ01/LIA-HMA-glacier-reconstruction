"""Prepare, launch, and evaluate the precipitation EnKF experiment-1 replay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MATLAB_EXE = Path(r"E:\Program Files\MATLAB\R2024b\bin\matlab.exe")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name]).reshape(-1)[0])


def main() -> int:
    root = project_root()
    driver_dir = Path(__file__).resolve().parent
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_replay_v1"
    )
    output.mkdir(parents=True, exist_ok=True)

    prepare_path = driver_dir / "prepare_input.py"
    spec = importlib.util.spec_from_file_location("prepare_precipitation_replay", prepare_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("prepare-module-load-failed")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    prepare.main()

    prepared = json.loads((output / "prepared_input_summary.json").read_text(encoding="utf-8"))
    source_hashes_before = prepared["source_hashes"]
    launch = driver_dir / "launch.m"
    prefdir = output / "matlab_prefdir"
    prefdir.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["MATLAB_PREFDIR"] = str(prefdir)
    completed = subprocess.run(
        [str(MATLAB_EXE), "-batch", f"run('{launch.as_posix()}')"],
        cwd=driver_dir,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    (output / "matlab_r2024b_stdout.log").write_text(
        completed.stdout, encoding="utf-8", errors="replace"
    )
    (output / "matlab_r2024b_stderr.log").write_text(
        completed.stderr, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"matlab-replay-failed:{completed.returncode}")

    replay_output = output / "replay_output.mat"
    if not replay_output.exists():
        raise FileNotFoundError(f"missing-replay-output:{replay_output}")
    with h5py.File(replay_output, "r") as handle:
        years = np.asarray(handle["run_years"], dtype=int).reshape(-1)
        active = np.asarray(handle["active_proxy_count"], dtype=int).reshape(-1)
        mean_by_year = np.asarray(handle["mean_max_abs_diff_by_year"], dtype=float).reshape(-1)
        sigma_by_year = np.asarray(handle["sigma_max_abs_diff_by_year"], dtype=float).reshape(-1)
        prct_by_year = np.asarray(handle["prct_max_abs_diff_by_year"], dtype=float).reshape(-1)
        metrics = {
            "mean_global_max_abs_diff_mm_day": scalar(handle, "mean_global_max_abs_diff"),
            "sigma_global_max_abs_diff_mm_day": scalar(handle, "sigma_global_max_abs_diff"),
            "percentile_global_max_abs_diff_mm_day": scalar(handle, "prct_global_max_abs_diff"),
            "mean_global_rmse_mm_day": scalar(handle, "mean_global_rmse"),
            "sigma_global_rmse_mm_day": scalar(handle, "sigma_global_rmse"),
            "percentile_global_rmse_mm_day": scalar(handle, "prct_global_rmse"),
            "runtime_seconds": scalar(handle, "runtime_seconds"),
            "active_proxy_minimum": int(active.min()),
            "active_proxy_maximum": int(active.max()),
            "mean_max_year": int(years[np.argmax(mean_by_year)]),
            "sigma_max_year": int(years[np.argmax(sigma_by_year)]),
            "percentile_max_year": int(years[np.argmax(prct_by_year)]),
        }

    archive = prepare.archive_root()
    source_paths = {
        "prior": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_precipitation_prior_recovery_preflight_v1/"
            "precipitation_prior_336x848_mm_day.mat"
        ),
        "indices": archive / "fixed_proxy_experiments_69" / "pr" / "indices" / "all_Xb_indices.mat",
        "proxy6": archive / "python_data" / "proxy_data_6.mat",
        "archived_result": archive / "fixed_proxy_experiments_69" / "pr" / "results" / "proxy6" / "EnKF_results.mat",
        "regem": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "enkf_reconstruction_source_recovery_audit_v2/regem_proxy6_exact_v1/regem_output.mat"
        ),
    }
    source_hashes_after = {name: sha256(path) for name, path in source_paths.items()}
    source_hashes_unchanged = source_hashes_before == source_hashes_after

    pass_bounds = {
        "mean_max_abs_mm_day": 0.05,
        "sigma_max_abs_mm_day": 0.005,
        "percentile_max_abs_mm_day": 0.06,
    }
    replay_gate = (
        metrics["mean_global_max_abs_diff_mm_day"] <= pass_bounds["mean_max_abs_mm_day"]
        and metrics["sigma_global_max_abs_diff_mm_day"] <= pass_bounds["sigma_max_abs_mm_day"]
        and metrics["percentile_global_max_abs_diff_mm_day"] <= pass_bounds["percentile_max_abs_mm_day"]
    )
    status = "pass" if replay_gate and source_hashes_unchanged else "fail"

    figure, axis = plt.subplots(figsize=(12.8, 5.2), constrained_layout=True)
    axis.plot(years, mean_by_year, lw=1.4, label="posterior mean")
    axis.plot(years, sigma_by_year, lw=1.2, label="posterior sigma")
    axis.plot(years, prct_by_year, lw=1.2, label="posterior percentiles")
    axis.axhline(pass_bounds["mean_max_abs_mm_day"], color="#777777", lw=0.8, ls="--")
    axis.set_xlabel("Year")
    axis.set_ylabel("Maximum absolute archive difference (mm/day)")
    axis.set_title("Proxy6 precipitation EnKF experiment-1 replay closure")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output / "figure_precipitation_enkf_exp1_replay.png", dpi=220)
    plt.close(figure)

    summary = {
        "experiment_id": "climate_reconstruction_precipitation_enkf_exp1_replay_v1",
        "date": "2026-08-13",
        "case": "proxy6 / precipitation / EnKF / experiment 1",
        "matlab": "R2024b",
        "years": [1000, 2000],
        "no_oggm": True,
        "rec_ref_used": False,
        "metrics": metrics,
        "pass_bounds": pass_bounds,
        "replay_gate": replay_gate,
        "source_hashes_unchanged": source_hashes_unchanged,
        "status": status,
        "decision": (
            "The recovered precipitation chain is sufficiently reproducible to proceed to a declared validation design."
            if status == "pass"
            else "Stop precipitation redevelopment and diagnose the replay discrepancy before validation or method changes."
        ),
        "interpretation_boundary": (
            "This is a close numerical replay test of one archived EnKF experiment, not a validation "
            "of precipitation skill and not evidence that the missing May 17 runtime snapshot has been recovered."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": source_hashes_before, "after": source_hashes_after}, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Precipitation EnKF Experiment-1 Replay Result",
        "",
        f"Status: **{status}**",
        "",
        f"- Posterior mean maximum difference: `{metrics['mean_global_max_abs_diff_mm_day']:.8g} mm/day`",
        f"- Posterior sigma maximum difference: `{metrics['sigma_global_max_abs_diff_mm_day']:.8g} mm/day`",
        f"- Posterior percentile maximum difference: `{metrics['percentile_global_max_abs_diff_mm_day']:.8g} mm/day`",
        f"- Active proxy range: `{metrics['active_proxy_minimum']}-{metrics['active_proxy_maximum']}`",
        f"- Runtime: `{metrics['runtime_seconds']:.2f} s`",
        f"- Source hashes unchanged: `{source_hashes_unchanged}`",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        summary["interpretation_boundary"],
        "",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if status != "pass":
        raise RuntimeError("precipitation-enkf-exp1-replay-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

