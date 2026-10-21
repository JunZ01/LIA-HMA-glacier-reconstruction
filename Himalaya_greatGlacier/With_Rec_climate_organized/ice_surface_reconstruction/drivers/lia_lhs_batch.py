"""Run the D-layer Latin-hypercube parameter search in isolated processes.

Each trial delegates to ``lia_lhs_single_chain`` and gets its own OGGM
working/output directory.  This keeps OGGM's global configuration out of the
parent process and makes failed trials explicit in the summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def _lhs(n: int, low: np.ndarray, high: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic Latin-hypercube values for one glacier."""
    u = (np.arange(n, dtype=float) + rng.random(n)) / n
    values = np.empty((n, len(low)), dtype=float)
    for j in range(len(low)):
        values[:, j] = low[j] + rng.permutation(u) * (high[j] - low[j])
    return values


def _run_trial(payload: dict) -> dict:
    output_dir = Path(payload["output_dir"])
    workdir = Path(payload["workdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        payload["python_exe"], "-m",
        "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_lhs_single_chain",
        "--system-id", str(payload["system_id"]),
        "--references", payload["references"],
        "--stage2-members", payload["stage2_members"],
        "--climate-pixels", payload["climate_pixels"],
        "--reconstruction-climate", payload["reconstruction_climate"],
        "--source-gdir-root", payload["source_gdir_root"],
        "--elevation-band-root", payload["elevation_band_root"],
        "--workdir", str(workdir),
        "--output-dir", str(output_dir),
        "--ensemble-role", "mid-turnover",
        "--climate-id", "ensemble_rec_ref",
        "--prcp-mode", "off",
        "--spinup-window-start", "1807",
        "--spinup-window-end", "1836",
        "--start-year", "1836",
        "--spinup-years", "500",
        "--spinup-checkpoint-years", "100",
        "--delta-t", repr(payload["delta_t"]),
        "--melt-f", repr(payload["melt_f"]),
        "--prcp-fac-multiplier", repr(payload["prcp_fac_multiplier"]),
        "--glen-a", repr(payload["glen_a"]),
        "--temp-melt", repr(payload["temp_melt"]),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = payload["project_root"]
    log_path = output_dir / "run.log"
    try:
        completed = subprocess.run(
            command,
            cwd=payload["project_root"],
            env=env,
            capture_output=True,
            text=True,
            timeout=payload["timeout_seconds"],
        )
    except Exception as exc:  # pragma: no cover - worker failure path
        log_path.write_text(f"worker-exception: {exc}\n", encoding="utf-8")
        return {**payload, "status": "crash", "error": f"worker-exception:{exc}"}
    log_path.write_text(
        (completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    if completed.returncode != 0 or not summary_path.exists():
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:]
        return {
            **payload,
            "status": "crash",
            "returncode": completed.returncode,
            "error": tail[0] if tail else "missing-summary",
        }
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {**payload, "status": "crash", "error": f"invalid-summary:{exc}"}
    metrics = summary.get("metrics", {})
    return {
        **payload,
        "status": summary.get("status", "crash"),
        "e_lia": metrics.get("e_lia"),
        "e_2000": metrics.get("e_2000"),
        "e_total": metrics.get("e_total"),
        "l_spinup_m": metrics.get("l_spinup_m"),
        "l_2000_m": metrics.get("l_2000_m"),
        "v_spinup_m3": metrics.get("v_spinup_m3"),
        "v_2000_m3": metrics.get("v_2000_m3"),
        "a_spinup_m2": metrics.get("a_spinup_m2"),
        "a_2000_m2": metrics.get("a_2000_m2"),
        "runtime_s": summary.get("timing_seconds", {}).get("total"),
        "error": "",
    }


def build_trials(args: argparse.Namespace) -> list[dict]:
    refs = pd.read_csv(args.references)
    stage2 = pd.read_csv(args.stage2_members)
    selected = refs[refs["lia_unique_id"].astype(int).isin(args.system_ids)].copy()
    if len(selected) != len(args.system_ids):
        raise ValueError("reference rows missing for selected systems")
    rows: list[dict] = []
    for system_id in args.system_ids:
        ref = selected[selected["lia_unique_id"].astype(int).eq(system_id)].iloc[0]
        rgi_id = str(ref["rgi_id"])
        candidates = stage2[
            stage2["lia_unique_id"].astype(int).eq(system_id)
            & stage2["rgi_id"].astype(str).eq(rgi_id)
            & stage2["ensemble_role"].astype(str).eq("mid-turnover")
        ]
        if len(candidates) != 1:
            raise ValueError(f"stage2 row count {system_id}/{rgi_id}: {len(candidates)}")
        base = candidates.iloc[0]
        rng = np.random.default_rng(args.seed + system_id)
        low = np.array([-3.0, 2.5, 0.8, 0.5, -2.0])
        high = np.array([1.5, 7.5, 1.2, 3.0, 0.0])
        values = _lhs(args.points, low, high, rng)
        for point, (delta_t, melt_f, prcp_mult, glen_mult, temp_melt) in enumerate(values):
            rows.append({
                "system_id": system_id,
                "rgi_id": rgi_id,
                "point": point,
                "delta_t": float(delta_t),
                "melt_f": float(melt_f),
                "prcp_fac_multiplier": float(prcp_mult),
                "glen_a_multiplier": float(glen_mult),
                "glen_a": float(args.glen_a_default * glen_mult),
                "temp_melt": float(temp_melt),
                "prcp_fac_initial": float(base["prcp_fac"]),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-ids", nargs="+", type=int, default=[1044, 1545, 3584])
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3584)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--glen-a-default", type=float, default=2.939387691339814e-24)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--source-gdir-root", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    project_root = str(Path(__file__).resolve().parents[4])
    trials = build_trials(args)
    design = pd.DataFrame(trials)
    design.to_csv(args.output_root / "parameter_design.csv", index=False)
    payloads = []
    for trial in trials:
        tag = f"system_{trial['system_id']}/point_{trial['point']:04d}"
        payloads.append({
            **trial,
            "python_exe": sys.executable,
            "project_root": project_root,
            "references": str(args.references.resolve()),
            "stage2_members": str(args.stage2_members.resolve()),
            "climate_pixels": str(args.climate_pixels.resolve()),
            "reconstruction_climate": str(args.reconstruction_climate.resolve()),
            "source_gdir_root": str(args.source_gdir_root.resolve()),
            "elevation_band_root": str(args.elevation_band_root.resolve()),
            "output_dir": str((args.output_root / tag).resolve()),
            "workdir": str((args.output_root / "work" / tag).resolve()),
            "timeout_seconds": args.timeout_seconds,
        })
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_trial, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"completed {index}/{len(futures)} system={result['system_id']} point={result['point']} status={result['status']}", flush=True)
    result_df = pd.DataFrame(results).sort_values(["system_id", "point"])
    result_df.to_csv(args.output_root / "results.csv", index=False)
    diagnostic = result_df.groupby("system_id", as_index=False).agg(
        failure_rate=("status", lambda s: float((s != "ok").mean())),
        min_e_lia=("e_lia", "min"),
        min_e_2000=("e_2000", "min"),
        min_e_total=("e_total", "min"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    diagnostic.to_csv(args.output_root / "diagnostics.csv", index=False)
    print(diagnostic.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
