"""Run a cached, class-level 2000-2020 diagnostic for frozen LIA curves.

This driver does not change historical-curve acceptance.  It continues each
frozen year-2000 state with the candidate parameters and the historical
GSWP3-W5E5 2000-2019 climate, then compares mean surface mass balance with the
existing system-level Hugonnet reference used by Stage 2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
REGISTRY = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_frozen_curve_registry_v1"
)
OUTPUT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_frozen_modern_validation_v1"
)
WORK = ROOT / "w191_frozen_modern_v1"
ROUTING = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06"
    / "stage2_freeze_stage3_routing_v0"
    / "stage3_routing_245.csv"
)
PROBE_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.lia_modern_mb_probe_saved_state"
)
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_dirs() -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in REGISTRY.glob("system_*"):
        if not path.is_dir():
            continue
        system_id = int(path.name.removeprefix("system_"))
        required = [
            path / "summary.json",
            path / "state_2000.pkl.gz",
            path / "candidate_record.json",
        ]
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            raise FileNotFoundError(f"frozen-candidate-incomplete:{system_id}:{missing}")
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        climate_file = Path(str(summary["climate"]["forward"]["climate_file"]))
        source_gdir = climate_file.parent
        if not (source_gdir / "climate_historical.nc").exists():
            raise FileNotFoundError(
                f"frozen-source-gdir-missing-climate:{system_id}:{source_gdir}"
            )
        candidates.append((system_id, path))
    candidates.sort()
    if len(candidates) != 11:
        raise RuntimeError(f"expected-11-frozen-systems-found:{len(candidates)}")
    return candidates


def _paths(system_id: int) -> tuple[Path, Path, Path, Path]:
    output_dir = OUTPUT / "output" / f"system_{system_id}"
    workdir = WORK / f"system_{system_id}"
    result = output_dir / f"system_{system_id}.json"
    log = OUTPUT / "logs" / f"system_{system_id}.log"
    return output_dir, workdir, result, log


def _command(system_id: int, candidate_dir: Path) -> list[str]:
    output_dir, workdir, _, _ = _paths(system_id)
    return [
        str(PYTHON),
        "-m",
        PROBE_MODULE,
        "--candidate-dir",
        str(candidate_dir),
        "--workdir",
        str(workdir),
        "--output-dir",
        str(output_dir),
        "--label",
        f"system_{system_id}",
    ]


def prepare() -> pd.DataFrame:
    if not PYTHON.exists():
        raise FileNotFoundError(f"verified-python-missing:{PYTHON}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for system_id, candidate_dir in _candidate_dirs():
        output_dir, workdir, result, log = _paths(system_id)
        rows.append(
            {
                "lia_unique_id": system_id,
                "candidate_dir": str(candidate_dir),
                "output_dir": str(output_dir),
                "workdir": str(workdir),
                "result_json": str(result),
                "log": str(log),
                "state": "completed_cached" if result.exists() else "planned",
                "command": json.dumps(_command(system_id, candidate_dir)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "manifest.csv", index=False)
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": _utc_now(),
                "question": (
                    "How do the eleven frozen historical curves continue from "
                    "their saved 2000 states to 2020 under unchanged candidate "
                    "parameters and historical GSWP3-W5E5 climate?"
                ),
                "system_count": len(frame),
                "system_ids": frame["lia_unique_id"].astype(int).tolist(),
                "historical_acceptance_changed": False,
                "modern_mb_role": "diagnostic-only",
                "max_workers": 11,
                "python_executable": str(PYTHON),
                "probe_module": PROBE_MODULE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame


def _run_one(row: dict) -> dict:
    system_id = int(row["lia_unique_id"])
    output_dir, workdir, result, log = _paths(system_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if result.exists():
        return {"lia_unique_id": system_id, "state": "completed_cached", "returncode": 0}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        json.loads(str(row["command"])),
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    log.write_text(
        (completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    state = "completed" if completed.returncode == 0 and result.exists() else "technical_failure"
    return {
        "lia_unique_id": system_id,
        "state": state,
        "returncode": int(completed.returncode),
        "error": "" if state == "completed" else (completed.stderr or completed.stdout)[-1000:],
    }


def run(max_workers: int, limit: int | None) -> pd.DataFrame:
    manifest = prepare()
    planned = manifest[manifest["state"].eq("planned")]
    if limit is not None:
        planned = planned.head(limit)
    if not planned.empty:
        workers = min(max_workers, 11, len(planned))
        if workers < 1 or workers > 12:
            raise ValueError("max-workers-must-be-1-to-12")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, row): int(row["lia_unique_id"])
                for row in planned.to_dict(orient="records")
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                print(
                    f"completed {index}/{len(futures)} system={result['lia_unique_id']} "
                    f"state={result['state']}",
                    flush=True,
                )
    return summarize()


def summarize() -> pd.DataFrame:
    manifest = prepare()
    routing = pd.read_csv(ROUTING)
    routing["lia_unique_id"] = pd.to_numeric(routing["lia_unique_id"]).astype(int)
    routing = routing.set_index("lia_unique_id")
    rows = []
    for item in manifest.itertuples(index=False):
        result_path = Path(item.result_json)
        if not result_path.exists():
            rows.append(
                {
                    "lia_unique_id": int(item.lia_unique_id),
                    "state": "missing",
                }
            )
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        system_id = int(result["lia_unique_id"])
        reference = routing.loc[system_id]
        target = float(reference["hugonnet_mb_kg_m2_yr"])
        sigma = 1000.0 * float(reference["hugonnet_err_fully_correlated_mwea"])
        modeled = float(result["model_surface_mb_mean_kg_m2_yr"])
        residual = modeled - target
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": result["rgi_id"],
                "state": "ok",
                "model_surface_mb_mean_kg_m2_yr": modeled,
                "model_geodetic_mb_kg_m2_yr": float(result["model_geodetic_mb_kg_m2_yr"]),
                "hugonnet_mb_kg_m2_yr": target,
                "hugonnet_sigma_kg_m2_yr": sigma,
                "surface_mb_residual_kg_m2_yr": residual,
                "surface_mb_abs_z": abs(residual) / sigma if sigma > 0 else math.nan,
                "length_change_2000_2020_m": float(result["length_change_m"]),
                "area_change_2000_2020_percent": (
                    100.0
                    * float(result["area_change_m2"])
                    / float(result["initial_2000"]["area_m2"])
                ),
                "volume_change_2000_2020_percent": (
                    100.0
                    * float(result["volume_change_m3"])
                    / float(result["initial_2000"]["volume_m3"])
                ),
                "runtime_seconds": float(result["runtime_seconds"]),
                "result_json": str(result_path),
                "timeseries_file": result["timeseries_file"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    frame.to_csv(OUTPUT / "modern_validation_summary.csv", index=False)
    ok = frame[frame["state"].eq("ok")]
    summary = {
        "schema_version": 1,
        "updated_at_utc": _utc_now(),
        "system_count": int(len(frame)),
        "completed_count": int(len(ok)),
        "technical_or_missing_count": int(len(frame) - len(ok)),
        "within_1sigma_count": int((ok["surface_mb_abs_z"] <= 1).sum()) if not ok.empty else 0,
        "within_2sigma_count": int((ok["surface_mb_abs_z"] <= 2).sum()) if not ok.empty else 0,
        "median_abs_z": float(ok["surface_mb_abs_z"].median()) if not ok.empty else math.nan,
        "historical_acceptance_changed": False,
        "diagnostic_role": "modern MB is diagnostic only",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--max-workers", type=int, default=11)
    run_parser.add_argument("--limit", type=int, default=None)
    subparsers.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare":
        frame = prepare()
        print(frame[["lia_unique_id", "state"]].to_string(index=False))
    elif args.command == "run":
        run(args.max_workers, args.limit)
    else:
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
