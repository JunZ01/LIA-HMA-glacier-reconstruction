"""Run explicit parameter-pair pilots on a spatial downstream domain.

This additive runner shares manifest construction and scoring with the verified
tradeoff pilot, but advances each task with the spatial-downstream single-chain
runner so the mapped observation-domain gate is retained.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .lia_191_controller import (
    _atomic_csv,
    _atomic_json,
    _utc_now,
    controller_root,
    load_config,
    task_command,
)
from .lia_191_glen_tradeoff_pilot import build_manifest
from .lia_191_tradeoff_pilot import print_status, sync as sync_tasks


SPATIAL_RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers."
    "lia_lhs_single_chain_spatial_downstream"
)


def _run_one_spatial(config: dict, task: dict) -> dict:
    output_dir = Path(task["output_dir"])
    workdir = Path(task["workdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    command = task_command(config, task)
    command[2] = SPATIAL_RUNNER
    _atomic_json(
        output_dir / "task.json",
        {**task, "command": command, "controller_config": str(config["_config_path"])},
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return {**task, "task_state": "completed_cached", "returncode": 0, "error": ""}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(config["_project_root"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=config["_project_root"],
            env=env,
            capture_output=True,
            text=True,
            timeout=int(config["execution"]["timeout_seconds"]),
        )
        (output_dir / "run.log").write_text(
            (completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        state = (
            "completed"
            if completed.returncode == 0 and summary_path.exists()
            else "technical_failure"
        )
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:]
        result = {
            **task,
            "task_state": state,
            "returncode": int(completed.returncode),
            "runtime_s_controller": time.perf_counter() - started,
            "error": "" if state == "completed" else (tail[0] if tail else "missing-summary"),
            "updated_at_utc": _utc_now(),
        }
    except Exception as exc:
        result = {
            **task,
            "task_state": "technical_failure",
            "returncode": math.nan,
            "runtime_s_controller": time.perf_counter() - started,
            "error": f"{type(exc).__name__}:{exc}",
            "updated_at_utc": _utc_now(),
        }
        (output_dir / "run.log").write_text(result["error"] + "\n", encoding="utf-8")
    _atomic_json(output_dir / "task_result.json", result)
    return result


def run(config: dict, limit: int | None) -> pd.DataFrame:
    if not config["execution"].get("allow_runs", False):
        raise RuntimeError("execution-disabled-in-controller-config")
    workers = int(config["execution"]["max_workers"])
    if workers > 12:
        raise ValueError("max-workers-exceeds-verified-limit-12")
    root = controller_root(config)
    manifest = pd.read_csv(root / "task_manifest.csv")
    planned = manifest[manifest["task_state"].eq("planned")]
    if limit is not None:
        planned = planned.head(limit)
    if planned.empty:
        return sync_tasks(config)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(planned))) as pool:
        futures = {
            pool.submit(_run_one_spatial, config, row._asdict()): row.task_id
            for row in planned.itertuples(index=False)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"completed {index}/{len(futures)} task={result['task_id']} "
                f"state={result['task_state']}",
                flush=True,
            )
    by_task = {row["task_id"]: row for row in results}
    updated = pd.DataFrame(
        [by_task.get(row["task_id"], row) for row in manifest.to_dict(orient="records")]
    )
    _atomic_csv(root / "task_manifest.csv", updated)
    return sync_tasks(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--limit", type=int, default=None)
    subparsers.add_parser("sync")
    subparsers.add_parser("status")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        manifest = build_manifest(config)
        print(manifest.groupby("lia_unique_id").size().to_string())
    elif args.command == "run":
        scores = run(config, args.limit)
        print(f"scored={len(scores)}")
    elif args.command == "sync":
        scores = sync_tasks(config)
        print(f"scored={len(scores)}")
    else:
        print_status(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
