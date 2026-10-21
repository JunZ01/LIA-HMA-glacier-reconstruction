"""Run the full Stage 0-A geometry and dual-product fs=0 inversions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06"
    / "stage0_bed_full_farinotti_v0"
)
STATUS_PATH = RUN_ROOT / "overnight_status.json"
LOG_PATH = RUN_ROOT / "overnight_full_v0.log"
GEOMETRY_SUMMARY_PATH = RUN_ROOT / "geometry_summary.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as target:
        json.dump(status, target, indent=2, ensure_ascii=False)


def full_geometry_is_ready() -> bool:
    if not GEOMETRY_SUMMARY_PATH.exists():
        return False
    with open(GEOMETRY_SUMMARY_PATH, encoding="utf-8") as source:
        summary = json.load(source)
    return (
        int(summary.get("member_count", 0)) == 1205
        and int(summary.get("geometry_ready_count", 0)) == 1205
        and int(summary.get("geometry_failed_count", -1)) == 0
    )


def main() -> None:
    os.environ["GDAL_DATA"] = (
        r"D:\Anaconda\envs\oggm_env2\Library\share\gdal"
    )
    os.environ["PROJ_LIB"] = (
        r"D:\Anaconda\envs\oggm_env2\Library\share\proj"
    )
    os.environ["PATH"] = os.pathsep.join(
        [
            r"D:\Anaconda\envs\oggm_env2",
            r"D:\Anaconda\envs\oggm_env2\Scripts",
            r"D:\Anaconda\envs\oggm_env2\Library\bin",
            os.environ.get("PATH", ""),
        ]
    )

    steps = [
        {
            "name": "full-geometry",
            "command": [
                sys.executable,
                "src/stage0_bed/build_modern_gdirs_pilot.py",
                "--config",
                "config/stage0a_bed_full_farinotti_v0.json",
                "--phase",
                "geometry",
            ],
        },
        {
            "name": "farinotti-climate-mb-inversion",
            "command": [
                sys.executable,
                "src/stage0_bed/run_inversion_pilot.py",
                "--config",
                "config/stage0a_bed_full_farinotti_v0.json",
                "--phase",
                "all",
            ],
        },
        {
            "name": "millan-inversion",
            "command": [
                sys.executable,
                "src/stage0_bed/run_inversion_pilot.py",
                "--config",
                "config/stage0a_bed_full_millan_v0.json",
                "--phase",
                "inversion",
            ],
        },
    ]
    if full_geometry_is_ready():
        steps = [step for step in steps if step["name"] != "full-geometry"]
        reused_steps = ["full-geometry"]
    else:
        reused_steps = []

    status = {
        "run_id": "stage0a-full-245-dual-product-v0",
        "pid": os.getpid(),
        "started_utc": now_utc(),
        "status": "running",
        "current_step": None,
        "completed_steps": reused_steps,
        "log_path": str(LOG_PATH),
    }
    write_status(status)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8", buffering=1) as log:
        for step in steps:
            status["current_step"] = step["name"]
            status["step_started_utc"] = now_utc()
            write_status(status)
            log.write(f"\n[{now_utc()}] START {step['name']}\n")
            try:
                subprocess.run(
                    step["command"],
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except Exception as exc:
                status.update(
                    {
                        "status": "failed",
                        "failed_step": step["name"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_utc": now_utc(),
                    }
                )
                write_status(status)
                log.write(f"[{now_utc()}] FAILED {step['name']}: {exc}\n")
                raise
            status["completed_steps"].append(step["name"])
            status["step_finished_utc"] = now_utc()
            write_status(status)
            log.write(f"[{now_utc()}] COMPLETE {step['name']}\n")

    status.update(
        {
            "status": "complete",
            "current_step": None,
            "finished_utc": now_utc(),
        }
    )
    write_status(status)


if __name__ == "__main__":
    main()
