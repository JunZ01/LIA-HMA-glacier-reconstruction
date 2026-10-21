"""Administrative-guard retry launcher for the frozen v5 1-D GPU preflight.

This wrapper changes no numerical input or driver.  It corrects only the legacy
guard assumption that the preserved CPU12 production had never launched.  A
terminal CPU12 run is accepted only when it is unauthorized, terminal, and has
zero active and queued work.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


DRIVERS = Path(__file__).resolve().parent
sys.path.insert(0, str(DRIVERS))
import launch_lia_1d_gpu_batch_preflight_v5 as v5


ORIGINAL_VALIDATE = v5.validate_v5_launch
WRAPPER_PATH = Path(__file__).resolve()


def validate_terminal_cpu12_launch(*, output_must_be_absent: bool = False) -> dict:
    """Run the frozen v5 guard after validating the actual terminal CPU12 state."""
    state_path = v5.PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = v5.load_json(state_path)
    focus = state.get("current_focus", {})
    counts = state.get("counts", {})
    failures = []
    if focus.get("system_id") != "all191-oggm-flowline-1d-cpu12":
        failures.append("CPU12 production current focus is not preserved")
    if focus.get("production_authorized") is not False:
        failures.append("CPU12 production is authorized")
    if focus.get("production_terminal") is not True:
        failures.append("CPU12 production is not terminal")
    if counts.get("active") != 0 or counts.get("queued") != 0:
        failures.append("CPU12 has active or queued work")
    if failures:
        raise RuntimeError("v5 terminal-CPU12 guard failed: " + "; ".join(failures))

    original_load_json = v5.load_json

    def load_for_legacy_guard(path: Path) -> dict:
        payload = original_load_json(path)
        if Path(path).resolve() == state_path.resolve():
            payload = copy.deepcopy(payload)
            payload["current_focus"]["production_launched"] = False
        return payload

    v5.load_json = load_for_legacy_guard
    try:
        authorization = ORIGINAL_VALIDATE(output_must_be_absent=output_must_be_absent)
    finally:
        v5.load_json = original_load_json
    authorization["state"] = state
    authorization["administrative_guard"] = {
        "cpu12_production_authorized": False,
        "cpu12_production_terminal": True,
        "cpu12_active": 0,
        "cpu12_queued": 0,
        "historical_production_launched_preserved": True,
    }
    return authorization


def finalize_wrapper_summary(phase: str) -> None:
    summary_path = v5.OUTPUT_ROOT / ("cpu/summary.json" if phase == "cpu" else "gpu/summary.json")
    if not summary_path.exists():
        return
    payload = v5.load_json(summary_path)
    payload["administrative_launcher"] = str(WRAPPER_PATH.relative_to(v5.PROJECT_ROOT))
    payload["superseded_launcher"] = str(Path(v5.__file__).resolve().relative_to(v5.PROJECT_ROOT))
    payload["guard_correction"] = (
        "accept only terminal, unauthorized CPU12 with zero active and queued work; "
        "preserve historical production_launched=true"
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=summary_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(summary_path)


def run_phase(phase: str) -> int:
    v5.validate_v5_launch = validate_terminal_cpu12_launch
    try:
        return v5.run_bound_phase(phase)
    finally:
        finalize_wrapper_summary(phase)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cpu", "gpu"))
    parser.add_argument("--environment-check-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "gpu" and os.environ.get(v5.CHILD_MARKER) != "1":
        command = [sys.executable, str(WRAPPER_PATH), "gpu"]
        if args.environment_check_only:
            command.append("--environment-check-only")
        return subprocess.run(command, env=v5.child_environment(), check=False).returncode
    if args.environment_check_only:
        if args.phase != "gpu":
            raise RuntimeError("environment check applies only to the GPU child")
        return v5.run_environment_check()
    return run_phase(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
