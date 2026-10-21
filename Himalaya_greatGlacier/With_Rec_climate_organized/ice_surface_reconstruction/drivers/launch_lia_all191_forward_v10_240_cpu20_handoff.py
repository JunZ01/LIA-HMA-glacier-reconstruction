"""Perform the one authorized lossless handoff from v10.237 CPU12 to v10.240 CPU20."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
TX = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_240_cpu20_handoff_transaction_v1"
EVIDENCE = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_240_cpu20_handoff_launch_v1"
OUTPUT_ROOT = ROOT / "w237f"
CONTROLLER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_flowline_forward_cpu20_handoff_controller_v10_240"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    if EVIDENCE.exists():
        raise FileExistsError("v10.240 handoff launch evidence already exists")
    validator = subprocess.run(
        [sys.executable, str(ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py")],
        cwd=ROOT, text=True, capture_output=True,
    )
    if validator.returncode != 0:
        raise RuntimeError(f"validator-failed:{validator.stdout}{validator.stderr}")
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    if state["authoritative_route"]["version"] != "v10.240" or not state["next_action"]["launch_authorized"]:
        raise RuntimeError("v10.240-handoff-not-authorized-in-current-state")
    manifest = json.loads((TX / "transaction_manifest.json").read_text(encoding="utf-8"))
    old_pid = int(manifest["handoff"]["source_parent_pid"])
    old_parent = psutil.Process(old_pid)
    old_parent.suspend()
    try:
        time.sleep(1.0)
        status = json.loads((OUTPUT_ROOT / "r/production_status.json").read_text(encoding="utf-8"))
        if int(status["parent_pid"]) != old_pid:
            raise RuntimeError("source-status-parent-mismatch")
        inherited = []
        for item in status["running"]:
            pid = int(item["pid"])
            inherited.append({
                "pid": pid, "member_key": item["member_key"],
                "lia_unique_id": int(item["lia_unique_id"]),
                "alive_at_handoff": psutil.pid_exists(pid), "started_at_utc": None,
            })
        EVIDENCE.mkdir(parents=True, exist_ok=False)
        snapshot = {
            "schema": "lia-all191-forward-cpu20-handoff-snapshot-v10.240",
            "created_at_utc": now(), "source_parent_pid": old_pid,
            "source_status": status, "inherited_workers": inherited,
            "terminal_results_before_handoff": len(list((OUTPUT_ROOT / "m").glob("f*/c/result.json"))),
        }
        snapshot_path = EVIDENCE / "handoff_snapshot.json"
        atomic_json(snapshot_path, snapshot)
        old_parent.terminate()
        try:
            old_parent.wait(timeout=10)
        except psutil.TimeoutExpired as exc:
            old_parent.resume()
            raise RuntimeError("source-parent-did-not-terminate") from exc
    except Exception:
        if old_parent.is_running():
            old_parent.resume()
        raise

    stdout_path = OUTPUT_ROOT / "l/production_v10_240_parent.out.log"
    stderr_path = OUTPUT_ROOT / "l/production_v10_240_parent.err.log"
    stdout_handle = stdout_path.open("x", encoding="utf-8")
    stderr_handle = stderr_path.open("x", encoding="utf-8")
    command = [
        manifest["runtime"]["python_executable"], "-m", CONTROLLER_MODULE,
        "--transaction-dir", str(TX.resolve()), "--output-root", str(OUTPUT_ROOT.resolve()),
        "--handoff-snapshot", str(snapshot_path.resolve()),
    ]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    controller = subprocess.Popen(
        command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle, creationflags=flags,
    )
    stdout_handle.close()
    stderr_handle.close()
    time.sleep(5.0)
    if controller.poll() is not None:
        raise RuntimeError(f"v10.240-controller-exited-early:{controller.returncode}")
    vm = psutil.virtual_memory()
    summary = {
        "schema": "lia-all191-forward-cpu20-handoff-launch-v10.240",
        "status": "active", "launched_at_utc": now(),
        "source_parent_pid": old_pid, "source_parent_alive": psutil.pid_exists(old_pid),
        "controller_pid": controller.pid, "target_workers": 20,
        "inherited_workers": inherited, "output_root": str(OUTPUT_ROOT.resolve()),
        "command": command, "stdout": str(stdout_path.resolve()), "stderr": str(stderr_path.resolve()),
        "memory_used_fraction": vm.percent / 100.0,
        "available_memory_gb": vm.available / (1024**3),
        "scientific_contract_unchanged": True, "spinup_recomputed": False,
        "duplicate_member_execution_allowed": False,
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
