"""Run the 30-year checkpoint preflight and hand off the unfinished CPU20 queue."""

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
TX = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1"
EVIDENCE = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_launch_v1"
OUTPUT_ROOT = ROOT / "w237f"
PREFLIGHT_ROOT = ROOT / "w243p"
WORKER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_flowline_forward_v10_243_30y"
)
CONTROLLER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_flowline_forward_cpu20_controller_v10_243"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    if EVIDENCE.exists() or PREFLIGHT_ROOT.exists():
        raise FileExistsError("v10.243 launch evidence or preflight root exists")
    validator = subprocess.run(
        [sys.executable, str(ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py")],
        cwd=ROOT, text=True, capture_output=True,
    )
    if validator.returncode != 0:
        raise RuntimeError(f"validator-failed:{validator.stdout}{validator.stderr}")
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    if state["authoritative_route"]["version"] != "v10.243" or not state["next_action"]["launch_authorized"]:
        raise RuntimeError("v10.243-not-authorized")
    manifest = json.loads((TX / "transaction_manifest.json").read_text(encoding="utf-8"))
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({k: str(v) for k, v in manifest["runtime"]["thread_environment"].items()})
    env["PYTHONPATH"] = str(ROOT)
    fingerprint = manifest["preflight"]["fingerprints"][0]
    preflight_out = EVIDENCE / "preflight.out.log"
    preflight_err = EVIDENCE / "preflight.err.log"
    preflight_command = [
        manifest["runtime"]["python_executable"], "-m", WORKER_MODULE,
        "--transaction-dir", str(TX.resolve()), "--fingerprint", fingerprint,
        "--output-root", str(PREFLIGHT_ROOT.resolve()), "--mode", "preflight",
    ]
    with preflight_out.open("x", encoding="utf-8") as out, preflight_err.open("x", encoding="utf-8") as err:
        preflight_process = subprocess.run(preflight_command, cwd=ROOT, env=env, stdout=out, stderr=err)
    preflight_result_path = PREFLIGHT_ROOT / "p/p0000/c/result.json"
    preflight_result = json.loads(preflight_result_path.read_text(encoding="utf-8"))
    if preflight_process.returncode != 0 or preflight_result.get("status") != "preflight-pass" or not preflight_result.get("resume_check", {}).get("passed"):
        atomic_json(EVIDENCE / "summary.json", {
            "schema": "lia-all191-forward-30y-launch-v10.243", "status": "preflight-failed",
            "preflight_returncode": preflight_process.returncode, "preflight_result": preflight_result,
            "source_controller_controlled": False, "production_started": False,
        })
        raise RuntimeError("30y-preflight-failed")

    old_pid = int(manifest["source"]["active_controller_pid"])
    old_parent = psutil.Process(old_pid)
    old_parent.suspend()
    try:
        time.sleep(1.0)
        status = json.loads((OUTPUT_ROOT / "r/production_status_v10_240.json").read_text(encoding="utf-8"))
        if int(status["parent_pid"]) != old_pid or len(status["running"]) != 20:
            raise RuntimeError("source-controller-state-mismatch")
        inherited = [{"pid": int(item["pid"]), "member_key": item["member_key"],
                      "kind": item.get("kind"), "alive_at_handoff": psutil.pid_exists(int(item["pid"]))}
                     for item in status["running"]]
        snapshot = {
            "schema": "lia-all191-forward-30y-handoff-snapshot-v10.243",
            "created_at_utc": now(), "source_parent_pid": old_pid,
            "source_status": status, "inherited_workers": inherited,
            "terminal_results_before_handoff": len(list((OUTPUT_ROOT / "m").glob("f*/c/result.json"))),
            "new_full_state_checkpoint_interval_years": 30,
        }
        snapshot_path = EVIDENCE / "handoff_snapshot.json"
        atomic_json(snapshot_path, snapshot)
        old_parent.terminate()
        try:
            old_parent.wait(timeout=10)
        except psutil.TimeoutExpired as exc:
            old_parent.resume()
            raise RuntimeError("source-controller-did-not-terminate") from exc
    except Exception:
        if old_parent.is_running():
            old_parent.resume()
        raise

    stdout_path = OUTPUT_ROOT / "l/production_v10_243_parent.out.log"
    stderr_path = OUTPUT_ROOT / "l/production_v10_243_parent.err.log"
    stdout_handle = stdout_path.open("x", encoding="utf-8")
    stderr_handle = stderr_path.open("x", encoding="utf-8")
    command = [
        manifest["runtime"]["python_executable"], "-m", CONTROLLER_MODULE,
        "--transaction-dir", str(TX.resolve()), "--output-root", str(OUTPUT_ROOT.resolve()),
        "--handoff-snapshot", str(snapshot_path.resolve()),
    ]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    controller = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout_handle,
                                  stderr=stderr_handle, creationflags=flags)
    stdout_handle.close(); stderr_handle.close()
    time.sleep(5.0)
    if controller.poll() is not None:
        raise RuntimeError(f"v10.243-controller-exited-early:{controller.returncode}")
    live_status = json.loads((OUTPUT_ROOT / "r/production_status_v10_243.json").read_text(encoding="utf-8"))
    summary = {
        "schema": "lia-all191-forward-30y-launch-v10.243", "status": "active",
        "launched_at_utc": now(), "preflight_returncode": preflight_process.returncode,
        "preflight_result": preflight_result, "source_parent_pid": old_pid,
        "source_parent_alive": psutil.pid_exists(old_pid), "controller_pid": controller.pid,
        "target_workers": 20, "inherited_workers": inherited,
        "full_state_checkpoint_interval_years_for_new_members": 30,
        "annual_diagnostics_retained": True, "output_root": str(OUTPUT_ROOT.resolve()),
        "preflight_root": str(PREFLIGHT_ROOT.resolve()), "command": command,
        "live_status": live_status, "scientific_contract_unchanged": True,
        "spinup_recomputed": False, "duplicate_member_execution_allowed": False,
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
