"""Consume the v10.239 launch authority and register the active GPU branch."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
STATE_PATH = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
FACTS_PATH = PROJECT_ROOT / "LIA_PROJECT_FACTS.md"
DECISION_LOG = PROJECT_ROOT / "LIA_DECISION_LOG.md"
TRANSACTION_PATH = RESULTS / "lia_191_gpu_spinup_v10_239_transaction_v1/transaction.json"
AUTHORIZATION_PATH = RESULTS / "lia_191_gpu_spinup_v10_239_authorization_v1/summary.json"
LAUNCH_DIR = RESULTS / "lia_191_gpu_spinup_v10_239_launch_v1"
LAUNCH_SUMMARY = LAUNCH_DIR / "summary.json"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_240"
ROUTE_PATH = ROUTE_DIR / "summary.json"
RUN_MANIFEST = PROJECT_ROOT / "wg239/run_manifest.json"
STATUS_PATH = PROJECT_ROOT / "wg239/status.json"
LAUNCHER_PID = 32524
GPU_PID = 29136


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("state_id") != "2026-09-04-v10.239-cpu12-forward-active-gpu191-spinup-authorized":
        raise RuntimeError(f"unexpected source state: {state.get('state_id')}")
    branch = state["research_branches"]["lia_191_gpu_spinup_v10_239"]
    if branch.get("launch_authorized") is not True:
        raise RuntimeError("GPU launch authority is not open")
    if not RUN_MANIFEST.is_file() or not STATUS_PATH.is_file():
        raise RuntimeError("GPU run manifest or status is missing")
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    transaction = json.loads(TRANSACTION_PATH.read_text(encoding="utf-8"))
    if run_manifest.get("transaction_sha256") != sha256(TRANSACTION_PATH):
        raise RuntimeError("running process transaction hash mismatch")
    if status.get("process_id") != GPU_PID or status.get("technical_failures") != 0:
        raise RuntimeError("GPU status does not identify the expected healthy child")

    launch = {
        "schema": "lia-191-gpu-spinup-launch-evidence-v1",
        "status": "active",
        "recorded_at_utc": now(),
        "transaction_id": transaction["transaction_id"],
        "transaction_path": relative(TRANSACTION_PATH),
        "transaction_sha256": sha256(TRANSACTION_PATH),
        "transaction_fingerprint_sha256": transaction["transaction_fingerprint_sha256"],
        "authorization_path": relative(AUTHORIZATION_PATH),
        "authorization_sha256": sha256(AUTHORIZATION_PATH),
        "launcher_pid": LAUNCHER_PID,
        "gpu_child_pid": GPU_PID,
        "command": "D:/Anaconda/envs/SS4/python.exe run_lia_191_gpu_spinup_v10_239.py --batch-size 512",
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/wg239",
        "run_manifest_path": relative(RUN_MANIFEST),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "launch_status_snapshot": status,
        "launch_status_snapshot_sha256": sha256(STATUS_PATH),
        "gpu_snapshot": {
            "memory_used_mib": 4258,
            "memory_free_mib": 11705,
            "utilization_percent": 98,
        },
        "cpu_mainline_modified_or_controlled": False,
        "accepted_count_effect": 0,
        "single_launch_authority_consumed": True,
    }
    write_new(LAUNCH_SUMMARY, launch)

    route = {
        "schema": "lia-191-current-route-v10-240",
        "status": "cpu12-forward-active-independent-gpu-full191-spinup-active",
        "population": state["population"],
        "accepted": state["counts"]["accepted"],
        "remaining": state["counts"]["remaining"],
        "active": state["counts"]["active"],
        "queue": state["counts"]["queued"],
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_239",
        "current_focus": state["current_focus"]["system_id"],
        "parent_pid": state["current_focus"]["parent_pid"],
        "workers": state["current_focus"]["workers"],
        "output_root": state["current_focus"]["output_root"],
        "cpu_mainline_preserved": True,
        "gpu_spinup_launch_authorized": False,
        "gpu_spinup_state": "active-monitor-only",
        "gpu_spinup_launcher_pid": LAUNCHER_PID,
        "gpu_spinup_child_pid": GPU_PID,
        "gpu_spinup_system_count": 191,
        "gpu_spinup_cell_count": 1337,
        "gpu_spinup_output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/wg239",
        "gpu_spinup_launch_evidence_path": relative(LAUNCH_SUMMARY),
        "gpu_spinup_launch_evidence_sha256": sha256(LAUNCH_SUMMARY),
        "gpu_cpu_comparison_authorized": False,
        "planform_2d_or_igm_allowed": False,
        "next_legal_action": "monitor both independent active branches; never duplicate launch or let GPU control CPU",
    }
    write_new(ROUTE_PATH, route)

    updated = deepcopy(state)
    updated["state_id"] = "2026-09-04-v10.240-cpu12-forward-active-gpu191-spinup-active"
    updated["updated_at"] = now()
    updated["scientific_contract"]["sha256"] = sha256(FACTS_PATH)
    updated["authoritative_route"] = {
        "version": "v10.240",
        "path": relative(ROUTE_PATH),
        "sha256": sha256(ROUTE_PATH),
        "supersedes": "v10.239",
    }
    active_branch = updated["research_branches"]["lia_191_gpu_spinup_v10_239"]
    active_branch.update({
        "status": "active-monitor-only",
        "launch_authorized": False,
        "launcher_pid": LAUNCHER_PID,
        "gpu_child_pid": GPU_PID,
        "batch_count": 3,
        "batch_size": 512,
        "completed_at_launch_snapshot": status["completed"],
        "pending_at_launch_snapshot": status["pending"],
        "technical_failures_at_launch_snapshot": status["technical_failures"],
        "evidence": {
            "path": relative(LAUNCH_SUMMARY),
            "sha256": sha256(LAUNCH_SUMMARY),
        },
        "next_legal_action": "monitor only; on termination classify complete, technical failure, or scientific outcomes",
    })
    updated["concurrent_gpu_spinup"].update({
        "state": "active-monitor-only",
        "launch_authorized": False,
        "launcher_pid": LAUNCHER_PID,
        "gpu_child_pid": GPU_PID,
        "completed_at_launch_snapshot": status["completed"],
        "pending_at_launch_snapshot": status["pending"],
        "technical_failures_at_launch_snapshot": status["technical_failures"],
        "evidence": {
            "path": relative(LAUNCH_SUMMARY),
            "sha256": sha256(LAUNCH_SUMMARY),
        },
    })
    atomic_json(STATE_PATH, updated)

    with DECISION_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "\n## 2026-09-04 v10.240: full-191 GPU spin-up launch consumed\n\n"
            "- Launcher PID: `32524`; SS4 GPU child PID: `29136`.\n"
            "- Launch snapshot: batch 0/3, 512 active cells, 1,337 pending, zero technical failures.\n"
            "- GPU snapshot: 4,258 MiB used, 98% utilization.\n"
            "- CPU12 mainline is unchanged and independent; duplicate GPU launch is prohibited.\n"
        )
    print(json.dumps({
        "status": "launch-consumed-active",
        "state_id": updated["state_id"],
        "launch_summary_sha256": sha256(LAUNCH_SUMMARY),
        "route_sha256": sha256(ROUTE_PATH),
        "gpu_child_pid": GPU_PID,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
