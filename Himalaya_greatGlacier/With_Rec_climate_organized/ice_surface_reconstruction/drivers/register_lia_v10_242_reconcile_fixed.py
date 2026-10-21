"""Reconcile concurrent CPU20 and GPU records without controlling either run."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE_DIR = BASE / "lia_191_current_route_v10_242"
GPU_MANIFEST = ROOT / "wg239/run_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def append_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8")
    if marker in current:
        raise RuntimeError(f"marker-already-present:{marker}")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if not current.endswith("\n"):
            handle.write("\n")
        handle.write(text)


def main() -> None:
    if ROUTE_DIR.exists():
        raise FileExistsError(ROUTE_DIR)
    state_path = ROOT / "LIA_CURRENT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["authoritative_route"]["version"] != "v10.241":
        raise RuntimeError("unexpected-authoritative-route")
    cpu = json.loads((ROOT / "w237f/r/production_status_v10_240.json").read_text(encoding="utf-8"))
    gpu = json.loads(GPU_MANIFEST.read_text(encoding="utf-8"))
    if len(cpu["running"]) != 20 or gpu["status"] != "running":
        raise RuntimeError("concurrent-runs-not-active")
    facts = f"""

## 542. Concurrent CPU20 and GPU control records are reconciled (2026-09-04)

- A read-only validator check found that the independent GPU task appended its active-run fact after the CPU20 state snapshot, making only the canonical facts hash stale. Neither numerical process, result, checkpoint, parameter, nor threshold was affected.
- Route v10.242 records CPU20 parent PID `{cpu['parent_pid']}` with `{cpu['completed']}` completed, `{len(cpu['running'])}` active, and `{cpu['pending']}` pending forward members, alongside independent GPU PID `{gpu['process_id']}` from immutable run-manifest SHA256 `{sha256(GPU_MANIFEST)}`.
- This reconciliation launches, stops, restarts, or modifies no process. Both runs remain monitoring-only and duplicate launch authority is false.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 542. Concurrent CPU20 and GPU control records", facts)
    decision = f"""

## 2026-09-04 v10.242 — reconcile concurrent CPU20/GPU active records

- Refresh the canonical facts hash after the independent GPU process appended its active-run evidence concurrently with CPU20 registration.
- Preserve CPU20 parent PID `{cpu['parent_pid']}` and GPU PID `{gpu['process_id']}` unchanged; perform no launch, stop, restart, model change, or accepted-count change.
- Keep both activities monitoring-only and consume all duplicate-launch authority.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.242", decision)
    route = {
        "schema": "lia-191-current-route-v10-242",
        "status": "cpu20-forward-and-independent-gpu-spinup-active",
        "population": 191, "accepted": 58, "remaining": 133,
        "active": len(cpu["running"]), "queue": int(cpu["pending"]),
        "launch_authorized": False, "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_241",
        "current_focus": "all191-oggm-flowline-1d-530-forward-cpu20-active",
        "cpu_parent_pid": int(cpu["parent_pid"]), "cpu_workers": 20,
        "cpu_completed": int(cpu["completed"]),
        "cpu_output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "gpu_pid": int(gpu["process_id"]), "gpu_state": "active",
        "gpu_output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/wg239",
        "duplicate_launch_authorized": False,
        "next_legal_action": "monitor both active processes; do not duplicate, restart, or terminate without a registered user decision",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)
    state["state_id"] = "2026-09-04-v10.242-cpu20-forward-gpu-spinup-active"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {
        "version": "v10.242", "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path), "supersedes": "v10.241",
    }
    state["counts"]["active"] = len(cpu["running"])
    state["counts"]["queued"] = int(cpu["pending"])
    state["current_focus"].update({
        "state": "production-active-monitor-only", "parent_pid": int(cpu["parent_pid"]),
        "workers": 20, "completed": int(cpu["completed"]), "active": len(cpu["running"]),
        "queue": int(cpu["pending"]), "full_historical_authorized": False,
    })
    branch = state["research_branches"]["lia_191_gpu_spinup_v10_239"]
    branch.update({
        "status": "production-active-monitor-only", "launch_authorized": False,
        "gpu_pid": int(gpu["process_id"]),
        "evidence": {"path": "wg239/run_manifest.json", "sha256": sha256(GPU_MANIFEST)},
        "next_legal_action": "monitor active GPU PID; no duplicate launch or CPU interaction",
    })
    state["concurrent_gpu_spinup"].update({
        "state": "production-active-monitor-only", "launch_authorized": False,
        "gpu_pid": int(gpu["process_id"]),
    })
    state["next_action"] = {
        "type": "monitor-active-cpu20-forward-and-gpu-spinup",
        "transaction_id": "lia-v10.242-concurrent-state-reconciliation",
        "launch_authorized": False,
        "description": "Monitor CPU20 forward and independent GPU spin-up; no duplicate launch or process control.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"state_id": state["state_id"], "route": route}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
