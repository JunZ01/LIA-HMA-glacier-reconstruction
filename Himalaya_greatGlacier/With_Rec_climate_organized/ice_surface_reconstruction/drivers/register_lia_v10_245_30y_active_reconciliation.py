"""Reconcile active-monitoring authorization semantics after the v10.243 handoff."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
LAUNCH = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_launch_v1/summary.json"
STATUS = ROOT / "w237f/r/production_status_v10_243.json"
ROUTE_DIR = BASE / "lia_191_current_route_v10_245"


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
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
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
    if state["authoritative_route"]["version"] != "v10.244":
        raise RuntimeError("unexpected-authoritative-route")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    parent_pid = int(status["parent_pid"])
    if parent_pid != 73724 or not psutil.pid_exists(parent_pid):
        raise RuntimeError("active-controller-not-found")
    if psutil.pid_exists(69820):
        raise RuntimeError("source-controller-still-active")
    running = list(status["running"])
    legacy = sum(item.get("kind") == "legacy-inflight" for item in running)
    new30 = sum(item.get("kind") == "new30" for item in running)

    facts = f"""

## 545. Active CPU20 monitoring semantics are reconciled after the 30-year handoff (2026-09-04)

- The first post-handoff route incorrectly kept `full_historical_authorized=true` while also consuming launch authority. The controller validator rejected that administrative combination; no numerical process, checkpoint, forcing, parameter, threshold, or output was affected.
- Route v10.245 records the already-running batch as monitoring-only: both new-launch and full-historical-launch authorization are false. Controller PID `{parent_pid}` remains active; source dispatcher PID `69820` remains absent.
- At reconciliation, `{status['completed']}` members are terminal, `{len(running)}` are active, and `{status['pending']}` are undispatched. The active set contains `{legacy}` adopted legacy members and `{new30}` new 30-year-checkpoint members; no pause or scientific failure is present.
- New-route members `f0273` and `f0274` completed successfully with checkpoints only at 1470, 30-year boundaries, and 2000 while preserving annual diagnostics. This confirms that production dispatch has transitioned to the requested cadence.
- `f0192` is classified as an isolated technical failure. Its original annual-format checkpoint through year 1621, with fingerprint `0C19B1A4CE401A275E0789E6EBB8D3560652D09E4BA17840D02747891A4E3245`, remains intact and was not overwritten. The batch continues; no retry is launched by this reconciliation.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 545. Active CPU20 monitoring semantics are reconciled", facts)
    decision = f"""

## 2026-09-04 v10.245 — reconcile active monitoring after the 30-year handoff

- Supersede v10.244 solely to represent the already-running historical-forward batch with launch authority consumed, as required by the controller validator.
- Continue controller PID `{parent_pid}` without interruption. Preserve completed results, adopted workers, 30-year members, the recoverable `f0192` year-1621 checkpoint, and the independent GPU branch.
- Authorize monitoring and ordinary controller completion only. Launch no duplicate batch or technical retry.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.245", decision)

    route = {
        "schema": "lia-191-current-route-v10-245",
        "status": "cpu20-forward-30y-controller-active-monitoring-only",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": len(running),
        "queue": int(status["pending"]),
        "completed": int(status["completed"]),
        "technical_failures": int(status["technical_failures"]),
        "scientific_failures": int(status["scientific_failures"]),
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_244",
        "parent_pid": parent_pid,
        "source_parent_pid": 69820,
        "workers": 20,
        "adopted_legacy_workers_at_snapshot": legacy,
        "new_30y_workers_at_snapshot": new30,
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "full_state_checkpoint_interval_years_for_new_members": 30,
        "annual_diagnostics_retained": True,
        "spinup_recomputation_authorized": False,
        "duplicate_member_execution_authorized": False,
        "technical_retry_authorized": False,
        "gpu_branch_control_authorized": False,
        "launch_evidence": str(LAUNCH.relative_to(ROOT)).replace("\\", "/"),
        "next_legal_action": "monitor PID 73724 and preserve all terminal and recoverable checkpoint evidence",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)

    state["state_id"] = "2026-09-04-v10.245-cpu20-forward-30y-active-monitoring"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {
        "version": "v10.245",
        "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path),
        "supersedes": "v10.244",
    }
    state["counts"]["active"] = len(running)
    state["counts"]["queued"] = int(status["pending"])
    focus = state["current_focus"]
    focus.update({
        "system_id": "all191-oggm-flowline-1d-530-forward-cpu20-30y-active",
        "state": "active-monitoring-only",
        "full_historical_authorized": False,
        "parent_pid": parent_pid,
        "source_parent_pid": 69820,
        "workers": 20,
        "completed": int(status["completed"]),
        "active": len(running),
        "queue": int(status["pending"]),
        "technical_failures": int(status["technical_failures"]),
        "scientific_failures": int(status["scientific_failures"]),
        "full_state_checkpoint_interval_years_for_new_members": 30,
        "annual_diagnostics_retained": True,
        "spinup_recomputation_authorized": False,
        "evidence": {"path": str(LAUNCH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(LAUNCH)},
    })
    state["next_action"] = {
        "type": "monitor-active-forward-cpu20-30y",
        "transaction_id": "lia-all191-forward-v10.243-30y",
        "launch_authorized": False,
        "description": "Monitor PID 73724; do not duplicate, retry, or alter the active batch.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"state_id": state["state_id"], "route": route}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
