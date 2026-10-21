"""Register the completed lossless handoff to the v10.243 30-year controller."""

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
LAUNCH_SUMMARY = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_launch_v1/summary.json"
TX_SUMMARY = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1/summary.json"
STATUS_PATH = ROOT / "w237f/r/production_status_v10_243.json"
ROUTE_DIR = BASE / "lia_191_current_route_v10_244"


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
    if state["authoritative_route"]["version"] != "v10.243":
        raise RuntimeError("unexpected-authoritative-route")
    launch = json.loads(LAUNCH_SUMMARY.read_text(encoding="utf-8"))
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    parent_pid = int(status["parent_pid"])
    if parent_pid != 73724 or not psutil.pid_exists(parent_pid):
        raise RuntimeError("target-controller-not-active")
    if psutil.pid_exists(69820):
        raise RuntimeError("source-controller-still-active")
    if launch["preflight_result"]["status"] != "preflight-pass":
        raise RuntimeError("preflight-not-passed")
    running = list(status["running"])
    legacy_running = sum(item.get("kind") == "legacy-inflight" for item in running)
    new_running = sum(item.get("kind") == "new30" for item in running)
    preflight = launch["preflight_result"]
    resume = preflight["resume_check"]

    facts = f"""

## 544. The CPU20 historical-forward batch is losslessly handed off to 30-year full-state checkpoints (2026-09-04)

- The one authorized 60-year preflight passed. It wrote full restart states at years 1470, 1500, and 1530; the saved 1500-to-1530 continuation matched the uninterrupted run with volume relative difference `{resume['volume_relative_difference']}`, area relative difference `{resume['area_relative_difference']}`, maximum thickness difference `{resume['maximum_thickness_absolute_difference_m']} m`, and length-grid difference `{resume['length_grid_difference']}`.
- Only after that pass, source dispatcher PID `69820` was stopped and confirmed absent. Replacement controller PID `{parent_pid}` is active with twenty worker slots under the unchanged output root `E:/3spyderProject/1_LIA_himalaya_oggm/w237f`.
- At this post-handoff snapshot, `{status['completed']}` members are terminal, `{len(running)}` are active, and `{status['pending']}` remain undispatched. The controller adopted `{legacy_running}` pre-existing in-flight members without stopping or duplicating them; `{new_running}` newly dispatched members are currently using the 30-year full-state route. As adopted slots finish, every newly dispatched member uses that route automatically.
- Annual scalar diagnostics remain retained. Completed members and spin-up states were not recomputed. The pre-existing isolated technical failure `f0226` remains recorded and was not retried; no scientific failure or pause is present at the snapshot.
- Immutable launch evidence is `validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_launch_v1/summary.json`, SHA256 `{sha256(LAUNCH_SUMMARY)}`. The independent GPU branch was not controlled or modified.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 544. The CPU20 historical-forward batch is losslessly handed off", facts)
    decision = f"""

## 2026-09-04 v10.244 — consume handoff authority and monitor the 30-year route

- Record the successful v10.243 preflight and the lossless dispatcher change from PID `69820` to PID `{parent_pid}`.
- Preserve the adopted in-flight members under their original annual storage fingerprints; dispatch all not-yet-started members with 30-year full-state checkpoints and retained annual scalar diagnostics.
- Consume the one handoff authorization. Only monitoring and ordinary completion are authorized; no duplicate launch, spin-up recomputation, retry of `f0226`, scientific change, or GPU control is authorized.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.244", decision)

    route = {
        "schema": "lia-191-current-route-v10-244",
        "status": "cpu20-forward-30y-controller-active",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": len(running),
        "queue": int(status["pending"]),
        "completed": int(status["completed"]),
        "technical_failures": int(status["technical_failures"]),
        "scientific_failures": int(status["scientific_failures"]),
        "launch_authorized": False,
        "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_243",
        "current_focus": "all191-oggm-flowline-1d-530-forward-cpu20-30y-active",
        "parent_pid": parent_pid,
        "source_parent_pid": 69820,
        "workers": 20,
        "adopted_legacy_workers_at_snapshot": legacy_running,
        "new_30y_workers_at_snapshot": new_running,
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "full_state_checkpoint_interval_years_for_new_members": 30,
        "annual_diagnostics_retained": True,
        "spinup_recomputation_authorized": False,
        "duplicate_member_execution_authorized": False,
        "gpu_branch_control_authorized": False,
        "transaction_evidence": str(TX_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        "launch_evidence": str(LAUNCH_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        "next_legal_action": "monitor PID 73724; newly dispatched members automatically use 30-year full-state checkpoints",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)

    state["state_id"] = "2026-09-04-v10.244-cpu20-forward-30y-active"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {
        "version": "v10.244",
        "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path),
        "supersedes": "v10.243",
    }
    state["counts"]["active"] = len(running)
    state["counts"]["queued"] = int(status["pending"])
    state["current_focus"] = {
        "system_id": "all191-oggm-flowline-1d-530-forward-cpu20-30y-active",
        "mainline_family": "all191-oggm-flowline-1d-historical-forward-v1",
        "state": "active-monitoring-only",
        "full_historical_authorized": True,
        "parent_pid": parent_pid,
        "source_parent_pid": 69820,
        "workers": 20,
        "completed": int(status["completed"]),
        "active": len(running),
        "queue": int(status["pending"]),
        "technical_failures": int(status["technical_failures"]),
        "scientific_failures": int(status["scientific_failures"]),
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "full_state_checkpoint_interval_years_for_new_members": 30,
        "annual_diagnostics_retained": True,
        "spinup_recomputation_authorized": False,
        "evidence": {
            "path": str(LAUNCH_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(LAUNCH_SUMMARY),
        },
    }
    state["next_action"] = {
        "type": "monitor-active-historical-forward-cpu20-30y",
        "transaction_id": "lia-all191-forward-v10.243-30y",
        "launch_authorized": False,
        "description": "Monitor PID 73724; adopted members finish unchanged and all newly dispatched members use 30-year full-state checkpoints.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"state_id": state["state_id"], "route": route}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
