"""Register the active v10.240 CPU20 forward handoff as route v10.241."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
LAUNCH = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_240_cpu20_handoff_launch_v1/summary.json"
ROUTE_DIR = BASE / "lia_191_current_route_v10_241"


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
    if state["authoritative_route"]["version"] != "v10.240":
        raise RuntimeError("unexpected-authoritative-route")
    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    status_path = ROOT / "w237f/r/production_status_v10_240.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if launch["status"] != "active" or int(status["parent_pid"]) != int(launch["controller_pid"]):
        raise RuntimeError("cpu20-controller-not-active")
    if len(status["running"]) != 20:
        raise RuntimeError("cpu20-worker-count-not-reached")

    facts_text = f"""

## 541. The 530-cell historical-forward batch is active at CPU20 (2026-09-04)

- The v10.240 handoff passed the controller validator and completed at `2026-09-04T07:03:17.937300+00:00`. Source dispatcher PID `25116` is absent; its twelve in-flight workers were preserved and adopted rather than interrupted or duplicated.
- New controller PID `{status['parent_pid']}` launched eight additional unfinished members, reaching twenty simultaneous workers. The startup snapshot has `{status['completed']}` terminal results, `{len(status['running'])}` active members, `{status['pending']}` undispatched members, zero technical failures, zero scientific failures, and no pause reason.
- The same `w237f` output root, 530 fingerprints, member keys, forward physics, climate, parameters, timestep, annual checkpoint cadence, scientific gates, and post-run curve screen remain in force. No completed member or spin-up was recomputed.
- Startup memory use was 48%, with 33.15 GB available. Parent stderr is empty. The independent GPU branch was neither stopped nor modified.
- Launch evidence is `validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_forward_v10_240_cpu20_handoff_launch_v1/summary.json`, SHA256 `{sha256(LAUNCH)}`.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 541. The 530-cell historical-forward batch is active at CPU20", facts_text)
    decision_text = f"""

## 2026-09-04 v10.241 — CPU20 historical-forward handoff active

- Consume the one v10.240 handoff authority after source parent PID `25116` is stopped and controller PID `{status['parent_pid']}` adopts twelve in-flight members plus eight new members.
- Register twenty active workers, `{status['completed']}` completed cells, `{status['pending']}` pending cells, and zero technical/scientific failures at the first stable snapshot.
- Permit monitoring only. Do not duplicate launch, recompute spin-up or completed members, change the scientific contract, or control the independent GPU branch.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.241", decision_text)

    route = {
        "schema": "lia-191-current-route-v10-241",
        "status": "530-forward-cpu20-production-active",
        "population": 191, "accepted": 58, "remaining": 133,
        "active": len(status["running"]), "queue": int(status["pending"]),
        "launch_authorized": False, "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_240",
        "current_focus": "all191-oggm-flowline-1d-530-forward-cpu20-active",
        "parent_pid": int(status["parent_pid"]), "workers": 20,
        "completed_at_launch_snapshot": int(status["completed"]),
        "technical_failures": int(status["technical_failures"]),
        "scientific_failures": int(status["scientific_failures"]),
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "launch_evidence": str(LAUNCH.relative_to(ROOT)).replace("\\", "/"),
        "scientific_contract_unchanged": True, "spinup_recomputation_authorized": False,
        "duplicate_launch_authorized": False, "gpu_branch_control_authorized": False,
        "next_legal_action": f"monitor active CPU20 parent PID {status['parent_pid']} and workers; do not duplicate or terminate without a registered user decision",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)

    state["state_id"] = "2026-09-04-v10.241-cpu20-forward-active"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {
        "version": "v10.241", "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path), "supersedes": "v10.240",
    }
    state["counts"]["active"] = len(status["running"])
    state["counts"]["queued"] = int(status["pending"])
    state["current_focus"] = {
        "system_id": "all191-oggm-flowline-1d-530-forward-cpu20",
        "mainline_family": "all191-oggm-flowline-1d-historical-forward-v1",
        "state": "production-active-monitor-only", "full_historical_authorized": False,
        "parent_pid": int(status["parent_pid"]), "workers": 20,
        "completed": int(status["completed"]), "active": len(status["running"]),
        "queue": int(status["pending"]), "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "spinup_recomputation_authorized": False,
        "evidence": {"path": str(LAUNCH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(LAUNCH)},
    }
    state["next_action"] = {
        "type": "monitor-active-historical-forward-cpu20",
        "transaction_id": "lia-all191-oggm-flowline-1d-forward-v10.240-cpu20-handoff",
        "launch_authorized": False,
        "description": f"Monitor active parent PID {status['parent_pid']} and twenty workers; do not duplicate launch or terminate without registered user decision.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"route": route, "state_id": state["state_id"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
