"""Register the user-authorized 30-year checkpoint handoff."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
TX_SUMMARY = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1/summary.json"
ROUTE_DIR = BASE / "lia_191_current_route_v10_243"


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
    if state["authoritative_route"]["version"] != "v10.242":
        raise RuntimeError("unexpected-authoritative-route")
    status = json.loads((ROOT / "w237f/r/production_status_v10_240.json").read_text(encoding="utf-8"))
    if int(status["parent_pid"]) != 69820 or len(status["running"]) != 20:
        raise RuntimeError("source-controller-not-active-at-20")
    facts = f"""

## 543. Thirty-year full-state forward continuation is authorized (2026-09-04)

- The user explicitly authorizes stopping CPU20 dispatcher PID `69820` and losslessly handing off its unfinished queue while preserving every completed and in-flight result. Members not yet started will save complete restart states every 30 years instead of every year; annual length, area, volume, and other scalar diagnostics remain retained.
- The OGGM model, geometry, initial equilibrium states, 1470-2000 forcing, parameters, timestep, scientific thresholds, curve screen, member identities, and 20-worker ceiling remain unchanged. Storage-cadence fingerprints change for new members; no completed member or spin-up is recomputed.
- One 60-year technical preflight must save at 1470, 1500, and 1530 and reproduce the 1500-to-1530 continuation within the existing restart tolerances before the source dispatcher may be stopped. A failed preflight leaves PID `69820` untouched.
- At authorization, `{status['completed']}` members are terminal, `{len(status['running'])}` are active, and `{status['pending']}` are pending. One terminal technical failure (`f0226`) is an isolated Windows diagnostics-file replacement denial and is not retried or counted as a scientific failure.
- Transaction evidence is `validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1/summary.json`, SHA256 `{sha256(TX_SUMMARY)}`.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 543. Thirty-year full-state forward continuation", facts)
    decision = """

## 2026-09-04 v10.243 — authorize 30-year full-state checkpoint handoff

- Bind the user's exact permission to stop PID `69820`, preserve its completed and in-flight members, and hand off only unfinished work to the 30-year full-state storage route.
- Retain annual scientific diagnostics, all physics and forcing, twenty workers, and the same output root. Change only restart-state and diagnostic-flush cadence for members not yet started.
- Require one 60-year checkpoint/resume preflight before controlling the source dispatcher. Do not retry the isolated existing `f0226` technical failure or interact with the independent GPU run.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.243", decision)
    route = {
        "schema": "lia-191-current-route-v10-243",
        "status": "cpu20-forward-30y-storage-handoff-authorized",
        "population": 191, "accepted": 58, "remaining": 133,
        "active": len(status["running"]), "queue": int(status["pending"]),
        "launch_authorized": True, "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_242",
        "current_focus": "all191-oggm-flowline-1d-530-forward-cpu20-30y-handoff",
        "source_parent_pid": 69820, "target_workers": 20,
        "completed_at_authorization": int(status["completed"]),
        "technical_failures": int(status["technical_failures"]),
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "full_state_checkpoint_interval_years_for_new_members": 30,
        "annual_diagnostics_retained": True, "spinup_recomputation_authorized": False,
        "duplicate_member_execution_authorized": False, "gpu_branch_control_authorized": False,
        "transaction_evidence": str(TX_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        "next_legal_action": "validate, run the one 60-year resume preflight, then perform the authorized lossless CPU20 dispatcher handoff",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)
    state["state_id"] = "2026-09-04-v10.243-cpu20-forward-30y-handoff-authorized"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {"version": "v10.243",
        "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path), "supersedes": "v10.242"}
    state["counts"]["active"] = len(status["running"])
    state["counts"]["queued"] = int(status["pending"])
    state["current_focus"] = {
        "system_id": "all191-oggm-flowline-1d-530-forward-cpu20-30y-handoff",
        "mainline_family": "all191-oggm-flowline-1d-historical-forward-v1",
        "state": "storage-handoff-authorized", "full_historical_authorized": True,
        "source_parent_pid": 69820, "workers": 20,
        "completed": int(status["completed"]), "active": len(status["running"]),
        "queue": int(status["pending"]), "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "spinup_recomputation_authorized": False,
        "evidence": {"path": str(TX_SUMMARY.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(TX_SUMMARY)},
    }
    state["next_action"] = {
        "type": "historical-forward-cpu20-30y-storage-handoff",
        "transaction_id": "lia-all191-forward-v10.243-30y",
        "launch_authorized": True,
        "description": "Run one resume preflight, then stop PID 69820 and hand off unfinished work while preserving all completed and in-flight members.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"state_id": state["state_id"], "route": route}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
