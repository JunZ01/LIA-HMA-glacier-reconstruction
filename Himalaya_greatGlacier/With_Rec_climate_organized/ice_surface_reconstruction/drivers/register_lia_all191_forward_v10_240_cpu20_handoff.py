"""Register the authorized v10.240 CPU20 handoff before launch."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
TX_SUMMARY = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_240_cpu20_handoff_transaction_v1/summary.json"
ROUTE_DIR = BASE / "lia_191_current_route_v10_240"


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
    if state["authoritative_route"]["version"] != "v10.239":
        raise RuntimeError("unexpected-authoritative-route")
    status = json.loads((ROOT / "w237f/r/production_status.json").read_text(encoding="utf-8"))
    if int(status["parent_pid"]) != 25116 or len(status["running"]) != 12:
        raise RuntimeError("source-controller-not-in-expected-active-state")

    facts_text = f"""

## 540. Lossless CPU12-to-CPU20 historical-forward handoff is authorized (2026-09-04)

- The user explicitly instructs `开始` after asking to raise the active historical-forward production from twelve to twenty CPU workers. This authorizes one controlled handoff of the existing v10.237 forward batch; it does not authorize a second batch.
- The 530-member matrix, OGGM one-dimensional model, 1470-2000 forcing, parameters, fixed timestep, annual checkpoints, output root, member keys, fingerprints, curve screen, failure rules, and all scientific thresholds remain unchanged. No spin-up or completed forward member may be recomputed.
- The machine has 20 physical cores, 28 logical processors, 63.8 GB physical memory, and 34.4 GB free memory at review. Each numerical-library thread remains fixed to one. The target is twenty simultaneous member processes, not twenty internal threads per glacier.
- Because the v10.237 controller has a fixed twelve-worker limit and no live resize interface, route v10.240 authorizes a lossless controller handoff: freeze the source dispatcher, snapshot its terminal and in-flight members, stop only source parent PID `25116`, adopt its still-running children, and launch only enough additional members to reach twenty total. Existing results and checkpoints stay under `E:/3spyderProject/1_LIA_himalaya_oggm/w237f`.
- The independent GPU v10.239 process is outside this handoff and may not be stopped, modified, or used to change CPU results. Memory above 80% still pauses new CPU dispatch for user decision rather than terminating the batch.
- Transaction evidence is `validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_forward_v10_240_cpu20_handoff_transaction_v1/summary.json`, SHA256 `{sha256(TX_SUMMARY)}`.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 540. Lossless CPU12-to-CPU20", facts_text)
    decision_text = """

## 2026-09-04 v10.240 — raise active historical-forward production to CPU20

- Bind the user's explicit `开始` to one lossless handoff of the existing v10.237 530-cell forward batch from twelve to twenty total CPU workers.
- Preserve every completed result and in-flight worker, reuse the same member fingerprints and `w237f` output root, and launch only unfinished members. Do not rerun spin-up or change any scientific input, threshold, timestep, climate, parameter, or curve rule.
- Stop only the fixed-limit source controller PID `25116` after a durable handoff snapshot; adopt its live children and fill free slots to twenty. Do not control the independent GPU branch.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.240", decision_text)

    route = {
        "schema": "lia-191-current-route-v10-240",
        "status": "cpu12-to-cpu20-forward-handoff-authorized",
        "population": 191, "accepted": 58, "remaining": 133,
        "active": len(status["running"]), "queue": int(status["pending"]),
        "launch_authorized": True, "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_239",
        "current_focus": "all191-oggm-flowline-1d-530-forward-cpu20-handoff",
        "source_parent_pid": 25116, "source_workers": 12, "target_workers": 20,
        "completed_at_authorization_snapshot": int(status["completed"]),
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "transaction_evidence": str(TX_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        "scientific_contract_unchanged": True, "spinup_recomputation_authorized": False,
        "duplicate_member_execution_authorized": False, "gpu_branch_control_authorized": False,
        "next_legal_action": "validate, preserve all terminal and in-flight members, stop only source controller PID 25116, and start one CPU20 handoff controller",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)

    timestamp = datetime.now(timezone.utc).isoformat()
    state["state_id"] = "2026-09-04-v10.240-cpu20-forward-handoff-authorized"
    state["updated_at"] = timestamp
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {
        "version": "v10.240", "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path), "supersedes": "v10.239",
    }
    state["counts"]["active"] = len(status["running"])
    state["counts"]["queued"] = int(status["pending"])
    state["current_focus"] = {
        "system_id": "all191-oggm-flowline-1d-530-forward-cpu20-handoff",
        "mainline_family": "all191-oggm-flowline-1d-historical-forward-v1",
        "state": "handoff-authorized", "full_historical_authorized": True,
        "source_parent_pid": 25116, "workers": 12, "target_workers": 20,
        "completed": int(status["completed"]), "active": len(status["running"]),
        "queue": int(status["pending"]), "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/w237f",
        "spinup_recomputation_authorized": False,
        "evidence": {"path": str(TX_SUMMARY.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(TX_SUMMARY)},
    }
    state["next_action"] = {
        "type": "historical-forward-cpu20-handoff",
        "transaction_id": "lia-all191-oggm-flowline-1d-forward-v10.240-cpu20-handoff",
        "launch_authorized": True,
        "description": "Perform one lossless CPU12-to-CPU20 handoff; preserve completed and in-flight members and do not control GPU.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"route": route, "state_id": state["state_id"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
