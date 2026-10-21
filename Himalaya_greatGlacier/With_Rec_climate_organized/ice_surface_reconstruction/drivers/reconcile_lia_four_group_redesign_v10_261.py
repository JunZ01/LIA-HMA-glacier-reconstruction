"""Reconcile the controller to design-only after both unused launches were cancelled."""

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
CANCEL_PATH = RESULTS / "lia_191_four_group_redesign_v10_261_cancelled_launches_v1/summary.json"
ROUTE_PATH = RESULTS / "lia_191_current_route_v10_261/summary.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main() -> int:
    state = load_json(STATE_PATH)
    if state.get("state_id") != "2026-09-04-v10.260-adaptive-spinup-round1-authorized":
        raise RuntimeError("current state changed; refusing reconciliation overwrite")
    if (PROJECT_ROOT / "wg248").exists():
        raise RuntimeError("wg248 exists; GPU terminal evidence must be reviewed")
    cpu_summary_path = PROJECT_ROOT / "w260a/r/summary.json"
    cpu_status_path = PROJECT_ROOT / "w260a/r/status.json"
    if not cpu_summary_path.is_file() or not cpu_status_path.is_file():
        raise RuntimeError("v10.260 terminal controller evidence is missing")
    cpu_summary = load_json(cpu_summary_path)
    cpu_status = load_json(cpu_status_path)
    if (
        cpu_summary.get("status") != "stopped"
        or cpu_summary.get("stop_reason") != "same-contract-preflight-failed"
        or cpu_summary.get("completed_cells") != 3
        or any(slot.get("state") != "idle" for slot in cpu_status.get("slots", []))
    ):
        raise RuntimeError("v10.260 is not the expected stopped three-member preflight")
    if CANCEL_PATH.exists() or ROUTE_PATH.exists():
        raise RuntimeError("v10.261 reconciliation path already exists")

    cancellation = {
        "schema": "lia-191-four-group-redesign-cancelled-launches-v1",
        "status": "gpu-cancelled-before-launch-cpu-preflight-terminal-no-production-release",
        "created_at_utc": utc_now(),
        "user_instruction": "别跑了，我现在有新的想法",
        "cancelled_before_launch": ["lia-191-gpu-spinup-cap1500-v10.248"],
        "cpu_preflight_terminal": {
            "transaction": "lia-all191-adaptive-spinup-v10.260-round1",
            "completed": 3,
            "joint_passes": 0,
            "scientific_nonconvergence": 1,
            "undispatched_cancelled": 146,
            "stop_reason": "same-contract-preflight-failed",
            "summary_path": relative(cpu_summary_path),
            "summary_sha256": sha256(cpu_summary_path),
        },
        "gpu_output_root_absent": "wg248",
        "accepted_ledger_effect": 0,
        "preserved_read_only_source": {
            "output_root": "wg239",
            "converged_members": 217,
            "checkpoint": "wg239/b/b000/checkpoints/year-1230.npz",
        },
        "next_legal_action": "freeze the four parameter groups and staged search rules; no further simulation launch",
    }
    write_json_new(CANCEL_PATH, cancellation)

    route = {
        "schema": "lia-191-current-route-v10-261",
        "status": "four-group-bruteforce-redesign-decision-required",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "simulation_authorized": False,
        "full_historical_authorized": False,
        "gpu_spinup_launch_authorized": False,
        "supersedes": "lia_191_current_route_v10_260",
        "current_focus": "all191-four-group-bruteforce-redesign-decision-required",
        "accepted_ledger_effect": 0,
        "evidence": relative(CANCEL_PATH),
        "evidence_sha256": sha256(CANCEL_PATH),
        "next_legal_action": "design and freeze a new staged four-group search; no launch without explicit approval",
    }
    write_json_new(ROUTE_PATH, route)

    new_state = deepcopy(state)
    new_state["state_id"] = "2026-09-04-v10.261-four-group-redesign-decision-required"
    new_state["updated_at"] = utc_now()
    new_state["scientific_contract"]["sha256"] = sha256(PROJECT_ROOT / "LIA_PROJECT_FACTS.md")
    new_state["authoritative_route"] = {
        "version": "v10.261",
        "path": relative(ROUTE_PATH),
        "sha256": sha256(ROUTE_PATH),
        "supersedes": "v10.260",
    }
    new_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    new_state["current_focus"] = {
        "system_id": "all191-four-group-bruteforce-redesign-decision-required",
        "mainline_family": "all191-oggm-flowline-1d-staged-four-group-search",
        "state": "decision-required",
        "full_historical_authorized": False,
        "active": 0,
        "queue": 0,
        "accepted_ledger_effect": 0,
        "evidence": {"path": relative(CANCEL_PATH), "sha256": sha256(CANCEL_PATH)},
    }
    new_state["next_action"] = {
        "type": "decision-required-freeze-four-group-search",
        "transaction_id": "lia-four-group-redesign-v10.261",
        "launch_authorized": False,
        "description": "Freeze parameter groups, grids, stage roles, retention, and curve gates before any launch.",
    }
    adaptive_branch = {
        "status": "terminal-three-member-preflight-failed-production-not-released",
        "system_count": 191,
        "cell_count": 149,
        "launch_authorized": False,
        "retry_authorized": False,
        "promoted_to_mainline": False,
        "accepted_count_effect": 0,
        "output_root": "w260a",
        "output_root_created": True,
        "completed_preflight": 3,
        "joint_preflight_passes": 0,
        "undispatched_cancelled": 146,
        "evidence": {"path": relative(CANCEL_PATH), "sha256": sha256(CANCEL_PATH)},
        "next_legal_action": "read-only evidence only; superseded by four-group redesign",
    }
    new_state["research_branches"]["lia_191_adaptive_spinup_v10_260"] = adaptive_branch
    gpu_branch = new_state["research_branches"]["lia_191_gpu_spinup_cap1500_v10_248"]
    gpu_branch["status"] = "cancelled-before-launch-by-latest-user-instruction"
    gpu_branch["launch_authorized"] = False
    gpu_branch["retry_authorized"] = False
    gpu_branch["evidence"] = {"path": relative(CANCEL_PATH), "sha256": sha256(CANCEL_PATH)}
    gpu_branch["next_legal_action"] = "none; superseded by four-group redesign"
    new_state["concurrent_gpu_spinup"] = {
        "transaction_id": "lia-191-gpu-spinup-cap1500-v10.248",
        "state": "cancelled-before-launch",
        "output_root": "wg248",
        "output_root_created": False,
        "launch_authorized": False,
        "cpu_mainline_modified_or_controlled": False,
    }
    atomic_json(STATE_PATH, new_state)
    print(json.dumps({
        "status": "decision-required-no-launch",
        "state_id": new_state["state_id"],
        "gpu_output_root_absent": "wg248",
        "cpu_preflight_completed": 3,
        "cpu_undispatched_cancelled": 146,
        "accepted": 58,
        "active": 0,
        "queued": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
