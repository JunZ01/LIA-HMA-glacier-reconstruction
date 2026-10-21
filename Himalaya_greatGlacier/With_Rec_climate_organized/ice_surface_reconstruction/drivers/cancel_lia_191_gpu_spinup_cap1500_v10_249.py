"""Revoke the unused v10.248 GPU continuation authorization."""

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
OUTPUT_ROOT = PROJECT_ROOT / "wg248"
CANCEL_PATH = RESULTS / "lia_191_gpu_spinup_cap1500_v10_249_cancelled_v1/summary.json"
ROUTE_PATH = RESULTS / "lia_191_current_route_v10_249/summary.json"


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
    if state.get("state_id") != "2026-09-04-v10.248-gpu191-spinup-cap1500-authorized":
        raise RuntimeError("current state changed; refusing cancellation overwrite")
    if OUTPUT_ROOT.exists():
        raise RuntimeError("wg248 exists; this cancellation script is only for a never-launched continuation")
    if CANCEL_PATH.exists() or ROUTE_PATH.exists():
        raise RuntimeError("v10.249 cancellation paths already exist")

    cancellation = {
        "schema": "lia-191-gpu-spinup-cap1500-cancellation-v1",
        "status": "cancelled-before-launch-by-user",
        "cancelled_at_utc": utc_now(),
        "user_instruction": "别跑了，我现在有新的想法",
        "cancelled_transaction": "lia-191-gpu-spinup-cap1500-v10.248",
        "output_root_created": False,
        "new_model_time_years": 0,
        "source_checkpoint_preserved": "wg239/b/b000/checkpoints/year-1230.npz",
        "inherited_results_preserved": 217,
        "cpu_mainline_modified_or_controlled": False,
        "next_legal_action": "design review only; no simulation launch without a new explicit authorization",
    }
    write_json_new(CANCEL_PATH, cancellation)

    route = {
        "schema": "lia-191-current-route-v10-249",
        "status": "forward-curve-screen-terminal-zero-pass-decision-required-gpu-redesign-only",
        "population": state["population"],
        "accepted": state["counts"]["accepted"],
        "remaining": state["counts"]["remaining"],
        "active": state["counts"]["active"],
        "queue": state["counts"]["queued"],
        "launch_authorized": False,
        "full_historical_authorized": False,
        "simulation_authorized": False,
        "gpu_spinup_launch_authorized": False,
        "supersedes": "lia_191_current_route_v10_248",
        "current_focus": state["current_focus"]["system_id"],
        "accepted_ledger_effect": 0,
        "evidence": state["current_focus"]["evidence"]["path"],
        "evidence_sha256": state["current_focus"]["evidence"]["sha256"],
        "gpu_cancellation_evidence": relative(CANCEL_PATH),
        "gpu_cancellation_evidence_sha256": sha256(CANCEL_PATH),
        "next_legal_action": "review and freeze a new four-layer parameter search; no automatic launch",
    }
    write_json_new(ROUTE_PATH, route)

    new_state = deepcopy(state)
    new_state["state_id"] = "2026-09-04-v10.249-gpu-redesign-decision-required"
    new_state["updated_at"] = utc_now()
    new_state["authoritative_route"] = {
        "version": "v10.249",
        "path": relative(ROUTE_PATH),
        "sha256": sha256(ROUTE_PATH),
        "supersedes": "v10.248",
    }
    branch = new_state["research_branches"]["lia_191_gpu_spinup_cap1500_v10_248"]
    branch["status"] = "cancelled-before-launch-by-user"
    branch["launch_authorized"] = False
    branch["retry_authorized"] = False
    branch["evidence"] = {"path": relative(CANCEL_PATH), "sha256": sha256(CANCEL_PATH)}
    branch["next_legal_action"] = "none; superseded by redesign discussion"
    new_state["concurrent_gpu_spinup"] = {
        "transaction_id": "lia-191-gpu-spinup-cap1500-v10.248",
        "state": "cancelled-before-launch-by-user",
        "system_count": 191,
        "cell_count": 1337,
        "output_root": relative(OUTPUT_ROOT),
        "output_root_created": False,
        "launch_authorized": False,
        "cpu_mainline_modified_or_controlled": False,
    }
    atomic_json(STATE_PATH, new_state)
    print(json.dumps({
        "status": "cancelled-before-launch-by-user",
        "state_id": new_state["state_id"],
        "output_root_created": False,
        "source_checkpoint_preserved": True,
        "inherited_results_preserved": 217,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
