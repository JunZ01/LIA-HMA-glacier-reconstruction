"""Freeze the terminal forward evidence and authorize its v3 curve screen."""

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
SCORER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/score_lia_all191_forward_essential_standard_v3.py"
MATRIX = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1/forward_matrix.csv"
TARGETS = BASE / "lia_191_controller_v1/core191_length_references.csv"
STANDARD = BASE / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"
PRODUCTION_SUMMARY = ROOT / "w237f/r/production_summary_v10_243.json"
SCREEN_OUTPUT = BASE / "lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_246_v1"
TX_DIR = BASE / "lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_246_transaction_v1"
ROUTE_DIR = BASE / "lia_191_current_route_v10_246"


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
    if TX_DIR.exists() or ROUTE_DIR.exists() or SCREEN_OUTPUT.exists():
        raise FileExistsError("v10.246-output-already-exists")
    state_path = ROOT / "LIA_CURRENT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["authoritative_route"]["version"] != "v10.245":
        raise RuntimeError("unexpected-authoritative-route")
    production = json.loads(PRODUCTION_SUMMARY.read_text(encoding="utf-8"))
    if not (production.get("batch_completed") and production.get("completed_cells") == 530
            and production.get("technical_failures") == 2 and production.get("scientific_failures") == 0):
        raise RuntimeError("unexpected-production-terminal-summary")
    if psutil.pid_exists(int(production["parent_pid"])):
        raise RuntimeError("production-parent-still-active")
    matrix = __import__("pandas").read_csv(MATRIX)
    completed = 0
    technical = 0
    for key in matrix["production_member_key"].astype(str):
        path = ROOT / "w237f/m" / key / "c/result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        technical += int(bool(result.get("technical_failure")))
        completed += int(result.get("status") == "completed" and result.get("terminal") is True)
    if completed != 528 or technical != 2:
        raise RuntimeError(f"result-inventory-mismatch:{completed}:{technical}")

    inputs = {
        "scorer_path": str(SCORER.relative_to(ROOT)).replace("\\", "/"),
        "scorer_sha256": sha256(SCORER),
        "matrix_path": str(MATRIX.relative_to(ROOT)).replace("\\", "/"),
        "matrix_sha256": sha256(MATRIX),
        "targets_path": str(TARGETS.relative_to(ROOT)).replace("\\", "/"),
        "targets_sha256": sha256(TARGETS),
        "standard_path": str(STANDARD.relative_to(ROOT)).replace("\\", "/"),
        "standard_sha256": sha256(STANDARD),
        "production_summary_path": str(PRODUCTION_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        "production_summary_sha256": sha256(PRODUCTION_SUMMARY),
    }
    transaction = {
        "schema": "lia-all191-forward-curve-screen-transaction-v10.246",
        "status": "authorized-not-yet-run",
        "authorization_text": "开始吧",
        "launch_authorized": True,
        "simulation_authorized": False,
        "evaluated_forward_cells": 528,
        "excluded_technical_failure_cells": 2,
        "accepted_ledger_effect_authorized": False,
        "independent_reproduction_authorized": False,
        "standard_role": "provisional-forward-screen-only",
        "inputs": inputs,
        "output": str(SCREEN_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "command": [str(Path(os.sys.executable)), "-m",
                    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.score_lia_all191_forward_essential_standard_v3",
                    "--transaction", str((TX_DIR / "transaction.json").resolve())],
    }
    TX_DIR.mkdir(parents=True, exist_ok=False)
    transaction_path = TX_DIR / "transaction.json"
    atomic_json(transaction_path, transaction)

    facts = f"""

## 546. The 530-cell forward batch is terminal and the frozen v3 curve screen is authorized (2026-09-04)

- CPU20 forward controller PID `73724` completed and is absent. All 530 members reached a terminal record: 528 completed normally, two are isolated technical failures (`f0192` and `f0226`), and there are zero scientific failures, active workers, or pending members.
- The user instructs `开始吧`, authorizing one read-only post-run screen of the 528 successful cells with the already-bound retained-32 essential curve standard v3. No OGGM simulation, failed-member retry, spin-up recomputation, or GPU control is authorized.
- The fixed gates are unchanged: joint LIA-peak and year-2000 endpoint bounds (10% total-length and 20% target-separation errors), 21-year reflected smoothing, 0-100-year lag with correlation at least 0.30 and sign consistency at least 0.55, at least two sustained growth and two sustained retreat phases of at least 20 years covering at least 60% of total smoothed length variation, and exclusion of terminal endpoint drops and numerical high-frequency oscillation.
- Passing cells are provisional only. The accepted ledger stays at 58; formal acceptance still requires inherited upstream evidence and an independent reproduction/controller transaction.
- Frozen screening transaction is `validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_246_transaction_v1/transaction.json`, SHA256 `{sha256(transaction_path)}`.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 546. The 530-cell forward batch is terminal", facts)
    decision = """

## 2026-09-04 v10.246 — authorize the terminal 528-cell forward curve screen

- Close CPU20 forward dynamics with 528 completed and two isolated technical failures.
- Apply the hash-bound essential curve standard v3 once to the successful results. Write a new immutable evidence package and do not alter accepted=58.
- Do not run OGGM, retry failed members, select a glacier by hand, or start independent reproduction under this authorization.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.246", decision)
    route = {
        "schema": "lia-191-current-route-v10-246",
        "status": "forward-terminal-curve-screen-authorized",
        "population": 191, "accepted": 58, "remaining": 133,
        "active": 0, "queue": 528, "completed_forward_cells": 528,
        "technical_failures": 2, "scientific_failures": 0,
        "launch_authorized": True, "full_historical_authorized": False,
        "simulation_authorized": False,
        "supersedes": "lia_191_current_route_v10_245",
        "current_focus": "all191-oggm-flowline-1d-528-forward-curve-screen-v3",
        "transaction": str(transaction_path.relative_to(ROOT)).replace("\\", "/"),
        "transaction_sha256": sha256(transaction_path),
        "output": str(SCREEN_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "accepted_ledger_effect_authorized": False,
        "independent_reproduction_authorized": False,
        "next_legal_action": "run the one authorized read-only v3 curve screen",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)
    state["state_id"] = "2026-09-04-v10.246-forward-terminal-curve-screen-authorized"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {"version": "v10.246",
        "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path), "supersedes": "v10.245"}
    state["counts"]["active"] = 0
    state["counts"]["queued"] = 528
    state["current_focus"] = {
        "system_id": "all191-oggm-flowline-1d-528-forward-curve-screen-v3",
        "mainline_family": "all191-oggm-flowline-1d-historical-forward-v1",
        "state": "curve-screen-authorized", "full_historical_authorized": False,
        "forward_completed": 528, "technical_failures": 2, "active": 0, "queue": 528,
        "simulation_authorized": False, "accepted_ledger_effect_authorized": False,
        "evidence": {"path": str(transaction_path.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": sha256(transaction_path)},
    }
    state["next_action"] = {
        "type": "forward-curve-screen-v3", "transaction_id": "lia-all191-forward-screen-v10.246",
        "launch_authorized": True,
        "description": "Run the one read-only v3 screen of 528 successful forward cells; accepted remains 58.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"route": route, "transaction_sha256": sha256(transaction_path)}, indent=2))


if __name__ == "__main__":
    main()
