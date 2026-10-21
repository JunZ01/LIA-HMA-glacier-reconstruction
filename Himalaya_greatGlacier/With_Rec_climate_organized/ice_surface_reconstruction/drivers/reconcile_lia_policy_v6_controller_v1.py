"""Atomically promote the already-reviewed Policy-V6 sidecars into controller state.

This is a one-shot control-plane reconciliation.  It does not import OGGM and
cannot launch a glacier simulation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = ROOT / "validation_data/results/paleo_surface_validation_v0"
POLICY_ROOT = RESULT_ROOT / "lia_191_separation_length20_policy_v6"
OUT = RESULT_ROOT / "lia_191_policy_v6_controller_reconciliation_v1"
ROUTE_DIR = RESULT_ROOT / "lia_191_current_route_v10_41"

FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
PENDING_FACTS = POLICY_ROOT / "LIA_PROJECT_FACTS_SECTION_352_PENDING.md"
PENDING_DECISION = POLICY_ROOT / "decision_log_entry_pending.md"
POLICY = POLICY_ROOT / "contract.json"

EXPECTED_FACTS_PRE_SHA = (
    "9E1EBEF5593EEC5898FB59FAC83F899410D0214B7C6449E4BC9C59E11259664B"
)
EXPECTED_STATE_ID = "2026-08-26-v10.32-system2005-length-decision"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dump_json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".policy_v6_reconcile.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)

    required = [FACTS, STATE, DECISIONS, PENDING_FACTS, PENDING_DECISION, POLICY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing reconciliation inputs: {missing}")

    facts_pre_sha = sha256(FACTS)
    if facts_pre_sha != EXPECTED_FACTS_PRE_SHA:
        raise RuntimeError(
            f"facts pre-hash changed: {facts_pre_sha} != {EXPECTED_FACTS_PRE_SHA}"
        )

    old_state = json.loads(STATE.read_text(encoding="utf-8"))
    if old_state.get("state_id") != EXPECTED_STATE_ID:
        raise RuntimeError(f"unexpected state id: {old_state.get('state_id')}")
    if old_state["counts"] != {"accepted": 54, "remaining": 137, "active": 0, "queued": 0}:
        raise RuntimeError(f"unexpected controller counts: {old_state['counts']}")

    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 352." in facts_text:
        raise RuntimeError("facts already contain Section 352")
    if "## 351. System-2005 +0.875 K authorized single-step diagnostic" not in facts_text:
        raise RuntimeError("expected Section 351 tail was not found")

    pending_text = PENDING_FACTS.read_text(encoding="utf-8")
    section_text = pending_text.split("\nPending reason:", 1)[0].rstrip()
    if not section_text.startswith("## 352. System-2005 Policy-V6"):
        raise RuntimeError("pending facts sidecar does not contain the expected section")

    decision_text = PENDING_DECISION.read_text(encoding="utf-8").strip()
    decisions_pre = DECISIONS.read_text(encoding="utf-8")
    if "## 2026-08-26 — Policy V6: first option selected" in decisions_pre:
        raise RuntimeError("Policy-V6 decision is already present in root log")

    for source, name in ((FACTS, "LIA_PROJECT_FACTS.pre.md"),
                         (STATE, "LIA_CURRENT_STATE.pre.json"),
                         (DECISIONS, "LIA_DECISION_LOG.pre.md")):
        shutil.copy2(source, OUT / name)

    facts_new = facts_text.rstrip() + "\n\n" + section_text + "\n"
    atomic_write(FACTS, facts_new.encode("utf-8"))
    facts_post_sha = sha256(FACTS)

    route = {
        "schema": "lia-191-current-route-v10-41",
        "status": "policy-v6-active-readonly-steady-feasibility-audit",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "remaining_mutually_exclusive": {
            "mechanism_closed": 123,
            "mathematically_unreachable": 13,
            "system2005_new_formal_steady_required": 1,
        },
        "active_systems": 0,
        "active": 0,
        "active_processes": 0,
        "active_process_ids": [],
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_40",
        "policy_contract": str(POLICY.relative_to(ROOT)).replace("\\", "/"),
        "registered_lia_length_m": 25985.52229946727,
        "minimum_peak_connected_root_length_m": 25450.125918037134,
        "prior_peak_connected_root_length_m": 25250.0,
        "shortfall_m": 200.1259180371344,
        "next_legal_route": (
            "read-only inventory and feasibility audit of existing system-2005 "
            "steady responses; no simulation until a versioned candidate contract "
            "and a new controller transition authorize it"
        ),
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, dump_json_bytes(route))
    route_sha = sha256(route_path)
    policy_sha = sha256(POLICY)

    state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.41-policy-v6-readonly-audit",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {
            "path": "LIA_PROJECT_FACTS.md",
            "sha256": facts_post_sha,
        },
        "execution_contract": {
            "path": "LIA_EXECUTION_CONTRACT.md",
            "version": "1.0",
        },
        "authoritative_route": {
            "version": "v10.41",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": route_sha,
            "supersedes": "v10.40",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 0},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-20pct-new-formal-steady",
            "state": "read-only-feasibility-audit",
            "registered_lia_length_m": 25985.52229946727,
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "prior_formal_steady_peak_connected_length_m": 25250.0,
            "shortfall_m": 200.1259180371344,
            "full_historical_authorized": False,
            "evidence": {
                "path": str(POLICY.relative_to(ROOT)).replace("\\", "/"),
                "sha256": policy_sha,
            },
        },
        "research_branches": old_state.get("research_branches", {}),
        "next_action": {
            "type": "system2005-steady-response-readonly-feasibility-audit",
            "launch_authorized": False,
            "description": (
                "Inventory existing responses and register a bounded new formal-steady "
                "candidate before any simulation launch."
            ),
        },
    }
    atomic_write(STATE, dump_json_bytes(state))

    decisions_new = decisions_pre.rstrip() + "\n\n" + decision_text + "\n"
    atomic_write(DECISIONS, decisions_new.encode("utf-8"))

    summary = {
        "schema": "lia-191-policy-v6-controller-reconciliation-v1",
        "status": "controller-reconciled-readonly-audit-only",
        "facts_pre_sha256": facts_pre_sha,
        "facts_post_sha256": facts_post_sha,
        "state_pre_id": old_state["state_id"],
        "state_post_id": state["state_id"],
        "route": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "route_sha256": route_sha,
        "policy_sha256": policy_sha,
        "accepted": 54,
        "remaining": 137,
        "simulation_launched": False,
        "full_historical_authorized": False,
    }
    atomic_write(OUT / "summary.json", dump_json_bytes(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
