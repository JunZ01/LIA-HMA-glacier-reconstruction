"""Revoke System-2005 routes derived from an artificial initial-thickness lift."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = RESULTS / "lia_191_system2005_no_initial_thickness_policy_v7"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_52"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"

EXPECTED_FACTS_SHA = "D919D4AB6A6DA6370B826F3F1BC4AF723BBD5AB4F6325CB53DB375487E2DA076"
EXPECTED_STATE_ID = "2026-08-26-v10.51-policy-v6-historical-pair-v3-authorized"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".no_thickness.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    if OUT.exists() or ROUTE_DIR.exists():
        raise RuntimeError("versioned output already exists")
    if sha256(FACTS) != EXPECTED_FACTS_SHA:
        raise RuntimeError("facts hash changed before reconciliation")
    old_state = json.loads(STATE.read_text(encoding="utf-8"))
    if old_state.get("state_id") != EXPECTED_STATE_ID:
        raise RuntimeError(f"unexpected state id: {old_state.get('state_id')}")
    if old_state.get("counts") != {"accepted": 54, "remaining": 137, "active": 0, "queued": 2}:
        raise RuntimeError(f"unexpected counts: {old_state.get('counts')}")

    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 363." in facts_text:
        raise RuntimeError("facts already contain Section 363")
    if "## 362. System-2005 Policy-V6 short historical output roots v3" not in facts_text:
        raise RuntimeError("expected Section 362 is missing")
    decisions_text = DECISIONS.read_text(encoding="utf-8")
    decision_heading = "## 2026-08-26 — System 2005 initial-thickness modification prohibited"
    if decision_heading in decisions_text:
        raise RuntimeError("decision is already recorded")

    OUT.mkdir(parents=True, exist_ok=False)
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    for source, name in (
        (FACTS, "LIA_PROJECT_FACTS.pre.md"),
        (STATE, "LIA_CURRENT_STATE.pre.json"),
        (DECISIONS, "LIA_DECISION_LOG.pre.md"),
    ):
        shutil.copy2(source, OUT / name)

    section = """## 363. System-2005 initial-thickness modification prohibited (2026-08-26)

- The user explicitly prohibits any artificial increase, decrease, scaling, or
  response-based recalculation of the initial ice thickness for system 2005.
  Only an unchanged, reproducibly generated natural model state may serve as a
  future steady or historical input.
- Policy-V6 candidates v2 and v3 used uniform active-node thickness lifts of
  `2.8052643 m` and `4.876919 m`. Candidate v4 changed only temperature relative
  to v2, but its input still inherited the v2 thickness lift. These outputs are
  preserved as technical evidence but cannot support formal steady acceptance,
  historical authorization, reproduction, or an accepted-count change under
  the new policy.
- The v10.51 authorization for the `+0.28125/+0.78125 K` historical endpoints is
  revoked before launch. Both expected PIDs were absent, no related model
  process was active, and neither short-root endpoint was started or resumed.
- Accepted count remains 54, remaining count remains 137, active and queued
  counts are both zero. No simulation is authorized. The next legal action is
  a read-only feasibility audit restricted to unchanged natural states; a new
  mechanism and route transaction are required before any future launch.
- Authoritative evidence and route:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_no_initial_thickness_policy_v7/summary.json`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_52/summary.json`.
"""
    facts_new = facts_text.rstrip() + "\n\n" + section
    atomic_write(FACTS, facts_new.encode("utf-8"))
    facts_sha = sha256(FACTS)

    evidence = {
        "schema": "lia-191-system2005-no-initial-thickness-policy-v7",
        "status": "active-policy-historical-authorization-revoked",
        "system_id": 2005,
        "user_decision": "initial-thickness-modification-prohibited",
        "prohibited": [
            "uniform thickness lift or reduction",
            "section scaling used to alter initial thickness",
            "response-corrected thickness recalculation",
            "any equivalent artificial initial-state thickness adjustment",
        ],
        "required_input": "unchanged reproducibly-generated natural model state",
        "ineligible_technical_outputs": {
            "candidate_v2_uniform_lift_m": 2.8052643058495814,
            "candidate_v3_uniform_lift_m": 4.876919,
            "candidate_v4_inherits_v2_lift": True,
            "v4_state_sha256": "DF85D2BB28D6C08049A630164EA56D482A2F324CF54920C1B9349051DB4AC439",
        },
        "revoked_route": "lia_191_current_route_v10_51",
        "revoked_endpoint_delta_t_c": [0.28125, 0.78125],
        "endpoint_process_ids_checked": [60236, 41584],
        "endpoint_processes_found": [],
        "endpoint_steps_run": 0,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queued": 0,
        "launch_authorized": False,
        "historical_authorized": False,
        "old_outputs_modified": False,
        "next_legal_action": "read-only unchanged-natural-state feasibility audit",
    }
    evidence_path = OUT / "summary.json"
    atomic_write(evidence_path, json_bytes(evidence))
    evidence_sha = sha256(evidence_path)

    route = {
        "schema": "lia-191-current-route-v10-52",
        "status": "system2005-no-initial-thickness-policy-active-readonly-only",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_51",
        "system_id": 2005,
        "initial_thickness_modification_allowed": False,
        "v4_steady_eligible_for_historical": False,
        "revoked_endpoint_delta_t_c": [0.28125, 0.78125],
        "evidence": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
        "next_legal_route": "read-only feasibility audit using unchanged natural states only",
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, json_bytes(route))
    route_sha = sha256(route_path)

    new_state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.52-system2005-no-initial-thickness",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": facts_sha},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.52",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": route_sha,
            "supersedes": "v10.51",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 0},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v7-unchanged-natural-state-only",
            "state": "hold-new-mechanism",
            "initial_thickness_modification_allowed": False,
            "full_historical_authorized": False,
            "evidence": {
                "path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": evidence_sha,
            },
        },
        "research_branches": old_state.get("research_branches", {}),
        "next_action": {
            "type": "system2005-unchanged-natural-state-readonly-feasibility-audit",
            "launch_authorized": False,
            "description": "Audit unchanged natural states only; no thickness adjustment or simulation launch.",
        },
    }
    atomic_write(STATE, json_bytes(new_state))

    decision = f"""{decision_heading}

- Artificial initial-thickness increases, decreases, section scaling, and
  response-based thickness recalculation are prohibited.
- V2/v3/v4 remain technical evidence but cannot authorize historical runs.
- Revoke v10.51 before launch; set active/queued to 0/0 and return to a
  read-only unchanged-natural-state feasibility audit.
"""
    atomic_write(DECISIONS, (decisions_text.rstrip() + "\n\n" + decision).encode("utf-8"))

    transaction = {
        "schema": "lia-191-system2005-no-initial-thickness-reconciliation-v1",
        "status": "complete",
        "state_pre_id": old_state["state_id"],
        "state_post_id": new_state["state_id"],
        "facts_sha256": facts_sha,
        "route_sha256": route_sha,
        "evidence_sha256": evidence_sha,
        "simulation_launched": False,
    }
    atomic_write(OUT / "transaction.json", json_bytes(transaction))
    print(json.dumps(transaction, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
