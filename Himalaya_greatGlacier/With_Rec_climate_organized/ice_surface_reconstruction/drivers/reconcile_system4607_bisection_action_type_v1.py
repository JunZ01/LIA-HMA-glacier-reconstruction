"""Reconcile the queued system-4607 bisection action-type label."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import psutil


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
DECISIONS = PROJECT / "LIA_DECISION_LOG.md"
ROUTE_121 = RESULTS / "lia_191_current_route_v10_121/summary.json"
PREPARATION = RESULTS / "lia_191_system4607_first_bracket_result_bisection1_preparation_v1/summary.json"
EVIDENCE_DIR = RESULTS / "lia_191_system4607_bisection_action_type_reconciliation_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_122"
EXPECTED_STATE = "2026-08-28-v10.121-system4607-first-bisection-queued"
NEW_ACTION = "launch-system4607-first-historical-bisection-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if EVIDENCE_DIR.exists() or ROUTE_DIR.exists():
        raise FileExistsError("v10.122-output-collision")
    controller = json.loads(STATE.read_text(encoding="utf-8"))
    route = json.loads(ROUTE_121.read_text(encoding="utf-8"))
    preparation = json.loads(PREPARATION.read_text(encoding="utf-8"))
    if controller["state_id"] != EXPECTED_STATE:
        raise RuntimeError(f"unexpected-state:{controller['state_id']}")
    if controller["counts"] != {"accepted": 57, "remaining": 134, "active": 0, "queued": 1}:
        raise RuntimeError("unexpected-counts")
    if route["registered_bisection_endpoint_c"] != 0.625:
        raise RuntimeError("midpoint-contract-failed")
    if route["preparation_sha256"] != sha256(PREPARATION):
        raise RuntimeError("preparation-hash-failed")
    config_path = PROJECT / route["config"]["path"]
    if sha256(config_path) != route["config"]["sha256"]:
        raise RuntimeError("config-hash-failed")
    output_root = PROJECT / route["config"]["output_root"]
    log_root = PROJECT / route["config"]["log_root"]
    if output_root.exists() or log_root.exists():
        raise RuntimeError("registered-output-collision")
    if any(psutil.pid_exists(int(pid)) for pid in (14776, 17532)):
        raise RuntimeError("old-process-still-active")
    state_hash = sha256(STATE)
    facts_hash = sha256(FACTS)
    decisions_hash = sha256(DECISIONS)

    EVIDENCE_DIR.mkdir(parents=True)
    ROUTE_DIR.mkdir(parents=True)
    evidence = {
        "schema": "lia-191-system4607-bisection-action-type-reconciliation-v1",
        "status": "control-only-reconciliation",
        "system_id": 4607,
        "old_action_type": controller["next_action"]["type"],
        "new_action_type": NEW_ACTION,
        "validator_failure_resolved": "historical authorization has a non-historical next action",
        "registered_endpoint_c": 0.625,
        "config_path": route["config"]["path"],
        "config_sha256": route["config"]["sha256"],
        "output_root": route["config"]["output_root"],
        "log_root": route["config"]["log_root"],
        "model_or_scientific_change": False,
        "process_started": False,
        "accepted_count_effect": 0,
    }
    atomic_json(EVIDENCE_DIR / "summary.json", evidence)
    evidence_hash = sha256(EVIDENCE_DIR / "summary.json")

    new_route = dict(route)
    new_route["schema"] = "lia-191-current-route-v10-122"
    new_route["status"] = "system4607-first-historical-bisection-queued-control-reconciled"
    new_route["supersedes"] = "lia_191_current_route_v10_121"
    new_route["control_reconciliation"] = (
        "v10.121 used a non-historical next-action type while full historical authorization was true; "
        "v10.122 changes only that action label"
    )
    new_route["reconciliation_evidence"] = str((EVIDENCE_DIR / "summary.json").relative_to(PROJECT)).replace("\\", "/")
    new_route["reconciliation_evidence_sha256"] = evidence_hash
    new_route["next_legal_action"] = (
        "after validator PASS and final duplicate/output checks, launch exactly one checkpointed +0.625 K historical midpoint process"
    )
    atomic_json(ROUTE_DIR / "summary.json", new_route)
    route_hash = sha256(ROUTE_DIR / "summary.json")

    if sha256(STATE) != state_hash or sha256(FACTS) != facts_hash or sha256(DECISIONS) != decisions_hash:
        raise RuntimeError("canonical-source-changed-before-commit")
    facts_text = FACTS.read_text(encoding="utf-8")
    decisions_text = DECISIONS.read_text(encoding="utf-8")
    facts_heading = "## 426. System-4607 midpoint action-type reconciled (2026-08-28)"
    decision_heading = "## 2026-08-28 - Route v10.122: system 4607 midpoint action type reconciled"
    if facts_heading in facts_text or decision_heading in decisions_text:
        raise RuntimeError("canonical-entry-exists")
    facts_text += (
        "\n\n" + facts_heading + "\n\n"
        "- Route v10.121 correctly registered the sole +0.625 K midpoint but named its next action without the required historical marker, so the controller validator withheld launch authorization.\n"
        "- Route v10.122 changes only the action-type label to `launch-system4607-first-historical-bisection-v1`. The endpoint, config and source hashes, output roots, equilibrium, climate, physics, solver, gates, and initial thickness are unchanged. No process was launched by this correction.\n"
        "- Counts remain 57 accepted and 134 remaining, active 0 and queued 1.\n"
    )
    decisions_text += (
        "\n\n" + decision_heading + "\n\n"
        "- Corrected only the queued action-type label so it satisfies the historical-authorization validator rule.\n"
        "- Preserved the sole +0.625 K midpoint, config hash, immutable roots, and all scientific settings. Counts remain 57 accepted and 134 remaining, active 0 and queued 1.\n"
    )
    atomic_text(FACTS, facts_text)
    atomic_text(DECISIONS, decisions_text)

    controller["state_id"] = "2026-08-28-v10.122-system4607-first-historical-bisection-queued-control-reconciled"
    controller["updated_at"] = datetime.now().astimezone().isoformat()
    controller["scientific_contract"]["sha256"] = sha256(FACTS)
    controller["authoritative_route"] = {
        "version": "v10.122",
        "path": str((ROUTE_DIR / "summary.json").relative_to(PROJECT)).replace("\\", "/"),
        "sha256": route_hash,
        "supersedes": "v10.121",
    }
    controller["current_focus"]["state"] = "first-historical-bisection-queued"
    controller["current_focus"]["evidence"] = {
        "path": str((EVIDENCE_DIR / "summary.json").relative_to(PROJECT)).replace("\\", "/"),
        "sha256": evidence_hash,
    }
    controller["next_action"] = {
        "type": NEW_ACTION,
        "launch_authorized": True,
        "description": "After validator PASS and final duplicate/output checks, launch exactly one checkpointed +0.625 K historical midpoint process.",
    }
    atomic_json(STATE, controller)
    print(json.dumps({"status": "complete", "route": "v10.122", "process_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
