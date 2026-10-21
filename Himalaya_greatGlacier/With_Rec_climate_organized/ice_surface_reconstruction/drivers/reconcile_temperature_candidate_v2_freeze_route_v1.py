from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
ROUTE112 = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_112/summary.json"
ROUTE113 = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_113"
FREEZE_SUMMARY = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/temperature_candidate_v2_formal_freeze_v1/summary.json"
SYSTEM709_RESULT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system709_first_historical_bracket_result_v1/summary.json"

EXPECTED = {
    STATE: "04C23703E124D936DFE186FF8D81469D83E75BBE7C7C8B04967117E486442D3A",
    FACTS: "78B6CAAEB8171E00E24C52E9184672A565DBFD49938B2E9442DBB325CC7E1052",
    DECISIONS: "0ABF62A1EBC5C04E3F173DA0740DFEEAF221B17E6513DDE74D6AF441243CA",
    ROUTE112: "A40EBCF5937532026ABE6FFD6A4668BE5EC84841D6F371633D77E05A31366F3E",
    FREEZE_SUMMARY: "BC790F9525D997EA0DA1D0305FB7DA3041FE5028F8972AA4A08494ACF0180F62",
    SYSTEM709_RESULT: "5DCCD0E9660611C998D5FAD3DA935AA13BE79227DCE3C2FA6BC23172A043A5A2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def replace_once(content: bytes, old: str, new: str) -> bytes:
    newline = "\r\n" if b"\r\n" in content else "\n"
    old_bytes = old.replace("\n", newline).encode("utf-8")
    new_bytes = new.replace("\n", newline).encode("utf-8")
    if content.count(old_bytes) != 1:
        raise RuntimeError(f"replacement-count:{old[:40]}:{content.count(old_bytes)}")
    return content.replace(old_bytes, new_bytes, 1)


def main() -> None:
    if ROUTE113.exists():
        raise FileExistsError(ROUTE113)
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"concurrent-change:{path}")

    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("state_id") != "2026-08-28-v10.112-temperature-candidate-v2-formal-freeze":
        raise RuntimeError("unexpected-state-id")
    route112 = json.loads(ROUTE112.read_text(encoding="utf-8"))
    if route112.get("status") != "system709-first-historical-bracket-too-warm":
        raise RuntimeError("unexpected-system709-route")

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = replace_once(
        facts_before,
        "## 415. HMA Temperature Candidate v2 formally frozen (2026-08-28)",
        "## 416. HMA Temperature Candidate v2 formally frozen (2026-08-28)",
    )
    decisions_after = replace_once(
        decisions_before,
        "## 2026-08-28 - Route v10.112: HMA Temperature Candidate v2 frozen",
        "## 2026-08-28 - Route v10.113: HMA Temperature Candidate v2 frozen and v10.112 reconciled",
    )
    decisions_after = replace_once(
        decisions_after,
        "- System 709 controller state is carried forward unchanged from v10.111:\n  accepted 57, remaining 134, active 2, queued 0. Monitoring and formal\n  classification of the registered endpoint processes remain the only LIA\n  mainline action; this policy-only transition authorizes no model launch.",
        "- The concurrent system-709 v10.112 result is retained as authoritative:\n  accepted 57, remaining 134, active 0, queued 0. Route v10.113 merges that\n  result with the temperature-candidate freeze and releases preparation only\n  for the registered colder return pair; no model launch is authorized.",
    )
    facts_hash = hashlib.sha256(facts_after).hexdigest().upper()

    route113 = dict(route112)
    route113["schema"] = "lia-191-current-route-v10-113"
    route113["status"] = "system709-first-historical-bracket-too-warm-temperature-candidate-v2-frozen"
    route113["supersedes"] = "lia_191_current_route_v10_112"
    route113["temperature_candidate"] = {
        "candidate_id": "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828",
        "status": "formally-frozen-candidate-not-production",
        "evidence": "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/temperature_candidate_v2_formal_freeze_v1/summary.json",
        "evidence_sha256": EXPECTED[FREEZE_SUMMARY],
        "field_sha256": "DF008575229772892BFC8A2B5C5ED8450E735F7CD5CC64CA7B8ACA9F58A71C90",
    }
    route113["production_temperature_modified"] = False
    route113["sealed_1921_1939_values_used"] = False
    route113["next_legal_action"] = (
        "prepare and hash immutable +0.625 K and +0.875 K return-endpoint configs from the same "
        "system-709 equilibrium; no launch authorized"
    )
    route_bytes = (json.dumps(route113, indent=2) + "\n").encode("utf-8")
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    new_state = dict(state)
    new_state["state_id"] = "2026-08-28-v10.113-system709-too-warm-temperature-candidate-v2-frozen"
    new_state["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    new_state["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": facts_hash}
    new_state["authoritative_route"] = {
        "version": "v10.113",
        "path": "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_113/summary.json",
        "sha256": route_hash,
        "supersedes": "v10.112",
    }
    new_state["counts"] = {"accepted": 57, "remaining": 134, "active": 0, "queued": 0}
    new_state["current_focus"] = {
        "system_id": "709",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "first-historical-bracket-too-warm",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {
            "path": "validation_data/results/paleo_surface_validation_v0/lia_191_system709_first_historical_bracket_result_v1/summary.json",
            "sha256": EXPECTED[SYSTEM709_RESULT],
        },
    }
    new_state["next_action"] = {
        "type": "prepare-system709-colder-return-bracket-v1",
        "launch_authorized": False,
        "description": (
            "Prepare and hash immutable +0.625 K and +0.875 K return-endpoint configs from the "
            "same equilibrium; create a new validated route before any launch."
        ),
    }
    state_bytes = (json.dumps(new_state, indent=2) + "\n").encode("utf-8")

    stage = Path(tempfile.mkdtemp(prefix=".route_v10_113_", dir=ROOT))
    backups = {FACTS: facts_before, DECISIONS: decisions_before, STATE: STATE.read_bytes()}
    try:
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(state_bytes)
        (stage / "route").mkdir()
        (stage / "route/summary.json").write_bytes(route_bytes)
        ROUTE113.mkdir(parents=False)
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "route/summary.json", ROUTE113 / "summary.json")
        os.replace(stage / "state", STATE)
    except BaseException:
        for path, content in backups.items():
            path.write_bytes(content)
        shutil.rmtree(ROUTE113, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps({
        "status": "complete",
        "state_id": new_state["state_id"],
        "facts_sha256": facts_hash,
        "route_sha256": route_hash,
        "system709_active": 0,
        "temperature_candidate": "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828",
        "production_modified": False,
        "launch_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
