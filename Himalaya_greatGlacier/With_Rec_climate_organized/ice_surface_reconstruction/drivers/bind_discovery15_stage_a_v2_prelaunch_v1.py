"""Bind the final Stage A v2 retry code to route v10.185."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE184 = BASE / "lia_191_current_route_v10_184/summary.json"
ADJUDICATION = BASE / "lia_191_discovery16_stage_a_v1_adjudication_v1/summary.json"
OUT = BASE / "lia_191_discovery15_stage_a_v2_prelaunch_binding_v1"
ROUTE185 = BASE / "lia_191_current_route_v10_185"
BASE_RUNNER = DRIVER.parent / "prepare_discovery16_equilibrium_centered_stage_a_v1.py"
RETRY_RUNNER = DRIVER.parent / "prepare_discovery15_equilibrium_centered_stage_a_v2.py"
EXPORTER = DRIVER.parent / "export_system2005_historical_monthly_ti_bridge_v1.py"
TEST = DRIVER.parents[1] / "tests/test_generic_planform2d_adapter_contract_v1.py"
SYSTEMS = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 5770, 2559]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"json-root-not-object:{path}")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append_decision(content: bytes, text: str) -> bytes:
    marker = "## 2026-09-01 - Route v10.185: bind final Stage A v2 retry code"
    if marker.encode("utf-8") in content:
        raise RuntimeError("decision-already-present")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def main() -> int:
    if OUT.exists() or ROUTE185.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}:{ROUTE185}")
    state = load(STATE)
    if state.get("state_id") != "2026-09-01-v10.184-discovery16-stage-a-v1-adjudicated-technical-retry":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(FACTS) != str(state["scientific_contract"]["sha256"]).upper():
        raise RuntimeError("facts-hash-mismatch")
    if sha256(ROUTE184) != str(state["authoritative_route"]["sha256"]).upper():
        raise RuntimeError("route184-hash-mismatch")
    adjudication = load(ADJUDICATION)
    if adjudication.get("status") != "stage-a-v1-adjudicated-one-scientific-failure-one-common-technical-retry-authorized":
        raise RuntimeError("retry-authorization-missing")

    recorded = now()
    code = {
        "base_runner": {"path": str(BASE_RUNNER.relative_to(ROOT)), "sha256": sha256(BASE_RUNNER)},
        "retry_runner": {"path": str(RETRY_RUNNER.relative_to(ROOT)), "sha256": sha256(RETRY_RUNNER)},
        "climate_exporter": {"path": str(EXPORTER.relative_to(ROOT)), "sha256": sha256(EXPORTER)},
        "adapter_test": {"path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
    }
    summary = {
        "schema": "lia-191-discovery15-stage-a-v2-prelaunch-binding-v1",
        "status": "stage-a-v2-final-code-bound-technical-retry-ready",
        "recorded_at": recorded,
        "authorization": {"path": str(ADJUDICATION.relative_to(ROOT)), "sha256": sha256(ADJUDICATION)},
        "systems": SYSTEMS,
        "code": code,
        "scientific_contract_changed": False,
        "dynamics_authorized": False,
        "model_run_performed": False,
        "next_legal_action": "run the single versioned Stage A v2 technical retry for the fixed 15 static-admissible systems; launch no dynamics",
    }
    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-185",
        "status": "discovery15-stage-a-v2-final-code-bound-technical-retry-ready",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 15,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_184",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": summary_hash,
        "technical_retry_systems": SYSTEMS,
        "retry_runner_sha256": code["retry_runner"]["sha256"],
        "base_runner_sha256": code["base_runner"]["sha256"],
        "climate_exporter_sha256": code["climate_exporter"]["sha256"],
        "accepted_count_effect": 0,
        "next_legal_action": summary["next_legal_action"],
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    state185 = dict(state)
    state185["state_id"] = "2026-09-01-v10.185-discovery15-stage-a-v2-prelaunch-binding"
    state185["updated_at"] = recorded
    state185["authoritative_route"] = {
        "version": "v10.185",
        "path": str((ROUTE185 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.184",
    }
    state185["current_focus"] = {
        "system_id": "discovery15-technical-retry",
        "mainline_family": "population93-equilibrium-centered-planform2d-v1",
        "state": "stage-a-v2-final-code-bound-ready",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "historical_run_active": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
    }
    branches = dict(state185.get("research_branches", {}))
    branch = dict(branches["population93_equilibrium_centered_planform2d_v1"])
    branch["status"] = "stage-a-v2-final-code-bound-technical-retry-ready"
    branch["evidence"] = {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash}
    branch["next_legal_action"] = summary["next_legal_action"]
    branches["population93_equilibrium_centered_planform2d_v1"] = branch
    state185["research_branches"] = branches
    state185["next_action"] = {
        "type": "retry-discovery15-equilibrium-centered-planform2d-stage-a-v2",
        "launch_authorized": False,
        "description": summary["next_legal_action"],
    }
    state_bytes = json_bytes(state185)
    decisions_before = DECISIONS.read_bytes()
    decisions_after = append_decision(decisions_before, """## 2026-09-01 - Route v10.185: bind final Stage A v2 retry code

- Preserve the user-approved v10.184 scientific adjudication and fixed fifteen-member retry list without change.
- Bind the final short-path runtime implementation, retry runner, base runner, exporter, and focused test hashes before execution.
- Keep all dynamics, gate changes, member replacement, and accepted-count changes unauthorized.""")

    stage = Path(tempfile.mkdtemp(prefix=".discovery185_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        (route_stage / "summary.json").write_bytes(route_bytes)
        state_before = STATE.read_bytes()
        shutil.move(str(evidence_stage), str(OUT))
        shutil.move(str(route_stage), str(ROUTE185))
        try:
            DECISIONS.write_bytes(decisions_after)
            STATE.write_bytes(state_bytes)
        except Exception:
            DECISIONS.write_bytes(decisions_before)
            STATE.write_bytes(state_before)
            shutil.rmtree(OUT, ignore_errors=True)
            shutil.rmtree(ROUTE185, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
