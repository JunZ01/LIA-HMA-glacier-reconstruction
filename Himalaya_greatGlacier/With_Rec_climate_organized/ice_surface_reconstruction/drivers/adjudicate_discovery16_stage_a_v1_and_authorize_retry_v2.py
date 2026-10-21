"""Adjudicate discovery16 Stage A v1 and authorize one technical retry."""

from __future__ import annotations

import csv
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
ROUTE183 = BASE / "lia_191_current_route_v10_183/summary.json"
STAGE_A_V1 = BASE / "lia_191_discovery16_equilibrium_centered_stage_a_v1"
STAGE_A_SUMMARY = STAGE_A_V1 / "summary.json"
STAGE_A_MEMBERS = STAGE_A_V1 / "member_preparation_results.csv"
OUT = BASE / "lia_191_discovery16_stage_a_v1_adjudication_v1"
ROUTE184 = BASE / "lia_191_current_route_v10_184"
EXPORTER = DRIVER.parent / "export_system2005_historical_monthly_ti_bridge_v1.py"
RETRY_DRIVER = DRIVER.parent / "prepare_discovery15_equilibrium_centered_stage_a_v2.py"

EXPECTED = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 4588, 5770, 2559]
SCIENTIFIC_FAILURE = 4588
TECHNICAL_RETRY = [sid for sid in EXPECTED if sid != SCIENTIFIC_FAILURE]


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


def append_section(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"section-already-present:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    if OUT.exists() or ROUTE184.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}:{ROUTE184}")
    state = load(STATE)
    if state.get("state_id") != "2026-09-01-v10.183-equilibrium-operator-discovery16-staged-steady-preparation":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(FACTS) != str(state["scientific_contract"]["sha256"]).upper():
        raise RuntimeError("facts-hash-mismatch")
    if sha256(ROUTE183) != str(state["authoritative_route"]["sha256"]).upper():
        raise RuntimeError("route183-hash-mismatch")

    stage_a = load(STAGE_A_SUMMARY)
    rows = read_csv(STAGE_A_MEMBERS)
    by_id = {int(row["system_id"]): row for row in rows}
    if list(by_id) != EXPECTED or stage_a.get("status") != "stage-a-fail":
        raise RuntimeError("unexpected-stage-a-v1-evidence")
    if not stage_a.get("source_files_immutable") or stage_a.get("model_launch_performed"):
        raise RuntimeError("stage-a-v1-safety-gate-failed")
    if by_id[SCIENTIFIC_FAILURE]["static_status"] != "static-fail":
        raise RuntimeError("system4588-static-classification-drift")
    if float(by_id[SCIENTIFIC_FAILURE]["registered_length_relative_error"]) <= 0.10:
        raise RuntimeError("system4588-length-gate-evidence-drift")
    if any(by_id[sid]["static_status"] != "static-pass" for sid in TECHNICAL_RETRY):
        raise RuntimeError("technical-retry-static-admissibility-drift")
    if any(by_id[sid]["climate_status"] != "not-run" for sid in EXPECTED):
        raise RuntimeError("unexpected-climate-result-in-v1")

    recorded = now()
    summary = {
        "schema": "lia-191-discovery16-stage-a-v1-adjudication-v1",
        "status": "stage-a-v1-adjudicated-one-scientific-failure-one-common-technical-retry-authorized",
        "recorded_at": recorded,
        "stage_a_v1": {
            "summary": str(STAGE_A_SUMMARY.relative_to(ROOT)),
            "summary_sha256": sha256(STAGE_A_SUMMARY),
            "member_results": str(STAGE_A_MEMBERS.relative_to(ROOT)),
            "member_results_sha256": sha256(STAGE_A_MEMBERS),
            "static_pass_count": 15,
            "climate_package_count": 0,
        },
        "scientific_failure": {
            "system_id": SCIENTIFIC_FAILURE,
            "stage": "static-adapter-length-gate",
            "registered_length_relative_error": float(by_id[SCIENTIFIC_FAILURE]["registered_length_relative_error"]),
            "threshold": 0.10,
            "disposition": "retained in discovery16 denominator; no tuning, replacement, or retry",
        },
        "technical_failure": {
            "systems": TECHNICAL_RETRY,
            "cause": "Windows destination path length during full OGGM glacier-directory copy",
            "correction": "use one shared short-path minimal runtime clone containing frozen climate and outline inputs",
            "scientific_contract_changed": False,
            "one_exact_contract_retry_authorized": True,
        },
        "accepted_count_effect": 0,
        "dynamics_authorized": False,
        "model_run_performed": False,
        "initial_thickness_modified": False,
        "old_output_modified": False,
        "next_legal_action": "run the single versioned Stage A v2 technical retry for the fixed 15 static-admissible systems; launch no dynamics",
    }

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = append_section(facts_before, "## 487. Discovery16 Stage A v1 adjudicated", """## 487. Discovery16 Stage A v1 adjudicated (2026-09-01)

- Discovery16 Stage A v1 launched no dynamics and left every source file unchanged. Fifteen of sixteen members passed the common two-dimensional static adapter gates.
- System 4588 failed the preregistered registered-length gate with relative error `0.1016724222` against the unchanged `0.10` threshold. It remains a discovery16 scientific failure in the denominator and receives no member-specific tuning, replacement, threshold relaxation, or retry.
- The other fifteen members encountered one shared technical failure while cloning full OGGM glacier directories into a Windows path longer than the runtime limit. Their static gates passed; no climate package was produced and no model was launched.
- Route v10.184 authorizes one exact-contract Stage A v2 technical retry for those fixed fifteen systems using a shared short-path minimal runtime clone. The equilibrium operator, forcing, geometry, thresholds, and initial thickness remain unchanged. Dynamic launch remains unauthorized.""")
    decisions_after = append_section(decisions_before, "## 2026-09-01 - Route v10.184: adjudicate discovery16 Stage A v1", """## 2026-09-01 - Route v10.184: adjudicate discovery16 Stage A v1

- Classify system 4588 as a scientific static-gate failure and retain it in the discovery16 denominator without rescue.
- Classify the common long-path copy error affecting the other fifteen members as a technical failure.
- Authorize one versioned, exact-scientific-contract Stage A v2 retry for the fixed fifteen static-admissible members using the shared short-path minimal clone.
- Keep all dynamics, historical runs, member replacement, gate changes, and accepted-count changes unauthorized.""")

    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-184",
        "status": "discovery16-stage-a-v1-adjudicated-technical-retry-authorized",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 15,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_183",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": summary_hash,
        "scientific_failure_systems": [SCIENTIFIC_FAILURE],
        "technical_retry_systems": TECHNICAL_RETRY,
        "accepted_count_effect": 0,
        "next_legal_action": summary["next_legal_action"],
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    state184 = dict(state)
    state184["state_id"] = "2026-09-01-v10.184-discovery16-stage-a-v1-adjudicated-technical-retry"
    state184["updated_at"] = recorded
    state184["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    state184["authoritative_route"] = {
        "version": "v10.184",
        "path": str((ROUTE184 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.183",
    }
    state184["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 15}
    state184["current_focus"] = {
        "system_id": "discovery15-technical-retry",
        "mainline_family": "population93-equilibrium-centered-planform2d-v1",
        "state": "stage-a-v2-technical-retry-authorized",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "historical_run_active": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
    }
    branches = dict(state184.get("research_branches", {}))
    branches["population93_equilibrium_centered_planform2d_v1"] = {
        "status": "stage-a-v1-one-scientific-failure-technical-retry-authorized",
        "systems": EXPECTED,
        "scientific_failure_systems": [SCIENTIFIC_FAILURE],
        "technical_retry_systems": TECHNICAL_RETRY,
        "promoted_to_mainline": False,
        "accepted_count_effect": 0,
        "dynamic_launch_authorized": False,
        "heldout16_sealed": True,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
        "next_legal_action": summary["next_legal_action"],
    }
    state184["research_branches"] = branches
    state184["next_action"] = {
        "type": "retry-discovery15-equilibrium-centered-planform2d-stage-a-v2",
        "launch_authorized": False,
        "description": summary["next_legal_action"],
    }
    state_bytes = json_bytes(state184)

    stage = Path(tempfile.mkdtemp(prefix=".discovery184_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        source_manifest = {
            "created_at": recorded,
            "driver": str(DRIVER.relative_to(ROOT)),
            "driver_sha256": sha256(DRIVER),
            "sources": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in [ROUTE183, STAGE_A_SUMMARY, STAGE_A_MEMBERS, EXPORTER, RETRY_DRIVER]
            ],
        }
        (evidence_stage / "source_manifest.json").write_bytes(json_bytes(source_manifest))
        (route_stage / "summary.json").write_bytes(route_bytes)
        backups = {path: path.read_bytes() for path in [FACTS, DECISIONS, STATE]}
        shutil.move(str(evidence_stage), str(OUT))
        shutil.move(str(route_stage), str(ROUTE184))
        try:
            FACTS.write_bytes(facts_after)
            DECISIONS.write_bytes(decisions_after)
            STATE.write_bytes(state_bytes)
        except Exception:
            for path, content in backups.items():
                path.write_bytes(content)
            shutil.rmtree(OUT, ignore_errors=True)
            shutil.rmtree(ROUTE184, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
