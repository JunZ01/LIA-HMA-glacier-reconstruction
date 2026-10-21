"""Build the authoritative remaining-133 mechanism-exhaustion inventory."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
CONTRACT = ROOT / "LIA_EXECUTION_CONTRACT.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = DRIVER.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE134 = BASE / "lia_191_current_route_v10_134/summary.json"
OLD_ROUTE = BASE / "lia_191_current_route_v10_33/lia_191_current_route.csv"
OLD_RESCREEN = BASE / "lia_191_remaining136_latest_rescreen_v2/remaining136_latest_rescreen.csv"
ACCEPTED58 = BASE / "lia_191_accepted_58_registry_v1/accepted_58_curve_metrics.csv"
OUT = BASE / "lia_191_remaining133_mechanism_exhaustion_after_system3910_v1"
ROUTE135 = BASE / "lia_191_current_route_v10_135"

OVERRIDES = {
    902: {
        "mechanism_class": "checkpointed-conservative-scalar-historical-forward-family",
        "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system902_first_historical_bracket_result_v1/summary.json",
        "closure_reason": "colder endpoint fails LIA/modern gates and warmer endpoint is dominated",
    },
    2005: {
        "mechanism_class": "unchanged-natural-state-1d-family-closed; planform2d-research-not-promoted",
        "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v7_unchanged_natural_state_feasibility_v1/summary.json",
        "closure_reason": "initial-thickness-modified route revoked; unchanged-natural-state 1D route exhausted",
    },
    3144: {
        "mechanism_class": "true-dem-domain-repair-linear-historical-family",
        "latest_evidence": "w743hist3144_endp0p9375_v1/historical_summary.json",
        "closure_reason": "+0.9375 K scientific endpoint failure; +0.4375 K independently repeats frozen 30 s CFL failure",
    },
    3910: {
        "mechanism_class": "checkpointed-conservative-scalar-historical-forward-family",
        "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system3910_historical_peak_hard_bound_result_v1/summary.json",
        "closure_reason": "both endpoints irreversibly exceed the frozen LIA peak upper gate",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
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
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evidence_exists(value: str) -> bool:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def main() -> None:
    if OUT.exists() or ROUTE135.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}:{ROUTE135}")
    state = load_json(STATE)
    if state.get("state_id") != "2026-08-29-v10.134-system3910-historical-family-closed":
        raise RuntimeError(f"unexpected-state-id:{state.get('state_id')}")
    if sha256(ROUTE134) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route134-hash-mismatch")

    accepted_rows = read_csv(ACCEPTED58)
    accepted_ids = {int(row["system_id"]) for row in accepted_rows}
    if len(accepted_ids) != 58:
        raise RuntimeError(f"accepted58-id-count:{len(accepted_ids)}")

    old_rescreen = {int(row["system_id"]): row for row in read_csv(OLD_RESCREEN)}
    old_route = {int(row["lia_unique_id"]): row for row in read_csv(OLD_ROUTE)}
    population_ids = accepted_ids | set(old_rescreen)
    remaining_ids = sorted(population_ids - accepted_ids)
    if len(population_ids) != 191 or len(remaining_ids) != 133:
        raise RuntimeError(f"population-ledger-mismatch:{len(population_ids)}:{len(remaining_ids)}")
    if set(old_route) != population_ids:
        raise RuntimeError("complete-route-population-does-not-match-ledger")

    rows: list[dict[str, Any]] = []
    for sid in remaining_ids:
        old = old_rescreen[sid]
        route = old_route[sid]
        category = old["category"]
        if sid == 3144:
            category = "mechanism_closed"
        if category not in {"mechanism_closed", "mathematically_unreachable"}:
            raise RuntimeError(f"unresolved-current-category:{sid}:{category}")
        override = OVERRIDES.get(sid, {})
        evidence = override.get("latest_evidence", route.get("latest_evidence", ""))
        reason = override.get("closure_reason", old.get("override_reason", ""))
        mechanism = override.get("mechanism_class", route.get("mechanism_class", ""))
        rows.append({
            "system_id": sid,
            "rgi_id": old["rgi_id"],
            "current_class": category,
            "controller_state": "closed-current-branch",
            "mechanism_class": mechanism,
            "closure_reason": reason,
            "latest_evidence": evidence,
            "evidence_path_exists": evidence_exists(evidence) if evidence else False,
            "current_mechanism_actionable": False,
            "new_mechanism_required": True,
        })

    class_counts = Counter(str(row["current_class"]) for row in rows)
    mechanism_counts = Counter(str(row["mechanism_class"]) for row in rows)
    unresolved = [row for row in rows if row["current_mechanism_actionable"]]
    technical = [row for row in rows if row["current_class"] == "technical_pending_retry"]
    if unresolved or technical:
        raise RuntimeError(f"inventory-not-exhausted:{len(unresolved)}:{len(technical)}")
    if class_counts != Counter({"mechanism_closed": 120, "mathematically_unreachable": 13}):
        raise RuntimeError(f"unexpected-class-counts:{dict(class_counts)}")

    missing_evidence = [int(row["system_id"]) for row in rows if not row["evidence_path_exists"]]
    summary = {
        "schema": "lia-191-remaining133-mechanism-exhaustion-after-system3910-v1",
        "status": "current-mainline-mechanisms-exhausted-new-mechanism-required",
        "recorded_at": now(),
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "inventory_unique_systems": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "mechanism_class_counts": dict(sorted(mechanism_counts.items())),
        "current_mechanism_actionable_count": 0,
        "technical_pending_retry_count": 0,
        "mathematically_unreachable_count": 13,
        "closed_current_mechanism_count": 120,
        "missing_latest_evidence_path_count": len(missing_evidence),
        "missing_latest_evidence_system_ids": missing_evidence,
        "key_reconciliations": {
            "3144": "stale technical-pending label removed; formal scientific/solver-floor closure retained",
            "902": "checkpointed scalar historical family closed by joint LIA-modern conflict",
            "3910": "checkpointed scalar historical family closed by irreversible peak hard bound",
            "2005": "stale active label removed; unchanged-natural-state 1D mainline closed and planform2d remains research only",
        },
        "scientific_conclusion": (
            "No legally untested member remains in the currently registered geometry, inversion, zero-MB, "
            "steady-state, scalar historical ramp, pulse, late-only, or reproduction mechanisms. This does not "
            "mean the 133 systems are impossible; it means a genuinely new physical/model mechanism is required."
        ),
        "next_legal_action": (
            "read-only system-2005 1D/2D climate and area-budget parity audit to identify why the planform2d "
            "research branch gains 68.76% volume in 50 years before defining any new calibration"
        ),
        "model_run_performed": False,
        "production_temperature_modified": False,
        "accepted_count_effect": 0,
        "prohibited_source_accessed": False,
    }

    route = {
        "schema": "lia-191-current-route-v10-135",
        "status": "remaining133-current-mechanisms-exhausted-readonly-parity-audit-next",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_134",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "current_mechanism_actionable_count": 0,
        "new_mechanism_required": True,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
        "next_legal_action": summary["next_legal_action"],
    }

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append_section(facts_before, "## 439. Remaining-133 current-mechanism exhaustion audit", """## 439. Remaining-133 current-mechanism exhaustion audit (2026-08-29)

- The current ledger reconciles exactly 191 unique systems: 58 accepted and 133 remaining. The remaining inventory contains 120 systems closed under their registered mechanisms and 13 systems mathematically incompatible with the frozen historical-maximum endpoint operator.
- No technical-pending retry remains. System `3144` is a scientific/solver-floor closure, not a technical hold. Stale labels for systems `2005`, `902`, and `3910` are replaced by their later formal evidence.
- No legally untested system remains in the currently registered geometry, inversion, zero-MB, steady, scalar historical-ramp, pulse, late-only, or reproduction mechanisms. This exhausts the current mechanism family; it does not prove the remaining systems physically impossible.
- Route v10.135 authorizes no model launch. The next legal action is the already permitted read-only system-2005 1D/2D climate and area-budget parity audit, which must explain the 2D branch's `+68.76%` 50-year volume drift before any new mechanism or calibration is defined.""")
    decisions_after = append_section(decisions_before, "## 2026-08-29 - Route v10.135: remaining-133 current mechanisms exhausted", """## 2026-08-29 - Route v10.135: remaining-133 current mechanisms exhausted

- Reconciled the 58 accepted and 133 remaining systems against the latest formal evidence.
- Current mechanism classes are exhausted: 120 closed and 13 mathematically unreachable; zero active, queued, or technical-retry systems remain.
- No simulation is authorized. Continue with a read-only 1D/2D climate and area-budget parity audit for system 2005 before proposing a genuinely new mechanism.""")

    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route["result_sha256"] = summary_hash
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    state135 = dict(state)
    state135["state_id"] = "2026-08-29-v10.135-remaining133-current-mechanisms-exhausted"
    state135["updated_at"] = now()
    state135["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    state135["authoritative_route"] = {
        "version": "v10.135",
        "path": str((ROUTE135 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.134",
    }
    state135["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state135["current_focus"] = {
        "system_id": "remaining133",
        "mainline_family": "current-mechanism-exhaustion-inventory",
        "state": "hold-new-mechanism",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
    }
    state135["next_action"] = {
        "type": "audit-system2005-1d-2d-climate-area-budget-parity-v1",
        "launch_authorized": False,
        "description": summary["next_legal_action"],
    }
    state_bytes = json_bytes(state135)

    report_lines = [
        "# Remaining-133 Current-Mechanism Exhaustion Audit",
        "",
        "## Result",
        "",
        "The ledger closes exactly: 58 accepted + 133 remaining = 191 unique LIA systems.",
        "",
        "- 120 systems: registered mechanism closed",
        "- 13 systems: mathematically incompatible with the frozen endpoint operator",
        "- 0 systems: active, queued, or technically pending retry",
        "- 0 systems: legally actionable inside the current registered mechanisms",
        "",
        "This is mechanism exhaustion, not proof that the remaining systems are physically impossible.",
        "",
        "## Next Route",
        "",
        "Perform the read-only system-2005 1D/2D climate and area-budget parity audit. No OGGM, EnKF, production-temperature change, or initial-thickness modification is authorized.",
    ]

    fields = [
        "system_id", "rgi_id", "current_class", "controller_state", "mechanism_class",
        "closure_reason", "latest_evidence", "evidence_path_exists",
        "current_mechanism_actionable", "new_mechanism_required",
    ]
    stage = Path(tempfile.mkdtemp(prefix=".remaining133_audit_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(summary_bytes)
        write_csv(stage / "evidence/remaining133_mechanism_inventory.csv", rows, fields)
        write_csv(
            stage / "evidence/mechanism_class_counts.csv",
            [{"mechanism_class": key, "count": value} for key, value in sorted(mechanism_counts.items())],
            ["mechanism_class", "count"],
        )
        (stage / "evidence/FINAL_RESULT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        manifest = {
            "created_at": now(),
            "driver": str(DRIVER.relative_to(ROOT)),
            "driver_sha256": sha256(DRIVER),
            "execution_contract_sha256": sha256(CONTRACT),
            "sources": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in [ROUTE134, OLD_ROUTE, OLD_RESCREEN, ACCEPTED58]
            ],
        }
        (stage / "evidence/source_manifest.json").write_bytes(json_bytes(manifest))
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(state_bytes)
        OUT.mkdir(parents=False)
        ROUTE135.mkdir(parents=False)
        for item in (stage / "evidence").iterdir():
            os.replace(item, OUT / item.name)
        os.replace(stage / "route/summary.json", ROUTE135 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE135, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({
        "status": "complete",
        "summary": summary,
        "route": "v10.135",
        "validator": json.loads(validation.stdout),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
