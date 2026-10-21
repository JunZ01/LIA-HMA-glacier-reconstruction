"""Adjudicate the fixed discovery15 Stage B short qualification batch."""

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
ROUTE187 = BASE / "lia_191_current_route_v10_187/summary.json"
AUTH = BASE / "lia_191_discovery15_stage_b_short_qualification_authorization_v1"
CONFIG_MANIFEST = AUTH / "config_manifest.json"
BATCH = BASE / "lia_191_discovery15_stage_b_short_qualification_v1"
BATCH_SUMMARY = BATCH / "summary.json"
OUT = BASE / "lia_191_discovery15_stage_b_terminal_adjudication_v1"
ROUTE188 = BASE / "lia_191_current_route_v10_188"
SYSTEMS = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 5770, 2559]
PASSED = [988, 3185, 801, 4959, 2570, 705, 2727, 4491, 3016, 3660, 2177, 5770, 2559]
TECHNICAL = [2086, 2534]


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


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append_section(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"section-already-present:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def passed_metrics(system_id: int, expected_hash: str) -> dict[str, Any]:
    path = BATCH / "members" / f"system_{system_id}" / "summary.json"
    if sha256(path) != expected_hash.upper():
        raise RuntimeError(f"member-summary-hash-mismatch:{system_id}")
    value = load(path)
    if value.get("status") != "qualification-preflight-pass":
        raise RuntimeError(f"member-not-pass:{system_id}")
    if not all(value["overall_checks"].values()):
        raise RuntimeError(f"member-overall-check-failed:{system_id}")
    for phase_name in ("zero_mb", "climate_response"):
        phase = value[phase_name]
        if phase.get("status") != "pass" or not all(phase["checks"].values()):
            raise RuntimeError(f"member-phase-check-failed:{system_id}:{phase_name}")
    if value.get("steady_run") or value.get("historical_run") or value.get("accepted_count_effect") != 0:
        raise RuntimeError(f"member-boundary-violation:{system_id}")
    zero = value["zero_mb"]
    climate = value["climate_response"]
    return {
        "system_id": system_id,
        "summary": relative(path),
        "summary_sha256": expected_hash.upper(),
        "zero_mb": {
            "completed_years": zero["completed_years"],
            "accepted_steps": zero["accepted_steps"],
            "rejected_steps": zero["rejected_steps"],
            "minimum_accepted_dt_seconds": zero["minimum_accepted_dt_seconds"],
            "initial_volume_m3": zero["initial_volume_m3"],
            "final_volume_m3": zero["final_volume_m3"],
            "cumulative_mass_budget_residual_m3": zero["cumulative_mass_budget_residual_m3"],
            "outside_domain_maximum_thickness_m": zero["outside_domain_maximum_thickness_m"],
            "checkpoint_manifest_sha256": zero["checkpoint_restore"]["manifest_sha256"],
        },
        "climate_response": {
            "completed_years": climate["completed_years"],
            "accepted_steps": climate["accepted_steps"],
            "rejected_steps": climate["rejected_steps"],
            "minimum_accepted_dt_seconds": climate["minimum_accepted_dt_seconds"],
            "initial_volume_m3": climate["initial_volume_m3"],
            "final_volume_m3": climate["final_volume_m3"],
            "cumulative_source_m3": climate["cumulative_source_m3"],
            "cumulative_mass_budget_residual_m3": climate["cumulative_mass_budget_residual_m3"],
            "outside_domain_maximum_thickness_m": climate["outside_domain_maximum_thickness_m"],
            "checkpoint_manifest_sha256": climate["checkpoint_restore"]["manifest_sha256"],
        },
        "peak_torch_memory_reserved_mib": value["peak_torch_memory_reserved_mib"],
    }


def technical_metrics(system_id: int) -> dict[str, Any]:
    member = BATCH / "members" / f"system_{system_id}"
    stderr = BATCH / "logs" / f"system_{system_id}.stderr.log"
    history = member / "zero_mb_0p1y_step_history.csv"
    text = stderr.read_text(encoding="utf-8", errors="replace")
    if "zero_mb_0p1y-minimum-adaptive-dt-exhausted" not in text:
        raise RuntimeError(f"unexpected-technical-failure:{system_id}")
    if (member / "summary.json").exists():
        raise RuntimeError(f"technical-failure-has-summary:{system_id}")
    rows = read_history(history)
    accepted = [row for row in rows if row["accepted"].lower() == "true"]
    rejected = [row for row in rows if row["accepted"].lower() == "false"]
    completed_years = max(float(row["elapsed_before_years"]) for row in rows)
    minimum_accepted_dt = (
        min(float(row["dt_seconds"]) for row in accepted) if accepted else None
    )
    return {
        "system_id": system_id,
        "classification": "technical-failure",
        "phase": "zero_mb_0p1y",
        "error": "minimum-adaptive-dt-exhausted",
        "completed_years_before_stop": completed_years,
        "accepted_history_rows": len(accepted),
        "rejected_history_rows": len(rejected),
        "minimum_accepted_dt_seconds": minimum_accepted_dt,
        "stderr": relative(stderr),
        "stderr_sha256": sha256(stderr),
        "history": relative(history),
        "history_sha256": sha256(history),
        "formal_scientific_summary_present": False,
        "retry_authorized": False,
    }


def main() -> int:
    if OUT.exists() or ROUTE188.exists():
        raise FileExistsError("v10.188-versioned-output-exists")
    state = load(STATE)
    if state.get("state_id") != "2026-09-01-v10.187-discovery15-stage-b-short-qualification-authorized":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(FACTS) != str(state["scientific_contract"]["sha256"]).upper():
        raise RuntimeError("facts-hash-mismatch")
    if sha256(ROUTE187) != str(state["authoritative_route"]["sha256"]).upper():
        raise RuntimeError("route187-hash-mismatch")
    batch = load(BATCH_SUMMARY)
    if batch.get("status") != "terminal-with-failures":
        raise RuntimeError("batch-not-terminal-with-failures")
    if batch.get("systems") != SYSTEMS or batch.get("pass_systems") != PASSED or batch.get("failed_systems") != TECHNICAL:
        raise RuntimeError("batch-terminal-membership-drift")
    if batch.get("pass_count") != 13 or batch.get("failed_count") != 2:
        raise RuntimeError("batch-terminal-count-drift")
    if batch.get("steady_run") or batch.get("historical_run") or batch.get("accepted_count_effect") != 0:
        raise RuntimeError("batch-boundary-violation")

    record_by_system = {int(record["system_id"]): record for record in batch["records"]}
    passed = [passed_metrics(system_id, record_by_system[system_id]["summary_sha256"]) for system_id in PASSED]
    technical = [technical_metrics(system_id) for system_id in TECHNICAL]

    manifest = load(CONFIG_MANIFEST)
    for item in manifest["configs"]:
        config_path = AUTH / item["config"]
        if sha256(config_path) != str(item["sha256"]).upper():
            raise RuntimeError(f"config-hash-drift:{item['system_id']}")
        config = load(config_path)
        for key in ("geology", "climate_package", "static_summary", "upstream_evidence", "upstream_config_snapshot", "launch_route"):
            if sha256(ROOT / config[key]) != str(config[f"{key}_sha256"]).upper():
                raise RuntimeError(f"source-hash-drift:{item['system_id']}:{key}")

    recorded = now()
    maximum_memory = max(float(item["peak_torch_memory_reserved_mib"]) for item in passed)
    summary = {
        "schema": "lia-191-discovery15-stage-b-terminal-adjudication-v1",
        "status": "stage-b-terminal-13-pass-2-technical-fail-retry-decision-required",
        "recorded_at": recorded,
        "discovery16_denominator": 16,
        "stage_a_scientific_failure_systems": [4588],
        "stage_b_denominator": 15,
        "stage_b_pass_systems": PASSED,
        "stage_b_technical_failure_systems": TECHNICAL,
        "stage_b_scientific_failure_systems": [],
        "stage_b_pass_count": 13,
        "stage_b_technical_failure_count": 2,
        "stage_b_observed_pass_fraction": 13 / 15,
        "passed_members": passed,
        "technical_failures": technical,
        "maximum_peak_torch_memory_reserved_mib_among_passes": maximum_memory,
        "all_passed_members_completed_zero_mb_years": 0.1,
        "all_passed_members_completed_climate_response_years": 1.0,
        "all_passed_members_exact_outside_domain_zero": all(
            item[phase]["outside_domain_maximum_thickness_m"] == 0.0
            for item in passed for phase in ("zero_mb", "climate_response")
        ),
        "all_registered_sources_immutable": True,
        "member_substitution_performed": False,
        "parameter_adjustment_performed": False,
        "retry_performed": False,
        "retry_authorized": False,
        "steady_authorized": False,
        "historical_authorized": False,
        "accepted_count_effect": 0,
        "batch_summary": {"path": relative(BATCH_SUMMARY), "sha256": sha256(BATCH_SUMMARY)},
        "next_legal_action": "explicit user decision whether to authorize one exact-contract Stage B retry for systems 2086 and 2534; no steady or historical launch",
    }
    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-188",
        "status": "discovery15-stage-b-terminal-retry-decision-required",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_187",
        "result": relative(OUT / "summary.json"),
        "result_sha256": summary_hash,
        "stage_b_pass_systems": PASSED,
        "stage_b_technical_failure_systems": TECHNICAL,
        "stage_b_scientific_failure_systems": [],
        "retry_authorized": False,
        "steady_authorized": False,
        "historical_authorized": False,
        "accepted_count_effect": 0,
        "next_legal_action": summary["next_legal_action"],
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = append_section(facts_before, "## 490. Discovery15 Stage B terminal result", """## 490. Discovery15 Stage B terminal result (2026-09-01)

- The fixed fifteen-member Stage B batch is terminal. Thirteen systems pass both the 0.1-year zero-MB and one-year equilibrium-centered MonthlyTI qualifications with every frozen input, runtime, finite/nonnegative, exterior-zero, mass-closure, adaptive-error, checkpoint, and source-immutability gate true.
- Systems 2086 and 2534 stop in the zero-MB phase because the registered minimum adaptive timestep is exhausted. They have no formal scientific summary and are classified as technical failures, not scientific failures. Their stderr and partial histories are retained; neither was retried, tuned, or replaced.
- System 4588 remains the prior Stage A scientific failure in the frozen discovery16 denominator. Stage B therefore has 13 observed passes among 15 launched members, two unresolved technical failures, and no Stage B scientific failure yet.
- Route v10.188 records the terminal result and requires an explicit user decision before any exact-contract retry. It authorizes no retry, 50-year steady run, historical chain, member substitution, parameter or gate change, thickness modification, promotion, or accepted-count change.""")
    decisions_after = append_section(decisions_before, "## 2026-09-01 - Route v10.188: adjudicate discovery15 Stage B terminal result", """## 2026-09-01 - Route v10.188: adjudicate discovery15 Stage B terminal result

- Record thirteen complete qualification passes and two zero-MB minimum-timestep technical failures.
- Preserve systems 2086 and 2534 as unresolved technical failures with their partial evidence; do not relabel them scientific failures and do not retry under this route.
- Require an explicit decision before any one-time exact-contract retry.
- Keep all steady, historical, tuning, substitution, thickness, promotion, and accepted-count actions unauthorized.""")

    state188 = dict(state)
    state188["state_id"] = "2026-09-01-v10.188-discovery15-stage-b-terminal-retry-decision-required"
    state188["updated_at"] = recorded
    state188["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state188["authoritative_route"] = {"version": "v10.188", "path": relative(ROUTE188 / "summary.json"), "sha256": route_hash, "supersedes": "v10.187"}
    state188["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state188["current_focus"] = {
        "system_id": "discovery15-stage-b",
        "mainline_family": "population93-equilibrium-centered-planform2d-v1",
        "state": "terminal-13-pass-2-technical-fail-retry-decision-required",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "historical_run_active": False,
        "evidence": {"path": relative(OUT / "summary.json"), "sha256": summary_hash},
    }
    branches = dict(state188["research_branches"])
    branch = dict(branches["population93_equilibrium_centered_planform2d_v1"])
    branch.update({
        "status": "stage-b-terminal-13-pass-2-technical-fail-retry-decision-required",
        "dynamic_launch_authorized": False,
        "stage_b_passed_systems": PASSED,
        "technical_retry_systems": TECHNICAL,
        "stage_b_scientific_failure_systems": [],
        "evidence": {"path": relative(OUT / "summary.json"), "sha256": summary_hash},
        "next_legal_action": summary["next_legal_action"],
    })
    branches["population93_equilibrium_centered_planform2d_v1"] = branch
    state188["research_branches"] = branches
    state188["next_action"] = {"type": "decision-required-discovery2-stage-b-exact-contract-retry", "launch_authorized": False, "description": summary["next_legal_action"]}
    state_bytes = json_bytes(state188)

    stage = Path(tempfile.mkdtemp(prefix=".discovery188_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        (evidence_stage / "source_manifest.json").write_bytes(json_bytes({
            "created_at": recorded,
            "driver": relative(DRIVER),
            "driver_sha256": sha256(DRIVER),
            "sources": [{"path": relative(path), "sha256": sha256(path)} for path in [ROUTE187, CONFIG_MANIFEST, BATCH_SUMMARY]],
        }))
        (route_stage / "summary.json").write_bytes(route_bytes)
        backups = {path: path.read_bytes() for path in [FACTS, DECISIONS, STATE]}
        shutil.move(str(evidence_stage), str(OUT))
        shutil.move(str(route_stage), str(ROUTE188))
        try:
            FACTS.write_bytes(facts_after)
            DECISIONS.write_bytes(decisions_after)
            STATE.write_bytes(state_bytes)
        except Exception:
            for path, content in backups.items():
                path.write_bytes(content)
            shutil.rmtree(OUT, ignore_errors=True)
            shutil.rmtree(ROUTE188, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
