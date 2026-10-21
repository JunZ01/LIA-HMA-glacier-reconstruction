"""Record the discovery16 Stage A terminal result and require a Stage B decision."""

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
ROUTE185 = BASE / "lia_191_current_route_v10_185/summary.json"
ADJUDICATION184 = BASE / "lia_191_discovery16_stage_a_v1_adjudication_v1/summary.json"
STAGE_A_V1 = BASE / "lia_191_discovery16_equilibrium_centered_stage_a_v1/summary.json"
STAGE_A_V2_DIR = BASE / "lia_191_discovery15_equilibrium_centered_stage_a_v2"
STAGE_A_V2 = STAGE_A_V2_DIR / "summary.json"
STAGE_A_V2_MEMBERS = STAGE_A_V2_DIR / "member_preparation_results.csv"
OUT = BASE / "lia_191_discovery16_stage_a_terminal_adjudication_v1"
ROUTE186 = BASE / "lia_191_current_route_v10_186"
PASSED = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 5770, 2559]
FAILED = [4588]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append_section(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"section-already-present:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def maximum(rows: list[dict[str, str]], key: str) -> float:
    return max(float(row[key]) for row in rows)


def minimum(rows: list[dict[str, str]], key: str) -> float:
    return min(float(row[key]) for row in rows)


def main() -> int:
    if OUT.exists() or ROUTE186.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}:{ROUTE186}")
    state = load(STATE)
    if state.get("state_id") != "2026-09-01-v10.185-discovery15-stage-a-v2-prelaunch-binding":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(FACTS) != str(state["scientific_contract"]["sha256"]).upper():
        raise RuntimeError("facts-hash-mismatch")
    if sha256(ROUTE185) != str(state["authoritative_route"]["sha256"]).upper():
        raise RuntimeError("route185-hash-mismatch")

    v2 = load(STAGE_A_V2)
    rows = read_csv(STAGE_A_V2_MEMBERS)
    if v2.get("status") != "stage-a-15-of-15-pass":
        raise RuntimeError("stage-a-v2-not-terminal-pass")
    if [int(row["system_id"]) for row in rows] != PASSED:
        raise RuntimeError("stage-a-v2-membership-drift")
    if any(row["stage_a_status"] != "pass" for row in rows):
        raise RuntimeError("stage-a-v2-member-failure")
    if not v2.get("source_files_immutable") or v2.get("model_launch_performed"):
        raise RuntimeError("stage-a-v2-safety-gate-failed")

    metrics = {
        "maximum_area_relative_error": maximum(rows, "area_relative_error"),
        "maximum_volume_relative_error": maximum(rows, "volume_relative_error"),
        "maximum_registered_length_relative_error": maximum(rows, "registered_length_relative_error"),
        "minimum_free_domain_m": minimum(rows, "minimum_ice_to_grid_edge_m"),
        "maximum_formula_absolute_error_m_ice_s": maximum(rows, "formula_absolute_error_m_ice_s"),
        "maximum_formula_relative_error": maximum(rows, "formula_relative_error"),
    }
    recorded = now()
    summary = {
        "schema": "lia-191-discovery16-stage-a-terminal-adjudication-v1",
        "status": "stage-a-terminal-15-pass-1-scientific-fail-stage-b-decision-required",
        "recorded_at": recorded,
        "discovery16_denominator": 16,
        "stage_a_passed_systems": PASSED,
        "stage_a_scientific_failure_systems": FAILED,
        "stage_a_passed_count": 15,
        "stage_a_scientific_failed_count": 1,
        "stage_a_pass_rate": 15 / 16,
        "metrics": metrics,
        "evidence": {
            "v1": {"path": str(STAGE_A_V1.relative_to(ROOT)), "sha256": sha256(STAGE_A_V1)},
            "v1_adjudication": {"path": str(ADJUDICATION184.relative_to(ROOT)), "sha256": sha256(ADJUDICATION184)},
            "v2": {"path": str(STAGE_A_V2.relative_to(ROOT)), "sha256": sha256(STAGE_A_V2)},
            "v2_members": {"path": str(STAGE_A_V2_MEMBERS.relative_to(ROOT)), "sha256": sha256(STAGE_A_V2_MEMBERS)},
        },
        "source_files_immutable": True,
        "dynamics_run": False,
        "stage_b_qualification_authorized": False,
        "accepted_count_effect": 0,
        "next_legal_action": "explicit user decision whether to authorize the fixed 15-member short zero-MB plus one-year equilibrium-centered qualification batch",
    }

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = append_section(facts_before, "## 488. Discovery16 Stage A terminal result", """## 488. Discovery16 Stage A terminal result (2026-09-01)

- The single authorized Stage A v2 technical retry passes for all fixed fifteen static-admissible systems. It creates 15 immutable two-dimensional static adapters and 15 exact 1441-1470 equilibrium-centered MonthlyTI packages; no dynamics ran.
- Across those fifteen, maximum area and volume adapter errors are exactly zero, maximum registered-length error is `5.559994%`, minimum free domain is `5.5165 km`, and maximum MonthlyTI formula absolute and relative errors are zero. Every source file remains unchanged.
- Discovery16 therefore closes Stage A with 15 passes and one preregistered scientific failure, system 4588, retained in the 16-member denominator. No member was replaced or tuned and the accepted count remains 58.
- Route v10.186 records the terminal Stage A result and requires an explicit user decision before any short zero-MB or one-year equilibrium-centered dynamic qualification. No Stage B, steady, historical, or accepted-count action is authorized.""")
    decisions_after = append_section(decisions_before, "## 2026-09-01 - Route v10.186: record discovery16 Stage A terminal result", """## 2026-09-01 - Route v10.186: record discovery16 Stage A terminal result

- Record 15 Stage A passes and one scientific failure in the frozen discovery16 denominator.
- Preserve the common operator, inputs, gates, and no-member-rescue policy.
- Set the branch to decision-required; do not launch Stage B or any later dynamics without explicit user authorization.
- Keep the accepted count at 58.""")
    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-186",
        "status": "discovery16-stage-a-terminal-stage-b-decision-required",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_185",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": summary_hash,
        "stage_a_passed_systems": PASSED,
        "stage_a_scientific_failure_systems": FAILED,
        "stage_b_qualification_authorized": False,
        "accepted_count_effect": 0,
        "next_legal_action": summary["next_legal_action"],
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    state186 = dict(state)
    state186["state_id"] = "2026-09-01-v10.186-discovery16-stage-a-terminal-stage-b-decision-required"
    state186["updated_at"] = recorded
    state186["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state186["authoritative_route"] = {"version": "v10.186", "path": str((ROUTE186 / "summary.json").relative_to(ROOT)), "sha256": route_hash, "supersedes": "v10.185"}
    state186["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state186["current_focus"] = {
        "system_id": "discovery16",
        "mainline_family": "population93-equilibrium-centered-planform2d-v1",
        "state": "decision-required-stage-b-short-qualification",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "historical_run_active": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
    }
    branches = dict(state186.get("research_branches", {}))
    branch = dict(branches["population93_equilibrium_centered_planform2d_v1"])
    branch.update({
        "status": "stage-a-terminal-15-pass-1-scientific-fail-stage-b-decision-required",
        "stage_a_passed_systems": PASSED,
        "scientific_failure_systems": FAILED,
        "technical_retry_systems": [],
        "dynamic_launch_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
        "next_legal_action": summary["next_legal_action"],
    })
    branches["population93_equilibrium_centered_planform2d_v1"] = branch
    state186["research_branches"] = branches
    state186["next_action"] = {"type": "decision-required-discovery15-stage-b-short-qualification", "launch_authorized": False, "description": summary["next_legal_action"]}
    state_bytes = json_bytes(state186)

    stage = Path(tempfile.mkdtemp(prefix=".discovery186_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        (evidence_stage / "source_manifest.json").write_bytes(json_bytes({
            "created_at": recorded,
            "driver": str(DRIVER.relative_to(ROOT)),
            "driver_sha256": sha256(DRIVER),
            "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [ROUTE185, STAGE_A_V1, ADJUDICATION184, STAGE_A_V2, STAGE_A_V2_MEMBERS]],
        }))
        (route_stage / "summary.json").write_bytes(route_bytes)
        backups = {path: path.read_bytes() for path in [FACTS, DECISIONS, STATE]}
        shutil.move(str(evidence_stage), str(OUT))
        shutil.move(str(route_stage), str(ROUTE186))
        try:
            FACTS.write_bytes(facts_after)
            DECISIONS.write_bytes(decisions_after)
            STATE.write_bytes(state_bytes)
        except Exception:
            for path, content in backups.items():
                path.write_bytes(content)
            shutil.rmtree(OUT, ignore_errors=True)
            shutil.rmtree(ROUTE186, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
