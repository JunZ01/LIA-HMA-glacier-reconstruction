"""Atomically authorize one bounded year-9 system-2005 materialization."""

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
OUT = RESULTS / "lia_191_system2005_policy_v6_year9_authorization_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_42"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
AUDIT = RESULTS / "lia_191_system2005_policy_v6_year9_candidate_audit_v1/summary.json"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_year9_materialization_v1.json"
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/materialize_system2005_policy_v6_year9_v1.py"
EXPECTED_FACTS_SHA = "BC18E5365F4FB8720959A503883FF28677F7A93CD2AA9B42DFB5AE05E8191FF9"
EXPECTED_STATE_ID = "2026-08-26-v10.41-policy-v6-readonly-audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".year9_authorize.tmp")
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
    for path in (FACTS, STATE, DECISIONS, AUDIT, CONFIG, RUNNER):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(FACTS) != EXPECTED_FACTS_SHA:
        raise RuntimeError(f"facts pre-hash changed: {sha256(FACTS)}")
    state_old = json.loads(STATE.read_text(encoding="utf-8"))
    if state_old.get("state_id") != EXPECTED_STATE_ID:
        raise RuntimeError(f"unexpected state: {state_old.get('state_id')}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "year9-unique-natural-candidate-selected":
        raise RuntimeError("year9 audit is not a pass")

    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 353." in facts_text:
        raise RuntimeError("facts already contain Section 353")
    section = """## 353. System-2005 Policy-V6 natural year-9 candidate selection (2026-08-26)

- A read-only audit of the unchanged 50-year cold `delta_t=-0.125 K` ledger
  selects source year 9 as the unique natural entry for the restored 20%
  separation gate. Its connected-root length is `25.550 km`, volume is
  `12,888,052,624.905638 m3`, area is `91,771,970.3021613 m2`, free downstream
  domain is `5.900 km`, terminal thickness is zero, and no internal gap is open.
- From that state, the unchanged source trajectory retains `25.550 km` for
  another 10, 20, 24, and 29 years, with volume changes of `-0.0631%`,
  `+0.1658%`, `-0.0142%`, and `-0.5306%`. At 30 years the length becomes
  `25.450 km`, below the continuous `25.450125918 km` Policy-V6 threshold.
  Therefore the year-9 state is an evidence-supported shooting entry, not a
  claimed 50-year steady pass under the same climate.
- Exactly one nine-year materialization from the original registered state is
  authorized, using the scalar-order-preserving graph-reuse solver already
  proven bitwise equal to the original arithmetic. It must reproduce the four
  registered year-9 metrics exactly, serialize and reload bitwise, and launch
  no 50-year steady or historical run. Independent reproduction remains gated
  on this first materialization.
- Evidence and frozen configuration are
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_year9_candidate_audit_v1/summary.json`
  and
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_year9_materialization_v1.json`.
""".rstrip()

    for source, name in ((FACTS, "LIA_PROJECT_FACTS.pre.md"),
                         (STATE, "LIA_CURRENT_STATE.pre.json"),
                         (DECISIONS, "LIA_DECISION_LOG.pre.md")):
        shutil.copy2(source, OUT / name)
    atomic_write(FACTS, (facts_text.rstrip() + "\n\n" + section + "\n").encode("utf-8"))
    facts_sha = sha256(FACTS)

    route = {
        "schema": "lia-191-current-route-v10-42",
        "status": "policy-v6-year9-natural-state-materialization-authorized",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queue": 0,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_41",
        "system_id": 2005,
        "candidate_source_year": 9,
        "candidate_connected_root_length_m": 25550.0,
        "minimum_connected_root_length_m": 25450.125918037134,
        "authorized_run": "one-nine-year-natural-state-materialization-only",
        "forbidden": ["formal-50y-steady", "historical", "duplicate-materialization"],
        "evidence": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, json_bytes(route))

    state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.42-year9-materialization-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": facts_sha},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.42",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(route_path),
            "supersedes": "v10.41",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 0},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-natural-year9-joint-shooting",
            "state": "materialization-authorized",
            "registered_lia_length_m": 25985.52229946727,
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "candidate_source_year": 9,
            "candidate_connected_root_length_m": 25550.0,
            "full_historical_authorized": False,
            "evidence": {
                "path": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(AUDIT),
            },
        },
        "research_branches": state_old.get("research_branches", {}),
        "next_action": {
            "type": "system2005-year9-natural-state-materialization",
            "launch_authorized": True,
            "description": "Run exactly one nine-year materialization; no 50-year steady or historical run.",
        },
    }
    atomic_write(STATE, json_bytes(state))

    decision = """## 2026-08-26 — System 2005 natural year-9 materialization authorized

- The unchanged cold-run ledger selects year 9 at 25.550 km with zero internal
  gaps as the unique Policy-V6 shooting entry.
- One nine-year materialization is authorized. Formal 50-year steady,
  historical execution, and duplicate materialization remain unauthorized.
""".strip()
    decisions = DECISIONS.read_text(encoding="utf-8")
    if "System 2005 natural year-9 materialization authorized" in decisions:
        raise RuntimeError("decision already recorded")
    atomic_write(DECISIONS, (decisions.rstrip() + "\n\n" + decision + "\n").encode("utf-8"))

    summary = {
        "schema": "lia-191-system2005-policy-v6-year9-authorization-v1",
        "status": "one-year9-materialization-authorized",
        "facts_sha256": facts_sha,
        "route_sha256": sha256(route_path),
        "audit_sha256": sha256(AUDIT),
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(RUNNER),
        "simulation_launched": False,
        "formal_50y_steady_authorized": False,
        "historical_authorized": False,
    }
    atomic_write(OUT / "summary.json", json_bytes(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
