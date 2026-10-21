"""Record the year-9 materialization pass and authorize one reproduction."""

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
OUT = RESULTS / "lia_191_system2005_policy_v6_year9_reproduction_authorization_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_43"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
MATERIAL = RESULTS / "lia_191_system2005_policy_v6_year9_materialization_v1/summary.json"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_year9_reproduction_v1.json"
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/reproduce_system2005_policy_v6_year9_v1.py"
EXPECTED_FACTS_SHA = "2A4F54E2DB04140AD38A3B469D40CBD76604C60F14892301F6E6BA1A7C10C525"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1024 * 1024), b""):
            d.update(b)
    return d.hexdigest().upper()


def jb(v: dict) -> bytes:
    return (json.dumps(v, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic(path: Path, data: bytes) -> None:
    t = path.with_name(path.name + ".year9_repro_auth.tmp")
    if t.exists():
        raise RuntimeError(f"stale temporary file: {t}")
    with t.open("xb") as h:
        h.write(data); h.flush(); os.fsync(h.fileno())
    os.replace(t, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    if sha256(FACTS) != EXPECTED_FACTS_SHA:
        raise RuntimeError("facts hash changed")
    old = json.loads(STATE.read_text(encoding="utf-8"))
    if old.get("state_id") != "2026-08-26-v10.42-year9-materialization-authorized":
        raise RuntimeError(f"unexpected state: {old.get('state_id')}")
    material = json.loads(MATERIAL.read_text(encoding="utf-8"))
    if material.get("status") != "year9-materialization-pass-reproduction-pending":
        raise RuntimeError("materialization is not a pass")
    for p in (CONFIG, RUNNER):
        if not p.exists():
            raise FileNotFoundError(p)
    for source, name in ((FACTS, "LIA_PROJECT_FACTS.pre.md"),
                         (STATE, "LIA_CURRENT_STATE.pre.json"),
                         (DECISIONS, "LIA_DECISION_LOG.pre.md")):
        shutil.copy2(source, OUT / name)

    section = """## 354. System-2005 Policy-V6 year-9 natural-state materialization (2026-08-26)

- The authorized nine-year replay completed `13,176` six-hour steps in
  `135.950 s` with maximum step closure `0.0055523 m3` and maximum KKT
  stationarity `4.9059e-6`. No internal gap occurred.
- The materialized volume `12,888,052,624.905638 m3`, area
  `91,771,970.3021613 m2`, connected-root length `25.550 km`, and zero gap
  count exactly equal the registered year-9 ledger values. The serialized
  state reloads with bitwise equality for section, thickness, surface, width,
  and bed arrays.
- This is a reproducible-state candidate, not yet a formal 50-year steady
  state. Exactly one independent nine-year replay is authorized to compare all
  five array families and registered diagnostics. No 50-year or historical run
  is authorized at this transition.
- Evidence:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_year9_materialization_v1/summary.json`.
""".rstrip()
    facts = FACTS.read_text(encoding="utf-8")
    if "## 354." in facts:
        raise RuntimeError("Section 354 already present")
    atomic(FACTS, (facts.rstrip() + "\n\n" + section + "\n").encode("utf-8"))

    route = {
        "schema": "lia-191-current-route-v10-43",
        "status": "policy-v6-year9-independent-reproduction-authorized",
        "population": 191, "accepted": 54, "remaining": 137,
        "active": 0, "queue": 0, "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_42",
        "system_id": 2005,
        "authorized_run": "one-independent-nine-year-reproduction-only",
        "materialization": str(MATERIAL.relative_to(ROOT)).replace("\\", "/"),
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "forbidden": ["formal-50y-steady", "historical", "duplicate-reproduction"],
    }
    rp = ROUTE_DIR / "summary.json"
    atomic(rp, jb(route))
    state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.43-year9-reproduction-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": sha256(FACTS)},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.43",
            "path": str(rp.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(rp), "supersedes": "v10.42",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 0},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-natural-year9-joint-shooting",
            "state": "independent-reproduction-authorized",
            "candidate_source_year": 9,
            "candidate_connected_root_length_m": 25550.0,
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "full_historical_authorized": False,
            "evidence": {"path": str(MATERIAL.relative_to(ROOT)).replace("\\", "/"),
                         "sha256": sha256(MATERIAL)},
        },
        "research_branches": old.get("research_branches", {}),
        "next_action": {
            "type": "system2005-year9-independent-reproduction",
            "launch_authorized": True,
            "description": "Run exactly one independent nine-year replay; no 50-year or historical run.",
        },
    }
    atomic(STATE, jb(state))
    decisions = DECISIONS.read_text(encoding="utf-8")
    note = """## 2026-08-26 — System 2005 year-9 materialization passed

- The natural year-9 state was materialized exactly and round-trips bitwise.
- One independent reproduction is authorized; longer runs remain gated.
""".strip()
    atomic(DECISIONS, (decisions.rstrip() + "\n\n" + note + "\n").encode("utf-8"))
    summary = {
        "schema": "lia-191-system2005-policy-v6-year9-reproduction-authorization-v1",
        "status": "one-independent-reproduction-authorized",
        "facts_sha256": sha256(FACTS), "route_sha256": sha256(rp),
        "materialization_sha256": sha256(MATERIAL), "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(RUNNER), "formal_50y_steady_authorized": False,
        "historical_authorized": False,
    }
    atomic(OUT / "summary.json", jb(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
