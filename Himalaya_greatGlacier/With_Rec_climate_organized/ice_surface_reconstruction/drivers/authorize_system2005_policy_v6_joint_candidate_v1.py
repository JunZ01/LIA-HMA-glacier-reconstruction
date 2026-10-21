"""Record year-9 reproduction and authorize one staged joint candidate."""

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
OUT = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_authorization_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_44"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
REPRO = RESULTS / "lia_191_system2005_policy_v6_year9_reproduction_v1/summary.json"
DESIGN = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_design_v1/summary.json"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_joint_candidate_v1.json"
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_policy_v6_joint_candidate_v1.py"
EXPECTED_FACTS_SHA = "B78FFEAAF472023EC3F05DA15010A674B80FCF6609A77D2A4AA5E98A3461AEF1"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1024 * 1024), b""):
            d.update(b)
    return d.hexdigest().upper()


def jb(v: dict) -> bytes:
    return (json.dumps(v, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic(path: Path, data: bytes) -> None:
    t = path.with_name(path.name + ".joint_auth.tmp")
    if t.exists():
        raise RuntimeError(f"stale temporary file: {t}")
    with t.open("xb") as h:
        h.write(data); h.flush(); os.fsync(h.fileno())
    os.replace(t, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    if sha256(FACTS) != EXPECTED_FACTS_SHA:
        raise RuntimeError(f"facts hash changed: {sha256(FACTS)}")
    old = json.loads(STATE.read_text(encoding="utf-8"))
    if old.get("state_id") != "2026-08-26-v10.43-year9-reproduction-authorized":
        raise RuntimeError(f"unexpected state: {old.get('state_id')}")
    reproduction = json.loads(REPRO.read_text(encoding="utf-8"))
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if reproduction.get("status") != "year9-independent-reproduction-pass":
        raise RuntimeError("reproduction not pass")
    if design.get("status") != "unique-bounded-joint-candidate-selected":
        raise RuntimeError("joint design not pass")
    for p in (CONFIG, RUNNER):
        if not p.exists():
            raise FileNotFoundError(p)
    for source, name in ((FACTS, "LIA_PROJECT_FACTS.pre.md"),
                         (STATE, "LIA_CURRENT_STATE.pre.json"),
                         (DECISIONS, "LIA_DECISION_LOG.pre.md")):
        shutil.copy2(source, OUT / name)

    section = """## 355. System-2005 Policy-V6 joint state/temperature candidate (2026-08-26)

- The independent nine-year reproduction exactly matches the first
  materialization in four registered metrics, ten numerical diagnostics, and
  all section, thickness, surface, width, and bed arrays; every maximum array
  difference is zero.
- The unchanged `delta_t=-0.125/+0.375 K` 50-year endpoints imply final-volume
  and root-length slopes of `-4.033455306e9 m3/K` and `-1500 m/K`. On the
  declared 0.0625 K grid, the unique coldest candidate whose linear volume
  prediction remains inside 2% after the capped state adjustment is
  `delta_t=-0.25 K` with all positive control-volume sections multiplied by
  exactly `1.02`. Its predicted volume change is `+1.2690%`; the next colder
  grid point predicts `+3.1867%` and is excluded before running.
- One staged candidate is authorized: static A/V/support/hypsometry/reverse-
  slope gates, then a 10-year preflight, then exactly one 50-year formal steady
  run only if every preceding gate passes. Bed, width, topology, all Stage-2
  parameters except the declared temperature offset, numerical tolerances, and
  scoring remain unchanged. Historical execution remains unauthorized.
- Evidence:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_year9_reproduction_v1/summary.json`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_joint_candidate_design_v1/summary.json`.
""".rstrip()
    facts = FACTS.read_text(encoding="utf-8")
    if "## 355." in facts:
        raise RuntimeError("Section 355 already present")
    atomic(FACTS, (facts.rstrip() + "\n\n" + section + "\n").encode("utf-8"))

    route = {
        "schema": "lia-191-current-route-v10-44",
        "status": "policy-v6-one-staged-joint-candidate-authorized",
        "population": 191, "accepted": 54, "remaining": 137,
        "active": 0, "queue": 0, "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_43",
        "system_id": 2005,
        "candidate": {"source_year": 9, "delta_t_c": -0.25, "active_section_scale": 1.02},
        "authorized_stages": ["static", "conditional-10y", "conditional-single-50y"],
        "candidate_count_max": 1,
        "historical_forbidden": True,
        "design": str(DESIGN.relative_to(ROOT)).replace("\\", "/"),
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
    }
    rp = ROUTE_DIR / "summary.json"
    atomic(rp, jb(route))
    state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.44-joint-candidate-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": sha256(FACTS)},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {"version": "v10.44",
            "path": str(rp.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(rp), "supersedes": "v10.43"},
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 0},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-natural-year9-joint-shooting",
            "state": "one-staged-joint-candidate-authorized",
            "candidate_source_year": 9, "candidate_delta_t_c": -0.25,
            "candidate_active_section_scale": 1.02,
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "full_historical_authorized": False,
            "evidence": {"path": str(DESIGN.relative_to(ROOT)).replace("\\", "/"),
                         "sha256": sha256(DESIGN)},
        },
        "research_branches": old.get("research_branches", {}),
        "next_action": {
            "type": "system2005-joint-candidate-static-preflight-formal50",
            "launch_authorized": True,
            "description": "Run static gates, conditional 10y, then at most one conditional 50y; no historical run.",
        },
    }
    atomic(STATE, jb(state))
    decisions = DECISIONS.read_text(encoding="utf-8")
    note = """## 2026-08-26 — System 2005 one joint steady candidate authorized

- Year-9 reproduction passed exactly.
- Candidate is delta_t=-0.25 K plus exact 1.02 active-section scale.
- Static and 10-year gates precede at most one 50-year run; historical remains forbidden.
""".strip()
    atomic(DECISIONS, (decisions.rstrip() + "\n\n" + note + "\n").encode("utf-8"))
    summary = {
        "schema": "lia-191-system2005-policy-v6-joint-candidate-authorization-v1",
        "status": "one-staged-joint-candidate-authorized",
        "facts_sha256": sha256(FACTS), "route_sha256": sha256(rp),
        "reproduction_sha256": sha256(REPRO), "design_sha256": sha256(DESIGN),
        "config_sha256": sha256(CONFIG), "runner_sha256": sha256(RUNNER),
        "historical_authorized": False,
    }
    atomic(OUT / "summary.json", jb(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
