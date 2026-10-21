"""Close uniform lift and authorize one temperature-only system-2005 candidate."""

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
OUT = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4_authorization"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_47"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
V3 = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_v3/summary.json"
DESIGN = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4_design_v2/summary.json"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_temperature_correction_v4.json"
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_policy_v6_temperature_correction_v4.py"
EXPECTED_FACTS_SHA = "D42259BCDB3DD3984AF79A4270A622365A1DEE1EF9BC80519441A31D7EC22E44"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".v4auth.tmp")
    if temp.exists():
        raise RuntimeError(f"stale temp:{temp}")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    if sha256(FACTS) != EXPECTED_FACTS_SHA:
        raise RuntimeError(f"facts changed:{sha256(FACTS)}")
    old_state = json.loads(STATE.read_text(encoding="utf-8"))
    if old_state.get("state_id") != "2026-08-26-v10.46-joint-candidate-v3-authorized":
        raise RuntimeError("controller state changed")
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if v3.get("status") != "formal-50y-fail-mechanism-closed":
        raise RuntimeError("v3 final result missing")
    if design.get("status") != "unique-temperature-only-candidate-selected":
        raise RuntimeError("temperature design not pass")
    for path in (CONFIG, RUNNER):
        if not path.exists():
            raise FileNotFoundError(path)
    for source, name in (
        (FACTS, "LIA_PROJECT_FACTS.pre.md"),
        (STATE, "LIA_CURRENT_STATE.pre.json"),
        (DECISIONS, "LIA_DECISION_LOG.pre.md"),
    ):
        shutil.copy2(source, OUT / name)

    section = """## 358. System-2005 Policy-V6 temperature-only correction v4 (2026-08-26)

- Response-corrected uniform-thickness v3 passed its static and 10-year gates,
  then completed 50 years with clean length (`25.550 km`), domain (`5.900 km`),
  terminal, finite, and nonnegative diagnostics. Its volume change was
  `+2.97653%`, so it fails the unchanged 2% gate. The uniform-lift family is
  closed; no further initial-thickness adjustment is authorized.
- Candidate v4 reuses the serialized v2 initial state byte-for-byte (SHA256
  `1B8D9E606538F69E5FE19BEAAD75ACB6DF0C8DD77731F3703BDC8700F1C4DA24`).
  It changes no geometry, thickness, bed, width, topology, precipitation,
  Stage-2 physics, numerical threshold, or score. The only change is the
  temperature offset from `-0.25000 K` to `-0.21875 K`.
- The registered cold/warm response predicts 50-year volume change `+1.7992%`
  and continuous/lattice root lengths `25.503/25.500 km`. The next warmer
  half-step maps to `25.450 km`, below the continuous Policy-V6 lower bound
  `25.450125918 km`, so no scalar scan or warmer continuation is authorized.
- Exactly one staged v4 is authorized: a 10-year preflight followed, only on
  pass, by one 50-year formal steady run. Historical execution remains
  forbidden. If v4 fails, this response-corrected temperature route closes.
- Evidence:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_joint_candidate_v3/summary.json`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_temperature_correction_v4_design_v2/summary.json`.
""".rstrip()
    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 358." in facts_text:
        raise RuntimeError("facts section 358 already exists")
    atomic_write(FACTS, (facts_text.rstrip() + "\n\n" + section + "\n").encode("utf-8"))

    route = {
        "schema": "lia-191-current-route-v10-47",
        "status": "policy-v6-temperature-only-candidate-v4-authorized",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queue": 0,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_46",
        "system_id": 2005,
        "candidate": {
            "initial_state_modified": False,
            "initial_state_sha256": design["candidate"]["initial_state_sha256"],
            "delta_t_c": design["candidate"]["delta_t_c"],
        },
        "authorized_stages": ["conditional-10y", "conditional-single-50y"],
        "candidate_count_max": 1,
        "family_closes_if_fail": True,
        "historical_forbidden": True,
        "design": str(DESIGN.relative_to(ROOT)).replace("\\", "/"),
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, json_bytes(route))
    new_state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.47-temperature-correction-v4-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": sha256(FACTS)},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.47",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(route_path),
            "supersedes": "v10.46",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 0},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-temperature-only-joint-shooting",
            "state": "one-staged-temperature-correction-v4-authorized",
            "candidate_source_year": 9,
            "candidate_initial_state_modified": False,
            "candidate_delta_t_c": design["candidate"]["delta_t_c"],
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "full_historical_authorized": False,
            "evidence": {
                "path": str(DESIGN.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(DESIGN),
            },
        },
        "research_branches": old_state.get("research_branches", {}),
        "next_action": {
            "type": "system2005-temperature-correction-v4-preflight-formal50",
            "launch_authorized": True,
            "description": "Reuse the frozen v2 state; run one -0.21875 K 10y preflight and conditional 50y formal candidate; no historical run.",
        },
    }
    atomic_write(STATE, json_bytes(new_state))
    decision_text = DECISIONS.read_text(encoding="utf-8")
    note = """## 2026-08-26 — System 2005 temperature-only correction v4

- Uniform initial-thickness corrections are closed after v2/v3 volume misses.
- Reuse v2 state without modification; change only delta-T to -0.21875 K.
- One conditional 10-year plus one 50-year run is authorized; no historical run.
""".strip()
    atomic_write(DECISIONS, (decision_text.rstrip() + "\n\n" + note + "\n").encode("utf-8"))
    summary = {
        "schema": "lia-191-system2005-policy-v6-temperature-correction-v4-authorization",
        "status": "temperature-only-v4-authorized",
        "facts_sha256": sha256(FACTS),
        "route_sha256": sha256(route_path),
        "design_sha256": sha256(DESIGN),
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(RUNNER),
        "initial_state_modified": False,
        "historical_authorized": False,
    }
    atomic_write(OUT / "summary.json", json_bytes(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

