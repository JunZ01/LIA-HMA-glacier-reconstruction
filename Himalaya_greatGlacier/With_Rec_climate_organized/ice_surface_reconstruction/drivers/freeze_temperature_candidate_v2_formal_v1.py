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
EXECUTION = ROOT / "LIA_EXECUTION_CONTRACT.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
FREEZE = LOOP / "temperature_candidate_v2_formal_freeze_v1"
ROUTE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_112"

EXPECTED_STATE_ID = "2026-08-28-v10.111-system709-hash-case-resume-retry-active"
EXPECTED_STATE_SHA256 = "DA88E87A377A46744415EF104AD9C416DCC3D8E3CA0C31F98C5B6F7CE266B1AC"
EXPECTED_FACTS_SHA256 = "1AA18A8D8284590202287A6C50872CF24D16C7248CBDA278991548E968C9FD51"
EXPECTED_FREEZE_HASHES = {
    "CANDIDATE_CONTRACT.md": "7856F19A069DC358BCD6BEEE029B95199BFA92FA1AB4F710AE2D484040503731",
    "AUTHORIZATION_RECEIPT.json": "112DEFBC7AE0A492FBA18FBDB9435298EF98AE80061893074C157638BEE746D5",
    "summary.json": "BC790F9525D997EA0DA1D0305FB7DA3041FE5028F8972AA4A08494ACF0180F62",
    "FINAL_RESULT.md": "3A8D5404E0022175B67A761974B3CDE08ABA5D587A4901BA6FEDDA1E28FF986B",
    "temperature_candidate_v2_frozen_allowed_years.nc": "DF008575229772892BFC8A2B5C5ED8450E735F7CD5CC64CA7B8ACA9F58A71C90",
    "hma_temperature_candidate_v2_allowed.csv": "59EA983F8B87BB6F9CB078499783581A2BA5F2DECAC095F1D295DCEED00573B8",
    "hma_temperature_candidate_v2_allowed.png": "DF1839B82D8A7D1E042B925AE6A8294108608C36D360109A467CBD026AA3E05B",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def append_bytes(original: bytes, text: str) -> bytes:
    newline = b"\r\n" if original.count(b"\r\n") > original.count(b"\n") // 2 else b"\n"
    normalized = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    encoded = normalized.replace("\n", newline.decode("ascii")).encode("utf-8")
    separator = newline * (2 if original.rstrip().endswith(b".") else 1)
    return original.rstrip() + separator + encoded + newline


def main() -> None:
    if ROUTE.exists():
        raise FileExistsError(ROUTE)
    if sha256(STATE) != EXPECTED_STATE_SHA256:
        raise RuntimeError("current-state-changed-before-freeze")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("state_id") != EXPECTED_STATE_ID:
        raise RuntimeError("current-state-id-changed-before-freeze")
    if sha256(FACTS) != EXPECTED_FACTS_SHA256:
        raise RuntimeError("facts-changed-before-freeze")
    if state.get("counts") != {"accepted": 57, "remaining": 134, "active": 2, "queued": 0}:
        raise RuntimeError("population-counts-changed-before-freeze")
    for name, expected in EXPECTED_FREEZE_HASHES.items():
        if sha256(FREEZE / name) != expected:
            raise RuntimeError(f"freeze-package-hash-mismatch:{name}")
    authorization = json.loads((FREEZE / "AUTHORIZATION_RECEIPT.json").read_text(encoding="utf-8"))
    if not authorization.get("authorized") or authorization.get("production_modification_authorized"):
        raise RuntimeError("authorization-boundary-invalid")

    facts_section = """
## 415. HMA Temperature Candidate v2 formally frozen (2026-08-28)

- By explicit user decision, Candidate `HMA-TEMP-CANDIDATE-V2-FROZEN-20260828`
  adopts the HMA-target-aligned hierarchical closure contract: maximum absolute
  HMA area-weighted 50-year mean discrepancy and maximum HMA area-weighted
  50-year spatial RMS discrepancy must each be at most `0.05 degC`; West,
  Central, and East regional means remain explicit safeguards interpreted
  against external-family dispersion. The former maximum-single-cell statistic
  remains diagnostic but is no longer the primary HMA acceptance gate.
- The frozen method is the equal, untuned mean of ModE-Sim, MPI-ESM1-2-LR,
  and MIROC-ES2L forced-background differences, represented by the existing
  31-year low-frequency correction for 1500-1920 and the stored cellwise
  constant correction for 1940-1950. The original project reconstruction is
  retained after 1950, and the archived posterior increment is unchanged.
- Verified closure values are `0.04905098 degC` for the HMA mean and
  `0.04987337 degC` for HMA spatial RMS. Regional maxima are `0.05086046`,
  `0.05116350`, and `0.04647750 degC` for West, Central, and East, versus a
  minimum external-family 50-year standard deviation of `0.19919943 degC`.
- Qcf absolute warm-bias improvements are `40.23%` for 1880-1900 and `63.90%`
  for 1901-1920; all frozen direct-station gates pass. Candidate D2 is rejected
  because it fails annual-correlation and modern-trend gates, and the `b1 >= 0`
  sign channel remains diagnostic only.
- The immutable candidate package is
  `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/temperature_candidate_v2_formal_freeze_v1/`.
  Production temperature files remain unchanged. No 1921-1939 value, EnKF,
  OGGM run, accepted-count change, or new LIA launch is authorized by this
  candidate freeze.
"""
    decision_section = """
## 2026-08-28 - Route v10.112: HMA Temperature Candidate v2 frozen

- Accepted the HMA mean/RMS hierarchical closure contract and formally froze
  `HMA-TEMP-CANDIDATE-V2-FROZEN-20260828` as the sole leading temperature
  candidate. Candidate D2 remains rejected and coefficient-sign filtering is
  diagnostic only.
- Production files, the sealed 1921-1939 interval, EnKF, OGGM, and all glacier
  acceptance counts remain untouched.
- System 709 controller state is carried forward unchanged from v10.111:
  accepted 57, remaining 134, active 2, queued 0. Monitoring and formal
  classification of the registered endpoint processes remain the only LIA
  mainline action; this policy-only transition authorizes no model launch.
"""

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = append_bytes(facts_before, facts_section)
    decisions_after = append_bytes(decisions_before, decision_section)
    facts_hash = hashlib.sha256(facts_after).hexdigest().upper()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat()

    route_summary = {
        "schema": "lia-191-current-route-v10-112",
        "status": "system709-resume-retry-active-temperature-candidate-v2-frozen",
        "population": 191,
        "accepted": 57,
        "remaining": 134,
        "active": 2,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_111",
        "system_id": 709,
        "active_runs": [
            {
                "endpoint_c": 1.125,
                "pid": 39324,
                "resumed_from_year": 1677,
                "output_root": "w927hist709rampendp1125v1",
                "log_root": "w927hist709rampendp1125v1_resume_v2_logs",
            },
            {
                "endpoint_c": 1.625,
                "pid": 4912,
                "resumed_from_year": 1706,
                "output_root": "w927hist709rampendp1625v1",
                "log_root": "w927hist709rampendp1625v1_resume_v2_logs",
            },
        ],
        "launch_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system709_hash_case_resume_retry_launch_v1/summary.json",
        "launch_evidence_sha256": "ABDB73C2E85F30426024B96875D30F281C715E25FFBAEE9D528FC6B037E73064",
        "temperature_candidate": {
            "candidate_id": "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828",
            "status": "formally-frozen-candidate-not-production",
            "evidence": "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/temperature_candidate_v2_formal_freeze_v1/summary.json",
            "evidence_sha256": EXPECTED_FREEZE_HASHES["summary.json"],
            "field_sha256": EXPECTED_FREEZE_HASHES["temperature_candidate_v2_frozen_allowed_years.nc"],
        },
        "technical_retry_consumed": True,
        "duplicate_launch_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "monitor and formally classify the v10.111 system-709 endpoint processes; no launch authorized",
    }
    route_bytes = (json.dumps(route_summary, indent=2) + "\n").encode("utf-8")
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    new_state = dict(state)
    new_state["state_id"] = "2026-08-28-v10.112-temperature-candidate-v2-formal-freeze"
    new_state["updated_at"] = timestamp
    new_state["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": facts_hash}
    new_state["authoritative_route"] = {
        "version": "v10.112",
        "path": "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_112/summary.json",
        "sha256": route_hash,
        "supersedes": "v10.111",
    }
    branches = dict(new_state.get("research_branches", {}))
    branches["hma_temperature_candidate_v2"] = {
        "candidate_id": "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828",
        "status": "formally-frozen-candidate-not-production",
        "promoted_to_production": False,
        "accepted_count_effect": 0,
        "evidence": {
            "path": "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/temperature_candidate_v2_formal_freeze_v1/summary.json",
            "sha256": EXPECTED_FREEZE_HASHES["summary.json"],
        },
        "next_legal_action": "explicit production decision only; no EnKF or OGGM authorized",
    }
    new_state["research_branches"] = branches
    state_bytes = (json.dumps(new_state, indent=2) + "\n").encode("utf-8")

    stage = Path(tempfile.mkdtemp(prefix=".candidate_v2_freeze_", dir=ROOT))
    backups = {FACTS: facts_before, DECISIONS: decisions_before, STATE: STATE.read_bytes()}
    try:
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(state_bytes)
        (stage / "route").mkdir()
        (stage / "route/summary.json").write_bytes(route_bytes)
        ROUTE.mkdir(parents=False)
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "route/summary.json", ROUTE / "summary.json")
        os.replace(stage / "state", STATE)
    except BaseException:
        for path, content in backups.items():
            path.write_bytes(content)
        shutil.rmtree(ROUTE, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    result = {
        "status": "complete",
        "candidate_id": "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828",
        "facts_sha256": facts_hash,
        "route_sha256": route_hash,
        "state_id": new_state["state_id"],
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_1921_1939_values_used": False,
        "lia_counts": new_state["counts"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
