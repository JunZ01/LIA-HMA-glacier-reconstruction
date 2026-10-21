"""Record pre-output path failures and authorize the path-corrected pair."""

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
OUT = RESULTS / "lia_191_system2005_policy_v6_historical_pair_v2_authorization"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_50"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
CONFIGS = (
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_historical_endp0p28125_v2.json",
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_historical_endp0p78125_v2.json",
)
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_policy_v6_historical_v2.py"
EXPECTED_FACTS_SHA = "160B318BE5B762860FBE4E55C2206AA0A54EAB5F2BA1385A00F46E81951E1AF1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".histpairv2.tmp")
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
    if old_state.get("state_id") != "2026-08-26-v10.49-policy-v6-historical-pair-authorized":
        raise RuntimeError("controller state changed")
    failed_outputs = (
        RESULTS / "lia_191_system2005_policy_v6_historical_endp0p28125_v1",
        RESULTS / "lia_191_system2005_policy_v6_historical_endp0p78125_v1",
    )
    if any(path.exists() for path in failed_outputs):
        raise RuntimeError("a v1 output unexpectedly exists")
    for path in (*CONFIGS, RUNNER):
        if not path.exists():
            raise FileNotFoundError(path)
    for source, name in (
        (FACTS, "LIA_PROJECT_FACTS.pre.md"),
        (STATE, "LIA_CURRENT_STATE.pre.json"),
        (DECISIONS, "LIA_DECISION_LOG.pre.md"),
    ):
        shutil.copy2(source, OUT / name)
    section = """## 361. System-2005 Policy-V6 historical path correction v2 (2026-08-26)

- The two v1 endpoint launch attempts exited before reading model state or
  creating output directories. The first used a package-relative config path;
  the second exposed the same wrapper's incorrect project-root calculation.
  Neither attempt ran a timestep or produced scientific evidence.
- V2 changes only path resolution: its wrapper derives the project root from
  the driver location, and both configs use absolute paths for immutable
  scientific inputs and distinct versioned outputs. Endpoint temperatures,
  solver, 12-hour nominal step, 30-second floor, fallback triggers, physics,
  Policy-V6 gates, and checkpoint contract are unchanged from Section 360.
- The same two endpoints `+0.28125/+0.78125 K` are authorized concurrently.
  No third endpoint, bisection, or reproduction is authorized yet.
""".rstrip()
    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 361." in facts_text:
        raise RuntimeError("facts section 361 already exists")
    atomic_write(FACTS, (facts_text.rstrip() + "\n\n" + section + "\n").encode("utf-8"))
    route = {
        "schema": "lia-191-current-route-v10-50",
        "status": "policy-v6-first-historical-endpoint-pair-v2-authorized",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queue": 2,
        "launch_authorized": True,
        "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_49",
        "system_id": 2005,
        "endpoint_delta_t_c": [0.28125, 0.78125],
        "path_correction_only": True,
        "parallel_distinct_outputs_authorized": True,
        "third_endpoint_authorized": False,
        "configs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in CONFIGS],
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, json_bytes(route))
    new_state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.50-policy-v6-historical-pair-v2-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": sha256(FACTS)},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.50",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(route_path),
            "supersedes": "v10.49",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 2},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-checkpointed-historical-first-bracket",
            "state": "two-path-corrected-endpoints-authorized",
            "steady_delta_t_c": -0.21875,
            "steady_state_sha256": "DF85D2BB28D6C08049A630164EA56D482A2F324CF54920C1B9349051DB4AC439",
            "endpoint_delta_t_c": [0.28125, 0.78125],
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "full_historical_authorized": True,
            "evidence": {
                "path": str((OUT / "summary.json").relative_to(ROOT)).replace("\\", "/"),
                "sha256": "PENDING_SELF_HASH_NOT_USED_BY_VALIDATOR_UNTIL_AFTER_WRITE"
            },
        },
        "research_branches": old_state.get("research_branches", {}),
        "next_action": {
            "type": "system2005-policy-v6-historical-first-endpoint-pair-v2",
            "launch_authorized": True,
            "description": "Launch the two path-corrected checkpointed endpoints concurrently; no third endpoint or reproduction.",
        },
    }
    summary = {
        "schema": "lia-191-system2005-policy-v6-historical-pair-v2-authorization",
        "status": "two-path-corrected-endpoints-authorized",
        "facts_sha256": sha256(FACTS),
        "route_sha256": sha256(route_path),
        "config_sha256": {path.name: sha256(path) for path in CONFIGS},
        "runner_sha256": sha256(RUNNER),
        "scientific_change": False,
        "path_correction_only": True,
    }
    summary_path = OUT / "summary.json"
    atomic_write(summary_path, json_bytes(summary))
    new_state["current_focus"]["evidence"]["sha256"] = sha256(summary_path)
    atomic_write(STATE, json_bytes(new_state))
    decisions = DECISIONS.read_text(encoding="utf-8")
    note = """## 2026-08-26 — System 2005 historical path correction v2

- Two pre-output path failures produced no scientific run or output.
- V2 corrects only paths and retains the authorized endpoint pair unchanged.
""".strip()
    atomic_write(DECISIONS, (decisions.rstrip() + "\n\n" + note + "\n").encode("utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

