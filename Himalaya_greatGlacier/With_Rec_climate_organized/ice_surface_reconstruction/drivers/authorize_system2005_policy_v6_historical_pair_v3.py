"""Record the 260-character NetCDF failure and authorize short output roots."""

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
OUT = RESULTS / "lia_191_system2005_policy_v6_historical_pair_v3_authorization"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_51"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
FAILED = (
    RESULTS / "lia_191_system2005_policy_v6_historical_endp0p28125_v2/historical_summary_unadjudicated.json",
    RESULTS / "lia_191_system2005_policy_v6_historical_endp0p78125_v2/historical_summary_unadjudicated.json",
)
CONFIGS = (
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_historical_endp0p28125_v3.json",
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_historical_endp0p78125_v3.json",
)
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_policy_v6_historical_v3.py"
EXPECTED_FACTS_SHA = "77EC3FF3DB897A1EBED5E33ABD1F7907D0F5E6B1A10355BD4C66013C93466B03"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".histpairv3.tmp")
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
    if old_state.get("state_id") != "2026-08-26-v10.50-policy-v6-historical-pair-v2-authorized":
        raise RuntimeError("controller state changed")
    for path in FAILED:
        failed = json.loads(path.read_text(encoding="utf-8"))
        if failed.get("status") != "technical-fail" or "PermissionError" not in failed.get("error", ""):
            raise RuntimeError(f"unexpected v2 result:{path}")
    for path in (*CONFIGS, RUNNER):
        if not path.exists():
            raise FileNotFoundError(path)
    if any((ROOT / name).exists() for name in ("w2005p6h281v3", "w2005p6h781v3")):
        raise RuntimeError("short output root already exists")
    for source, name in (
        (FACTS, "LIA_PROJECT_FACTS.pre.md"),
        (STATE, "LIA_CURRENT_STATE.pre.json"),
        (DECISIONS, "LIA_DECISION_LOG.pre.md"),
    ):
        shutil.copy2(source, OUT / name)
    section = """## 362. System-2005 Policy-V6 short historical output roots v3 (2026-08-26)

- Both v2 endpoints passed their controller and steady-state gates, then failed
  before the first dynamics step while NetCDF attempted to create the isolated
  ramp file. The failing paths are exactly 260 characters; directory ACLs allow
  modification and no target files were created. This is a Windows path-limit
  failure, not a scientific or numerical result.
- V3 changes only the versioned output roots to `w2005p6h281v3` and
  `w2005p6h781v3`, reducing the longest immediate NetCDF target to 167
  characters. All inputs, endpoint temperatures, physics, solver, timesteps,
  fallback triggers, checkpoints, and Policy-V6 gates remain unchanged.
- The same two endpoints are authorized concurrently. V1/v2 failures are
  retained; no third endpoint, bisection, or reproduction is authorized yet.
""".rstrip()
    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 362." in facts_text:
        raise RuntimeError("facts section 362 already exists")
    atomic_write(FACTS, (facts_text.rstrip() + "\n\n" + section + "\n").encode("utf-8"))
    route = {
        "schema": "lia-191-current-route-v10-51",
        "status": "policy-v6-short-root-historical-pair-v3-authorized",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queue": 2,
        "launch_authorized": True,
        "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_50",
        "system_id": 2005,
        "endpoint_delta_t_c": [0.28125, 0.78125],
        "short_output_root_only": True,
        "parallel_distinct_outputs_authorized": True,
        "third_endpoint_authorized": False,
        "configs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in CONFIGS],
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, json_bytes(route))
    summary = {
        "schema": "lia-191-system2005-policy-v6-historical-pair-v3-authorization",
        "status": "two-short-root-endpoints-authorized",
        "facts_sha256": sha256(FACTS),
        "route_sha256": sha256(route_path),
        "config_sha256": {path.name: sha256(path) for path in CONFIGS},
        "runner_sha256": sha256(RUNNER),
        "scientific_change": False,
        "path_length_correction_only": True,
    }
    summary_path = OUT / "summary.json"
    atomic_write(summary_path, json_bytes(summary))
    new_state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.51-policy-v6-historical-pair-v3-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": sha256(FACTS)},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.51",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(route_path),
            "supersedes": "v10.50",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 2},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-checkpointed-historical-first-bracket",
            "state": "two-short-root-endpoints-authorized",
            "steady_delta_t_c": -0.21875,
            "steady_state_sha256": "DF85D2BB28D6C08049A630164EA56D482A2F324CF54920C1B9349051DB4AC439",
            "endpoint_delta_t_c": [0.28125, 0.78125],
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "full_historical_authorized": True,
            "evidence": {
                "path": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(summary_path),
            },
        },
        "research_branches": old_state.get("research_branches", {}),
        "next_action": {
            "type": "system2005-policy-v6-historical-first-endpoint-pair-v3",
            "launch_authorized": True,
            "description": "Launch the two short-root checkpointed endpoints concurrently; no third endpoint or reproduction.",
        },
    }
    atomic_write(STATE, json_bytes(new_state))
    decisions = DECISIONS.read_text(encoding="utf-8")
    note = """## 2026-08-26 — System 2005 short historical roots v3

- V2 failed pre-step at the 260-character NetCDF target path.
- V3 shortens only output paths; the two scientific endpoints remain unchanged.
""".strip()
    atomic_write(DECISIONS, (decisions.rstrip() + "\n\n" + note + "\n").encode("utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

