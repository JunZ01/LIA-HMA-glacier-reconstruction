"""Atomically fix the validator's unconditional historical-run rejection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
VALIDATOR = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py"
STATE = ROOT / "LIA_CURRENT_STATE.json"
ROUTE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_49/summary.json"
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_execution_validator_historical_authorization_upgrade_v1"
EXPECTED_VALIDATOR_SHA = "9A18CA9452E12CA86A49B88451AA050D1FDF2E974F70E3FA5F9EF26224BE7B7C"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".historical-auth-upgrade.tmp")
    if temp.exists():
        raise RuntimeError(f"stale temp:{temp}")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    if sha256(VALIDATOR) != EXPECTED_VALIDATOR_SHA:
        raise RuntimeError(f"validator changed:{sha256(VALIDATOR)}")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    if state.get("state_id") != "2026-08-26-v10.49-policy-v6-historical-pair-authorized":
        raise RuntimeError("controller state changed")
    if route.get("full_historical_authorized") is not True or route.get("launch_authorized") is not True:
        raise RuntimeError("authoritative route does not authorize history")
    if "## 360. System-2005 Policy-V6 first historical endpoint pair" not in (
        ROOT / "LIA_PROJECT_FACTS.md"
    ).read_text(encoding="utf-8"):
        raise RuntimeError("scientific authorization section missing")
    shutil.copy2(VALIDATOR, OUT / "validate_lia_execution_state_v1.pre.py")
    old = VALIDATOR.read_text(encoding="utf-8")
    source = '''    if state["current_focus"]["full_historical_authorized"]:
        failures.append("current evidence does not authorize full historical run")
'''
    replacement = '''    if state["current_focus"]["full_historical_authorized"]:
        if route.get("full_historical_authorized") is not True:
            failures.append("route does not authorize full historical run")
        if route.get("launch_authorized") is not True:
            failures.append("route historical authorization is not launchable")
        if "historical" not in str(state["next_action"].get("type", "")):
            failures.append("historical authorization has a non-historical next action")
'''
    if old.count(source) != 1:
        raise RuntimeError(f"validator replacement count:{old.count(source)}")
    updated = old.replace(source, replacement)
    atomic_write(VALIDATOR, updated.encode("utf-8"))
    summary = {
        "schema": "lia-191-execution-validator-historical-authorization-upgrade-v1",
        "status": "validator-upgrade-applied",
        "old_sha256": EXPECTED_VALIDATOR_SHA,
        "new_sha256": sha256(VALIDATOR),
        "changed_semantics": "historical authorization now requires matching authoritative route and historical next action",
        "scientific_threshold_modified": False,
        "model_modified": False,
        "route_modified": False,
        "explicit_user_authorization": True,
    }
    atomic_write(
        OUT / "summary.json",
        (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

