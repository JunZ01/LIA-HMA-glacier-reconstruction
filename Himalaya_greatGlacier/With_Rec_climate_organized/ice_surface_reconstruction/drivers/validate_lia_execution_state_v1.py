"""Read-only validator for the LIA 191 controller state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    state_path = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = load_json("LIA_CURRENT_STATE.json")
    failures: list[str] = []

    for key in ("scientific_contract", "authoritative_route"):
        item = state[key]
        path = PROJECT_ROOT / item["path"]
        if not path.exists():
            failures.append(f"missing {key}: {path}")
        elif sha256(path) != item["sha256"].upper():
            failures.append(f"hash mismatch for {key}: {path}")

    focus_evidence = state["current_focus"]["evidence"]
    focus_path = PROJECT_ROOT / focus_evidence["path"]
    if not focus_path.exists():
        failures.append(f"missing current-focus evidence: {focus_path}")
    elif sha256(focus_path) != focus_evidence["sha256"].upper():
        failures.append(f"hash mismatch for current-focus evidence: {focus_path}")

    for name, branch in state.get("research_branches", {}).items():
        evidence = branch["evidence"]
        path = PROJECT_ROOT / evidence["path"]
        if not path.exists():
            failures.append(f"missing research evidence for {name}: {path}")
        elif sha256(path) != evidence["sha256"].upper():
            failures.append(f"hash mismatch for research evidence {name}: {path}")

    route = load_json(state["authoritative_route"]["path"])
    counts = state["counts"]
    if route.get("population") != state["population"]:
        failures.append("route population differs from current state")
    if route.get("accepted") != counts["accepted"]:
        failures.append("route accepted count differs from current state")
    if route.get("remaining") != counts["remaining"]:
        failures.append("route remaining count differs from current state")
    if counts["accepted"] + counts["remaining"] != state["population"]:
        failures.append("accepted + remaining does not equal population")
    if route.get("active") != counts["active"]:
        failures.append("route active count differs from current state")
    if route.get("queue") != counts["queued"]:
        failures.append("route queue count differs from current state")

    decision_required = state["current_focus"]["state"] == "decision-required"
    launch_authorized = state["next_action"]["launch_authorized"]
    if decision_required and launch_authorized:
        failures.append("decision-required state cannot authorize a launch")
    if state["current_focus"]["full_historical_authorized"]:
        if route.get("full_historical_authorized") is not True:
            failures.append("route does not authorize full historical run")
        if route.get("launch_authorized") is not True:
            failures.append("route historical authorization is not launchable")
        if "historical" not in str(state["next_action"].get("type", "")):
            failures.append("historical authorization has a non-historical next action")

    report = {
        "schema": "lia-191-controller-validation-v1",
        "state_path": str(state_path),
        "state_id": state["state_id"],
        "status": "pass" if not failures else "fail",
        "launch_authorized": bool(launch_authorized and not failures),
        "failures": failures,
        "counts": counts,
        "current_focus": state["current_focus"]["system_id"],
        "next_action_type": state["next_action"]["type"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

