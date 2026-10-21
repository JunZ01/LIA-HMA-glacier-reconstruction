"""V3 controller wrapper around the independently rechecking staged v2 engine."""

import json

from . import run_system2005_policy_v6_joint_candidate_v2 as engine


def controller_gate():
    state = json.loads((engine.ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "state_id": state.get("state_id") == "2026-08-26-v10.46-joint-candidate-v3-authorized",
        "route": state.get("authoritative_route", {}).get("version") == "v10.46",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type") == "system2005-joint-candidate-v3-static-preflight-formal50",
        "full_historical_not_authorized": state.get("current_focus", {}).get("full_historical_authorized") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


engine.controller_gate = controller_gate

if __name__ == "__main__":
    raise SystemExit(engine.main())
