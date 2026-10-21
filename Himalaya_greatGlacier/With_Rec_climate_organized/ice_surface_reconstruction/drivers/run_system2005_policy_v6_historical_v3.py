"""Short-output-root wrapper for the Policy-V6 endpoint pair."""

from __future__ import annotations

import json

from . import run_system2005_policy_v6_historical_v2 as engine


def controller_gate(config: dict) -> dict:
    state = json.loads((engine.ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    endpoint = float(config["forward_delta_t_end_c"])
    checks = {
        "state_id": state.get("state_id") == "2026-08-26-v10.51-policy-v6-historical-pair-v3-authorized",
        "route": state.get("authoritative_route", {}).get("version") == "v10.51",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type")
        == "system2005-policy-v6-historical-first-endpoint-pair-v3",
        "full_historical_authorized": state.get("current_focus", {}).get("full_historical_authorized")
        is True,
        "endpoint_registered": endpoint in engine.ALLOWED_ENDPOINTS,
        "start_temperature_frozen": float(config["stage2"]["delta_t_c"]) == -0.21875,
        "separation_gate_frozen": float(config["formal_gates"]["separation_length_error_max"])
        == 0.2,
        "timestep_floor_frozen": float(config["minimum_timestep_seconds"]) == 30.0,
        "short_output_root": len(str(config["output_root"])) < 100,
    }
    steady_root = engine.Path(config["steady_root"])
    steady_summary = json.loads((steady_root / "steady_50y_summary_formal.json").read_text(encoding="utf-8"))
    checks["steady_pass"] = steady_summary.get("status") == "steady-50y-pass" and all(
        steady_summary.get("checks", {}).values()
    )
    checks["steady_state_hash"] = (
        engine.engine._sha256(steady_root / "lia_equilibrium_state_50y.pkl.gz")
        == engine.EXPECTED_STEADY_SHA
    )
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


engine.controller_gate = controller_gate

if __name__ == "__main__":
    engine.main()

