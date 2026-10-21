"""Controller-gated Policy-V6 historical endpoint wrapper for system 2005."""

from __future__ import annotations

import json
import sys

from . import run_system2005_topology_adaptive_historical_v7_floor_adaptive_checkpoint as engine


ROOT = engine.resolve(".")
ALLOWED_ENDPOINTS = {0.28125, 0.78125}
EXPECTED_STEADY_SHA = "df85d2bb28d6c08049a630164ea56d482a2f324cf54920c1b9349051db4ac439"


def config_path_from_argv():
    try:
        return sys.argv[sys.argv.index("--config") + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--config is required") from exc


def controller_gate(config: dict) -> dict:
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    endpoint = float(config["forward_delta_t_end_c"])
    checks = {
        "state_id": state.get("state_id") == "2026-08-26-v10.49-policy-v6-historical-pair-authorized",
        "route": state.get("authoritative_route", {}).get("version") == "v10.49",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type")
        == "system2005-policy-v6-historical-first-endpoint-pair",
        "full_historical_authorized": state.get("current_focus", {}).get("full_historical_authorized")
        is True,
        "endpoint_registered": endpoint in ALLOWED_ENDPOINTS,
        "start_temperature_frozen": float(config["stage2"]["delta_t_c"]) == -0.21875,
        "separation_gate_frozen": float(config["formal_gates"]["separation_length_error_max"])
        == 0.2,
        "timestep_floor_frozen": float(config["minimum_timestep_seconds"]) == 30.0,
        "checkpointing_enabled": int(config["end_year"]) == 2000,
    }
    steady_root = engine.resolve(config["steady_root"])
    steady_summary = json.loads((steady_root / "steady_50y_summary_formal.json").read_text(encoding="utf-8"))
    checks["steady_pass"] = steady_summary.get("status") == "steady-50y-pass" and all(
        steady_summary.get("checks", {}).values()
    )
    checks["steady_state_hash"] = (
        engine._sha256(steady_root / "lia_equilibrium_state_50y.pkl.gz") == EXPECTED_STEADY_SHA
    )
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


def main() -> None:
    config = json.loads(engine.resolve(config_path_from_argv()).read_text(encoding="utf-8"))
    gate = controller_gate(config)
    output = engine.resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    print(json.dumps({"status": "controller-gate-pass", "checks": gate}, ensure_ascii=False), flush=True)
    engine.main()


if __name__ == "__main__":
    main()

