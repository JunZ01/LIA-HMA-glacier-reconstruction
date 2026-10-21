"""Run one temperature-only correction using the frozen v2 candidate state."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
import sys

from . import run_system2005_policy_v6_joint_candidate_v2 as engine


def controller_gate():
    state = json.loads((engine.ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "state_id": state.get("state_id") == "2026-08-26-v10.47-temperature-correction-v4-authorized",
        "route": state.get("authoritative_route", {}).get("version") == "v10.47",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type")
        == "system2005-temperature-correction-v4-preflight-formal50",
        "full_historical_not_authorized": state.get("current_focus", {}).get("full_historical_authorized")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


def build_frozen_candidate(_source, config):
    path = engine.base.resolve(config["candidate_initial_state"])
    expected = str(config["candidate_initial_state_sha256"]).upper()
    actual = engine.sha256(path).upper()
    if actual != expected:
        raise RuntimeError(f"frozen candidate hash mismatch:{actual}")
    with gzip.open(path, "rb") as stream:
        candidate = pickle.load(stream)
    return copy.deepcopy(candidate)


def config_path_from_argv():
    try:
        return sys.argv[sys.argv.index("--config") + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--config is required") from exc


def main() -> int:
    config = json.loads(engine.base.resolve(config_path_from_argv()).read_text(encoding="utf-8"))
    design = json.loads(engine.base.resolve(config["design_summary"]).read_text(encoding="utf-8"))
    if design.get("status") != "unique-temperature-only-candidate-selected":
        raise RuntimeError("temperature-only design not authorized")
    if design["candidate"].get("initial_state_modified") is not False:
        raise RuntimeError("candidate state modification is forbidden")
    if float(config["stage2"]["delta_t_c"]) != float(design["candidate"]["delta_t_c"]):
        raise RuntimeError("temperature mismatch")
    engine.controller_gate = controller_gate
    engine.build_candidate = build_frozen_candidate
    rc = engine.main()
    output = engine.base.resolve(config["output_root"])
    primary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    wrapper = {
        "schema": "lia-191-system2005-policy-v6-temperature-correction-v4",
        "status": primary.get("status"),
        "initial_state_modified": False,
        "candidate_initial_state_sha256": engine.sha256(engine.base.resolve(config["candidate_initial_state"])),
        "delta_t_c": config["stage2"]["delta_t_c"],
        "other_stage2_modified": False,
        "preflight": primary.get("preflight"),
        "formal": primary.get("formal"),
        "historical_run": False,
        "primary_summary": str((output / "summary.json").relative_to(engine.ROOT)).replace("\\", "/"),
    }
    engine.write_json(output / "temperature_correction_summary.json", wrapper)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

