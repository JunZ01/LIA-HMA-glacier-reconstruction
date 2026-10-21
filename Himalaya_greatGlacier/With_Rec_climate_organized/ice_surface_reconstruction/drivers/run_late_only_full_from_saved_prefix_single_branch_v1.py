"""Run one saved-prefix endpoint after a single-branch preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import run_late_only_full_from_saved_prefix_v1 as full


def _validate_single_branch(
    config: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    endpoint = float(config["late_delta_t_2000_c"])
    endpoints = [float(value) for value in config["registered_endpoints_c"]]
    if endpoints != [endpoint]:
        raise ValueError("single-branch-endpoint-registration-mismatch")
    if int(config["start_year"]) >= int(config["resume_year"]):
        raise ValueError("invalid-start-resume-years")
    if int(config["resume_year"]) >= int(config["end_year"]):
        raise ValueError("invalid-resume-end-years")
    if config.get("full_historical_authorized") is not True:
        raise RuntimeError("full-historical-not-authorized-in-config")
    if config.get("initial_thickness_modification_allowed") is not False:
        raise RuntimeError("initial-thickness-modification-policy-violation")
    if config.get("prohibited_source_accessed") is not False:
        raise RuntimeError("prohibited-source-policy-violation")

    prefix_root = full.preflight._resolve(config["prefix_root"])
    prefix_state = prefix_root / config["prefix_state_filename"]
    prefix_curve = prefix_root / config["prefix_curve_filename"]
    preflight_summary = prefix_root / config["preflight_summary_filename"]
    for path, expected in (
        (prefix_state, config["prefix_state_sha256"]),
        (prefix_curve, config["prefix_curve_sha256"]),
        (preflight_summary, config["preflight_summary_sha256"]),
    ):
        if full.preflight._sha256(path) != expected:
            raise RuntimeError(f"registered-input-hash-mismatch:{path.name}")

    result = full.preflight._read_json(preflight_summary)
    if result.get("status") != "preflight-pass":
        raise RuntimeError("registered-preflight-not-pass")
    if int(result.get("system_id", -1)) != int(config["system_id"]):
        raise RuntimeError("preflight-system-id-mismatch")
    if result.get("prefix_equivalence_exact") is not True:
        raise RuntimeError("preflight-prefix-equivalence-not-exact")
    branch = result.get("branch_result")
    if not isinstance(branch, dict):
        raise RuntimeError("single-branch-result-missing")
    if float(branch.get("late_delta_t_2000_c")) != endpoint:
        raise RuntimeError("single-branch-endpoint-mismatch")
    if branch.get("status") != "preflight-pass":
        raise RuntimeError("single-branch-preflight-not-pass")
    if branch.get("positive_cooling_response") is not True:
        raise RuntimeError("single-branch-response-not-positive")
    return prefix_root, prefix_state, prefix_curve, preflight_summary


def main() -> int:
    full._validate_config_and_inputs = _validate_single_branch
    return full.main()


if __name__ == "__main__":
    raise SystemExit(main())
