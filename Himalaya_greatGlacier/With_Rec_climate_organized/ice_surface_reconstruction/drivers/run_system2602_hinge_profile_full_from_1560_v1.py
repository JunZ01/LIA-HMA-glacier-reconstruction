"""Run the registered system-2602 hinge profile from the saved 1560 state.

The numerical continuation is delegated to the established saved-prefix
driver.  This adapter freezes the scientific interpretation: the copied warm
reference climate is unchanged through 1560 and receives one linear
temperature correction of -1.0625 K by 2000, so its registered absolute 2000
endpoint is +0.125 K.  It does not reconstruct or modify the supplied ice
thickness.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from . import run_late_only_full_from_saved_prefix_v1 as full


def _validate_hinge_inputs(
    config: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    if int(config["system_id"]) != 2602:
        raise RuntimeError("unexpected-system-id")
    if config.get("mechanism") != "single-fixed-hinge-temperature-profile":
        raise RuntimeError("unexpected-mechanism")
    if int(config["start_year"]) >= int(config["resume_year"]):
        raise ValueError("invalid-start-resume-years")
    if int(config["resume_year"]) >= int(config["end_year"]):
        raise ValueError("invalid-resume-end-years")
    if int(config["resume_year"]) != int(config["split_year"]):
        raise ValueError("resume-split-year-mismatch")
    if config.get("full_historical_authorized") is not True:
        raise RuntimeError("full-historical-not-authorized-in-config")
    if config.get("initial_thickness_modification_allowed") is not False:
        raise RuntimeError("initial-thickness-modification-policy-violation")
    if config.get("prohibited_source_accessed") is not False:
        raise RuntimeError("prohibited-source-policy-violation")

    reference_endpoint = float(config["reference_endpoint_c"])
    hinge_endpoint = float(config["hinge_endpoint_c"])
    correction = float(config["late_delta_t_2000_c"])
    if abs(correction - (hinge_endpoint - reference_endpoint)) > 1.0e-12:
        raise ValueError("hinge-correction-endpoint-mismatch")
    endpoints = [float(value) for value in config["registered_endpoints_c"]]
    if endpoints != [hinge_endpoint]:
        raise ValueError("single-hinge-endpoint-registration-mismatch")

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
    if int(result.get("system_id", -1)) != 2602:
        raise RuntimeError("preflight-system-id-mismatch")
    if result.get("prefix_equivalence_exact") is not True:
        raise RuntimeError("preflight-prefix-equivalence-not-exact")
    if result.get("prefix_state_sha256") != config["prefix_state_sha256"]:
        raise RuntimeError("preflight-prefix-state-hash-mismatch")
    if float(result.get("reference_endpoint_c")) != reference_endpoint:
        raise RuntimeError("preflight-reference-endpoint-mismatch")
    if float(result.get("hinge_endpoint_c")) != hinge_endpoint:
        raise RuntimeError("preflight-hinge-endpoint-mismatch")
    if result.get("full_historical_started") is not False:
        raise RuntimeError("preflight-already-started-full-historical")
    if result.get("initial_thickness_modified") is not False:
        raise RuntimeError("preflight-initial-thickness-policy-violation")
    if result.get("prohibited_source_accessed") is not False:
        raise RuntimeError("preflight-prohibited-source-policy-violation")
    branch = result.get("branch_result")
    if not isinstance(branch, dict):
        raise RuntimeError("hinge-branch-result-missing")
    if branch.get("status") != "preflight-pass":
        raise RuntimeError("hinge-branch-preflight-not-pass")
    if branch.get("positive_cooling_response") is not True:
        raise RuntimeError("hinge-branch-response-not-positive")
    if int(branch.get("final", {}).get("year", -1)) != int(
        config["preflight_end_year"]
    ):
        raise RuntimeError("hinge-branch-preflight-end-year-mismatch")
    return prefix_root, prefix_state, prefix_curve, preflight_summary


def _decorate_summary(
    summary: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    summary["schema"] = "lia-system2602-hinge-profile-full-v1"
    summary["mechanism"] = config["mechanism"]
    summary["reference_endpoint_c"] = float(config["reference_endpoint_c"])
    summary["hinge_endpoint_c"] = float(config["hinge_endpoint_c"])
    summary["hinge_correction_2000_c"] = float(config["late_delta_t_2000_c"])
    summary["split_year"] = int(config["split_year"])
    summary["initial_thickness_modified"] = False
    summary["prohibited_source_accessed"] = False
    return summary


def run(config_path: Path, resume: bool = False) -> dict[str, Any]:
    config = full.preflight._read_json(config_path)
    full._validate_config_and_inputs = _validate_hinge_inputs
    result = full.run(config_path, resume=resume)
    result = _decorate_summary(result, config)
    output = full.preflight._resolve(config["output_root"])
    full.preflight._atomic_json(output / "historical_summary_formal.json", result)
    full.preflight._atomic_json(output / "historical_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.config.resolve(), resume=args.resume)
    except Exception as exc:
        config = full.preflight._read_json(args.config.resolve())
        output = full.preflight._resolve(config["output_root"])
        if output.exists():
            failure = {
                "schema": "lia-system2602-hinge-profile-full-v1",
                "status": "technical-fail",
                "system_id": int(config["system_id"]),
                "mechanism": config["mechanism"],
                "reference_endpoint_c": float(config["reference_endpoint_c"]),
                "hinge_endpoint_c": float(config["hinge_endpoint_c"]),
                "hinge_correction_2000_c": float(config["late_delta_t_2000_c"]),
                "runtime_error": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(),
                "checkpoint_resume_supported": True,
                "accepted_count_effect": 0,
                "initial_thickness_modified": False,
                "prohibited_source_accessed": False,
                "full_historical_started": True,
            }
            full.preflight._atomic_json(
                output / "historical_summary_formal.json", failure
            )
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
