"""Ten-year staged preflight for the full system-2005 historical ramps."""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from ..paths import resolve
from . import run_system2005_policy5pct_historical_v3 as full_runner


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    full_end_year = int(config["preflight_full_end_year"])
    full_end_delta = float(config["preflight_full_end_delta_t_c"])
    start_delta = float(config["stage2"]["delta_t_c"])
    if end_year - start_year != 10:
        raise RuntimeError("preflight-duration-must-be-10-years")
    expected_end_delta = start_delta + (full_end_delta - start_delta) * (
        (end_year - start_year) / (full_end_year - start_year)
    )
    if abs(float(config["forward_delta_t_end_c"]) - expected_end_delta) > 1e-12:
        raise RuntimeError("preflight-ramp-slope-does-not-match-full-run")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], "--config", str(config_path)]
        try:
            full_runner.main()
        except SystemExit:
            pass
    finally:
        sys.argv = old_argv

    output = resolve(config["output_root"])
    underlying_path = output / "historical_summary_formal.json"
    underlying = json.loads(underlying_path.read_text(encoding="utf-8"))
    if underlying.get("status") == "technical-fail":
        summary = {
            "schema": "lia-191-system2005-policy5pct-historical-10y-preflight-v1",
            "status": "preflight-technical-fail",
            "system_id": 2005,
            "underlying_error": underlying.get("error"),
            "full_historical_authorized": False,
        }
    else:
        separation = abs(
            float(config["lia_registered_length_m"])
            - float(config["rgi_registered_length_m"])
        )
        peak_upper = min(
            float(config["lia_registered_length_m"])
            * (1.0 + float(config["formal_gates"]["total_length_error_max"])),
            float(config["lia_registered_length_m"])
            + float(config["formal_gates"]["separation_length_error_max"]) * separation,
        )
        peak_length = float(underlying["peak"]["length_m"])
        checks = {
            "completed_10_year_target": bool(
                underlying["numerical_checks"]["completed_target"]
            ),
            "all_numerical_gates": all(underlying["numerical_checks"].values()),
            "aggregate_tributary_policy": bool(
                underlying["formal_checks"]["tributary_separation_policy"]
            ),
            "initial_and_early_peak_hard_bound": peak_length <= peak_upper,
            "full_ramp_slope_reproduced": True,
        }
        passed = all(checks.values())
        summary = {
            "schema": "lia-191-system2005-policy5pct-historical-10y-preflight-v1",
            "status": "preflight-pass" if passed else "preflight-fail",
            "system_id": 2005,
            "start_year": start_year,
            "end_year": end_year,
            "full_end_year": full_end_year,
            "start_delta_t_c": start_delta,
            "preflight_end_delta_t_c": float(config["forward_delta_t_end_c"]),
            "full_end_delta_t_c": full_end_delta,
            "checks": checks,
            "peak_length_m": peak_length,
            "joint_peak_upper_bound_m": peak_upper,
            "connectivity_extrema": underlying["historical_connectivity_extrema"],
            "numerical_checks": underlying["numerical_checks"],
            "full_endpoint_length_checks_intentionally_not_applied": True,
            "full_historical_authorized": passed,
        }
        state = output / "state_2000.pkl.gz"
        if state.exists():
            shutil.copy2(state, output / f"state_{end_year}_preflight.pkl.gz")
    write_json(output / "preflight_summary.json", summary)
    print(json.dumps(summary, allow_nan=False))
    if summary.get("status") != "preflight-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
