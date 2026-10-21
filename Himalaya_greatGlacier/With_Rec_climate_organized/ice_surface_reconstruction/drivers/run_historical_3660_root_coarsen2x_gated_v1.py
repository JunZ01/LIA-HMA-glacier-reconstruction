"""Strictly gated historical template for the 3660 root-coarsen2x state.

The runner refuses before creating an output directory unless the registered
delta-T=-0.375 K steady summary and equilibrium state both exist and pass.
No command in this module is executed merely by preparing the templates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ..paths import resolve
from . import synthetic_lia_conservative_junction_historical_ramp_v1 as historical


ALLOWED_STAGES = {
    "historical-template-end-steady-plus-0.5K": 0.125,
    "historical-template-end-steady-plus-1.0K": 0.625,
}


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _preflight(config: dict) -> tuple[Path, dict]:
    if config.get("experiment") != "lia-191-3660-root-coarsen2x-historical-gated-v1":
        raise ValueError("unexpected-experiment")
    if int(config.get("system_id", -1)) != 3660:
        raise ValueError("only-system-3660-authorized")
    stage = str(config.get("stage"))
    if stage not in ALLOWED_STAGES:
        raise ValueError("unregistered-historical-template")
    if float(config["forward_delta_t_end_c"]) != ALLOWED_STAGES[stage]:
        raise ValueError("forward-end-delta-conflict")
    if int(config["start_year"]) != 1470 or int(config["end_year"]) != 2000:
        raise ValueError("historical-year-contract-conflict")
    if float(config["minimum_timestep_seconds"]) != 30.0:
        raise ValueError("minimum-timestep-contract-conflict")
    if float(config["stage2"]["delta_t_c"]) != -0.375:
        raise ValueError("steady-start-delta-conflict")
    if str(config["stage2"]["prcp_anomaly_mode"]) != "off":
        raise ValueError("precipitation-mode-conflict")
    authorization = config.get("authorization") or {}
    if authorization != {
        "requires_steady_status": "steady-50y-pass",
        "requires_steady_delta_t_c": -0.375,
        "requires_equilibrium_state": True,
        "minimum_timestep_seconds": 30.0,
        "precipitation_anomaly": "off",
        "template_prepared_not_launched": True,
    }:
        raise ValueError("authorization-contract-changed")
    steady_root = resolve(config["steady_root"])
    summary_path = steady_root / "steady_50y_summary.json"
    state_path = steady_root / "lia_equilibrium_state_50y.pkl.gz"
    if not summary_path.is_file():
        raise RuntimeError("automatic-refusal:steady-summary-missing")
    steady = json.loads(summary_path.read_text(encoding="utf-8"))
    if steady.get("status") != "steady-50y-pass":
        raise RuntimeError(f"automatic-refusal:steady-status-{steady.get('status')}")
    if not state_path.is_file():
        raise RuntimeError("automatic-refusal:steady-equilibrium-state-missing")
    snapshot = steady.get("config_snapshot")
    if snapshot and Path(str(snapshot)).is_file():
        steady_config = json.loads(Path(str(snapshot)).read_text(encoding="utf-8"))
        if float(steady_config["stage2"]["delta_t_c"]) != -0.375:
            raise RuntimeError("automatic-refusal:steady-delta-not-minus-0.375")
    if _finite(steady.get("minimum_timestep_seconds")) not in (None, 30.0):
        raise RuntimeError("automatic-refusal:steady-timestep-not-30s")
    return steady_root, steady


def _formalize(config: dict, summary: dict, output_root: Path) -> dict:
    lia = float(config["lia_registered_length_m"])
    rgi = float(config["rgi_registered_length_m"])
    separation = abs(lia - rgi)
    initial = summary.get("initial") or {}
    peak = summary.get("peak") or {}
    final = summary.get("final") or {}
    e_lia_total = _finite(summary.get("e_lia_total"))
    e_rgi_total = _finite(summary.get("e_rgi_total"))
    e_lia_sep = _finite(summary.get("e_lia_separation"))
    e_rgi_sep = _finite(summary.get("e_rgi_separation"))
    total_pass = bool(
        e_lia_total is not None and e_rgi_total is not None
        and e_lia_total <= 0.1 and e_rgi_total <= 0.1
    )
    separation_pass = bool(
        e_lia_sep is not None and e_rgi_sep is not None
        and e_lia_sep <= 0.2 and e_rgi_sep <= 0.2
    )
    domain_reasons = []
    for label, state in (("initial", initial), ("peak", peak), ("final", final)):
        if state.get("finite_geometry") is not True:
            domain_reasons.append(f"{label}-nonfinite")
        if int(_finite(state.get("negative_section_count")) or -1) != 0:
            domain_reasons.append(f"{label}-negative-section")
        if (_finite(state.get("root_array_end_thickness_m")) or 0.0) > 1e-9:
            domain_reasons.append(f"{label}-root-boundary-ice")
        free = _finite(state.get("remaining_free_downstream_m"))
        if free is None or free < 0:
            domain_reasons.append(f"{label}-observation-domain-overflow")
    numerical_pass = bool(
        summary.get("runtime_error") in (None, "")
        and int(_finite(final.get("year")) or -1) == 2000
        and summary.get("status") == "historical-pass"
    )
    initial_length = _finite(initial.get("length_m"))
    peak_joint_upper = min(lia * 1.1, lia + 0.2 * separation)
    initial_peak_bound_pass = bool(initial_length is not None and initial_length <= peak_joint_upper)
    flags = []
    if not total_pass:
        flags.append("total-length-gate")
    if not separation_pass:
        flags.append("separation-length-gate")
    if domain_reasons:
        flags.extend(domain_reasons)
    if not numerical_pass:
        flags.append("numerical-or-completion-gate")
    if not initial_peak_bound_pass:
        flags.append("initial-peak-hard-bound")
    formal = {
        **summary,
        "status": "historical-pass" if not flags else "historical-fail",
        "formal_total_length_pass": total_pass,
        "formal_separation_length_pass": separation_pass,
        "formal_observation_domain_pass": not domain_reasons,
        "formal_numerical_completion_pass": numerical_pass,
        "formal_initial_peak_hard_bound_pass": initial_peak_bound_pass,
        "formal_peak_joint_upper_m": peak_joint_upper,
        "formal_review_flags": list(dict.fromkeys(flags)),
    }
    (output_root / "historical_summary_formal.json").write_text(
        json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return formal


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _preflight(config)
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"versioned-output-exists:{output_root}")
    summary = historical.run(config_path)
    return _formalize(config, summary, output_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.preflight_only:
        try:
            _preflight(config)
            result = {"status": "ready-steady-pass", "historical_run": False}
        except RuntimeError as exc:
            result = {"status": "automatic-refusal", "reason": str(exc), "historical_run": False}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
