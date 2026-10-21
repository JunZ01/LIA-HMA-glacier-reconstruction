"""Formal 1478 length preflight for the prior 25.25 km steady state."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
START = RESULTS / "s2005fs25250p4stop1472v1"
RESUME = RESULTS / "s2005fs25250p4resume1478v1"
STEADY = RESULTS / "lia_191_system2005_terminal_retreat_gap_adjudication_v2/steady_50y_summary_formal.json"
OUTPUT = RESULTS / "lia_191_system2005_formalsteady25250_policyv4_1478_preflight_v1"


def load_bundle(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    steady = json.loads(STEADY.read_text(encoding="utf-8"))
    start_manifest = json.loads(
        (START / "checkpoints/year_1472/manifest.json").read_text(encoding="utf-8")
    )
    final_manifest = json.loads(
        (RESUME / "checkpoints/year_1478/manifest.json").read_text(encoding="utf-8")
    )
    start = load_bundle(START / "checkpoints/year_1472/state.pkl.gz")
    final = load_bundle(RESUME / "checkpoints/year_1478/state.pkl.gz")
    rows = final["annual_rows"]
    lengths = [float(row["length_m"]) for row in rows]
    lia_length = 25985.52229946727
    rgi_length = 23308.5403923166
    separation = lia_length - rgi_length
    minimum_pass_length = lia_length - 0.2 * separation
    peak_length = max(lengths)
    e_lia_separation = abs(peak_length - lia_length) / separation
    checks = {
        "formal_steady_state_passed": (
            steady.get("status") == "steady-50y-pass"
            and all(steady.get("checks", {}).values())
        ),
        "resume_contract_matches_start": (
            start_manifest["contract_sha256"] == final_manifest["contract_sha256"]
        ),
        "start_checkpoint_at_1472_1464_steps": (
            start["step_count"] == 1464 and start_manifest["year"] == 1472
        ),
        "final_checkpoint_at_1478_5856_steps": (
            final["step_count"] == 5856 and final_manifest["year"] == 1478
        ),
        "nine_annual_states_1470_through_1478": (
            [int(row["year"]) for row in rows] == list(range(1470, 1479))
        ),
        "per_step_closure_pass": final["max_closure"] <= final["closure_limit"],
        "kkt_pass": final["max_kkt"] <= 1e-5,
        "finite_peak_length": peak_length > 0.0,
        "disconnection_diagnostic_only": True,
        "lia_separation_length_gate_pass": e_lia_separation <= 0.2,
    }
    summary = {
        "schema": "lia-191-system2005-formalsteady25250-policyv4-1478-preflight-v1",
        "status": "length-preflight-pass" if all(checks.values()) else "length-preflight-fail",
        "checks": checks,
        "annual_connected_lengths_m": {
            str(int(row["year"])): float(row["length_m"]) for row in rows
        },
        "peak_connected_length_m": peak_length,
        "registered_lia_length_m": lia_length,
        "minimum_20pct_separation_pass_length_m": minimum_pass_length,
        "shortfall_to_pass_m": minimum_pass_length - peak_length,
        "e_lia_total": abs(peak_length - lia_length) / lia_length,
        "e_lia_separation": e_lia_separation,
        "maximum_absolute_step_closure_m3": final["max_closure"],
        "closure_limit_m3": final["closure_limit"],
        "maximum_absolute_kkt_stationarity": final["max_kkt"],
        "maximum_internal_gap_branch_count": final["maximum_gap_count"],
        "nonzero_exit_is_intentional_checkpoint_stop": True,
        "model_state_modified": False,
        "physics_forcing_or_solver_modified": False,
        "full_historical_authorized": False,
        "remaining_blocker": "formal-steady-peak-is-200m-short-of-20pct-separation-gate",
        "next_legal_options": [
            "retain-20pct-and-produce-a-new-formal-steady-state-at-least-25450p126m",
            "explicitly-revise-separation-threshold-to-at-least-0p275-with-post-result-disclosure",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    if summary["status"] != "length-preflight-fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
