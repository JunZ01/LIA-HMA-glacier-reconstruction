"""Audit that policy V4 changes diagnostics but not the 1472 model state."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
REFERENCE = RESULTS / "s2005g50v9ckv1/checkpoints/year_1472/state.pkl.gz"
CANDIDATE = RESULTS / "s2005p4stop1472v1/checkpoints/year_1472/state.pkl.gz"
OUTPUT = RESULTS / "lia_191_system2005_disconnection_policy_v4_stop1472_equivalence_v1"


def load(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    reference = load(REFERENCE)
    candidate = load(CANDIDATE)
    array_names = ("section", "thick", "surface_h", "widths_m", "bed_h")
    array_equal = {name: True for name in array_names}
    array_maxima = {name: 0.0 for name in array_names}
    for reference_flowline, candidate_flowline in zip(
        reference["flowlines"], candidate["flowlines"]
    ):
        for name in array_names:
            left = np.asarray(getattr(reference_flowline, name))
            right = np.asarray(getattr(candidate_flowline, name))
            array_equal[name] &= bool(np.array_equal(left, right))
            array_maxima[name] = max(
                array_maxima[name], float(np.max(np.abs(left - right)))
            )

    state_keys = (
        "closure_limit", "elapsed", "step_count", "cumulative_mb",
        "cumulative_overmelt", "max_closure", "max_kkt",
        "max_active_constraints", "maximum_gap_count", "events",
        "current_event", "trace_rows", "next_record_year",
    )
    state_equal = {key: reference[key] == candidate[key] for key in state_keys}
    common_annual_keys = (
        "year", "volume_m3", "area_m2", "finite_geometry",
        "negative_section_count", "disconnected_tributary_count",
        "internal_gap_branch_count", "root_gap_branch_count",
        "tributary_gap_branch_count", "aggregate_detached_volume_fraction",
        "aggregate_detached_area_fraction",
        "newly_activated_initial_zero_node_count", "terminal_thickness_m",
    )
    annual_equal = {
        key: [row[key] for row in reference["annual_rows"]]
        == [row[key] for row in candidate["annual_rows"]]
        for key in common_annual_keys
    }
    old_lengths = [float(row["length_m"]) for row in reference["annual_rows"]]
    new_lengths = [float(row["length_m"]) for row in candidate["annual_rows"]]
    expected_length_change = (
        old_lengths == [24300.0, 24300.0, 24300.0]
        and new_lengths == [24300.0, 24100.0, 24100.0]
    )
    checks = {
        "both_at_1472_1464_steps": (
            reference["step_count"] == candidate["step_count"] == 1464
        ),
        "all_geometry_arrays_bitwise_equal": all(array_equal.values()),
        "all_model_accumulators_and_events_equal": all(state_equal.values()),
        "all_common_annual_science_fields_equal": all(annual_equal.values()),
        "only_expected_connected_length_change": expected_length_change,
        "candidate_has_root_remnant_diagnostics": all(
            "root_downstream_remnant_volume_fraction" in row
            and "root_downstream_remnant_area_fraction" in row
            for row in candidate["annual_rows"]
        ),
    }
    summary = {
        "schema": "lia-191-system2005-disconnection-policy-v4-stop1472-equivalence-v1",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "model_state_equality": state_equal,
        "common_annual_field_equality": annual_equal,
        "reference_lengths_m": old_lengths,
        "connected_component_lengths_m": new_lengths,
        "scientific_state_modified": False,
        "physics_forcing_or_solver_modified": False,
        "policy_or_scoring_fields_modified_only": True,
        "full_historical_authorized": False,
        "next_gate": "rescore-complete-50y-result-under-policy-v4",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    if summary["status"] != "equivalence-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
