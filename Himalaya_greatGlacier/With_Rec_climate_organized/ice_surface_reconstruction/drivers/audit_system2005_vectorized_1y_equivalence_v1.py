"""One-year adjudication of cached V6 against vectorized V7."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
BASELINE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/s2005p5prof0375v3_12h_cached"
)
VECTORIZED = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/s2005p5prof0375v5_12h_vectorized"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_vectorized_1y_equivalence_v1/summary.json"
)


def load_state(root: Path):
    with gzip.open(root / "state_2000.pkl.gz", "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    if OUTPUT.parent.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT.parent}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    baseline_state = load_state(BASELINE)
    vectorized_state = load_state(VECTORIZED)
    names = ("section", "thick", "surface_h", "widths_m", "bed_h")
    array_equal = {name: True for name in names}
    array_maxima = {name: 0.0 for name in names}
    for baseline_flowline, vectorized_flowline in zip(baseline_state, vectorized_state):
        for name in names:
            baseline_array = np.asarray(getattr(baseline_flowline, name))
            vectorized_array = np.asarray(getattr(vectorized_flowline, name))
            array_equal[name] &= bool(np.array_equal(baseline_array, vectorized_array))
            array_maxima[name] = max(
                array_maxima[name],
                float(np.max(np.abs(baseline_array - vectorized_array))),
            )
    baseline_summary = json.loads(
        (BASELINE / "historical_summary_formal.json").read_text(encoding="utf-8")
    )
    vectorized_summary = json.loads(
        (VECTORIZED / "historical_summary_formal.json").read_text(encoding="utf-8")
    )
    baseline_final = baseline_summary["final"]
    vectorized_final = vectorized_summary["final"]
    volume_difference = float(vectorized_final["volume_m3"] - baseline_final["volume_m3"])
    area_difference = float(vectorized_final["area_m2"] - baseline_final["area_m2"])
    policy_classification_equal = all(
        baseline_final[key] == vectorized_final[key]
        for key in (
            "length_m",
            "disconnected_tributary_count",
            "internal_gap_branch_count",
            "root_gap_branch_count",
            "tributary_gap_branch_count",
        )
    )
    checks = {
        "both_numerical_gate_sets_pass": (
            all(baseline_summary["numerical_checks"].values())
            and all(vectorized_summary["numerical_checks"].values())
        ),
        "policy_classification_equal": policy_classification_equal,
        "all_final_state_arrays_bitwise_equal": all(array_equal.values()),
    }
    summary = {
        "schema": "lia-191-system2005-vectorized-1y-equivalence-v1",
        "status": "vectorized-not-authorized-bitwise-trajectory-gate-failed",
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "final_volume_difference_m3": volume_difference,
        "final_volume_relative_difference": volume_difference / float(baseline_final["volume_m3"]),
        "final_area_difference_m2": area_difference,
        "final_area_relative_difference": area_difference / float(baseline_final["area_m2"]),
        "baseline_runtime_seconds": float(baseline_summary["runtime_seconds"]),
        "vectorized_runtime_seconds": float(vectorized_summary["runtime_seconds"]),
        "runtime_speedup": (
            float(baseline_summary["runtime_seconds"])
            / float(vectorized_summary["runtime_seconds"])
        ),
        "scientific_parameters_modified": False,
        "historical_acceptance_policy_modified": False,
        "vectorized_full_historical_authorized": False,
        "next_action": "retain-v6-and-test-scalar-order-preserving-graph-structure-reuse",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
