"""Ten-year trajectory equivalence gate for V5 and full-array-cache V6."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
BASELINE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/s2005p5pf0375v8_12h"
)
OPTIMIZED = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/s2005p5pf0375v9_12h_cached"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_12h_cached_10y_equivalence_v1/summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(root: Path):
    with gzip.open(root / "state_1480_preflight.pkl.gz", "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    if OUTPUT.parent.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT.parent}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    baseline_state = load_state(BASELINE)
    optimized_state = load_state(OPTIMIZED)
    names = ("section", "thick", "surface_h", "widths_m", "bed_h")
    array_equal = {name: True for name in names}
    array_maxima = {name: 0.0 for name in names}
    for baseline_flowline, optimized_flowline in zip(baseline_state, optimized_state):
        for name in names:
            baseline_array = np.asarray(getattr(baseline_flowline, name))
            optimized_array = np.asarray(getattr(optimized_flowline, name))
            array_equal[name] &= bool(np.array_equal(baseline_array, optimized_array))
            array_maxima[name] = max(
                array_maxima[name],
                float(np.max(np.abs(baseline_array - optimized_array))),
            )

    baseline_summary_path = BASELINE / "historical_summary_formal.json"
    optimized_summary_path = OPTIMIZED / "historical_summary_formal.json"
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    optimized_summary = json.loads(optimized_summary_path.read_text(encoding="utf-8"))
    science_keys = (
        "step_count",
        "initial",
        "peak",
        "final",
        "length_checks",
        "numerical_checks",
        "events",
        "maximum_internal_gap_branch_count",
        "final_internal_gap_branch_count",
        "maximum_absolute_step_closure_m3",
        "maximum_absolute_KKT_stationarity",
        "maximum_active_constraint_count",
        "closure_limit_m3",
        "cumulative_accepted_mb_m3",
        "cumulative_prevented_overmelt_m3",
        "volume_change_fraction_start_end",
        "tributary_separation_checks",
        "historical_connectivity_extrema",
    )
    science_equal = {
        key: baseline_summary.get(key) == optimized_summary.get(key) for key in science_keys
    }
    checks = {
        "flowline_count_equal": len(baseline_state) == len(optimized_state),
        "all_final_state_arrays_bitwise_equal": all(array_equal.values()),
        "all_saved_science_diagnostics_equal": all(science_equal.values()),
        "step_count_is_7320": optimized_summary.get("step_count") == 7320,
        "optimized_numerical_gates_pass": all(
            optimized_summary.get("numerical_checks", {}).values()
        ),
    }
    baseline_runtime = float(baseline_summary["runtime_seconds"])
    optimized_runtime = float(optimized_summary["runtime_seconds"])
    runtime_speedup = baseline_runtime / optimized_runtime
    projected_530y_seconds = optimized_runtime * 53.0
    summary = {
        "schema": "lia-191-system2005-12h-cached-10y-equivalence-v1",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "saved_science_diagnostic_equality": science_equal,
        "baseline_runtime_seconds": baseline_runtime,
        "optimized_runtime_seconds": optimized_runtime,
        "runtime_speedup": runtime_speedup,
        "linear_projected_530y_runtime_seconds": projected_530y_seconds,
        "linear_projected_530y_runtime_hours": projected_530y_seconds / 3600.0,
        "projection_is_not_full_run_authorization": True,
        "baseline_state_sha256": sha256(BASELINE / "state_1480_preflight.pkl.gz"),
        "optimized_state_sha256": sha256(OPTIMIZED / "state_1480_preflight.pkl.gz"),
        "baseline_summary_sha256": sha256(baseline_summary_path),
        "optimized_summary_sha256": sha256(optimized_summary_path),
        "simulation_scope": "1470-1480 historical climate, 12-hour maximum step, 7320 steps",
        "scientific_parameters_modified": False,
        "historical_acceptance_policy_modified": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "equivalence-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
