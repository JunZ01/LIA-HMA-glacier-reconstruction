"""Ten-year trajectory equivalence gate for cached V6 and graph-reuse V8."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
BASELINE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/s2005p5pf0375v9_12h_cached"
)
GRAPH_REUSE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/s2005p5pf0375v10_12h_graph_reuse"
)
ORIGINAL_12H = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/s2005p5pf0375v8_12h"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_12h_graph_reuse_10y_equivalence_v1/summary.json"
)


def load_state(root: Path):
    with gzip.open(root / "state_1480_preflight.pkl.gz", "rb") as stream:
        return pickle.load(stream)


def load_summary(root: Path):
    return json.loads((root / "historical_summary_formal.json").read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.parent.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT.parent}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    baseline_state = load_state(BASELINE)
    graph_state = load_state(GRAPH_REUSE)
    names = ("section", "thick", "surface_h", "widths_m", "bed_h")
    array_equal = {name: True for name in names}
    array_maxima = {name: 0.0 for name in names}
    for baseline_flowline, graph_flowline in zip(baseline_state, graph_state):
        for name in names:
            baseline_array = np.asarray(getattr(baseline_flowline, name))
            graph_array = np.asarray(getattr(graph_flowline, name))
            array_equal[name] &= bool(np.array_equal(baseline_array, graph_array))
            array_maxima[name] = max(
                array_maxima[name], float(np.max(np.abs(baseline_array - graph_array)))
            )
    baseline_summary = load_summary(BASELINE)
    graph_summary = load_summary(GRAPH_REUSE)
    original_summary = load_summary(ORIGINAL_12H)
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
        key: baseline_summary.get(key) == graph_summary.get(key) for key in science_keys
    }
    checks = {
        "all_final_state_arrays_bitwise_equal": all(array_equal.values()),
        "all_saved_science_diagnostics_equal": all(science_equal.values()),
        "step_count_is_7320": graph_summary.get("step_count") == 7320,
        "graph_reuse_numerical_gates_pass": all(graph_summary["numerical_checks"].values()),
    }
    runtime = float(graph_summary["runtime_seconds"])
    projected_seconds = runtime * 53.0
    summary = {
        "schema": "lia-191-system2005-12h-graph-reuse-10y-equivalence-v1",
        "status": (
            "equivalence-pass-provisional-optimization-baseline"
            if all(checks.values())
            else "equivalence-fail"
        ),
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "saved_science_diagnostic_equality": science_equal,
        "original_12h_runtime_seconds": float(original_summary["runtime_seconds"]),
        "cached_v6_runtime_seconds": float(baseline_summary["runtime_seconds"]),
        "graph_reuse_v8_runtime_seconds": runtime,
        "speedup_vs_original_12h": float(original_summary["runtime_seconds"]) / runtime,
        "speedup_vs_cached_v6": float(baseline_summary["runtime_seconds"]) / runtime,
        "linear_projected_530y_runtime_seconds": projected_seconds,
        "linear_projected_530y_runtime_hours": projected_seconds / 3600.0,
        "projection_is_not_full_run_authorization": True,
        "scientific_parameters_modified": False,
        "historical_acceptance_policy_modified": False,
        "full_historical_authorized": False,
        "authorization_blockers": [
            "root-gap-policy-conflict-remains",
            "separate-0p1m-realized-reverse-slope-tolerance-not-yet-implemented",
            "annual-resume-checkpoints-not-yet-implemented",
            "50y-trajectory-equivalence-gate-not-yet-run"
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
