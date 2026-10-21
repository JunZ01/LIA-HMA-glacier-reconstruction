"""Fifty-year equivalence gate for V8 and V9 annual-checkpoint solver."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from oggm import cfg


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
V8 = RESULTS / "s2005g50v8v1"
V9 = RESULTS / "s2005g50v9ckv1"
OUTPUT = RESULTS / "lia_191_system2005_v8_v9_checkpoint_50y_equivalence_v1"


def load_state(root: Path):
    with gzip.open(root / "state_2000.pkl.gz", "rb") as stream:
        return pickle.load(stream)


def load_summary(root: Path):
    return json.loads((root / "historical_summary_formal.json").read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    v8_state = load_state(V8)
    v9_state = load_state(V9)
    names = ("section", "thick", "surface_h", "widths_m", "bed_h")
    array_equal = {name: True for name in names}
    array_maxima = {name: 0.0 for name in names}
    for v8_flowline, v9_flowline in zip(v8_state, v9_state):
        for name in names:
            v8_array = np.asarray(getattr(v8_flowline, name))
            v9_array = np.asarray(getattr(v9_flowline, name))
            array_equal[name] &= bool(np.array_equal(v8_array, v9_array))
            array_maxima[name] = max(
                array_maxima[name], float(np.max(np.abs(v8_array - v9_array)))
            )
    v8_summary = load_summary(V8)
    v9_summary = load_summary(V9)
    science_keys = (
        "step_count", "initial", "peak", "final", "length_checks",
        "numerical_checks", "events", "maximum_internal_gap_branch_count",
        "final_internal_gap_branch_count", "maximum_absolute_step_closure_m3",
        "maximum_absolute_KKT_stationarity", "maximum_active_constraint_count",
        "closure_limit_m3", "cumulative_accepted_mb_m3",
        "cumulative_prevented_overmelt_m3", "volume_change_fraction_start_end",
        "tributary_separation_checks", "historical_connectivity_extrema",
    )
    science_equal = {
        key: v8_summary.get(key) == v9_summary.get(key) for key in science_keys
    }
    manifests = sorted((V9 / "checkpoints").glob("year_*/manifest.json"))
    manifest_values = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    contract_hashes = {value.get("contract_sha256") for value in manifest_values}
    temporary_files = list((V9 / "checkpoints").rglob("*.tmp"))
    checks = {
        "both_completed_50y_36600_steps": (
            v8_summary.get("step_count") == v9_summary.get("step_count") == 36600
        ),
        "both_numerical_gate_sets_pass": (
            all(v8_summary["numerical_checks"].values())
            and all(v9_summary["numerical_checks"].values())
        ),
        "all_final_state_arrays_bitwise_equal": all(array_equal.values()),
        "all_saved_science_diagnostics_equal": all(science_equal.values()),
        "exactly_50_complete_annual_checkpoints": (
            len(manifests) == 50
            and all(value.get("status") == "checkpoint-complete" for value in manifest_values)
        ),
        "checkpoint_years_1471_through_1520": (
            [value.get("year") for value in manifest_values] == list(range(1471, 1521))
        ),
        "checkpoint_steps_732_through_36600": (
            [value.get("step_count") for value in manifest_values]
            == [732 * year for year in range(1, 51)]
        ),
        "single_checkpoint_contract_hash": len(contract_hashes) == 1,
        "checkpoint_contract_matches_summary": (
            contract_hashes == {v9_summary.get("checkpoint_contract_sha256")}
        ),
        "no_temporary_checkpoint_files": len(temporary_files) == 0,
        "realized_reverse_tolerance_is_0p1m": (
            v9_summary.get("realized_reverse_tolerance_m") == 0.1
        ),
        "other_numerical_tolerances_unchanged": (
            v9_summary.get("other_numerical_tolerances_modified") is False
        ),
    }
    runtime = float(v9_summary["runtime_seconds"])
    summary = {
        "schema": "lia-191-system2005-v8-v9-checkpoint-50y-equivalence-v1",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "science_diagnostic_equality": science_equal,
        "v8_runtime_seconds": float(v8_summary["runtime_seconds"]),
        "v9_checkpoint_runtime_seconds": runtime,
        "checkpoint_runtime_ratio_vs_v8": runtime / float(v8_summary["runtime_seconds"]),
        "linear_projected_530y_runtime_hours": runtime * (530.0 / 50.0) / 3600.0,
        "projection_is_not_full_run_authorization": True,
        "checkpoint_manifest_count": len(manifests),
        "checkpoint_contract_hashes": sorted(contract_hashes),
        "full_historical_authorized": False,
        "remaining_blocker": "root-gap-policy-conflict-remains",
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
