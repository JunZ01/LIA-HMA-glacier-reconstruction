"""Two-year continuous-versus-resumed equivalence gate for annual checkpoints."""

from __future__ import annotations

import gzip
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from . import run_system2005_topology_adaptive_historical_v2_checkpoint as checkpointed


PROJECT = Path(__file__).resolve().parents[4]
BASE_CONFIG = PROJECT / "config/system2005_policy5pct_historical_10y_preflight_endp0p375_v10_12h_graph_reuse.json"
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_checkpoint_resume_equivalence_v2"
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_state(path: Path, flowlines) -> None:
    with gzip.open(path, "wb") as stream:
        pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)


def run_case(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config_snapshot.json", config)
    summary, rows, trace, flowlines = checkpointed.run(config, output)
    write_json(output / "summary.json", summary)
    save_state(output / "state_final.pkl.gz", flowlines)
    return summary, rows, trace, flowlines


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base.update({
        "experiment": "lia-191-system2005-checkpoint-resume-equivalence-v2",
        "stage": "1470-1472-continuous-versus-annual-resume",
        "end_year": 1472,
        "forward_delta_t_end_c": -0.12311320754716981,
    })

    continuous_config = dict(base)
    continuous_config["output_root"] = str(OUTPUT / "continuous_2y")
    continuous_summary, continuous_rows, continuous_trace, continuous_flowlines = run_case(
        continuous_config, OUTPUT / "continuous_2y"
    )

    interrupted_output = OUTPUT / "interrupted_at_1471"
    interrupted_output.mkdir(parents=True, exist_ok=False)
    interrupted_config = dict(base)
    interrupted_config.update({
        "output_root": str(interrupted_output),
        "stop_after_checkpoint_year": 1471,
    })
    write_json(interrupted_output / "config_snapshot.json", interrupted_config)
    interrupted_error = None
    try:
        checkpointed.run(interrupted_config, interrupted_output)
    except RuntimeError as exc:
        interrupted_error = str(exc)
    manifest_path = interrupted_output / "checkpoints/year_1471/manifest.json"

    resumed_config = dict(base)
    resumed_config.update({
        "output_root": str(OUTPUT / "resumed_to_1472"),
        "resume_checkpoint_manifest": str(manifest_path),
    })
    resumed_summary, resumed_rows, resumed_trace, resumed_flowlines = run_case(
        resumed_config, OUTPUT / "resumed_to_1472"
    )

    names = ("section", "thick", "surface_h", "widths_m", "bed_h")
    array_equal = {name: True for name in names}
    array_maxima = {name: 0.0 for name in names}
    for continuous_flowline, resumed_flowline in zip(continuous_flowlines, resumed_flowlines):
        for name in names:
            continuous_array = np.asarray(getattr(continuous_flowline, name))
            resumed_array = np.asarray(getattr(resumed_flowline, name))
            array_equal[name] &= bool(np.array_equal(continuous_array, resumed_array))
            array_maxima[name] = max(
                array_maxima[name],
                float(np.max(np.abs(continuous_array - resumed_array))),
            )

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
    )
    science_equal = {
        key: continuous_summary.get(key) == resumed_summary.get(key)
        for key in science_keys
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    temporary_files = list(OUTPUT.rglob("*.tmp"))
    checks = {
        "intentional_stop_observed_after_checkpoint": (
            interrupted_error is not None
            and interrupted_error.startswith("intentional-stop-after-checkpoint:1471:")
        ),
        "checkpoint_manifest_complete": manifest.get("status") == "checkpoint-complete",
        "checkpoint_year_1471": manifest.get("year") == 1471,
        "checkpoint_step_732": manifest.get("step_count") == 732,
        "resume_contract_hash_equal": (
            continuous_summary["checkpoint_contract_sha256"]
            == resumed_summary["checkpoint_contract_sha256"]
            == manifest["contract_sha256"]
        ),
        "resumed_source_recorded": resumed_summary["resumed_from_checkpoint"] is True,
        "all_final_state_arrays_bitwise_equal": all(array_equal.values()),
        "all_science_diagnostics_equal": all(science_equal.values()),
        "annual_rows_equal": continuous_rows == resumed_rows,
        "event_trace_equal": continuous_trace == resumed_trace,
        "no_temporary_checkpoint_files": len(temporary_files) == 0,
    }
    summary = {
        "schema": "lia-191-system2005-checkpoint-resume-equivalence-v2",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "checks": checks,
        "array_bitwise_equality": array_equal,
        "maximum_absolute_array_differences": array_maxima,
        "science_diagnostic_equality": science_equal,
        "continuous_runtime_seconds": continuous_summary["runtime_seconds"],
        "resumed_cumulative_runtime_seconds": resumed_summary["runtime_seconds"],
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_state_sha256": manifest["state_sha256"],
        "temporary_files": [str(path) for path in temporary_files],
        "full_historical_authorized": False,
        "next_action": "run-50y-v8-v9-checkpoint-equivalence-gate",
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary))
    if summary["status"] != "equivalence-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
