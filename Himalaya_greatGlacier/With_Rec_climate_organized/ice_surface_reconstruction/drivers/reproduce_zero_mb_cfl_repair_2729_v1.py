"""Independent zero-MB reproduction from the saved repaired 2729 flowlines."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from .pilot_zero_mb_cfl_local_continuity_v1 import _parse_required, _run_zero_mb


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    pilot_root = args.pilot_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    state_path = pilot_root / "repaired_model_flowlines.pkl.gz"
    with gzip.open(state_path, "rb") as handle:
        flowlines = pickle.load(handle)
    config = json.loads(args.source_config.resolve().read_text(encoding="utf-8"))
    initial, steps, checkpoints, error, tb, model = _run_zero_mb(
        flowlines, float(config["glen_a"]), float(config["fs"])
    )
    pd.DataFrame(steps).to_csv(output_root / "zero_mb_steps.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
    limit = max(1e-5, 1e-14 * initial)
    maximum = float(pd.DataFrame(steps)["closure_residual_m3"].abs().max()) if steps else None
    completed = float(checkpoints[-1]["year"]) if checkpoints else 0.0
    drift = float(checkpoints[-1]["volume_drift_fraction"]) if checkpoints else None
    terminal = float([fl for fl in model.fls if fl.flows_to is None][0].thick[-1])
    passed = bool(error is None and completed == 5.0 and drift is not None and abs(drift) <= 0.001 and
                  maximum is not None and maximum <= limit and terminal <= 10.0)
    pilot = json.loads((pilot_root / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "experiment": "independent-zero-mb-cfl-repair-2729-reproduction-v1",
        "system_id": 2729, "status": "zero-mb-pass" if passed else "zero-mb-fail",
        "minimum_timestep_seconds": 60.0, "repaired_state_sha256": _hash(state_path),
        "completed_checkpoint_year": completed, "volume_drift_fraction": drift,
        "maximum_absolute_step_closure_residual_m3": maximum,
        "per_step_closure_limit_m3": limit, "root_terminal_thickness_m": terminal,
        "runtime_error": error, "required_timestep_seconds_if_failed": _parse_required(error),
        "failure_traceback": tb,
        "exact_metric_reproduction": {
            "volume_drift": drift == pilot["volume_drift_fraction"],
            "maximum_step_residual": maximum == pilot["maximum_absolute_step_closure_residual_m3"],
            "completed_year": completed == pilot["completed_checkpoint_year"],
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
