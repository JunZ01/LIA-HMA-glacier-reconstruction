"""Checkpointed V9 launcher for disconnection-diagnostic policy V4."""

from __future__ import annotations

import argparse
import json

from ..paths import resolve
from . import network_topology_adaptive_split_system2005_zero_mb_v9 as solver_v9
from . import run_system2005_disconnection_diagnostic_historical_v1 as policy_v4
from . import run_system2005_topology_adaptive_historical_v3_checkpoint as checkpointed


def _write_provenance() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _unknown = parser.parse_known_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    summary_path = resolve(config["output_root"]) / "historical_summary_formal.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "schema": "lia-191-system2005-disconnection-diagnostic-historical-v2-checkpoint",
        "solver": "network-topology-adaptive-split-system2005-zero-mb-v9",
        "realized_reverse_tolerance_m": 0.1,
        "other_numerical_tolerances_modified": False,
        "annual_checkpoint_schema": checkpointed.CHECKPOINT_SCHEMA,
        "resume_supported": True,
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    solver_v9.reset_structure_cache()
    exit_error = None
    try:
        try:
            policy_v4.main()
        except SystemExit as exc:
            exit_error = exc
    finally:
        solver_v9.reset_structure_cache()
    _write_provenance()
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()
