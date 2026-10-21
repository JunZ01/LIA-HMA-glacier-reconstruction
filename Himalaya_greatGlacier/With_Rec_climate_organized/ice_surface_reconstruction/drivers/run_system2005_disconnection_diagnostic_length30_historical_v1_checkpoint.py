"""V9 checkpoint launcher for disconnection diagnostics and 30% length gate."""

from __future__ import annotations

import argparse
import json

from ..paths import resolve
from . import run_system2005_disconnection_diagnostic_historical_v1 as base


POLICY_SCHEMA = "lia-191-disconnection-diagnostic-length30-policy-v5"


def _write_provenance():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _unknown = parser.parse_known_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    summary_path = resolve(config["output_root"]) / "historical_summary_formal.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "schema": "lia-191-system2005-disconnection-diagnostic-length30-historical-v1-checkpoint",
        "policy_schema": POLICY_SCHEMA,
        "separation_normalized_length_error_max": 0.3,
        "total_length_error_max": 0.1,
        "solver": "network-topology-adaptive-split-system2005-zero-mb-v9",
        "realized_reverse_tolerance_m": 0.1,
        "other_numerical_tolerances_modified": False,
        "resume_supported": True,
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    previous_schema = base.POLICY_SCHEMA
    base.POLICY_SCHEMA = POLICY_SCHEMA
    exit_error = None
    try:
        try:
            base.main()
        except SystemExit as exc:
            exit_error = exc
    finally:
        base.POLICY_SCHEMA = previous_schema
    _write_provenance()
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()
