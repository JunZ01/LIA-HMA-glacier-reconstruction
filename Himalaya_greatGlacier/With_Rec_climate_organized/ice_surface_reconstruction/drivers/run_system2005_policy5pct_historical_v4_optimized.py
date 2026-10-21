"""Versioned historical launcher using the full-array-cache V6 solver."""

from __future__ import annotations

import argparse
import json

from ..paths import resolve
from . import network_topology_adaptive_split_system2005_zero_mb_v6 as optimized_solver
from . import run_system2005_policy5pct_historical_v3 as historical_v3


def _write_optimized_provenance() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _unknown = parser.parse_known_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    summary_path = resolve(config["output_root"]) / "historical_summary_formal.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "schema": "lia-191-system2005-policy5pct-historical-v4-optimized",
        "mixedbed_property_access": (
            "all-dynamic-arrays-cached-once-per-branch-per-step"
        ),
        "solver_optimization": (
            "property-cache-only; scalar conductance and solver arithmetic unchanged"
        ),
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    original_solver = historical_v3.cached_solver
    historical_v3.cached_solver = optimized_solver
    exit_error = None
    try:
        try:
            historical_v3.main()
        except SystemExit as exc:
            exit_error = exc
    finally:
        historical_v3.cached_solver = original_solver
    _write_optimized_provenance()
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()