"""V2 numerical-boundary correction for topology-adaptive split pilot."""

from __future__ import annotations

import argparse
import csv
import json
import traceback

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_positive_conservative_system2005_failure_state_step_v1 as active_set
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as v1


def run(config):
    technical = json.loads(base.resolve(config["technical_predecessor"]).read_text(encoding="utf-8"))
    error = technical.get("error", "")
    if "active-section-interior-floor:9.79764790132176e-09" not in error:
        raise RuntimeError("v1-is-not-the-registered-numerical-interior-floor-failure")
    original = active_set.solve_affine_active_set

    def tight_solver(matrix, rhs, constraint_matrix, lower, volume_row, volume_target, **_kwargs):
        return original(
            matrix,
            rhs,
            constraint_matrix,
            lower,
            volume_row,
            volume_target,
            tolerance=float(config["active_set_tolerance"]),
            maximum_iterations=400,
        )

    active_set.solve_affine_active_set = tight_solver
    try:
        summary, rows = v1.run(config)
    finally:
        active_set.solve_affine_active_set = original
    summary["schema"] = "lia-191-network-topology-adaptive-split-system2005-zero-mb-v2"
    summary["technical_correction"] = (
        "physical section>=0 boundary; no positive-support reserve; active-set tolerance 1e-13"
    )
    summary["technical_predecessor_sha256"] = base.sha256(base.resolve(config["technical_predecessor"]))
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = base.resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        summary, rows = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-v2",
            "status": "zero-mb-technical-or-scientific-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "source_modified": False,
            "zero_mb_5y_run": False,
            "steady_50y_run": False,
            "historical_run": False,
        }
        rows = []
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "checkpoints.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary))
    if summary.get("status") != "zero-mb-0p1y-pass-topology-adaptive-split":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
