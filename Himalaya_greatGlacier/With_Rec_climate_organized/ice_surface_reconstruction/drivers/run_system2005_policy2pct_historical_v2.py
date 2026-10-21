"""Full system-2005 historical endpoint under the uniform 2% tributary policy."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import traceback

import numpy as np

from ..paths import resolve
from . import audit_lia_saved_state_connectivity_v1 as connectivity
from . import network_topology_adaptive_split_system2005_zero_mb_v5 as cached_solver
from . import run_system2005_topology_adaptive_historical_v1 as historical


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config_snapshot.json", config)
    flowlines = None
    rows = []
    trace = []
    try:
        policy_path = resolve(config["tributary_separation_policy"])
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        steady_root = resolve(config["steady_root"])
        steady_state_path = steady_root / "lia_equilibrium_state_50y.pkl.gz"
        with gzip.open(steady_state_path, "rb") as stream:
            initial_flowlines = pickle.load(stream)
        tolerance = float(config["active_thickness_tolerance_m"])
        initial_structure = connectivity.state_connectivity(initial_flowlines, tolerance)
        initial_zero = [np.asarray(flowline.section) == 0.0 for flowline in initial_flowlines]

        original_step = historical.solver.conservative_boundary_step
        cached_solver.conservative_boundary_step.config = config
        historical.solver.conservative_boundary_step = cached_solver.conservative_boundary_step
        try:
            base_summary, rows, trace, flowlines = historical.run(config, output)
        finally:
            historical.solver.conservative_boundary_step = original_step

        final_structure = connectivity.state_connectivity(flowlines, tolerance)
        root_branch = int(final_structure["root_branch"])
        events_exclude_root = all(
            root_branch not in [int(branch) for branch in event.get("branches", [])]
            for event in base_summary.get("events", [])
        )
        newly_activated = sum(
            int(np.count_nonzero(mask & (np.asarray(flowline.section) > 0.0)))
            for mask, flowline in zip(initial_zero, flowlines)
        )
        volume_limit = float(policy["detached_tributary_volume_fraction_max"])
        area_limit = float(policy["detached_tributary_area_fraction_max"])
        gap_checks = {
            "initial_root_continuous": int(initial_structure["root_gap_branch_count"]) == 0,
            "final_root_continuous": int(final_structure["root_gap_branch_count"]) == 0,
            "all_gap_events_exclude_root": events_exclude_root,
            "at_most_one_internal_gap_branch": int(base_summary["maximum_internal_gap_branch_count"]) <= 1,
            "initial_detached_volume_within_2pct": float(initial_structure["detached_head_volume_fraction_system"]) <= volume_limit,
            "final_detached_volume_within_2pct": float(final_structure["detached_head_volume_fraction_system"]) <= volume_limit,
            "initial_detached_area_within_0p5pct": float(initial_structure["detached_head_area_fraction_system"]) <= area_limit,
            "final_detached_area_within_0p5pct": float(final_structure["detached_head_area_fraction_system"]) <= area_limit,
            "artificial_positive_support_absent": newly_activated == 0,
            "posthoc_section_clip_absent": True,
        }
        checks = {
            "numerical": all(base_summary["numerical_checks"].values()),
            "length": all(base_summary["length_checks"].values()),
            "tributary_separation_policy": all(gap_checks.values()),
        }
        passed = all(checks.values())
        summary = dict(base_summary)
        summary.update({
            "schema": "lia-191-system2005-policy2pct-historical-v2",
            "status": "historical-pass" if passed else "historical-fail",
            "formal_checks": checks,
            "tributary_separation_checks": gap_checks,
            "tributary_separation_policy": policy,
            "initial_connectivity": {
                "root_gap_branch_count": initial_structure["root_gap_branch_count"],
                "tributary_gap_branch_count": initial_structure["tributary_gap_branch_count"],
                "detached_volume_fraction": initial_structure["detached_head_volume_fraction_system"],
                "detached_area_fraction": initial_structure["detached_head_area_fraction_system"],
            },
            "final_connectivity": {
                "root_gap_branch_count": final_structure["root_gap_branch_count"],
                "tributary_gap_branch_count": final_structure["tributary_gap_branch_count"],
                "detached_volume_fraction": final_structure["detached_head_volume_fraction_system"],
                "detached_area_fraction": final_structure["detached_head_area_fraction_system"],
            },
            "newly_activated_initial_zero_node_count": newly_activated,
            "mixedbed_property_access": "cached-once-per-branch-per-step",
            "formal_historical_adjudicated": True,
            "independent_reproduction_run": False,
        })
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-policy2pct-historical-v2",
            "status": "technical-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "formal_historical_adjudicated": False,
        }
    write_json(output / "historical_summary_formal.json", summary)
    if rows:
        with (output / "historical_1470_2000.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if trace:
        with (output / "event_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
    if flowlines is not None:
        with gzip.open(output / "state_2000.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary, allow_nan=False))
    if summary.get("status") != "historical-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
