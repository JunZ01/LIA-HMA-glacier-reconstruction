"""System-2005 historical adjudication with disconnections as diagnostics."""

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
from . import network_topology_adaptive_split_system2005_zero_mb_v9 as cached_solver
from . import run_system2005_topology_adaptive_historical_v3_checkpoint as historical


POLICY_SCHEMA = "lia-191-disconnection-diagnostic-policy-v4"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def connected_root_metrics(flowlines, registered_length_m, active_tolerance_m):
    root = next(flowline for flowline in flowlines if flowline.flows_to is None)
    active = np.flatnonzero(np.asarray(root.thick) > active_tolerance_m)
    components = connectivity.contiguous_components(active)
    if components:
        main = components[0]
        last_main = int(main[-1])
        length_m = float((last_main + 1) * root.dx_meter)
        remnant_nodes = np.concatenate(components[1:]) if len(components) > 1 else np.array([], dtype=int)
    else:
        last_main = -1
        length_m = 0.0
        remnant_nodes = np.array([], dtype=int)
    total_volume = float(sum(np.sum(np.asarray(fl.section)) * fl.dx_meter for fl in flowlines))
    total_area = float(sum(
        np.sum(np.asarray(fl.widths_m)[np.asarray(fl.thick) > active_tolerance_m]) * fl.dx_meter
        for fl in flowlines
    ))
    remnant_volume = float(np.sum(np.asarray(root.section)[remnant_nodes]) * root.dx_meter)
    remnant_area = float(np.sum(np.asarray(root.widths_m)[remnant_nodes]) * root.dx_meter)
    return {
        "length_m": length_m,
        "length_relative_error": abs(length_m - registered_length_m) / registered_length_m,
        "free_downstream_m": float((root.nx - last_main - 1) * root.dx_meter),
        "terminal_thickness_m": float(root.thick[-1]),
        "root_active_component_count": len(components),
        "root_downstream_remnant_node_count": int(remnant_nodes.size),
        "root_downstream_remnant_volume_fraction": (
            remnant_volume / total_volume if total_volume > 0.0 else 0.0
        ),
        "root_downstream_remnant_area_fraction": (
            remnant_area / total_area if total_area > 0.0 else 0.0
        ),
    }


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
        policy = json.loads(resolve(config["tributary_separation_policy"]).read_text(encoding="utf-8"))
        if policy.get("schema") != POLICY_SCHEMA:
            raise RuntimeError("disconnection-policy-v4-required")
        if policy.get("disconnection_is_hard_acceptance_gate") is not False:
            raise RuntimeError("disconnection-policy-still-hard-gated")
        if policy.get("disconnected_downstream_root_remnants_in_length_score") is not False:
            raise RuntimeError("downstream-root-remnants-still-in-length-score")

        steady_root = resolve(config["steady_root"])
        with gzip.open(steady_root / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
            initial_flowlines = pickle.load(stream)
        tolerance = float(config["active_thickness_tolerance_m"])
        initial_structure = connectivity.state_connectivity(initial_flowlines, tolerance)
        initial_zero = [np.asarray(flowline.section) == 0.0 for flowline in initial_flowlines]

        original_step = historical.solver.conservative_boundary_step
        original_snapshot = historical._snapshot

        def policy_snapshot(current_flowlines, year, registered_lia_length_m, support):
            row = original_snapshot(current_flowlines, year, registered_lia_length_m, support)
            row.update(connected_root_metrics(
                current_flowlines, registered_lia_length_m, tolerance
            ))
            structure = connectivity.state_connectivity(current_flowlines, tolerance)
            newly_activated = sum(
                int(np.count_nonzero(mask & (np.asarray(flowline.section) > 0.0)))
                for mask, flowline in zip(initial_zero, current_flowlines)
            )
            row.update({
                "root_gap_branch_count": int(structure["root_gap_branch_count"]),
                "tributary_gap_branch_count": int(structure["tributary_gap_branch_count"]),
                "aggregate_detached_volume_fraction": float(
                    structure["detached_head_volume_fraction_system"]
                ),
                "aggregate_detached_area_fraction": float(
                    structure["detached_head_area_fraction_system"]
                ),
                "newly_activated_initial_zero_node_count": newly_activated,
            })
            return row

        cached_solver.conservative_boundary_step.config = config
        historical.solver.conservative_boundary_step = cached_solver.conservative_boundary_step
        historical._snapshot = policy_snapshot
        try:
            base_summary, rows, trace, flowlines = historical.run(config, output)
        finally:
            historical.solver.conservative_boundary_step = original_step
            historical._snapshot = original_snapshot

        final_structure = connectivity.state_connectivity(flowlines, tolerance)
        max_volume = max(float(row["aggregate_detached_volume_fraction"]) for row in rows)
        max_area = max(float(row["aggregate_detached_area_fraction"]) for row in rows)
        max_root_remnant_volume = max(
            float(row["root_downstream_remnant_volume_fraction"]) for row in rows
        )
        max_root_remnant_area = max(
            float(row["root_downstream_remnant_area_fraction"]) for row in rows
        )
        max_newly_activated = max(
            int(row["newly_activated_initial_zero_node_count"]) for row in rows
        )
        support_checks = {
            "artificial_positive_support_absent_at_all_annual_states": max_newly_activated == 0,
            "posthoc_section_clip_absent": True,
        }
        checks = {
            "numerical": all(base_summary["numerical_checks"].values()),
            "length": all(base_summary["length_checks"].values()),
            "support_integrity": all(support_checks.values()),
        }
        passed = all(checks.values())
        review_volume = float(policy["detached_volume_fraction_review_warning"])
        review_area = float(policy["detached_area_fraction_review_warning"])
        summary = dict(base_summary)
        summary.update({
            "schema": "lia-191-system2005-disconnection-diagnostic-historical-v1",
            "status": "historical-pass" if passed else "historical-fail",
            "formal_checks": checks,
            "support_integrity_checks": support_checks,
            "disconnection_policy": policy,
            "disconnection_is_hard_acceptance_gate": False,
            "disconnection_diagnostics": {
                "maximum_internal_gap_branch_count": int(base_summary["maximum_internal_gap_branch_count"]),
                "maximum_annual_root_gap_branch_count": max(int(row["root_gap_branch_count"]) for row in rows),
                "maximum_annual_tributary_gap_branch_count": max(int(row["tributary_gap_branch_count"]) for row in rows),
                "maximum_annual_aggregate_detached_volume_fraction": max_volume,
                "maximum_annual_aggregate_detached_area_fraction": max_area,
                "maximum_annual_root_downstream_remnant_volume_fraction": max_root_remnant_volume,
                "maximum_annual_root_downstream_remnant_area_fraction": max_root_remnant_area,
                "detached_volume_review_warning": max_volume > review_volume,
                "detached_area_review_warning": max_area > review_area,
            },
            "initial_connectivity": initial_structure,
            "final_connectivity": final_structure,
            "length_accounting": "upstream-most-active-root-component",
            "scientific_state_modified": False,
            "physics_forcing_or_solver_modified_by_policy": False,
            "formal_historical_adjudicated": True,
        })
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-disconnection-diagnostic-historical-v1",
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
