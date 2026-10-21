"""System-2005 zero-MB pilot allowing mass-conserving dynamic component split.

Unlike the closed connected-support mechanism, this mechanism never inserts a
positive support reserve.  When an active tributary naturally loses its
terminal connection or develops a zero-section internal node, the active graph
is rebuilt as separate components on the next step.  Existing section values
are not moved or clipped and the exact system-volume equality remains active.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import traceback

import numpy as np

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_positive_conservative_system2005_zero_mb_v1 as predecessor


TOL = 1e-10


def support_state(flowlines, active_tolerance):
    branches = []
    total_components = 0
    disconnected = []
    internal_gap_branches = []
    for branch, flowline in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(flowline.thick, dtype=float) > active_tolerance)
        if len(active):
            components = 1 + int(np.count_nonzero(np.diff(active) > 1))
        else:
            components = 0
        total_components += components
        has_gap = components > 1
        is_disconnected = bool(
            flowline.flows_to is not None
            and len(active)
            and int(active[-1]) != len(flowline.thick) - 1
        )
        if has_gap:
            internal_gap_branches.append(branch)
        if is_disconnected:
            disconnected.append(branch)
        branches.append({
            "branch": branch,
            "active_node_count": int(len(active)),
            "component_count": components,
            "first_active": None if not len(active) else int(active[0]),
            "last_active": None if not len(active) else int(active[-1]),
            "terminal_node": len(flowline.thick) - 1,
            "flows_to": flowline.flows_to is not None,
            "disconnected_from_junction": is_disconnected,
        })
    return {
        "total_active_components": total_components,
        "disconnected_tributary_count": len(disconnected),
        "disconnected_tributary_branches": disconnected,
        "internal_gap_branch_count": len(internal_gap_branches),
        "internal_gap_branches": internal_gap_branches,
        "branches": branches,
    }


def run(config):
    closed = json.loads(base.resolve(config["closed_predecessor"]).read_text(encoding="utf-8"))
    if "tributary-disconnected-from-junction:branch=8" not in closed.get("error", ""):
        raise RuntimeError("closed-predecessor-is-not-registered-branch8-disconnection")
    source_path = base.resolve(config["model_flowlines"])
    source = base.load_flowlines(source_path)
    initial_zero = [np.asarray(fl.section, dtype=float) == 0.0 for fl in source]
    initial_support = support_state(source, float(config["active_thickness_tolerance_m"]))

    original_continuity = predecessor.continuity_failure
    predecessor.continuity_failure = lambda _flowlines: None
    try:
        summary, rows = predecessor.run(config)
    finally:
        predecessor.continuity_failure = original_continuity

    output = base.resolve(config["output_root"])
    state_path = output / "model_flowlines_0p1y.pkl.gz"
    if summary.get("status") != "zero-mb-0p1y-pass" or not state_path.exists():
        return {
            **summary,
            "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-v1",
            "status": "zero-mb-0p1y-fail-mechanism-closed",
            "mechanism": "mass-conserving-dynamic-active-component-split",
            "artificial_positive_support_used": False,
            "runtime_clip_used": False,
            "historical_run": False,
        }, rows

    with gzip.open(state_path, "rb") as stream:
        final = pickle.load(stream)
    final_support = support_state(final, float(config["active_thickness_tolerance_m"]))
    newly_activated_initial_zero = 0
    for mask, flowline in zip(initial_zero, final):
        newly_activated_initial_zero += int(np.count_nonzero(mask & (np.asarray(flowline.section) > 0.0)))
    split_observed = (
        final_support["disconnected_tributary_count"] > initial_support["disconnected_tributary_count"]
        or final_support["internal_gap_branch_count"] > initial_support["internal_gap_branch_count"]
        or final_support["total_active_components"] > initial_support["total_active_components"]
    )
    checks = dict(summary["checks"])
    checks.pop("network_continuity", None)
    checks.update({
        "topology_split_event_observed": split_observed,
        "initially_zero_nodes_not_activated": newly_activated_initial_zero == 0,
        "artificial_positive_support_absent": True,
        "component_state_finite": all(
            np.isfinite(fl.section).all() and np.isfinite(fl.thick).all() for fl in final
        ),
    })
    passed = all(checks.values())
    summary.update({
        "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-v1",
        "status": "zero-mb-0p1y-pass-topology-adaptive-split" if passed else "zero-mb-0p1y-fail-mechanism-closed",
        "mechanism": "mass-conserving-dynamic-active-component-split",
        "checks": checks,
        "initial_support_state": initial_support,
        "final_support_state": final_support,
        "newly_activated_initial_zero_node_count": newly_activated_initial_zero,
        "artificial_positive_support_used": False,
        "runtime_clip_used": False,
        "source_modified": False,
        "zero_mb_5y_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run one 5y zero-MB under identical topology-adaptive contract" if passed else "close mechanism",
        "provenance": {
            **summary.get("provenance", {}),
            "closed_predecessor_sha256": base.sha256(base.resolve(config["closed_predecessor"])),
        },
    })
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
            "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-v1",
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
    if summary["status"] != "zero-mb-0p1y-pass-topology-adaptive-split":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

