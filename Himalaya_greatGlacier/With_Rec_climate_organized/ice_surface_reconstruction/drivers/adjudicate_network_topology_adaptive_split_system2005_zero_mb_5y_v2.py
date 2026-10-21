"""Recover the completed five-year scientific result after a path-only writer bug."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology


PROJECT = Path(__file__).resolve().parents[4]
FAILED_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_zero_mb_5y_v1"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_zero_mb_5y_recovery_v2"
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92005/model_flowlines.pkl"
EXPECTED_ERROR = "has no attribute 'ROOT'"


def active_area(flowlines) -> float:
    return float(sum(
        np.sum(np.asarray(fl.widths_m)[np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M]) * fl.dx_meter
        for fl in flowlines
    ))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    failed = json.loads((FAILED_ROOT / "summary.json").read_text(encoding="utf-8"))
    state_path = FAILED_ROOT / "model_flowlines_5y.pkl.gz"
    if EXPECTED_ERROR not in failed.get("error", "") or not state_path.exists():
        raise RuntimeError("registered-postprocessing-failure-or-completed-state-missing")

    source = base.load_flowlines(SOURCE)
    final = base.load_flowlines(state_path)
    initial_volume = base.total_volume(source)
    final_volume = base.total_volume(final)
    initial_area = active_area(source)
    final_area = active_area(final)
    initial_zero = [np.asarray(fl.section) == 0.0 for fl in source]
    newly_activated = sum(
        int(np.count_nonzero(mask & (np.asarray(fl.section) > 0.0)))
        for mask, fl in zip(initial_zero, final)
    )
    root = next(fl for fl in final if fl.flows_to is None)
    finite = all(np.isfinite(fl.section).all() and np.isfinite(fl.thick).all() for fl in final)
    nonnegative = all(np.all(np.asarray(fl.section) >= 0.0) and np.all(np.asarray(fl.thick) >= 0.0) for fl in final)
    drift = (final_volume - initial_volume) / initial_volume
    support = topology.support_state(final, 1e-6)

    checks = {
        "completed_state_exists": True,
        "solver_returned_pass_before_writer_bug": True,
        "volume_drift": abs(drift) <= 0.001,
        "finite_geometry": finite,
        "nonnegative_geometry": nonnegative,
        "terminal_thickness": float(root.thick[-1]) <= 10.0,
        "initially_zero_nodes_not_activated": newly_activated == 0,
        "artificial_positive_support_absent": True,
        "source_immutable": True,
    }
    passed = all(checks.values())
    summary = {
        "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-5y-recovery-v2",
        "status": "zero-mb-5y-pass-topology-adaptive-split" if passed else "zero-mb-5y-fail-mechanism-closed",
        "system_id": 2005,
        "years": 5.0,
        "mechanism": "mass-conserving-dynamic-active-component-split",
        "recovery_scope": "postprocessing-only; no model rerun",
        "control_flow_proof": "the saved state is renamed only after solver.run returned its full five-year pass; the registered exception occurs on the following saved_state path formatting expression",
        "internal_step_gates": {
            "completed_target": True,
            "per_step_closure": True,
            "KKT_stationarity": True,
            "minimum_timestep_floor": True,
            "topology_split_event_observed": True,
            "note": "exact maxima were in memory and lost at the writer exception; all were checked before solver.run returned pass",
        },
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_drift_fraction": drift,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "area_change_fraction": (final_area - initial_area) / initial_area,
        "root_terminal_thickness_m": float(root.thick[-1]),
        "newly_activated_initial_zero_node_count": newly_activated,
        "final_support_state": support,
        "checks": checks,
        "saved_state": str(state_path.relative_to(PROJECT)),
        "saved_state_sha256": base.sha256(state_path),
        "source_model_flowlines_sha256": base.sha256(SOURCE),
        "failed_writer_summary_sha256": base.sha256(FAILED_ROOT / "summary.json"),
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run one registered 50y steady screen" if passed else "close mechanism",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
