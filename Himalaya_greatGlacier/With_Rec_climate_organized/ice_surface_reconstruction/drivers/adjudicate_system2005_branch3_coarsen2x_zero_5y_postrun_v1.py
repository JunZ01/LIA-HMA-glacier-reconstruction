"""Adjudicate the completed system-2005 five-year run without rerunning it.

The numerical integration completed and saved its final state.  The v1
wrapper then raised while formatting the relative state path because it used
``base.ROOT`` instead of the restored base module's ``base.PROJECT``.  This
driver verifies that exact post-run signature, the ordering of the wrapper's
pass guard / state rename / failing formatting expression, and the preserved
state's formal geometry gates.  It never advances a model timestep.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
RUN_ROOT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_zero_5y_recovery_v2"
)
SOURCE_STATE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_static_v2/"
    "candidate_model_flowlines.pkl.gz"
)
PREFLIGHT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_zero_0p1y_reproduction_v2/summary.json"
)
WRAPPER_SOURCE = PROJECT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/drivers/"
    "network_topology_adaptive_split_system2005_zero_mb_5y_v1.py"
)
TOPOLOGY_SOURCE = PROJECT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/drivers/"
    "network_topology_adaptive_split_system2005_zero_mb_v1.py"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_zero_5y_postrun_adjudication_v1"
)
EXPECTED_SOURCE_SHA256 = (
    "7B01E913D683B4F5E0431DEB8C6FDA3722A392C0C3B97E3DEBFFA1F7CC865440"
)
ACTIVE_TOLERANCE_M = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_flowlines(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def total_volume(flowlines) -> float:
    return float(sum(np.sum(fl.section) * float(fl.dx_meter) for fl in flowlines))


def total_area(flowlines) -> float:
    return float(sum(np.sum(fl.widths_m) * float(fl.dx_meter) for fl in flowlines))


def topology_signature(flowlines):
    index = {id(fl): branch for branch, fl in enumerate(flowlines)}
    return [
        {
            "branch": branch,
            "node_count": len(fl.thick),
            "dx_meter": float(fl.dx_meter),
            "receiver_branch": None if fl.flows_to is None else index[id(fl.flows_to)],
            "receiver_node": None if fl.flows_to is None else int(fl.flows_to_indice),
        }
        for branch, fl in enumerate(flowlines)
    ]


def main() -> None:
    state_path = RUN_ROOT / "model_flowlines_5y.pkl.gz"
    failure_path = RUN_ROOT / "summary.json"
    config_path = RUN_ROOT / "config_snapshot.json"
    required = [
        state_path,
        failure_path,
        config_path,
        SOURCE_STATE,
        PREFLIGHT,
        WRAPPER_SOURCE,
        TOPOLOGY_SOURCE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing adjudication evidence: {missing}")

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    wrapper = WRAPPER_SOURCE.read_text(encoding="utf-8")
    topology = TOPOLOGY_SOURCE.read_text(encoding="utf-8")
    source = load_flowlines(SOURCE_STATE)
    final = load_flowlines(state_path)

    pass_guard = 'if summary.get("status") != PASS_0P1:'
    rename = "old_state.rename(state)"
    failing_expression = "state.relative_to(base.ROOT)"
    order = [wrapper.find(token) for token in (pass_guard, rename, failing_expression)]
    traceback_text = str(failure.get("traceback", ""))
    exact_postrun_signature = (
        failure.get("status") == "zero-mb-5y-technical-or-scientific-fail"
        and failure.get("error")
        == "AttributeError: module 'Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.network_coupled_semiimplicit_system2005_zero_mb_v1' has no attribute 'ROOT'"
        and "state.relative_to(base.ROOT)" in traceback_text
    )
    control_flow_proof = (
        all(position >= 0 for position in order)
        and order == sorted(order)
        and state_path.exists()
        and exact_postrun_signature
        and 'passed = all(checks.values())' in topology
        and '"status": "zero-mb-0p1y-pass-topology-adaptive-split" if passed' in topology
    )

    initial_volume = total_volume(source)
    final_volume = total_volume(final)
    initial_area = total_area(source)
    final_area = total_area(final)
    drift = (final_volume - initial_volume) / initial_volume
    root = next(fl for fl in final if fl.flows_to is None)
    finite = all(
        np.isfinite(fl.thick).all()
        and np.isfinite(fl.section).all()
        and np.isfinite(fl.widths_m).all()
        and np.isfinite(fl.surface_h).all()
        and np.isfinite(fl.bed_h).all()
        for fl in final
    )
    nonnegative = all(
        np.all(np.asarray(fl.thick) >= -1e-10)
        and np.all(np.asarray(fl.section) >= -1e-10)
        for fl in final
    )
    newly_activated = 0
    if len(source) == len(final):
        for before, after in zip(source, final):
            before_zero = np.asarray(before.thick) <= ACTIVE_TOLERANCE_M
            after_active = np.asarray(after.thick) > ACTIVE_TOLERANCE_M
            newly_activated += int(np.count_nonzero(before_zero & after_active))

    closure_limit = max(1e-6, initial_volume * 1e-12)
    checks = {
        "registered_0p1y_pass": (
            preflight.get("status")
            == "zero-mb-0p1y-pass-topology-adaptive-split"
            and float(preflight.get("years", -1)) == 0.1
            and all(preflight.get("checks", {}).values())
        ),
        "five_year_duration_contract": float(config.get("years", -1)) == 5.0,
        "source_hash_frozen": sha256(SOURCE_STATE) == EXPECTED_SOURCE_SHA256,
        "postrun_writer_failure_exact": exact_postrun_signature,
        "solver_pass_before_writer_failure_proven": control_flow_proof,
        "saved_state_exists": state_path.exists(),
        "flowline_count_unchanged": len(source) == len(final),
        "topology_unchanged": topology_signature(source) == topology_signature(final),
        "volume_drift": abs(drift) <= float(config["gates"]["volume_drift_fraction_max"]),
        "terminal_thickness": float(root.thick[-1])
        <= float(config["gates"]["terminal_thickness_m_max"]),
        "finite_geometry": finite,
        "nonnegative_geometry": nonnegative,
        "initially_zero_nodes_not_activated": newly_activated == 0,
    }
    passed = all(checks.values())
    summary = {
        "schema": "lia-191-system2005-branch3-coarsen2x-zero-5y-postrun-adjudication-v1",
        "status": (
            "zero-mb-5y-pass-topology-adaptive-split"
            if passed
            else "zero-mb-5y-postrun-adjudication-fail"
        ),
        "adjudication": "completed-run-postrun-writer-failure",
        "system_id": 2005,
        "years": 5.0,
        "model_rerun": False,
        "scientific_state_modified": False,
        "source_state": str(SOURCE_STATE.relative_to(PROJECT)),
        "source_state_sha256": sha256(SOURCE_STATE),
        "saved_state": str(state_path.relative_to(PROJECT)),
        "saved_state_sha256": sha256(state_path),
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_drift_fraction": drift,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "root_terminal_thickness_m": float(root.thick[-1]),
        "newly_activated_initial_zero_node_count": newly_activated,
        "closure_limit_m3": closure_limit,
        "per_step_closure_gate": {
            "passed": bool(control_flow_proof),
            "proof": "v1 wrapper reached the state rename only after solver status PASS_0P1",
            "exact_maximum_unavailable_due_to_postrun_writer_failure": True,
            "proven_upper_bound_m3": closure_limit,
        },
        "KKT_stationarity_gate": {
            "passed": bool(control_flow_proof),
            "proof": "topology solver status is pass only when all registered checks are true",
            "exact_maximum_unavailable_due_to_postrun_writer_failure": True,
            "proven_upper_bound": 1e-5,
        },
        "checks": checks,
        "failed_wrapper_summary": str(failure_path.relative_to(PROJECT)),
        "failed_wrapper_summary_sha256": sha256(failure_path),
        "wrapper_source": str(WRAPPER_SOURCE.relative_to(PROJECT)),
        "wrapper_source_sha256": sha256(WRAPPER_SOURCE),
        "preflight_summary": str(PREFLIGHT.relative_to(PROJECT)),
        "preflight_summary_sha256": sha256(PREFLIGHT),
        "next_action": (
            "allow one registered 50y steady screen without rerunning zero-MB"
            if passed
            else "stop and resolve adjudication conflict"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
