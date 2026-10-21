"""Independently replay and compare the Policy-V6 year-9 natural state."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import time
import traceback

import numpy as np

from . import diagnose_system2005_internal_gap_ledger_v2 as ledger
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v8 as solver_v8
from .materialize_system2005_policy_v6_year9_v1 import bitwise_reload_equal, sha256, write_json


ROOT = __import__("pathlib").Path(__file__).resolve().parents[4]
EXPECTED_STATE_ID = "2026-08-26-v10.43-year9-reproduction-authorized"


def controller_gate() -> dict:
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "state_id": state.get("state_id") == EXPECTED_STATE_ID,
        "route": state.get("authoritative_route", {}).get("version") == "v10.43",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type") == "system2005-year9-independent-reproduction",
        "full_historical_not_authorized": state.get("current_focus", {}).get("full_historical_authorized") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    auth = config.get("authorization", {})
    if auth.get("independent_reproduction") is not True or any(
        auth.get(key) for key in ("natural_year9_materialization", "formal_50y_steady", "historical")
    ):
        raise RuntimeError("out-of-scope-authorization")
    if float(config["years"]) != 9.0 or float(config["maximum_timestep_seconds"]) != 21600.0:
        raise RuntimeError("frozen-duration-or-timestep-mismatch")
    controller_checks = controller_gate()
    reference_summary_path = base.resolve(config["reference_materialization_summary"])
    reference_state_path = base.resolve(config["reference_materialized_state"])
    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    if reference_summary.get("status") != "year9-materialization-pass-reproduction-pending":
        raise RuntimeError("reference-materialization-not-pass")
    if sha256(reference_state_path) != reference_summary.get("state_sha256"):
        raise RuntimeError("reference-state-hash-mismatch")

    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config_snapshot.json", config)
    write_json(output / "controller_gate.json", controller_checks)
    started = time.perf_counter()
    try:
        solver_v8.reset_structure_cache()
        ledger.solver = solver_v8
        raw, annual_rows, trace_rows, flowlines = ledger.run(config, output)
        write_json(output / "raw_run_summary.json", raw)
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(annual_rows[0]))
            writer.writeheader(); writer.writerows(annual_rows)
        if trace_rows:
            with (output / "event_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
                writer.writeheader(); writer.writerows(trace_rows)

        state_path = output / "lia_natural_state_year9_reproduction.pkl.gz"
        with gzip.open(state_path, "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(state_path, "rb") as stream:
            recovered = pickle.load(stream)
        roundtrip_equal, roundtrip_max = bitwise_reload_equal(flowlines, recovered)
        with gzip.open(reference_state_path, "rb") as stream:
            reference_flowlines = pickle.load(stream)
        independent_equal, independent_max = bitwise_reload_equal(reference_flowlines, flowlines)

        observed = {
            "volume_m3": float(annual_rows[-1]["volume_m3"]),
            "area_m2": float(annual_rows[-1]["area_m2"]),
            "length_m": float(annual_rows[-1]["length_m"]),
            "internal_gap_branch_count": int(annual_rows[-1]["internal_gap_branch_count"]),
        }
        expected = reference_summary["observed"]
        metric_equal = {key: observed[key] == expected[key] for key in observed}
        raw_fields = (
            "step_count", "maximum_absolute_step_closure_m3",
            "maximum_absolute_KKT_stationarity", "maximum_active_constraint_count",
            "initial_volume_m3", "final_volume_m3", "initial_area_m2", "final_area_m2",
            "cumulative_accepted_mb_m3", "cumulative_prevented_overmelt_m3",
        )
        reference_raw = json.loads((reference_summary_path.parent / "raw_run_summary.json").read_text(encoding="utf-8"))
        diagnostic_equal = {key: raw.get(key) == reference_raw.get(key) for key in raw_fields}
        checks = {
            "raw_core_status_pass": raw.get("status") == "full-50y-ledger-complete-core-gates-pass",
            "completed_exactly_9y": float(raw.get("years", -1)) == 9.0,
            "four_registered_metrics_exact": all(metric_equal.values()),
            "five_flowline_array_families_bitwise_equal": independent_equal,
            "registered_diagnostics_exact": all(diagnostic_equal.values()),
            "reproduction_state_roundtrip_bitwise": roundtrip_equal,
            "connected_length_passes_policy_v6": observed["length_m"] >= float(config["gates"]["minimum_connected_root_length_m"]),
            "no_internal_gap": observed["internal_gap_branch_count"] == 0,
        }
        status = "year9-independent-reproduction-pass" if all(checks.values()) else "year9-independent-reproduction-fail"
        summary = {
            "schema": "lia-191-system2005-policy-v6-year9-reproduction-v1",
            "status": status,
            "system_id": 2005,
            "observed": observed,
            "metric_bitwise_equality": metric_equal,
            "diagnostic_equality": diagnostic_equal,
            "checks": checks,
            "independent_array_maximum_absolute_differences": independent_max,
            "roundtrip_array_maximum_absolute_differences": roundtrip_max,
            "reference_state_sha256": sha256(reference_state_path),
            "reproduction_state_sha256": sha256(state_path),
            "runtime_seconds": time.perf_counter() - started,
            "formal_50y_steady_run": False,
            "historical_run": False,
        }
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-policy-v6-year9-reproduction-v1",
            "status": "technical-fail",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
            "formal_50y_steady_run": False,
            "historical_run": False,
        }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary.get("status") == "year9-independent-reproduction-pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
