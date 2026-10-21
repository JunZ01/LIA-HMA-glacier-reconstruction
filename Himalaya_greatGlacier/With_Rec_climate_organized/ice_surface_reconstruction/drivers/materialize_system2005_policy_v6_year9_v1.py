"""Materialize the unique Policy-V6 natural year-9 system-2005 state."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from . import diagnose_system2005_internal_gap_ledger_v2 as ledger
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v8 as solver_v8


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_STATE_ID = "2026-08-26-v10.42-year9-materialization-authorized"
EXPECTED_ROUTE = "v10.42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def bitwise_reload_equal(original, recovered) -> tuple[bool, dict[str, float]]:
    fields = ("section", "thick", "surface_h", "widths_m", "bed_h")
    maxima = {field: 0.0 for field in fields}
    equal = len(original) == len(recovered)
    if not equal:
        return False, maxima
    for left, right in zip(original, recovered):
        for field in fields:
            a = np.asarray(getattr(left, field))
            b = np.asarray(getattr(right, field))
            equal = equal and np.array_equal(a, b)
            if a.size:
                maxima[field] = max(maxima[field], float(np.max(np.abs(a - b))))
    return bool(equal), maxima


def controller_gate() -> dict:
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "state_id": state.get("state_id") == EXPECTED_STATE_ID,
        "route": state.get("authoritative_route", {}).get("version") == EXPECTED_ROUTE,
        "focus": state.get("current_focus", {}).get("system_id") == 2005,
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type") == "system2005-year9-natural-state-materialization",
        "full_historical_not_authorized": state.get("current_focus", {}).get("full_historical_authorized") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = base.resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("authorization", {}).get("natural_year9_materialization") is not True:
        raise RuntimeError("materialization-not-authorized-in-config")
    if any(config.get("authorization", {}).get(key) for key in
           ("independent_reproduction", "formal_50y_steady", "historical")):
        raise RuntimeError("config-authorizes-out-of-scope-run")
    if float(config["years"]) != 9.0 or float(config["maximum_timestep_seconds"]) != 21600.0:
        raise RuntimeError("frozen-duration-or-timestep-mismatch")

    controller_checks = controller_gate()
    audit_path = base.resolve(config["reference_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "year9-unique-natural-candidate-selected":
        raise RuntimeError("reference-audit-not-pass")
    equivalence_path = base.resolve(config["optimized_solver_equivalence"])
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if not all(equivalence.get("checks", {}).values()):
        raise RuntimeError("optimized-solver-equivalence-not-pass")

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
        if annual_rows:
            with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(annual_rows[0]))
                writer.writeheader()
                writer.writerows(annual_rows)
        if trace_rows:
            with (output / "event_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
                writer.writeheader()
                writer.writerows(trace_rows)

        state_path = output / "lia_natural_state_year9.pkl.gz"
        with gzip.open(state_path, "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(state_path, "rb") as stream:
            recovered = pickle.load(stream)
        reload_equal, reload_maxima = bitwise_reload_equal(flowlines, recovered)

        final_row = annual_rows[-1]
        expected = config["expected_year9"]
        observed = {
            "volume_m3": float(final_row["volume_m3"]),
            "area_m2": float(final_row["area_m2"]),
            "length_m": float(final_row["length_m"]),
            "internal_gap_branch_count": int(final_row["internal_gap_branch_count"]),
            "free_downstream_m": float(final_row["free_downstream_m"]),
            "terminal_thickness_m": float(final_row["terminal_thickness_m"]),
        }
        exact = {
            key: observed[key] == expected[key]
            for key in ("volume_m3", "area_m2", "length_m", "internal_gap_branch_count")
        }
        checks = {
            "raw_core_status_pass": raw.get("status") == "full-50y-ledger-complete-core-gates-pass",
            "completed_exactly_9y": float(raw.get("years", -1)) == 9.0,
            "reference_metrics_bitwise_exact": all(exact.values()),
            "connected_length_passes_policy_v6": observed["length_m"] >= float(config["gates"]["minimum_connected_root_length_m"]),
            "no_internal_gap": observed["internal_gap_branch_count"] == 0,
            "terminal_thickness_pass": observed["terminal_thickness_m"] <= 10.0,
            "free_domain_pass": observed["free_downstream_m"] >= float(config["gates"]["minimum_free_downstream_m"]),
            "serialized_state_bitwise_roundtrip": reload_equal,
        }
        status = "year9-materialization-pass-reproduction-pending" if all(checks.values()) else "year9-materialization-fail"
        summary = {
            "schema": "lia-191-system2005-policy-v6-year9-materialization-v1",
            "status": status,
            "system_id": 2005,
            "observed": observed,
            "expected": expected,
            "exact_metric_checks": exact,
            "checks": checks,
            "step_count": raw.get("step_count"),
            "maximum_absolute_step_closure_m3": raw.get("maximum_absolute_step_closure_m3"),
            "maximum_absolute_KKT_stationarity": raw.get("maximum_absolute_KKT_stationarity"),
            "state": str(state_path.relative_to(ROOT)).replace("\\", "/"),
            "state_sha256": sha256(state_path),
            "state_roundtrip_maximum_absolute_differences": reload_maxima,
            "reference_audit_sha256": sha256(audit_path),
            "optimized_solver_equivalence_sha256": sha256(equivalence_path),
            "runtime_seconds": time.perf_counter() - started,
            "formal_50y_steady_run": False,
            "historical_run": False,
        }
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-policy-v6-year9-materialization-v1",
            "status": "technical-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
            "formal_50y_steady_run": False,
            "historical_run": False,
        }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary.get("status") == "year9-materialization-pass-reproduction-pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
