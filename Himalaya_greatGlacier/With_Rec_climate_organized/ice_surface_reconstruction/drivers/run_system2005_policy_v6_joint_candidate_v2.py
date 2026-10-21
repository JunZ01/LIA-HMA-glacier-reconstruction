"""V2 staged candidate using a uniform lift on formally active thickness."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from .materialize_system2005_policy_v6_year9_v1 import bitwise_reload_equal, sha256, write_json
from .run_system2005_policy_v6_joint_candidate_v1 import dynamic_run, hypsometry_cdf, reverse_counts


ROOT = Path(__file__).resolve().parents[4]


def controller_gate():
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "state_id": state.get("state_id") == "2026-08-26-v10.45-joint-candidate-v2-authorized",
        "route": state.get("authoritative_route", {}).get("version") == "v10.45",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type") == "system2005-joint-candidate-v2-static-preflight-formal50",
        "full_historical_not_authorized": state.get("current_focus", {}).get("full_historical_authorized") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


def build_candidate(source, config):
    candidate = copy.deepcopy(source)
    lift = float(config["uniform_active_thickness_lift_m"])
    tolerance = float(config["active_thickness_tolerance_m"])
    for fl in candidate:
        thick = np.asarray(fl.thick, dtype=float).copy()
        active = thick > tolerance
        thick[active] += lift
        fl.thick = thick
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    auth = config.get("authorization", {})
    if not (auth.get("static_candidate_construction") and auth.get("conditional_10y_preflight")
            and auth.get("conditional_single_formal_50y")):
        raise RuntimeError("staged candidate not authorized")
    if auth.get("historical") or int(auth.get("candidate_count_max", 0)) != 1:
        raise RuntimeError("out-of-scope authorization")
    control = controller_gate()
    source_summary_path = base.resolve(config["source_materialization_summary"])
    source_state_path = base.resolve(config["source_materialized_state"])
    repro_path = base.resolve(config["source_reproduction_summary"])
    static_evidence_path = base.resolve(config["static_feasibility"])
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    reproduction = json.loads(repro_path.read_text(encoding="utf-8"))
    static_evidence = json.loads(static_evidence_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "year9-materialization-pass-reproduction-pending":
        raise RuntimeError("source materialization not pass")
    if reproduction.get("status") != "year9-independent-reproduction-pass":
        raise RuntimeError("reproduction not pass")
    if static_evidence.get("status") != "static-feasible":
        raise RuntimeError("static feasibility not pass")
    if sha256(source_state_path) != source_summary["state_sha256"]:
        raise RuntimeError("source state hash mismatch")

    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config_snapshot.json", config)
    write_json(output / "controller_gate.json", control)
    started = time.perf_counter()
    try:
        with gzip.open(source_state_path, "rb") as stream:
            source = pickle.load(stream)
        candidate = build_candidate(source, config)
        tolerance = float(config["active_thickness_tolerance_m"])
        source_volume = base.total_volume(source)
        volume = base.total_volume(candidate)
        source_area = climate.ice_area(source)
        area = climate.ice_area(candidate)
        source_root = climate.root_metrics(source, float(config["registered_lia_length_m"]))
        root = climate.root_metrics(candidate, float(config["registered_lia_length_m"]))
        source_support = topology.support_state(source, tolerance)
        support = topology.support_state(candidate, tolerance)
        source_reverse = reverse_counts(source)
        reverse = reverse_counts(candidate)
        cdf = hypsometry_cdf(source, candidate)
        static_checks = {
            "volume_scale_exact": abs(volume / source_volume - float(config["target_volume_scale"])) <= float(config["gates"]["static_relative_tolerance"]),
            "area_change": abs(area / source_area - 1.0) <= float(config["gates"]["static_area_change_fraction_max"]),
            "support_unchanged": support == source_support,
            "connected_length_unchanged": root["length_m"] == source_root["length_m"],
            "free_domain_unchanged": root["free_downstream_m"] == source_root["free_downstream_m"],
            "within_reverse_unchanged": reverse["within"] == source_reverse["within"],
            "junction_reverse_unchanged": reverse["junction"] == source_reverse["junction"],
            "hypsometry_cdf": cdf <= float(config["gates"]["static_hypsometry_cdf_max"]),
        }
        static = {
            "schema": "lia-191-system2005-policy-v6-joint-candidate-v2-static",
            "status": "static-pass" if all(static_checks.values()) else "static-fail",
            "checks": static_checks,
            "uniform_thickness_lift_m": config["uniform_active_thickness_lift_m"],
            "volume_scale": volume / source_volume,
            "area_change_fraction": area / source_area - 1.0,
            "source_root": source_root, "candidate_root": root,
            "source_reverse": source_reverse, "candidate_reverse": reverse,
            "hypsometry_cdf_maximum": cdf,
        }
        write_json(output / "static_summary.json", static)
        if not all(static_checks.values()):
            summary = {"schema": "lia-191-system2005-policy-v6-joint-candidate-v2",
                       "status": "static-fail-mechanism-closed", "static": static,
                       "preflight_run": False, "formal_50y_run": False, "historical_run": False}
        else:
            candidate_path = output / "joint_candidate_initial_state.pkl.gz"
            with gzip.open(candidate_path, "wb") as stream:
                pickle.dump(candidate, stream, protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open(candidate_path, "rb") as stream:
                recovered = pickle.load(stream)
            roundtrip, maxima = bitwise_reload_equal(candidate, recovered)
            if not roundtrip:
                raise RuntimeError(f"candidate roundtrip fail:{maxima}")
            pre_dir = output / "preflight10"; pre_dir.mkdir()
            pre_raw, pre_rows, _ = dynamic_run(config, candidate, float(config["preflight_years"]), pre_dir)
            pre_final = pre_rows[-1]
            pre_dv = float(pre_raw["volume_change_fraction"])
            pre_checks = {
                "raw_core_status_pass": pre_raw.get("status") == "full-50y-ledger-complete-core-gates-pass",
                "volume_change": abs(pre_dv) <= float(config["gates"]["preflight_volume_change_fraction_max"]),
                "connected_length": float(pre_final["length_m"]) >= float(config["gates"]["preflight_connected_root_length_min_m"]),
                "free_domain": float(pre_final["free_downstream_m"]) >= float(config["gates"]["minimum_free_downstream_m"]),
                "finite_nonnegative": pre_raw["core_checks_excluding_gap_policy"]["finite_nonnegative"],
            }
            preflight = {
                "schema": "lia-191-system2005-policy-v6-joint-candidate-v2-preflight10",
                "status": "preflight-pass" if all(pre_checks.values()) else "preflight-fail",
                "volume_change_fraction": pre_dv,
                "final_length_m": float(pre_final["length_m"]),
                "final_area_m2": float(pre_final["area_m2"]),
                "final_free_downstream_m": float(pre_final["free_downstream_m"]),
                "final_internal_gap_branch_count": int(pre_final["internal_gap_branch_count"]),
                "step_count": pre_raw["step_count"], "checks": pre_checks,
            }
            write_json(output / "preflight10_summary.json", preflight)
            if not all(pre_checks.values()):
                summary = {"schema": "lia-191-system2005-policy-v6-joint-candidate-v2",
                           "status": "preflight-fail-mechanism-closed", "static": static,
                           "preflight": preflight, "formal_50y_run": False, "historical_run": False}
            else:
                formal_dir = output / "formal50"; formal_dir.mkdir()
                formal_raw, formal_rows, formal_flowlines = dynamic_run(config, candidate, float(config["formal_years"]), formal_dir)
                final = formal_rows[-1]
                dv = float(formal_raw["volume_change_fraction"])
                formal_checks = {
                    "completed_50y": float(formal_raw["years"]) == 50.0,
                    "raw_core_status_pass": formal_raw.get("status") == "full-50y-ledger-complete-core-gates-pass",
                    "volume_change": abs(dv) <= float(config["gates"]["formal_volume_change_fraction_max"]),
                    "connected_length": float(final["length_m"]) >= float(config["gates"]["formal_connected_root_length_min_m"]),
                    "total_length_relative_error": float(final["length_relative_error"]) <= float(config["gates"]["lia_length_relative_error_max"]),
                    "free_domain": float(final["free_downstream_m"]) >= float(config["gates"]["minimum_free_downstream_m"]),
                    "terminal_thickness": float(final["terminal_thickness_m"]) <= 10.0,
                    "finite_nonnegative": formal_raw["core_checks_excluding_gap_policy"]["finite_nonnegative"],
                }
                state_path = formal_dir / "lia_equilibrium_state_50y.pkl.gz"
                if all(formal_checks.values()):
                    with gzip.open(state_path, "wb") as stream:
                        pickle.dump(formal_flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
                formal = {
                    "schema": "lia-191-system2005-policy-v6-joint-candidate-v2-formal50",
                    "status": "formal-50y-steady-pass" if all(formal_checks.values()) else "formal-50y-steady-fail",
                    "volume_change_fraction": dv,
                    "final_length_m": float(final["length_m"]),
                    "final_area_m2": float(final["area_m2"]),
                    "final_free_downstream_m": float(final["free_downstream_m"]),
                    "final_internal_gap_branch_count": int(final["internal_gap_branch_count"]),
                    "step_count": formal_raw["step_count"], "checks": formal_checks,
                    "state_saved": state_path.exists(),
                    "state_sha256": sha256(state_path) if state_path.exists() else None,
                }
                write_json(output / "formal50_summary.json", formal)
                summary = {"schema": "lia-191-system2005-policy-v6-joint-candidate-v2",
                           "status": "formal-50y-pass-historical-pending" if all(formal_checks.values()) else "formal-50y-fail-mechanism-closed",
                           "static": static, "preflight": preflight, "formal": formal,
                           "historical_run": False}
        summary["runtime_seconds"] = time.perf_counter() - started
        summary["source_state_sha256"] = sha256(source_state_path)
        summary["static_feasibility_sha256"] = sha256(static_evidence_path)
    except Exception as exc:
        summary = {"schema": "lia-191-system2005-policy-v6-joint-candidate-v2",
                   "status": "technical-fail", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(),
                   "runtime_seconds": time.perf_counter() - started, "historical_run": False}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["status"] == "technical-fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
