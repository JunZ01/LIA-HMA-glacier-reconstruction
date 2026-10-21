"""Static -> 10y -> one 50y Policy-V6 joint state/temperature candidate."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from . import diagnose_system2005_internal_gap_ledger_v2 as ledger
from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from . import network_topology_adaptive_split_system2005_zero_mb_v8 as solver_v8
from .materialize_system2005_policy_v6_year9_v1 import bitwise_reload_equal, sha256, write_json


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_STATE_ID = "2026-08-26-v10.44-joint-candidate-authorized"


def controller_gate() -> dict:
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "state_id": state.get("state_id") == EXPECTED_STATE_ID,
        "route": state.get("authoritative_route", {}).get("version") == "v10.44",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type") == "system2005-joint-candidate-static-preflight-formal50",
        "full_historical_not_authorized": state.get("current_focus", {}).get("full_historical_authorized") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


def reverse_counts(flowlines, tolerance=1e-8):
    within = 0
    junction = 0
    details = []
    for branch, fl in enumerate(flowlines):
        section = np.asarray(fl.section, dtype=float)
        active = np.flatnonzero(section > 0.0)
        surface = np.asarray(fl.surface_h, dtype=float)
        if active.size > 1:
            consecutive = active[1:] == active[:-1] + 1
            rises = surface[active[1:]] - surface[active[:-1]]
            count = int(np.count_nonzero(consecutive & (rises > tolerance)))
            within += count
            if count:
                details.append({"branch": branch, "type": "within", "count": count,
                                "maximum_rise_m": float(np.max(rises[consecutive]))})
        if fl.flows_to is not None and active.size:
            receiver = flowlines.index(fl.flows_to)
            receiver_index = int(fl.flows_to_indice)
            rise = float(np.asarray(fl.flows_to.surface_h)[receiver_index] - surface[active[-1]])
            if rise > tolerance:
                junction += 1
                details.append({"branch": branch, "receiver": receiver,
                                "receiver_index": receiver_index, "type": "junction",
                                "rise_m": rise})
    return {"within": within, "junction": junction, "details": details}


def hypsometry_cdf(left, right, bin_m=100.0):
    elevations = []
    for flowlines in (left, right):
        values = []
        weights = []
        for fl in flowlines:
            active = np.asarray(fl.section) > 0.0
            values.extend(np.asarray(fl.surface_h)[active].tolist())
            weights.extend((np.asarray(fl.widths_m)[active] * float(fl.dx_meter)).tolist())
        elevations.append((np.asarray(values), np.asarray(weights)))
    low = np.floor(min(np.min(elevations[0][0]), np.min(elevations[1][0])) / bin_m) * bin_m
    high = np.ceil(max(np.max(elevations[0][0]), np.max(elevations[1][0])) / bin_m) * bin_m + bin_m
    edges = np.arange(low, high + 0.5 * bin_m, bin_m)
    cdfs = []
    for values, weights in elevations:
        hist, _ = np.histogram(values, bins=edges, weights=weights)
        cdf = np.cumsum(hist) / np.sum(hist)
        cdfs.append(cdf)
    return float(np.max(np.abs(cdfs[0] - cdfs[1])))


def save_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def dynamic_run(config, candidate, years, output):
    run_config = copy.deepcopy(config)
    run_config["years"] = float(years)
    run_config["gates"] = {
        "volume_change_fraction_max": float(config["gates"]["formal_volume_change_fraction_max"]),
        "lia_length_relative_error_max": float(config["gates"]["lia_length_relative_error_max"]),
        "minimum_free_downstream_m": float(config["gates"]["minimum_free_downstream_m"]),
    }
    original_loader = base.load_flowlines
    base.load_flowlines = lambda _path: copy.deepcopy(candidate)
    ledger.solver = solver_v8
    solver_v8.reset_structure_cache()
    try:
        raw, annual, trace, final = ledger.run(run_config, output)
    finally:
        base.load_flowlines = original_loader
    write_json(output / "raw_run_summary.json", raw)
    save_rows(output / "timeseries.csv", annual)
    save_rows(output / "event_trace.csv", trace)
    return raw, annual, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    auth = config.get("authorization", {})
    if not (auth.get("static_candidate_construction") and auth.get("conditional_10y_preflight")
            and auth.get("conditional_single_formal_50y")):
        raise RuntimeError("staged candidate not fully authorized")
    if auth.get("historical") or int(auth.get("candidate_count_max", 0)) != 1:
        raise RuntimeError("out-of-scope authorization")
    control = controller_gate()
    source_summary_path = base.resolve(config["source_materialization_summary"])
    source_state_path = base.resolve(config["source_materialized_state"])
    repro_path = base.resolve(config["source_reproduction_summary"])
    design_path = base.resolve(config["design_summary"])
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    reproduction = json.loads(repro_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "year9-materialization-pass-reproduction-pending":
        raise RuntimeError("source materialization not pass")
    if reproduction.get("status") != "year9-independent-reproduction-pass":
        raise RuntimeError("source reproduction not pass")
    if design.get("status") != "unique-bounded-joint-candidate-selected":
        raise RuntimeError("joint design not pass")
    if sha256(source_state_path) != source_summary["state_sha256"]:
        raise RuntimeError("source materialized state hash mismatch")

    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config_snapshot.json", config)
    write_json(output / "controller_gate.json", control)
    started = time.perf_counter()
    try:
        with gzip.open(source_state_path, "rb") as stream:
            source = pickle.load(stream)
        candidate = copy.deepcopy(source)
        scale = float(config["active_section_scale"])
        for fl in candidate:
            section = np.asarray(fl.section, dtype=float).copy()
            section[section > 0.0] *= scale
            fl.section = section

        source_volume = base.total_volume(source)
        candidate_volume = base.total_volume(candidate)
        source_area = climate.ice_area(source)
        candidate_area = climate.ice_area(candidate)
        source_root = climate.root_metrics(source, float(config["registered_lia_length_m"]))
        candidate_root = climate.root_metrics(candidate, float(config["registered_lia_length_m"]))
        source_support = topology.support_state(source, float(config["active_thickness_tolerance_m"]))
        candidate_support = topology.support_state(candidate, float(config["active_thickness_tolerance_m"]))
        source_reverse = reverse_counts(source)
        candidate_reverse = reverse_counts(candidate)
        cdf = hypsometry_cdf(source, candidate)
        relative_tolerance = float(config["gates"]["static_relative_tolerance"])
        static_checks = {
            "volume_scale_exact": abs(candidate_volume / source_volume - scale) <= relative_tolerance,
            "area_unchanged": abs(candidate_area / source_area - 1.0) <= relative_tolerance,
            "connected_length_unchanged": candidate_root["length_m"] == source_root["length_m"],
            "free_domain_unchanged": candidate_root["free_downstream_m"] == source_root["free_downstream_m"],
            "support_state_unchanged": candidate_support == source_support,
            "within_reverse_not_increased": candidate_reverse["within"] <= source_reverse["within"],
            "junction_reverse_not_increased": candidate_reverse["junction"] <= source_reverse["junction"],
            "hypsometry_cdf": cdf <= float(config["gates"]["static_hypsometry_cdf_max"]),
            "finite_nonnegative": all(np.isfinite(fl.section).all() and np.all(np.asarray(fl.section) >= 0.0) for fl in candidate),
        }
        static = {
            "schema": "lia-191-system2005-policy-v6-joint-candidate-static-v1",
            "status": "static-pass" if all(static_checks.values()) else "static-fail",
            "checks": static_checks,
            "source_volume_m3": source_volume,
            "candidate_volume_m3": candidate_volume,
            "volume_scale": candidate_volume / source_volume,
            "source_area_m2": source_area,
            "candidate_area_m2": candidate_area,
            "source_root": source_root,
            "candidate_root": candidate_root,
            "source_reverse": source_reverse,
            "candidate_reverse": candidate_reverse,
            "hypsometry_cdf_maximum": cdf,
        }
        write_json(output / "static_summary.json", static)
        if not all(static_checks.values()):
            summary = {
                "schema": "lia-191-system2005-policy-v6-joint-candidate-v1",
                "status": "static-fail-mechanism-closed",
                "static": static,
                "preflight_run": False, "formal_50y_run": False, "historical_run": False,
            }
        else:
            candidate_path = output / "joint_candidate_initial_state.pkl.gz"
            with gzip.open(candidate_path, "wb") as stream:
                pickle.dump(candidate, stream, protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open(candidate_path, "rb") as stream:
                recovered = pickle.load(stream)
            roundtrip, maxima = bitwise_reload_equal(candidate, recovered)
            if not roundtrip:
                raise RuntimeError(f"candidate state roundtrip failed: {maxima}")

            pre_dir = output / "preflight10"
            pre_dir.mkdir()
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
                "schema": "lia-191-system2005-policy-v6-joint-candidate-preflight10-v1",
                "status": "preflight-pass" if all(pre_checks.values()) else "preflight-fail",
                "volume_change_fraction": pre_dv,
                "final_length_m": float(pre_final["length_m"]),
                "final_area_m2": float(pre_final["area_m2"]),
                "final_free_downstream_m": float(pre_final["free_downstream_m"]),
                "final_internal_gap_branch_count": int(pre_final["internal_gap_branch_count"]),
                "step_count": pre_raw["step_count"],
                "checks": pre_checks,
            }
            write_json(output / "preflight10_summary.json", preflight)
            if not all(pre_checks.values()):
                summary = {
                    "schema": "lia-191-system2005-policy-v6-joint-candidate-v1",
                    "status": "preflight-fail-mechanism-closed",
                    "static": static, "preflight": preflight,
                    "formal_50y_run": False, "historical_run": False,
                }
            else:
                formal_dir = output / "formal50"
                formal_dir.mkdir()
                formal_raw, formal_rows, formal_flowlines = dynamic_run(
                    config, candidate, float(config["formal_years"]), formal_dir
                )
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
                formal_state = formal_dir / "lia_equilibrium_state_50y.pkl.gz"
                if all(formal_checks.values()):
                    with gzip.open(formal_state, "wb") as stream:
                        pickle.dump(formal_flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
                formal = {
                    "schema": "lia-191-system2005-policy-v6-joint-candidate-formal50-v1",
                    "status": "formal-50y-steady-pass" if all(formal_checks.values()) else "formal-50y-steady-fail",
                    "volume_change_fraction": dv,
                    "final_length_m": float(final["length_m"]),
                    "final_area_m2": float(final["area_m2"]),
                    "final_free_downstream_m": float(final["free_downstream_m"]),
                    "final_internal_gap_branch_count": int(final["internal_gap_branch_count"]),
                    "step_count": formal_raw["step_count"],
                    "checks": formal_checks,
                    "state_saved": formal_state.exists(),
                    "state_sha256": sha256(formal_state) if formal_state.exists() else None,
                }
                write_json(output / "formal50_summary.json", formal)
                summary = {
                    "schema": "lia-191-system2005-policy-v6-joint-candidate-v1",
                    "status": "formal-50y-pass-historical-pending" if all(formal_checks.values()) else "formal-50y-fail-mechanism-closed",
                    "static": static, "preflight": preflight, "formal": formal,
                    "historical_run": False,
                }
        summary["runtime_seconds"] = time.perf_counter() - started
        summary["design_sha256"] = sha256(design_path)
        summary["source_state_sha256"] = sha256(source_state_path)
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-policy-v6-joint-candidate-v1",
            "status": "technical-fail",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
            "historical_run": False,
        }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["status"] == "technical-fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
