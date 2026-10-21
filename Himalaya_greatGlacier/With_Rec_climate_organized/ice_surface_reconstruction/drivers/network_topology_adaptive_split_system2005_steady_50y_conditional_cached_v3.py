"""Exact 50-year conditional reproduction using the bitwise-equivalent cached solver."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import traceback

import numpy as np

from . import audit_lia_saved_state_connectivity_v1 as connectivity
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_climate_v3 as climate


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def reproduction(reference_summary, reference_rows, summary, rows):
    summary_fields = (
        "step_count", "initial_volume_m3", "final_volume_m3",
        "volume_change_fraction", "initial_area_m2", "final_area_m2",
        "area_change_fraction", "maximum_absolute_step_closure_m3",
        "maximum_absolute_KKT_stationarity",
    )
    summary_differences = {
        field: float(summary[field]) - float(reference_summary[field])
        for field in summary_fields
    }
    row_fields = (
        "year", "volume_m3", "area_m2", "length_m",
        "length_relative_error", "free_downstream_m",
        "terminal_thickness_m", "disconnected_tributary_count",
        "internal_gap_branch_count", "cumulative_accepted_mb_m3",
    )
    row_count_equal = len(reference_rows) == len(rows)
    maxima = {}
    if row_count_equal:
        for field in row_fields:
            old = np.asarray([float(row[field]) for row in reference_rows])
            new = np.asarray([float(row[field]) for row in rows])
            maxima[field] = float(np.max(np.abs(new - old)))
    exact = (
        row_count_equal
        and all(value == 0.0 for value in summary_differences.values())
        and all(value == 0.0 for value in maxima.values())
    )
    return {
        "reference_row_count": len(reference_rows),
        "new_row_count": len(rows),
        "row_count_equal": row_count_equal,
        "summary_differences": summary_differences,
        "timeseries_maximum_absolute_difference": maxima,
        "byte_numeric_exact": exact,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        integrity_path = base.resolve(config["cached_solver_gates"]["array_integrity_summary"])
        equivalence_path = base.resolve(config["cached_solver_gates"]["one_step_equivalence_summary"])
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
        if integrity.get("status") != "integrity-pass" or not all(integrity.get("checks", {}).values()):
            raise RuntimeError("mixedbed-array-integrity-gate-not-pass")
        if equivalence.get("status") != "equivalence-pass" or not all(equivalence.get("checks", {}).values()):
            raise RuntimeError("cached-step-equivalence-gate-not-pass")
        convergence_path = base.resolve(config["timestep_convergence_summary"])
        convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
        if convergence.get("status") != "six-hour-converged-authorized-for-50y":
            raise RuntimeError("six-hour-convergence-gate-not-passed")
        if float(config["years"]) != 50.0 or float(config["maximum_timestep_seconds"]) != 21600.0:
            raise RuntimeError("registered-steady-duration-or-timestep-mismatch")
        reference_summary_path = base.resolve(config["exact_reproduction"]["reference_summary"])
        reference_timeseries_path = base.resolve(config["exact_reproduction"]["reference_timeseries"])
        reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
        reference_rows = read_csv(reference_timeseries_path)
        summary, rows, flowlines = climate.run(config)
        base_status = summary.get("status")
        structure = connectivity.state_connectivity(flowlines, float(config["active_thickness_tolerance_m"]))
        impact = config["conditional_tributary_separation"]
        non_gap_checks = {
            key: bool(value)
            for key, value in summary["steady_checks"].items()
            if key != "internal_active_gaps_absent"
        }
        checks = {
            "base_failed_only_by_internal_gap": base_status == "steady-50y-fail" and all(non_gap_checks.values()),
            "unique_root_continuous": int(structure["root_gap_branch_count"]) == 0,
            "exactly_one_tributary_gap_branch": int(structure["tributary_gap_branch_count"]) == 1,
            "detached_volume_fraction_within_limit": float(structure["detached_head_volume_fraction_system"]) <= float(impact["detached_head_volume_fraction_max"]),
            "detached_area_fraction_within_limit": float(structure["detached_head_area_fraction_system"]) <= float(impact["detached_head_area_fraction_max"]),
            "artificial_positive_support_absent": summary.get("artificial_positive_support_used") is False,
            "posthoc_section_clip_absent": summary.get("posthoc_section_clip_used") is False,
        }
        exact = reproduction(reference_summary, reference_rows, summary, rows)
        checks["exact_reproduction"] = bool(exact["byte_numeric_exact"])
        passed = all(checks.values())
        summary.update({
            "schema": "lia-191-system2005-steady-50y-conditional-cached-v3",
            "base_v1_status": base_status,
            "status": "steady-50y-conditional-pass-tributary-separation" if passed else "steady-50y-conditional-fail",
            "conditional_tributary_separation": {
                "checks": checks,
                "detached_head_volume_fraction_system": structure["detached_head_volume_fraction_system"],
                "detached_head_area_fraction_system": structure["detached_head_area_fraction_system"],
                "root_branch": structure["root_branch"],
                "root_gap_branch_count": structure["root_gap_branch_count"],
                "tributary_gap_branch_count": structure["tributary_gap_branch_count"],
                "policy": impact,
            },
            "exact_reproduction": exact,
            "array_integrity_summary_sha256": base.sha256(integrity_path),
            "cached_step_equivalence_summary_sha256": base.sha256(equivalence_path),
            "reference_summary_sha256": base.sha256(reference_summary_path),
            "reference_timeseries_sha256": base.sha256(reference_timeseries_path),
            "timestep_convergence_summary_sha256": base.sha256(convergence_path),
            "historical_run": False,
        })
    except Exception as exc:
        summary = {
            "schema": "lia-191-system2005-steady-50y-conditional-cached-v3",
            "status": "steady-50y-conditional-technical-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "historical_run": False,
        }
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if summary.get("status") == "steady-50y-conditional-pass-tributary-separation" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary))
    if summary.get("status") != "steady-50y-conditional-pass-tributary-separation":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
