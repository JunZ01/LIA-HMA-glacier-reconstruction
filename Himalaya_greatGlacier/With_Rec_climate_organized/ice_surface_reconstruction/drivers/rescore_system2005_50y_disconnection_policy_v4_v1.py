"""Read-only policy-V4 rescore of the completed system-2005 50-year run."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

from oggm import cfg

from .run_system2005_disconnection_diagnostic_historical_v1 import (
    connected_root_metrics,
)


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "s2005g50v9ckv1"
STEADY = RESULTS / "lia_191_system2005_policy2pct_steady_adapter_v1/lia_equilibrium_state_50y.pkl.gz"
POLICY = RESULTS / "lia_191_disconnection_diagnostic_policy_v4/contract.json"
OUTPUT = RESULTS / "lia_191_system2005_50y_disconnection_policy_v4_rescore_v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_flowlines(path, bundle=False):
    with gzip.open(path, "rb") as stream:
        value = pickle.load(stream)
    return value["flowlines"] if bundle else value


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    source_summary = json.loads(
        (SOURCE / "historical_summary_formal.json").read_text(encoding="utf-8")
    )
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    config = json.loads((SOURCE / "config_snapshot.json").read_text(encoding="utf-8"))
    tolerance = float(config["active_thickness_tolerance_m"])
    lia_length = float(config["lia_registered_length_m"])
    rgi_length = float(config["rgi_registered_length_m"])
    separation = abs(lia_length - rgi_length)
    with (SOURCE / "historical_1470_2000.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        source_rows = list(csv.DictReader(stream))
    source_by_year = {int(row["year"]): row for row in source_rows}

    rescored_rows = []
    state_hashes = {}
    for year in range(1470, 1521):
        if year == 1470:
            state_path = STEADY
            flowlines = load_flowlines(state_path)
        else:
            state_path = SOURCE / f"checkpoints/year_{year}/state.pkl.gz"
            flowlines = load_flowlines(state_path, bundle=True)
        state_hashes[str(year)] = sha256(state_path)
        old = source_by_year[year]
        connected = connected_root_metrics(flowlines, lia_length, tolerance)
        rescored_rows.append({
            "year": year,
            "old_length_m": float(old["length_m"]),
            "connected_length_m": float(connected["length_m"]),
            "length_correction_m": float(connected["length_m"]) - float(old["length_m"]),
            "root_active_component_count": connected["root_active_component_count"],
            "root_downstream_remnant_node_count": connected["root_downstream_remnant_node_count"],
            "root_downstream_remnant_volume_fraction": connected["root_downstream_remnant_volume_fraction"],
            "root_downstream_remnant_area_fraction": connected["root_downstream_remnant_area_fraction"],
            "aggregate_detached_volume_fraction": float(old["aggregate_detached_volume_fraction"]),
            "aggregate_detached_area_fraction": float(old["aggregate_detached_area_fraction"]),
            "newly_activated_initial_zero_node_count": int(old["newly_activated_initial_zero_node_count"]),
        })

    peak = max(rescored_rows, key=lambda row: row["connected_length_m"])
    final = rescored_rows[-1]
    e_lia_total = abs(peak["connected_length_m"] - lia_length) / lia_length
    e_lia_separation = abs(peak["connected_length_m"] - lia_length) / separation
    e_rgi_total = abs(final["connected_length_m"] - rgi_length) / rgi_length
    e_rgi_separation = abs(final["connected_length_m"] - rgi_length) / separation
    length_checks = {
        "lia_total": e_lia_total <= float(config["formal_gates"]["total_length_error_max"]),
        "lia_separation": e_lia_separation <= float(config["formal_gates"]["separation_length_error_max"]),
        "rgi_total": e_rgi_total <= float(config["formal_gates"]["total_length_error_max"]),
        "rgi_separation": e_rgi_separation <= float(config["formal_gates"]["separation_length_error_max"]),
    }
    max_tributary_volume = max(row["aggregate_detached_volume_fraction"] for row in rescored_rows)
    max_tributary_area = max(row["aggregate_detached_area_fraction"] for row in rescored_rows)
    max_root_volume = max(row["root_downstream_remnant_volume_fraction"] for row in rescored_rows)
    max_root_area = max(row["root_downstream_remnant_area_fraction"] for row in rescored_rows)
    changed_years = [row["year"] for row in rescored_rows if row["length_correction_m"] != 0.0]
    checks = {
        "source_completed_50y_36600_steps": source_summary.get("step_count") == 36600,
        "source_numerical_checks_pass": all(source_summary["numerical_checks"].values()),
        "all_51_states_materialized": len(state_hashes) == 51,
        "disconnection_removed_from_hard_gate": policy["disconnection_is_hard_acceptance_gate"] is False,
        "length_uses_upstream_connected_root_component": policy["root_length_component"] == "upstream-most-active-component",
        "no_artificial_support": max(row["newly_activated_initial_zero_node_count"] for row in rescored_rows) == 0,
        "no_review_warning_over_20pct": (
            max(max_tributary_volume, max_root_volume) <= policy["detached_volume_fraction_review_warning"]
            and max(max_tributary_area, max_root_area) <= policy["detached_area_fraction_review_warning"]
        ),
    }
    summary = {
        "schema": "lia-191-system2005-50y-disconnection-policy-v4-rescore-v1",
        "status": "policy-rescore-complete" if all(checks.values()) else "policy-rescore-fail",
        "checks": checks,
        "formal_checks_under_policy_v4": {
            "numerical": all(source_summary["numerical_checks"].values()),
            "length": all(length_checks.values()),
            "support_integrity": checks["no_artificial_support"],
        },
        "length_checks": length_checks,
        "connected_length_metrics": {
            "peak_year": peak["year"],
            "peak_length_m": peak["connected_length_m"],
            "final_year": final["year"],
            "final_length_m": final["connected_length_m"],
            "e_lia_total": e_lia_total,
            "e_lia_separation": e_lia_separation,
            "e_rgi_total": e_rgi_total,
            "e_rgi_separation": e_rgi_separation,
            "years_changed_from_last-active-node_length": changed_years,
            "maximum_length_correction_m": min(row["length_correction_m"] for row in rescored_rows),
        },
        "disconnection_diagnostics": {
            "maximum_tributary_detached_volume_fraction": max_tributary_volume,
            "maximum_tributary_detached_area_fraction": max_tributary_area,
            "maximum_root_downstream_remnant_volume_fraction": max_root_volume,
            "maximum_root_downstream_remnant_area_fraction": max_root_area,
            "review_warning_fraction": policy["detached_volume_fraction_review_warning"],
            "hard_failure": False,
        },
        "source_summary": str((SOURCE / "historical_summary_formal.json").resolve()),
        "source_state_sha256_by_year": state_hashes,
        "scientific_state_modified": False,
        "physics_forcing_or_solver_modified": False,
        "accepted_count": 54,
        "remaining_count": 137,
        "full_historical_authorized": False,
        "remaining_blocker": "lia-separation-length-gate-fails",
        "next_legal_action": "adjudicate-separation-normalized-length-gate-before-530y-launch",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "annual_connected_length_rescore.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rescored_rows[0]))
        writer.writeheader()
        writer.writerows(rescored_rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    if summary["status"] != "policy-rescore-complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
