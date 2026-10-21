"""Static and checkpoint-contract validation for the system-2005 2D history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import run_system2005_masked_historical_branch_v1 as historical
from . import run_system2005_masked_steady50_checkpointed_v1 as engine


PROJECT = Path(__file__).resolve().parents[4]
CONFIGS = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/"
    "system2005_masked_historical_preflight_endp0p834281_v1.json",
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/"
    "system2005_masked_historical_preflight_endp1p334281_v1.json",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    configs = [json.loads((PROJECT / path).read_text(encoding="utf-8")) for path in CONFIGS]
    common = configs[0]
    package_path = PROJECT / common["climate_package"]
    with np.load(package_path, allow_pickle=False) as package:
        sequence_indices = np.asarray(package["sequence_indices"], dtype=np.int64)
        sequence_years = np.asarray(package["sequence_years"], dtype=np.int64)
        sequence_months = np.asarray(package["sequence_months"], dtype=np.int64)
        source_years = np.asarray(package["years"], dtype=np.int64)
        source_months = np.asarray(package["months"], dtype=np.int64)
    state_path = PROJECT / common["historical_start_state"]
    with np.load(state_path, allow_pickle=False) as state:
        thickness = np.asarray(state["thickness_m"], dtype=np.float64)
        mask = np.asarray(state["domain_mask"], dtype=bool)

    loaded = engine.load_inputs(common)
    path_metrics = engine._path_metrics(
        thickness,
        loaded["x"],
        loaded["y"],
        loaded["path_xy"],
        loaded["controls"],
        float(common["lia_registered_length_m"]),
    )
    fixture_root = output / "checkpoint_fixture"
    fixture_root.mkdir()
    fixture_counters = {
        "accepted_steps": 17,
        "rejected_steps": 2,
        "solver_calls": 102,
        "projection_calls_accepted_path": 3,
        "cumulative_source_m3": 123.5,
        "cumulative_prevented_overmelt_m3": 0.0,
        "maximum_step_closure_m3": 1e-6,
        "maximum_budget_residual_m3": 1e-6,
        "maximum_error_relative_l2": 0.01,
        "minimum_accepted_dt_seconds": 30.0,
        "maximum_accepted_dt_seconds": 86400.0,
    }
    fixture_metrics = {
        "1": {
            "elapsed_years": 1.0,
            "root_path_metrics": path_metrics,
        }
    }
    fixture_contract = "A" * 64
    historical.historical_checkpoint_writer.config = common
    original_schema = engine.CHECKPOINT_SCHEMA
    engine.CHECKPOINT_SCHEMA = historical.CHECKPOINT_SCHEMA
    try:
        manifest_path = historical.historical_checkpoint_writer(
            fixture_root,
            year=1,
            state=thickness,
            elapsed_seconds=31_536_000.0,
            proposed_dt_seconds=43_200.0,
            counters=fixture_counters,
            checkpoint_metrics=fixture_metrics,
            contract_hash=fixture_contract,
        )
        restored = engine.load_checkpoint(manifest_path, fixture_contract)
    finally:
        engine.CHECKPOINT_SCHEMA = original_schema

    refusal_checks = []
    for path in CONFIGS:
        try:
            historical.run(PROJECT / path)
        except RuntimeError as exc:
            refusal_checks.append(str(exc) == "historical-branch-launch-not-authorized")
        else:
            refusal_checks.append(False)

    endpoint_offsets = [float(config["temperature_offset_2000_k"]) for config in configs]
    checks = {
        "two_versioned_branch_configs": len(configs) == 2,
        "configs_share_frozen_inputs": all(
            config[key] == common[key]
            for config in configs
            for key in (
                "geology_sha256",
                "static_summary_sha256",
                "climate_package_sha256",
                "observation_path_sha256",
                "historical_start_state_sha256",
                "historical_start_thickness_array_sha256",
                "historical_start_domain_mask_array_sha256",
                "glen_a",
                "lia_registered_length_m",
                "rgi_registered_length_m",
                "formal_gates",
            )
        ),
        "launch_refused_before_route": all(refusal_checks),
        "new_output_roots_absent": all(not (PROJECT / config["output_root"]).exists() for config in configs),
        "historical_package_hash": engine.sha256(package_path) == common["climate_package_sha256"],
        "calendar_has_6372_months": sequence_indices.size == 6372,
        "calendar_exact_bounds": (
            int(sequence_years[0]) == 1470
            and int(sequence_months[0]) == 1
            and int(sequence_years[-1]) == 2000
            and int(sequence_months[-1]) == 12
        ),
        "calendar_source_rows_match": bool(
            np.array_equal(source_years[sequence_indices], sequence_years)
            and np.array_equal(source_months[sequence_indices], sequence_months)
        ),
        "start_state_file_hash": engine.sha256(state_path) == common["historical_start_state_sha256"],
        "start_thickness_array_hash": historical.array_sha256(thickness)
        == common["historical_start_thickness_array_sha256"],
        "start_mask_array_hash": historical.array_sha256(mask)
        == common["historical_start_domain_mask_array_sha256"],
        "start_state_domain_exact": bool(np.all(thickness[~mask] == 0.0)),
        "start_state_finite_nonnegative": bool(np.isfinite(thickness).all() and np.all(thickness >= 0.0)),
        "checkpoint_calendar_year": json.loads(manifest_path.read_text(encoding="utf-8"))[
            "calendar_year"
        ]
        == 1471,
        "checkpoint_state_exact": bool(np.array_equal(restored["state"], thickness)),
        "checkpoint_counters_exact": restored["counters"] == fixture_counters,
        "checkpoint_metrics_exact": restored["checkpoint_metrics"] == fixture_metrics,
        "registered_scoring_basis": (
            float(path_metrics["registered_lia_length_m"])
            == float(common["lia_registered_length_m"])
            and float(common["rgi_registered_length_m"]) == 23308.5403923166
        ),
        "endpoint_order_registered": endpoint_offsets == sorted(endpoint_offsets)
        and endpoint_offsets[0] < endpoint_offsets[1],
        "unchanged_1470_offset": all(
            float(config["temperature_offset_1470_k"]) == 0.3342811518826305
            for config in configs
        ),
    }
    summary = {
        "schema": "lia-191-system2005-masked-historical-preparation-v1",
        "status": "historical-preparation-pass" if all(checks.values()) else "historical-preparation-fail",
        "system_id": 2005,
        "configs": [
            {
                "path": path,
                "sha256": engine.sha256(PROJECT / path),
                "temperature_offset_2000_k": config["temperature_offset_2000_k"],
                "output_root": config["output_root"],
            }
            for path, config in zip(CONFIGS, configs)
        ],
        "driver": common["driver"],
        "driver_sha256": common["driver_sha256"],
        "climate_package": common["climate_package"],
        "climate_package_sha256": common["climate_package_sha256"],
        "historical_start_state": common["historical_start_state"],
        "historical_start_state_sha256": common["historical_start_state_sha256"],
        "start_path_length_m": path_metrics["observed_active_control_volume_length_m"],
        "registered_lia_length_m": common["lia_registered_length_m"],
        "registered_rgi_length_m": common["rgi_registered_length_m"],
        "preflight_calendar": [1470, 1480],
        "full_calendar": [1470, 2000],
        "estimated_seconds_per_preflight_branch": 1457.11,
        "estimated_seconds_per_full_branch": 77226.83,
        "checkpoint_fixture_manifest": str(manifest_path),
        "checkpoint_fixture_manifest_sha256": engine.sha256(manifest_path),
        "checks": checks,
        "model_run": False,
        "launch_authorized": False,
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "next_action": "version a launch route and launch the registered two-branch 1470-1480 response preflight without changing inputs or endpoint offsets",
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT
        / "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_system2005_masked_historical_preparation_v1",
    )
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if summary["status"] != "historical-preparation-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
