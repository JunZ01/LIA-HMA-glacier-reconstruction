"""Build the first formal dual-gate delta-T bisection batch.

The batch targets five saved near-misses whose LIA peak and year-2000 endpoint
move in a direction that can plausibly enter the 20% separation gate under a
small colder delta-T perturbation. Stage-2 physics and all forcing choices are
frozen. System 2928 keeps its verified spatial-downstream-v2 geometry.
"""

from __future__ import annotations

import json

import pandas as pd

from .build_lia_191_iterative_loop import project_root
from .build_lia_191_local_refine_991_5143_v1 import _base_config


EXPERIMENT_NAME = "lia_191_formal_delta_bisection_batch1_v1"
REGULAR_BLOCK = "B_formal_delta_bisection_batch1_v1"
SPATIAL_BLOCK = "B_formal_delta_bisection_2928_v1"
ROOT = project_root()
OUTPUT_ROOT = (
    ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT_NAME
)
CONFIG_DIR = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
    / EXPERIMENT_NAME
)
INVENTORY = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_analog_v1"
    / "chain_dual_gate_inventory.csv"
)

REGULAR_POINTS = {
    1546: [0.0937500, 0.1015625, 0.1093750],
    2815: [0.3437500, 0.3593750, 0.3671875],
    2902: [-0.2968750, -0.2812500, -0.2656250],
    2806: [0.2187500, 0.2265625, 0.2343750],
}
SPATIAL_POINTS = {2928: [0.3906250, 0.4062500, 0.4218750]}
ANCHOR_TASKS = {
    1546: "1546-B_delta_boundary_extension_v1-e91c06e37143e825",
    2815: "2815-B_delta_boundary_extension_v1-a670bf6486fef55f",
    2902: "2902-B_delta_boundary_extension_v1-855f4d987daf5d9d",
    2806: "2806-B_delta_boundary_extension_v1-09d9410c76ddfde2",
    2928: "2928-B_curve_loop_delta_boundary_batch2_v1-spatial-downstream-v2-e069e8f7ee7719d8",
}


def _systems(points: dict[int, list[float]]) -> dict[str, dict]:
    return {
        str(system_id): {
            "parameter_pairs": [
                {
                    "delta_t": value,
                    "melt_f_stage2_multiplier": 1.0,
                    "glen_a_stage2_multiplier": 1.0,
                }
                for value in values
            ]
        }
        for system_id, values in points.items()
    }


def _selection_diagnostics() -> pd.DataFrame:
    chains = pd.read_csv(INVENTORY)
    rows = []
    for system_id, task_id in ANCHOR_TASKS.items():
        selected = chains[
            chains["lia_unique_id"].astype(int).eq(system_id)
            & chains["task_id"].astype(str).eq(task_id)
        ]
        if len(selected) != 1:
            raise ValueError(f"anchor task changed for system {system_id}: {len(selected)} rows")
        row = selected.iloc[0].copy()
        if not bool(row["stable_chain"]) or not bool(row["audit_length_observation_valid"]):
            raise ValueError(f"anchor is no longer stable and observation-valid: {system_id}")
        if bool(row["formal_dual_pass"]):
            raise ValueError(f"system {system_id} already passes the formal dual gate")
        separation = float(row["lia_rgi_separation_m_formal"])
        row["signed_lia_sep"] = (float(row["forward_peak_m"]) - float(row["lia_target_m"])) / separation
        row["signed_rgi_sep"] = (float(row["forward_2000_m"]) - float(row["rgi_target_m"])) / separation
        row["planned_delta_t_values"] = json.dumps(
            (REGULAR_POINTS | SPATIAL_POINTS)[system_id]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lia_unique_id")


def _configured(
    systems: dict[str, dict],
    *,
    block: str,
    output_subdir: str,
    work_root: str,
    workers: int,
    purpose: str,
    spatial: bool,
) -> dict:
    config = _base_config()
    config["inputs"]["declared_preflight_result"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    )
    if spatial:
        config["inputs"]["elevation_band_root"] = (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_elevation_band_spatial_downstream_pilot_v2"
        )
    config["acceptance"]["report_total_length_10pct"] = True
    config["acceptance"]["freeze_requires_both_total_and_separation_gates"] = True
    config["pilot_design"]["block"] = block
    config["pilot_design"]["purpose"] = purpose
    config["pilot_design"]["systems"] = systems
    config["pilot_design"]["prcp_fac_multiplier"] = 1.0
    config["pilot_design"]["glen_a_source"] = "stage2-frozen"
    config["pilot_design"]["temp_melt"] = -1.0
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/{output_subdir}"
    )
    config["execution"]["work_root"] = work_root
    config["execution"]["max_workers"] = workers
    return config


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics = _selection_diagnostics()
    diagnostics.to_csv(OUTPUT_ROOT / "selection_diagnostics.csv", index=False)

    regular_systems = _systems(REGULAR_POINTS)
    spatial_systems = _systems(SPATIAL_POINTS)
    regular = _configured(
        regular_systems,
        block=REGULAR_BLOCK,
        output_subdir="regular",
        work_root="w191_formal_dt_bisect_b1_regular_v1",
        workers=12,
        purpose=(
            "Four-system fine delta-T bisection around stable formal near-misses; "
            "Stage-2 physical parameters remain frozen."
        ),
        spatial=False,
    )
    spatial = _configured(
        spatial_systems,
        block=SPATIAL_BLOCK,
        output_subdir="spatial_2928",
        work_root="w191_formal_dt_bisect_b1_2928_v1",
        workers=3,
        purpose=(
            "System-2928 fine delta-T bisection on its verified spatial-downstream-v2 domain; "
            "Stage-2 physical parameters remain frozen."
        ),
        spatial=True,
    )

    regular_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_regular.json"
    spatial_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_spatial_2928.json"
    regular_path.write_text(json.dumps(regular, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    spatial_path.write_text(json.dumps(spatial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    regular_preflight = _configured(
        _systems({2815: [0.3593750]}),
        block=REGULAR_BLOCK,
        output_subdir="regular",
        work_root="w191_formal_dt_bisect_b1_regular_v1",
        workers=1,
        purpose="Mandatory one-chain preflight for the regular four-system fine delta-T batch.",
        spatial=False,
    )
    spatial_preflight = _configured(
        _systems({2928: [0.4218750]}),
        block=SPATIAL_BLOCK,
        output_subdir="spatial_2928",
        work_root="w191_formal_dt_bisect_b1_2928_v1",
        workers=1,
        purpose="Mandatory one-chain preflight for system 2928 on spatial-downstream-v2 geometry.",
        spatial=True,
    )
    regular_preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_regular_preflight.json"
    spatial_preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_spatial_2928_preflight.json"
    regular_preflight_path.write_text(
        json.dumps(regular_preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    spatial_preflight_path.write_text(
        json.dumps(spatial_preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-15",
        "experiment_question": (
            "Can one bounded fine-delta-T interpolation layer move five stable total-length near-misses "
            "through both formal 20% separation endpoints without changing Stage-2 physics?"
        ),
        "source_inventory": str(INVENTORY),
        "selected_systems": sorted(ANCHOR_TASKS),
        "task_count": 15,
        "parameter_design": {
            "regular": REGULAR_POINTS,
            "spatial_2928": SPATIAL_POINTS,
            "changed_parameter": "delta_t only",
        },
        "frozen": {
            "melt_f": "Stage-2 x1.0",
            "prcp_fac": "Stage-2",
            "glen_a": "Stage-2 x1.0",
            "temp_melt_c": -1.0,
            "precipitation_anomaly_mode": "off",
            "spinup_window": "B 1441-1470",
            "forward_period": "1470-2000",
            "climate": "W5E5 1961-1990 plus ensemble_rec_ref temperature anomalies",
        },
        "formal_acceptance": {
            "technical_clean": True,
            "spinup_stability_fraction_max": 0.02,
            "valid_observation_domain": True,
            "total_length_endpoint_tolerance": 0.10,
            "separation_endpoint_tolerance": 0.20,
            "freeze_requires_all": True,
        },
        "preflights": {
            "regular": {"system": 2815, "delta_t": 0.3593750},
            "spatial_2928": {"system": 2928, "delta_t": 0.4218750},
            "release_gate": (
                "status=ok; converged; last-200y volume change<=2%; observation-domain valid; "
                "one worker"
            ),
        },
        "max_workers": 12,
        "configs": {
            "regular": str(regular_path),
            "regular_preflight": str(regular_preflight_path),
            "spatial_2928": str(spatial_path),
            "spatial_2928_preflight": str(spatial_preflight_path),
        },
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "systems": sorted(ANCHOR_TASKS),
        "task_count": 15,
        "configs": contract["configs"],
        "contract_json": str(OUTPUT_ROOT / "experiment_contract.json"),
    }
    (OUTPUT_ROOT / "design_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
