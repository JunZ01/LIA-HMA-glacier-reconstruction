"""Build the 991 bridge-refine batch in the narrow delta_T / melt_f gap.

This batch follows the 2026-08-14 local, micro, and fixed-melt bracket runs.
System 991 remains a near-miss under the formal separation rule, with the
best finite candidates alternating between peak-short / endpoint-good and
peak-good / endpoint-long. The bounded nine-point grid below resolves the
specific bridge suggested by those diagnostics: delta_T around -0.05 K and
melt_f between 1.12 and 1.14 times the Stage-2 value.
"""

from __future__ import annotations

import json
from pathlib import Path

from .build_lia_191_local_refine_991_5143_v1 import _base_config
from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_bridge_refine_991_v1"
BLOCK = "B_bridge_refine_991_v1"
CONFIG_DIR = (
    project_root()
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
    / EXPERIMENT_NAME
)
OUTPUT_ROOT = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / EXPERIMENT_NAME
)

POINTS = [
    {"delta_t": -0.045, "melt_f_stage2_multiplier": 1.12},
    {"delta_t": -0.045, "melt_f_stage2_multiplier": 1.13},
    {"delta_t": -0.045, "melt_f_stage2_multiplier": 1.14},
    {"delta_t": -0.050, "melt_f_stage2_multiplier": 1.12},
    {"delta_t": -0.050, "melt_f_stage2_multiplier": 1.13},
    {"delta_t": -0.050, "melt_f_stage2_multiplier": 1.14},
    {"delta_t": -0.055, "melt_f_stage2_multiplier": 1.12},
    {"delta_t": -0.055, "melt_f_stage2_multiplier": 1.13},
    {"delta_t": -0.055, "melt_f_stage2_multiplier": 1.14},
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": "Does system 991 contain a formal separation-accepted solution inside the narrow bridge between the 2026-08-14 finite near-miss branches, when delta_T is searched around -0.05 K and melt_f around 1.12-1.14 times Stage-2?",
        "source_results": [
            "validation_data/results/paleo_surface_validation_v0/lia_191_local_refine_991_5143_v1/scores.csv",
            "validation_data/results/paleo_surface_validation_v0/lia_191_micro_refine_991_v1/scores.csv",
            "validation_data/results/paleo_surface_validation_v0/lia_191_bracket_refine_991_v1/scores.csv"
        ],
        "execution_driver": "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_tradeoff_pilot",
        "systems": [991],
        "task_count": 9,
        "design_rule": {
            "delta_t_values_k": [-0.045, -0.05, -0.055],
            "melt_f_stage2_multipliers": [1.12, 1.13, 1.14],
            "logic": "search the user-approved bridge region between the finite endpoint-favoring branch and the finite peak-favoring branch without reopening any other parameter family"
        },
        "frozen": {
            "prcp_fac": "Stage-2",
            "glen_a": "Stage-2",
            "temp_melt_c": -1.0,
            "precipitation_anomaly_mode": "off",
            "spinup_window": "B 1441-1470",
            "forward_period": "1470-2000",
            "spinup_max_years": 1500,
            "stability_limit_last_200_years": 0.02
        },
        "formal_acceptance": {
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "lia_anchor": "forward_peak",
            "observation_domain_gate": True
        },
        "preflight": {
            "lia_unique_id": 991,
            "delta_t": -0.05,
            "melt_f_stage2_multiplier": 1.13,
            "release_gate": "technical_failure=false and spinup_converged=true and spinup_stability_fraction<=0.02 and observation_domain_overflow=false and observation_domain_unresolved=false"
        },
        "max_workers": 12
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    full = _base_config()
    full["inputs"]["declared_preflight_result"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    full["pilot_design"]["block"] = BLOCK
    full["pilot_design"]["purpose"] = "991-only bridge refine at the user-approved narrow delta_T / melt_f region around -0.05 K and x1.12-1.14 melt_f. Freeze geometry, forcing, prcp_fac, glen_a, temp_melt, and observation-domain gate."
    full["pilot_design"]["systems"] = {"991": {"parameter_pairs": POINTS}}
    full["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    full["execution"]["work_root"] = "w191_bridge_refine_991_v1"
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight = _base_config()
    preflight["inputs"]["declared_preflight_result"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    preflight["pilot_design"]["block"] = BLOCK
    preflight["pilot_design"]["purpose"] = "One-chain preflight for the 991 bridge refine batch."
    preflight["pilot_design"]["systems"] = {"991": {"parameter_pairs": [{"delta_t": -0.05, "melt_f_stage2_multiplier": 1.13}]}}
    preflight["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    preflight["execution"]["work_root"] = "w191_bridge_refine_991_v1"
    preflight["execution"]["max_workers"] = 1
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "task_count": 9,
        "config_full": str(full_path),
        "config_preflight": str(preflight_path),
        "contract_json": str(OUTPUT_ROOT / "experiment_contract.json")
    }
    (OUTPUT_ROOT / "design_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
