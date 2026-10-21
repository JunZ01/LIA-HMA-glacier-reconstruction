"""Build the 991 bracket-refine batch along fixed melt_f x1.1.

This batch follows the 2026-08-14 micro-refine result that identified a clean
formal bracket on system 991:
- delta_T=0.0 K with melt_f x1.1 keeps the year-2000 endpoint within the
  formal separation tolerance but leaves the LIA peak too short;
- delta_T=-0.03125 K with the same melt factor passes the LIA peak gate but
  makes the year-2000 endpoint too long.

The four-point batch resolves the narrow interval between these two chains.
"""

from __future__ import annotations

import json
from pathlib import Path

from .build_lia_191_local_refine_991_5143_v1 import _base_config
from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_bracket_refine_991_v1"
BLOCK = "B_bracket_refine_991_v1"
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
    {"delta_t": -0.015625, "melt_f_stage2_multiplier": 1.10},
    {"delta_t": -0.01953125, "melt_f_stage2_multiplier": 1.10},
    {"delta_t": -0.0234375, "melt_f_stage2_multiplier": 1.10},
    {"delta_t": -0.02734375, "melt_f_stage2_multiplier": 1.10},
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": "Does system 991 contain a formal separation-accepted point inside the fixed-melt x1.1 bracket between delta_T=0.0 K and delta_T=-0.03125 K?",
        "source_results": [
            "validation_data/results/paleo_surface_validation_v0/lia_191_local_refine_991_5143_v1/scores.csv",
            "validation_data/results/paleo_surface_validation_v0/lia_191_micro_refine_991_v1/scores.csv"
        ],
        "execution_driver": "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_tradeoff_pilot",
        "systems": [991],
        "task_count": 4,
        "design_rule": {
            "delta_t_values_k": [-0.015625, -0.01953125, -0.0234375, -0.02734375],
            "melt_f_stage2_multiplier": 1.10,
            "logic": "hold melt_f at the fixed x1.1 bracket and resolve the narrow delta_T interval where the peak and year-2000 endpoint gates swap order"
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
            "delta_t": -0.0234375,
            "melt_f_stage2_multiplier": 1.10,
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
    full["pilot_design"]["purpose"] = "991-only bracket refine at fixed melt_f x1.1, resolving the narrow delta_T interval between the 0.0/x1.1 endpoint-pass chain and the -0.03125/x1.1 peak-pass chain."
    full["pilot_design"]["systems"] = {"991": {"parameter_pairs": POINTS}}
    full["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    full["execution"]["work_root"] = "w191_bracket_refine_991_v1"
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight = _base_config()
    preflight["inputs"]["declared_preflight_result"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    preflight["pilot_design"]["block"] = BLOCK
    preflight["pilot_design"]["purpose"] = "One-chain preflight for the 991 fixed-melt bracket refine."
    preflight["pilot_design"]["systems"] = {"991": {"parameter_pairs": [{"delta_t": -0.0234375, "melt_f_stage2_multiplier": 1.10}]}}
    preflight["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    preflight["execution"]["work_root"] = "w191_bracket_refine_991_v1"
    preflight["execution"]["max_workers"] = 1
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "task_count": 4,
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
