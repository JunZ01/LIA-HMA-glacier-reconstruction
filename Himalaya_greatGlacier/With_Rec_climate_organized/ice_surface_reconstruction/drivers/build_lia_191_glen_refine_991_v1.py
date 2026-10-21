"""Build a 991-only glen_a refine batch on two established near-miss branches."""

from __future__ import annotations

import json

from .build_lia_191_local_refine_991_5143_v1 import _base_config
from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_glen_refine_991_v1"
BLOCK = "B_glen_refine_991_v1"
CONFIG_DIR = project_root() / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / EXPERIMENT_NAME
OUTPUT_ROOT = project_root() / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT_NAME

POINTS = [
    {"delta_t": -0.0234375, "melt_f_stage2_multiplier": 1.10, "glen_a_stage2_multiplier": value}
    for value in (0.80, 0.90, 1.00, 1.10, 1.20)
] + [
    {"delta_t": -0.045, "melt_f_stage2_multiplier": 1.14, "glen_a_stage2_multiplier": value}
    for value in (0.80, 0.90, 1.00, 1.10, 1.20)
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": "Can system 991 resolve its formal separation tradeoff when glen_a is opened around Stage-2 while the two best existing delta_T/melt_f branches remain fixed?",
        "source_results": [
            "validation_data/results/paleo_surface_validation_v0/lia_191_bracket_refine_991_v1/scores.csv",
            "validation_data/results/paleo_surface_validation_v0/lia_191_bridge_refine_991_v1/scores.csv",
        ],
        "execution_driver": "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_glen_tradeoff_pilot",
        "systems": [991],
        "task_count": len(POINTS),
        "design_rule": {
            "anchor_branches": [
                {"delta_t": -0.0234375, "melt_f_stage2_multiplier": 1.10, "role": "peak-near-LIA but year-2000 too long"},
                {"delta_t": -0.045, "melt_f_stage2_multiplier": 1.14, "role": "year-2000 near RGI but peak too short"},
            ],
            "glen_a_stage2_multipliers": [0.80, 0.90, 1.00, 1.10, 1.20],
            "logic": "open one new parameter family only; keep geometry, forcing, prcp_fac, temp_melt, and formal observation-domain scoring unchanged",
        },
        "frozen": {
            "prcp_fac": "Stage-2",
            "temp_melt_c": -1.0,
            "precipitation_anomaly_mode": "off",
            "spinup_window": "B 1441-1470",
            "forward_period": "1470-2000",
            "spinup_max_years": 1500,
            "stability_limit_last_200_years": 0.02,
        },
        "formal_acceptance": {
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "lia_anchor": "forward_peak",
            "observation_domain_gate": True,
        },
        "preflight": {
            "lia_unique_id": 991,
            "delta_t": -0.045,
            "melt_f_stage2_multiplier": 1.14,
            "glen_a_stage2_multiplier": 1.0,
        },
        "max_workers": 12,
        "off_contract_note": "User-approved 991-only follow-up after the bounded delta_T/melt_f bridge returned no formal acceptance.",
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    full = _base_config()
    full["inputs"]["declared_preflight_result"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    full["pilot_design"]["block"] = BLOCK
    full["pilot_design"]["purpose"] = "991-only glen_a refine after the bounded delta_T/melt_f bridge. Freeze geometry, forcing, prcp_fac, temp_melt, and observation-domain gate."
    full["pilot_design"]["systems"] = {"991": {"parameter_pairs": POINTS}}
    full["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    full["execution"]["work_root"] = "w191_glen_refine_991_v1"
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight = _base_config()
    preflight["inputs"]["declared_preflight_result"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    preflight["pilot_design"]["block"] = BLOCK
    preflight["pilot_design"]["purpose"] = "One-chain preflight for the 991 glen_a refine batch."
    preflight["pilot_design"]["systems"] = {"991": {"parameter_pairs": [{"delta_t": -0.045, "melt_f_stage2_multiplier": 1.14, "glen_a_stage2_multiplier": 1.0}]}}
    preflight["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    preflight["execution"]["work_root"] = "w191_glen_refine_991_v1"
    preflight["execution"]["max_workers"] = 1
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {"launched_oggm": False, "experiment_name": EXPERIMENT_NAME, "task_count": len(POINTS), "config_full": str(full_path), "config_preflight": str(preflight_path), "contract_json": str(OUTPUT_ROOT / "experiment_contract.json")}
    (OUTPUT_ROOT / "design_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
