"""Build the final three-point glen_a interpolation for system 991."""

from __future__ import annotations

import json

from .build_lia_191_local_refine_991_5143_v1 import _base_config
from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_finish_refine_991_v1"
BLOCK = "B_finish_refine_991_v1"
CONFIG_DIR = project_root() / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / EXPERIMENT_NAME
OUTPUT_ROOT = project_root() / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT_NAME
DELTA_T = -0.125
MELT_MULTIPLIER = 1.10
GLEN_MULTIPLIERS = (2.70, 2.75, 2.80)
POINTS = [
    {
        "delta_t": DELTA_T,
        "melt_f_stage2_multiplier": MELT_MULTIPLIER,
        "glen_a_stage2_multiplier": glen_multiplier,
    }
    for glen_multiplier in GLEN_MULTIPLIERS
]


def _configured(points: list[dict], *, workers: int, purpose: str) -> dict:
    config = _base_config()
    config["inputs"]["declared_preflight_result"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    config["pilot_design"]["block"] = BLOCK
    config["pilot_design"]["purpose"] = purpose
    config["pilot_design"]["systems"] = {"991": {"parameter_pairs": points}}
    config["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    config["execution"]["work_root"] = "w191_finish_refine_991_v1"
    config["execution"]["max_workers"] = workers
    return config


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": "Does the real OGGM system-991 chain formally pass at the interpolated glen_a multiplier near x2.75?",
        "source_results": "validation_data/results/paleo_surface_validation_v0/lia_191_response_target_991_v1/scores.csv",
        "execution_driver": "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_glen_tradeoff_pilot",
        "systems": [991],
        "task_count": len(POINTS),
        "design_rule": {
            "delta_t_k": DELTA_T,
            "melt_f_stage2_multiplier": MELT_MULTIPLIER,
            "glen_a_stage2_multipliers": list(GLEN_MULTIPLIERS),
            "evidence": {
                "glen_x2.55": {"forward_peak_m": 11650.0, "forward_2000_m": 10100.0},
                "glen_x2.65": {"forward_peak_m": 11600.0, "forward_2000_m": 10000.0},
                "interpolated_glen_x2.75": {"forward_peak_m": 11550.0, "forward_2000_m": 9900.0},
            },
            "logic": "three-point real-model interpolation around the predicted formal pass",
        },
        "frozen": {
            "delta_t_k": DELTA_T,
            "melt_f_stage2_multiplier": MELT_MULTIPLIER,
            "prcp_fac": "Stage-2",
            "temp_melt_c": -1.0,
            "precipitation_anomaly_mode": "off",
            "spinup_window": "B 1441-1470",
            "forward_period": "1470-2000",
            "spinup_max_years": 1500,
            "stability_limit_last_200_years": 0.02,
            "geometry_and_observation_domain": "unchanged",
        },
        "formal_acceptance": {"endpoint_tolerance_fraction_of_separation": 0.2, "lia_anchor": "forward_peak", "observation_domain_gate": True},
        "preflight": {"lia_unique_id": 991, "delta_t": DELTA_T, "melt_f_stage2_multiplier": MELT_MULTIPLIER, "glen_a_stage2_multiplier": 2.75},
        "max_workers": 3,
        "authorization_note": "Continuation of the user-approved 991-only iterative parameter-estimation loop.",
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    full = _configured(POINTS, workers=3, purpose="991-only final three-point glen_a interpolation at fixed delta_T=-0.125 K and melt_f x1.10.")
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight_point = [{"delta_t": DELTA_T, "melt_f_stage2_multiplier": MELT_MULTIPLIER, "glen_a_stage2_multiplier": 2.75}]
    preflight = _configured(preflight_point, workers=1, purpose="One-chain preflight at the predicted system-991 formal pass.")
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {"launched_oggm": False, "experiment_name": EXPERIMENT_NAME, "task_count": len(POINTS), "config_full": str(full_path), "config_preflight": str(preflight_path), "contract_json": str(OUTPUT_ROOT / "experiment_contract.json")}
    (OUTPUT_ROOT / "design_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
