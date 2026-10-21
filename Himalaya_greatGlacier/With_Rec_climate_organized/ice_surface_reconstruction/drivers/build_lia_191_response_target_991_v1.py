"""Build a response-surface-targeted nine-point batch for system 991.

A linear fit to the completed 3x3 high-glen grid predicts that the formal
discrete-length target (about 11550 m peak and 9900 m in 2000) lies near
delta_T=-0.137 K and glen_a=2.57 times Stage-2 when melt_f remains x1.10.
This batch verifies a small real-OGGM grid around that prediction.
"""

from __future__ import annotations

import json

from .build_lia_191_local_refine_991_5143_v1 import _base_config
from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_response_target_991_v1"
BLOCK = "B_response_target_991_v1"
CONFIG_DIR = project_root() / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / EXPERIMENT_NAME
OUTPUT_ROOT = project_root() / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT_NAME

DELTA_T_VALUES = (-0.125, -0.135, -0.145)
GLEN_MULTIPLIERS = (2.45, 2.55, 2.65)
MELT_MULTIPLIER = 1.10
POINTS = [
    {
        "delta_t": delta_t,
        "melt_f_stage2_multiplier": MELT_MULTIPLIER,
        "glen_a_stage2_multiplier": glen_multiplier,
    }
    for delta_t in DELTA_T_VALUES
    for glen_multiplier in GLEN_MULTIPLIERS
]


def _configured(points: list[dict], *, workers: int, purpose: str) -> dict:
    config = _base_config()
    config["inputs"]["declared_preflight_result"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    config["pilot_design"]["block"] = BLOCK
    config["pilot_design"]["purpose"] = purpose
    config["pilot_design"]["systems"] = {"991": {"parameter_pairs": points}}
    config["execution"]["output_root"] = f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    config["execution"]["work_root"] = "w191_response_target_991_v1"
    config["execution"]["max_workers"] = workers
    return config


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": "Does the real OGGM model confirm the local response-surface prediction of a formal separation-accepted system-991 curve near delta_T=-0.137 K and glen_a x2.57?",
        "source_results": "validation_data/results/paleo_surface_validation_v0/lia_191_high_glen_recenter_991_v1/scores.csv",
        "execution_driver": "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_glen_tradeoff_pilot",
        "systems": [991],
        "task_count": len(POINTS),
        "response_fit": {
            "peak_m": [11808.333333333338, -7500.000000000002, -500.0000000000004],
            "year_2000_m": [10758.333333333338, -12500.00000000001, -1000.0000000000008],
            "coefficient_order": ["intercept", "delta_t", "glen_a_stage2_multiplier"],
            "continuous_target_solution": {"delta_t": -0.1366666666666652, "glen_a_stage2_multiplier": 2.5666666666666518},
            "target_discrete_lengths_m": {"forward_peak": 11550.0, "year_2000": 9900.0},
        },
        "design_rule": {
            "delta_t_values_k": list(DELTA_T_VALUES),
            "glen_a_stage2_multipliers": list(GLEN_MULTIPLIERS),
            "melt_f_stage2_multiplier": MELT_MULTIPLIER,
            "logic": "real-model verification grid around the fitted continuous target",
        },
        "frozen": {
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
        "formal_acceptance": {
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "lia_anchor": "forward_peak",
            "observation_domain_gate": True,
        },
        "preflight": {
            "lia_unique_id": 991,
            "delta_t": -0.135,
            "melt_f_stage2_multiplier": MELT_MULTIPLIER,
            "glen_a_stage2_multiplier": 2.55,
        },
        "max_workers": 12,
        "authorization_note": "Continuation of the user-approved 991-only iterative parameter-estimation loop.",
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    full = _configured(POINTS, workers=12, purpose="991-only real-OGGM verification of the response-surface target near delta_T=-0.137 K and glen_a x2.57; keep all other controls frozen.")
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight_point = [{"delta_t": -0.135, "melt_f_stage2_multiplier": MELT_MULTIPLIER, "glen_a_stage2_multiplier": 2.55}]
    preflight = _configured(preflight_point, workers=1, purpose="One-chain preflight at the fitted system-991 response target.")
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "task_count": len(POINTS),
        "config_full": str(full_path),
        "config_preflight": str(preflight_path),
        "contract_json": str(OUTPUT_ROOT / "experiment_contract.json"),
    }
    (OUTPUT_ROOT / "design_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
