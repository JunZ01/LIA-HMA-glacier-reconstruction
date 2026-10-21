"""Build a nine-point bridge for system 991 after the glen_a pilot.

The first glen_a block found no formal acceptance but identified a narrow
tradeoff: glen_a x1.10-x1.20 moves the modern endpoint toward its tolerance
while the LIA peak remains close. This follow-up samples three diagonal
delta_T/melt_f interpolants between the two saved near-miss anchors at three
fixed glen_a levels. No other parameter family is opened.
"""

from __future__ import annotations

import json

from .build_lia_191_local_refine_991_5143_v1 import _base_config
from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_glen_bridge_refine_991_v2"
BLOCK = "B_glen_bridge_refine_991_v2"
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

BRIDGE_PAIRS = (
    (-0.028828125, 1.11),
    (-0.034218750, 1.12),
    (-0.039609375, 1.13),
)
GLEN_MULTIPLIERS = (1.10, 1.15, 1.20)
POINTS = [
    {
        "delta_t": delta_t,
        "melt_f_stage2_multiplier": melt_multiplier,
        "glen_a_stage2_multiplier": glen_multiplier,
    }
    for delta_t, melt_multiplier in BRIDGE_PAIRS
    for glen_multiplier in GLEN_MULTIPLIERS
]


def _configured(points: list[dict], *, workers: int, purpose: str) -> dict:
    config = _base_config()
    config["inputs"]["declared_preflight_result"] = (
        f"validation_data/results/paleo_surface_validation_v0/"
        f"{EXPERIMENT_NAME}/experiment_contract.json"
    )
    config["pilot_design"]["block"] = BLOCK
    config["pilot_design"]["purpose"] = purpose
    config["pilot_design"]["systems"] = {
        "991": {"parameter_pairs": points}
    }
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    )
    config["execution"]["work_root"] = "w191_glen_bridge_refine_991_v2"
    config["execution"]["max_workers"] = workers
    return config


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": (
            "Does system 991 contain a formal separation-accepted curve along "
            "the diagonal delta_T/melt_f bridge when glen_a is fixed at "
            "1.10, 1.15, or 1.20 times Stage-2?"
        ),
        "source_results": [
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_bracket_refine_991_v1/scores.csv",
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_bridge_refine_991_v1/scores.csv",
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_glen_refine_991_v1/scores.csv",
        ],
        "execution_driver": (
            "Himalaya_greatGlacier.With_Rec_climate_organized."
            "ice_surface_reconstruction.drivers.lia_191_glen_tradeoff_pilot"
        ),
        "systems": [991],
        "task_count": len(POINTS),
        "design_rule": {
            "source_anchor_peak_branch": {
                "delta_t": -0.0234375,
                "melt_f_stage2_multiplier": 1.10,
            },
            "source_anchor_modern_branch": {
                "delta_t": -0.045,
                "melt_f_stage2_multiplier": 1.14,
            },
            "bridge_pairs": [
                {"delta_t": delta_t, "melt_f_stage2_multiplier": melt}
                for delta_t, melt in BRIDGE_PAIRS
            ],
            "glen_a_stage2_multipliers": list(GLEN_MULTIPLIERS),
            "logic": (
                "three 25/50/75-percent diagonal interpolants between the two "
                "saved near-miss branches, crossed with three glen_a levels"
            ),
        },
        "frozen": {
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
            "delta_t": -0.03421875,
            "melt_f_stage2_multiplier": 1.12,
            "glen_a_stage2_multiplier": 1.15,
        },
        "max_workers": 12,
        "authorization_note": (
            "User approved the follow-up after reviewing the completed "
            "991-only glen_a diagnostic."
        ),
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    full = _configured(
        POINTS,
        workers=12,
        purpose=(
            "991-only nine-point diagonal delta_T/melt_f bridge at "
            "glen_a x1.10/x1.15/x1.20. Keep all other controls frozen."
        ),
    )
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    preflight_point = [
        {
            "delta_t": -0.03421875,
            "melt_f_stage2_multiplier": 1.12,
            "glen_a_stage2_multiplier": 1.15,
        }
    ]
    preflight = _configured(
        preflight_point,
        workers=1,
        purpose="One-chain preflight for the 991 nine-point glen bridge.",
    )
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "task_count": len(POINTS),
        "config_full": str(full_path),
        "config_preflight": str(preflight_path),
        "contract_json": str(OUTPUT_ROOT / "experiment_contract.json"),
    }
    (OUTPUT_ROOT / "design_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
