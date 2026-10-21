"""Build a high-glen recenter grid for system 991.

All saved 991 chains remain outside the formal separation tolerance. The
nearest chain misses both endpoints by less than one 100 m-scale adjustment,
but its peak-to-2000 amplitude is about 200 m too small for the discrete 50 m
length observations to pass both endpoints together. The preceding bounded
glen_a test increased that amplitude by about 100 m between x1.0 and x1.2.
This batch therefore tests x1.4-x1.6 and uses delta_T only to recenter the
curve. Melt factor and every other control remain fixed.
"""

from __future__ import annotations

import json

from .build_lia_191_local_refine_991_5143_v1 import _base_config
from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_high_glen_recenter_991_v1"
BLOCK = "B_high_glen_recenter_991_v1"
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

DELTA_T_VALUES = (-0.04, -0.05, -0.06)
GLEN_MULTIPLIERS = (1.40, 1.50, 1.60)
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
    config["execution"]["work_root"] = "w191_high_glen_recenter_991_v1"
    config["execution"]["max_workers"] = workers
    return config


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": (
            "Can higher glen_a create the approximately 200 m additional "
            "peak-to-2000 amplitude needed for system 991 while delta_T "
            "recenters both endpoints?"
        ),
        "source_results": [
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_glen_refine_991_v1/scores.csv",
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_glen_bridge_refine_991_v2/scores.csv",
        ],
        "execution_driver": (
            "Himalaya_greatGlacier.With_Rec_climate_organized."
            "ice_surface_reconstruction.drivers.lia_191_glen_tradeoff_pilot"
        ),
        "systems": [991],
        "task_count": len(POINTS),
        "design_rule": {
            "delta_t_values_k": list(DELTA_T_VALUES),
            "glen_a_stage2_multipliers": list(GLEN_MULTIPLIERS),
            "melt_f_stage2_multiplier": MELT_MULTIPLIER,
            "logic": (
                "increase dynamic amplitude with glen_a; use a narrow colder "
                "delta_T range to recenter the curve; do not open a new family"
            ),
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
            "delta_t": -0.05,
            "melt_f_stage2_multiplier": MELT_MULTIPLIER,
            "glen_a_stage2_multiplier": 1.50,
        },
        "max_workers": 12,
        "authorization_note": (
            "Continuation of the user-approved 991-only glen_a parameter-family loop."
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
            "991-only high-glen amplitude test at x1.4/x1.5/x1.6, crossed "
            "with delta_T -0.04/-0.05/-0.06 K; melt_f fixed at x1.10."
        ),
    )
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    preflight_point = [
        {
            "delta_t": -0.05,
            "melt_f_stage2_multiplier": MELT_MULTIPLIER,
            "glen_a_stage2_multiplier": 1.50,
        }
    ]
    preflight = _configured(
        preflight_point,
        workers=1,
        purpose="One-chain preflight for the 991 high-glen recenter grid.",
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
