"""Build the one permitted glen_a block for four Layer-2 analog systems."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root
from .build_lia_191_local_refine_991_5143_v1 import _base_config


SOURCE_EXPERIMENT = "lia_191_analog_melt_delta_after_991_v1"
EXPERIMENT_NAME = "lia_191_analog_glen_after_991_v1"
BLOCK = "B_analog_glen_after_991_v1"
GLEN_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)
EXPECTED_SYSTEMS = {214, 2569, 2622, 4631}

SOURCE_ROOT = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / SOURCE_EXPERIMENT
)
OUTPUT_ROOT = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / EXPERIMENT_NAME
)
CONFIG_DIR = (
    project_root()
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
    / EXPERIMENT_NAME
)


def _selected() -> pd.DataFrame:
    selected = pd.read_csv(SOURCE_ROOT / "layer3_glen_a_candidates.csv")
    ids = set(pd.to_numeric(selected["lia_unique_id"], errors="raise").astype(int))
    if ids != EXPECTED_SYSTEMS or len(selected) != len(EXPECTED_SYSTEMS):
        raise ValueError(f"unexpected Layer-3 selection: {sorted(ids)}")
    if not selected["total_length_pass"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    ).all():
        raise ValueError("every Layer-3 candidate must pass the total-length gate")
    if selected["separation_pass"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    ).any():
        raise ValueError("a separation-passing chain must be frozen, not sent to Layer 3")
    return selected.sort_values("lia_unique_id").reset_index(drop=True)


def _systems(selected: pd.DataFrame) -> dict[str, dict]:
    systems: dict[str, dict] = {}
    for row in selected.itertuples(index=False):
        system_id = int(row.lia_unique_id)
        points = [
            {
                "delta_t": float(row.delta_t),
                "melt_f_stage2_multiplier": float(row.melt_f_stage2_multiplier),
                "glen_a_stage2_multiplier": glen_multiplier,
            }
            for glen_multiplier in GLEN_MULTIPLIERS
        ]
        systems[str(system_id)] = {"parameter_pairs": points}
    return systems


def _configured(systems: dict[str, dict], workers: int, purpose: str) -> dict:
    config = _base_config()
    config["inputs"]["declared_preflight_result"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json"
    )
    config["acceptance"]["report_total_length_10pct"] = True
    config["acceptance"]["freeze_requires_both_total_and_separation_gates"] = True
    config["pilot_design"]["block"] = BLOCK
    config["pilot_design"]["purpose"] = purpose
    config["pilot_design"]["systems"] = systems
    config["pilot_design"]["glen_a_source"] = "explicit-stage2-multiplier"
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    )
    config["execution"]["work_root"] = "w191_analog_glen_991_v1"
    config["execution"]["max_workers"] = workers
    return config


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    selected = _selected()
    selected.to_csv(OUTPUT_ROOT / "layer3_source_selection.csv", index=False)
    systems = _systems(selected)
    task_count = sum(len(value["parameter_pairs"]) for value in systems.values())

    base_points = {
        str(int(row.lia_unique_id)): {
            "delta_t": float(row.delta_t),
            "melt_f_stage2_multiplier": float(row.melt_f_stage2_multiplier),
            "source_task_id": row.task_id,
            "source_e_lia_sep": float(row.e_lia_sep),
            "source_e_rgi_sep": float(row.e_rgi_sep),
            "source_amplitude_ratio": float(row.amplitude_ratio_to_target),
        }
        for row in selected.itertuples(index=False)
    }
    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": (
            "Can a bounded glen_a increase at each system's best Layer-2 "
            "melt_f-delta_T point close the remaining LIA separation error "
            "without losing the RGI endpoint?"
        ),
        "source_results": (
            f"validation_data/results/paleo_surface_validation_v0/{SOURCE_EXPERIMENT}/"
            "layer3_glen_a_candidates.csv"
        ),
        "systems": sorted(EXPECTED_SYSTEMS),
        "task_count": task_count,
        "base_points": base_points,
        "design_rule": {
            "glen_a_stage2_multipliers": list(GLEN_MULTIPLIERS),
            "delta_t_and_melt_f": "frozen at each system's best Layer-2 formal objective",
            "logic": (
                "all four systems already pass both total-length endpoints but "
                "remain short at the LIA peak under separation normalization; "
                "991 required glen_a near x2.9 to close the same response gap"
            ),
        },
        "frozen": {
            "prcp_fac": "Stage-2",
            "temp_melt_c": -1.0,
            "precipitation_anomaly_mode": "off",
            "spinup_window": "B 1441-1470",
            "forward_period": "1470-2000",
            "geometry_and_observation_operator": "unchanged",
        },
        "acceptance_reporting": {
            "formal_separation_tolerance": 0.20,
            "active_total_length_tolerance": 0.10,
            "freeze_only_if_both_pass": True,
        },
        "stop_rule": (
            "one four-point glen_a block only; if no pass, allow at most one "
            "delta-T recenter around a demonstrated bracket, otherwise close"
        ),
        "preflight": {
            "lia_unique_id": 2569,
            "delta_t": -0.500,
            "melt_f_stage2_multiplier": 1.20,
            "glen_a_stage2_multiplier": 2.0,
        },
        "max_workers": 12,
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    full = _configured(
        systems,
        12,
        "Four-system Layer-3 glen_a block after the completed analog melt_f-delta_T batch.",
    )
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    source_2569 = selected[selected["lia_unique_id"].astype(int).eq(2569)].iloc[0]
    if not np.isclose(float(source_2569["delta_t"]), -0.5) or not np.isclose(
        float(source_2569["melt_f_stage2_multiplier"]), 1.2
    ):
        raise ValueError("system 2569 preflight base point changed")
    preflight_systems = {
        "2569": {
            "parameter_pairs": [
                {
                    "delta_t": -0.5,
                    "melt_f_stage2_multiplier": 1.2,
                    "glen_a_stage2_multiplier": 2.0,
                }
            ]
        }
    }
    preflight = _configured(
        preflight_systems,
        1,
        "Mandatory one-chain preflight for the four-system glen_a block.",
    )
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "systems": sorted(EXPECTED_SYSTEMS),
        "task_count": task_count,
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
