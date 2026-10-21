"""Build a five-system batch testing the mechanism that resolved system 991.

The selected systems are the five nearest unresolved Layer-1 responses to
system 991 before its local refinement: a stable RGI endpoint, a short LIA
peak, and an amplitude ratio between 0.3 and 0.85.  The experiment first tests
the Layer-2 mechanism only: colder delta-T together with higher melt_f.  It
does not change glen_a yet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root
from .build_lia_191_local_refine_991_5143_v1 import _base_config


EXPERIMENT_NAME = "lia_191_analog_melt_delta_after_991_v1"
BLOCK = "B_analog_melt_delta_after_991_v1"
ROUTING_TABLE = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "iterative_loop_spatial_downstream_delta_extension_v2"
    / "core191_iterative_loop_table.csv"
)
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

ANCHORS = {
    2569: -0.125,
    2645: 0.375,
    214: -0.250,
    4631: 0.250,
    2622: 0.250,
}
DELTA_OFFSETS = (-0.125, -0.250, -0.375)
MELT_MULTIPLIERS = (1.10, 1.20)


def _candidate_ranking() -> pd.DataFrame:
    routing = pd.read_csv(ROUTING_TABLE)
    base = routing.loc[routing["lia_unique_id"].eq(991)].iloc[0]
    candidates = routing[
        routing["next_route"].eq("hold_after_layer1")
        & routing["target_direction"].eq("retreat")
        & routing["best_stable_e_rgi_total"].le(0.10)
        & routing["best_stable_e_lia_total"].gt(0.10)
        & routing["best_stable_e_lia_total"].le(0.25)
        & routing["best_stable_amplitude_ratio_to_target"].between(0.30, 0.85)
        & ~routing["lia_unique_id"].eq(991)
    ].copy()
    candidates["similarity_distance_to_991"] = np.sqrt(
        ((candidates["best_stable_e_lia_total"] - base["best_stable_e_lia_total"]) / 0.15) ** 2
        + ((candidates["best_stable_e_rgi_total"] - base["best_stable_e_rgi_total"]) / 0.10) ** 2
        + ((candidates["best_stable_amplitude_ratio_to_target"] - base["best_stable_amplitude_ratio_to_target"]) / 0.55) ** 2
    )
    ranked = candidates.sort_values(
        ["similarity_distance_to_991", "lia_unique_id"]
    ).reset_index(drop=True)
    selected = ranked.head(5).copy()
    if set(selected["lia_unique_id"].astype(int)) != set(ANCHORS):
        raise ValueError(
            "the declared five-system selection no longer matches the read-only ranking"
        )
    for row in selected.itertuples(index=False):
        expected = ANCHORS[int(row.lia_unique_id)]
        if not np.isclose(float(row.best_stable_delta_t), expected):
            raise ValueError(
                f"anchor changed for {row.lia_unique_id}: {row.best_stable_delta_t} != {expected}"
            )
    return selected


def _systems() -> dict[str, dict]:
    systems: dict[str, dict] = {}
    for system_id, anchor in ANCHORS.items():
        points = []
        for offset in DELTA_OFFSETS:
            for melt_multiplier in MELT_MULTIPLIERS:
                points.append(
                    {
                        "delta_t": round(anchor + offset, 6),
                        "melt_f_stage2_multiplier": melt_multiplier,
                        "glen_a_stage2_multiplier": 1.0,
                    }
                )
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
    config["pilot_design"]["glen_a_source"] = "stage2-frozen"
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}"
    )
    config["execution"]["work_root"] = "w191_analog_md_991_v1"
    config["execution"]["max_workers"] = workers
    return config


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    selected = _candidate_ranking()
    selected.to_csv(OUTPUT_ROOT / "analog_system_selection.csv", index=False)
    systems = _systems()
    task_count = sum(len(item["parameter_pairs"]) for item in systems.values())

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": (
            "Does the bounded mechanism that first improved system 991—colder "
            "delta-T paired with higher melt_f—also improve five quantitatively "
            "similar unresolved systems before any glen_a layer is opened?"
        ),
        "source_routing": str(ROUTING_TABLE),
        "selection_rule": {
            "route": "hold_after_layer1",
            "target_direction": "retreat",
            "rgi_total_error_max": 0.10,
            "lia_total_error_range": [0.10, 0.25],
            "amplitude_ratio_range": [0.30, 0.85],
            "ranking": "Euclidean distance to system 991 in scaled LIA error, RGI error, and amplitude ratio",
            "selected_systems": list(ANCHORS),
        },
        "execution_driver": (
            "Himalaya_greatGlacier.With_Rec_climate_organized."
            "ice_surface_reconstruction.drivers.lia_191_glen_tradeoff_pilot"
        ),
        "task_count": task_count,
        "design_rule": {
            "delta_t": "each Layer-1 anchor plus -0.125/-0.250/-0.375 K",
            "melt_f_stage2_multipliers": list(MELT_MULTIPLIERS),
            "glen_a_stage2_multiplier": 1.0,
            "logic": (
                "colder delta-T raises the missing LIA peak while higher melt_f "
                "tests whether the year-2000 endpoint can remain near RGI"
            ),
        },
        "frozen": {
            "prcp_fac": "Stage-2",
            "glen_a": "Stage-2",
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
        "preflight": {
            "lia_unique_id": 2569,
            "delta_t": -0.250,
            "melt_f_stage2_multiplier": 1.10,
            "release_gate": (
                "technical_failure=false; spinup_converged=true; "
                "spinup_stability_fraction<=0.02; observation-domain valid"
            ),
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
        "Five-system analog Layer-2 batch after the accepted 991 mechanism; delta-T and melt_f only.",
    )
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    preflight_systems = {
        "2569": {
            "parameter_pairs": [
                {
                    "delta_t": -0.250,
                    "melt_f_stage2_multiplier": 1.10,
                    "glen_a_stage2_multiplier": 1.0,
                }
            ]
        }
    }
    preflight = _configured(
        preflight_systems,
        1,
        "Mandatory one-chain preflight for the five-system analog batch.",
    )
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "systems": list(ANCHORS),
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
