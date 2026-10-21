"""Build regular-geometry representative pilots for response classes C1-C3.

The class router supplies per-system anchors.  This builder changes one
parameter family per class and does not advance OGGM.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DRIVER_ROOT = (
    PROJECT_ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
)
QUEUE = (
    PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "total_length_only_batch_router_v1"
    / "representative_pilot_queue.csv"
)
BASE_CONFIG = (
    DRIVER_ROOT
    / "config"
    / "lia_191_analog_glen_after_991_v1"
    / "lia_191_analog_glen_after_991_v1.json"
)


DESIGNS = {
    "C1_peak_short_rgi_long": {
        "experiment": "lia_191_class_C1_glen_pilot_v1",
        "block": "B_cls_C1_ga_v1",
        "family": "glen_a",
        "multipliers": [0.75, 1.25],
        "purpose": (
            "Class C1 representative probe: peak short and year-2000 long. "
            "Test both glen_a directions at each saved anchor to determine "
            "whether retreat amplitude can increase without losing endpoints."
        ),
    },
    "C2_peak_short_rgi_near": {
        "experiment": "lia_191_class_C2_glen_pilot_v1",
        "block": "B_cls_C2_ga_v1",
        "family": "glen_a",
        "multipliers": [0.75, 1.25],
        "purpose": (
            "Class C2 representative probe: year 2000 near RGI and peak short. "
            "Test both glen_a directions at each saved anchor to determine "
            "whether the historical peak can rise while retaining the endpoint."
        ),
    },
    "C3_both_short": {
        "experiment": "lia_191_class_C3_prcp_pilot_v1",
        "block": "B_cls_C3_pf_v1",
        "family": "prcp_fac",
        "multipliers": [1.10, 1.20],
        "purpose": (
            "Class C3 representative probe: both endpoints short. Increase the "
            "fixed-climatology prcp_fac at each saved anchor to test whether the "
            "curve level rises while the response amplitude is retained. "
            "Precipitation anomalies remain off."
        ),
    },
}


def _finite_or(value: object, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if math.isfinite(float(numeric)) else default


def _parameter_pairs(row: pd.Series, design: dict) -> list[dict]:
    delta_t = float(row["delta_t"])
    melt_multiplier = _finite_or(row["melt_f_stage2_multiplier"], 1.0)
    glen_multiplier = _finite_or(row.get("glen_a_stage2_multiplier"), 1.0)
    prcp_multiplier = _finite_or(row["prcp_fac_multiplier"], 1.0)
    temp_melt = _finite_or(row["temp_melt"], -1.0)
    pairs = []
    for multiplier in design["multipliers"]:
        item = {
            "delta_t": delta_t,
            "melt_f_stage2_multiplier": melt_multiplier,
            "glen_a_stage2_multiplier": glen_multiplier,
            "prcp_fac_multiplier": prcp_multiplier,
            "temp_melt": temp_melt,
        }
        if design["family"] == "glen_a":
            item["glen_a_stage2_multiplier"] = glen_multiplier * float(multiplier)
        elif design["family"] == "prcp_fac":
            item["prcp_fac_multiplier"] = prcp_multiplier * float(multiplier)
        else:
            raise ValueError(design["family"])
        pairs.append(item)
    return pairs


def main() -> None:
    queue = pd.read_csv(QUEUE)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    built = []

    for class_name, design in DESIGNS.items():
        selected = queue.loc[queue["response_class"].eq(class_name)].copy()
        if selected.empty:
            raise RuntimeError(f"No representatives selected for {class_name}")
        nonregular = selected.loc[
            selected["geometry_variant"].fillna("").astype(str).str.contains(
                "spatial", case=False
            )
        ]
        if not nonregular.empty:
            raise RuntimeError(
                f"{class_name} contains spatial geometry systems: "
                f"{nonregular['lia_unique_id'].astype(int).tolist()}"
            )

        experiment = design["experiment"]
        config = deepcopy(base)
        config["pilot_design"] = {
            "block": design["block"],
            "purpose": design["purpose"],
            "systems": {
                str(int(row["lia_unique_id"])): {
                    "parameter_pairs": _parameter_pairs(row, design)
                }
                for _, row in selected.iterrows()
            },
            "prcp_fac_multiplier": 1.0,
            "glen_a_source": "stage2-times-explicit-anchor-and-class-multiplier",
            "temp_melt": -1.0,
        }
        config["execution"] = {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": (
                f"validation_data/results/paleo_surface_validation_v0/{experiment}"
            ),
            "work_root": {
                "C1_peak_short_rgi_long": "w191_c1_ga",
                "C2_peak_short_rgi_near": "w191_c2_ga",
                "C3_both_short": "w191_c3_pf",
            }[class_name],
            "allow_runs": True,
            "max_workers": 12,
            "timeout_seconds": 1200,
        }
        config["inputs"].pop("declared_preflight_result", None)

        config_dir = DRIVER_ROOT / "config" / experiment
        result_dir = (
            PROJECT_ROOT
            / "validation_data"
            / "results"
            / "paleo_surface_validation_v0"
            / experiment
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{experiment}.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        representative_ids = selected["lia_unique_id"].astype(int).tolist()
        contract = {
            "schema_version": 1,
            "experiment": experiment,
            "response_class": class_name,
            "question": design["purpose"],
            "parameter_family": design["family"],
            "family_multipliers": design["multipliers"],
            "representative_system_ids": representative_ids,
            "planned_task_count": 2 * len(representative_ids),
            "anchor_policy": "best stable total-length-pass chain by formal separation objective",
            "fixed_contract": {
                "window": "B:1441-1470 spin-up;1470-2000 forward",
                "precipitation_anomalies": "off",
                "adaptive_spinup_stability": "preceding-200y volume change <=2%",
                "formal_gate": "technical+stable+observation-valid+10% total endpoints+20% separation endpoints",
                "max_workers": 12,
            },
            "release_rule": (
                "Run one representative task first. Release the remainder only "
                "if it is technically clean, spin-up stable, and observation-valid."
            ),
            "promotion_rule": (
                "Promote the class template only if at least two representatives "
                "show the same improving direction and the class technical failure "
                "rate is no greater than 30%."
            ),
            "config": str(config_path),
        }
        (result_dir / "experiment_contract.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        built.append(contract)

    print(json.dumps(built, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
