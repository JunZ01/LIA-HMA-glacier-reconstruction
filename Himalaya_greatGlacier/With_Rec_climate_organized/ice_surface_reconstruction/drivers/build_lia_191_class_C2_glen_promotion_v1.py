"""Promote the improving C2 glen_a x0.75 direction to the remaining class."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import build_lia_191_class_representative_pilots_v1 as _base


EXPERIMENT = "lia_191_class_C2_glen_promotion_v1"
CLASS_NAME = "C2_peak_short_rgi_near"
CLASSIFICATION = (
    _base.PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "total_length_only_batch_router_v1"
    / "system_classification.csv"
)
PILOT_SCORES = (
    _base.PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_class_C2_glen_pilot_v1"
    / "scores.csv"
)


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _finite(value: object, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if math.isfinite(float(numeric)) else default


def main() -> None:
    systems = pd.read_csv(CLASSIFICATION)
    pilot = pd.read_csv(PILOT_SCORES)
    pilot["glen_a_stage2_multiplier"] = pd.to_numeric(
        pilot["glen_a_stage2_multiplier"], errors="coerce"
    )
    direction = pilot.loc[
        pilot["glen_a_stage2_multiplier"].sub(0.75).abs().lt(1e-12)
        & _bool(pilot["spinup_converged"])
        & _bool(pilot["length_observation_valid"])
    ].copy()
    anchors = systems.set_index("lia_unique_id")["objective_sep_formal"]
    direction["anchor_objective"] = direction["lia_unique_id"].map(anchors)
    direction["improved"] = direction["objective_sep"] < direction["anchor_objective"]
    improvement_count = int(direction["improved"].sum())
    if improvement_count < 2:
        raise RuntimeError(
            f"C2 promotion requires >=2 improving representatives; found {improvement_count}"
        )

    selected = systems.loc[
        systems["response_class"].eq(CLASS_NAME)
        & systems["batch_state"].eq("eligible_class_pilot")
        & ~_bool(systems["is_representative"])
    ].copy()
    if len(selected) != 5:
        raise RuntimeError(f"Expected five remaining C2 systems, found {len(selected)}")

    config = json.loads(_base.BASE_CONFIG.read_text(encoding="utf-8"))
    config["pilot_design"] = {
        "block": "B_cls_C2_ga_prom_v1",
        "purpose": (
            "Promote the C2 representative-pilot improving glen_a x0.75 "
            "direction to the five remaining eligible systems, retaining each "
            "system's saved delta-T, melt_f, prcp_fac, and temp_melt anchor."
        ),
        "systems": {},
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-times-anchor-times-class-promotion-0.75",
        "temp_melt": -1.0,
    }
    for _, row in selected.iterrows():
        anchor_glen = _finite(row.get("glen_a_stage2_multiplier"), 1.0)
        config["pilot_design"]["systems"][str(int(row["lia_unique_id"]))] = {
            "parameter_pairs": [
                {
                    "delta_t": float(row["delta_t"]),
                    "melt_f_stage2_multiplier": _finite(
                        row.get("melt_f_stage2_multiplier"), 1.0
                    ),
                    "glen_a_stage2_multiplier": anchor_glen * 0.75,
                    "prcp_fac_multiplier": _finite(row["prcp_fac_multiplier"], 1.0),
                    "temp_melt": _finite(row["temp_melt"], -1.0),
                }
            ]
        }
    config["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}",
        "work_root": "w191_c2_prom",
        "allow_runs": True,
        "max_workers": 5,
        "timeout_seconds": 1200,
    }
    config["inputs"].pop("declared_preflight_result", None)

    config_dir = _base.DRIVER_ROOT / "config" / EXPERIMENT
    result_dir = (
        _base.PROJECT_ROOT
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{EXPERIMENT}.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "response_class": CLASS_NAME,
        "representative_x0p75_improvement_count": improvement_count,
        "representative_x0p75_evaluated_count": int(len(direction)),
        "promoted_system_ids": selected["lia_unique_id"].astype(int).tolist(),
        "planned_task_count": int(len(selected)),
        "tested_family": "glen_a",
        "tested_stage2_multiplier": 0.75,
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+20% separation endpoints"
        ),
        "config": str(config_path),
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    direction.to_csv(result_dir / "representative_direction_evidence.csv", index=False)
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
