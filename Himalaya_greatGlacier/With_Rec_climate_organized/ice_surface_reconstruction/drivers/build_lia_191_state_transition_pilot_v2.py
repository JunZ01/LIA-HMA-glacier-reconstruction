"""Build current-controller configs for the state-transition midpoint pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config"
BASE_CONFIG = CONFIG_DIR / "lia_191_controller_v1.json"
EXPERIMENT = "lia_191_state_transition_midpoint_pilot_v2"
OUTPUT = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT
GROUPS = {
    "cold_m0p25": {"systems": [2378], "delta_t": -0.25},
    "cold_p0p25": {"systems": [551, 660, 2595], "delta_t": 0.25},
    "cold_p0p75": {"systems": [4369], "delta_t": 0.75},
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    targets_path = ROOT / base["inputs"]["system_targets"]
    targets = pd.read_csv(targets_path)
    configs: list[str] = []
    for name, spec in GROUPS.items():
        ids = set(spec["systems"])
        subset = targets[targets["lia_unique_id"].astype(int).isin(ids)].copy()
        if set(subset["lia_unique_id"].astype(int)) != ids:
            raise RuntimeError(f"missing target in {name}")
        group_output = OUTPUT / name
        group_output.mkdir(parents=True, exist_ok=True)
        subset_path = group_output / "system_targets.csv"
        subset.to_csv(subset_path, index=False)

        config = json.loads(json.dumps(base))
        config["inputs"]["system_targets"] = str(subset_path.relative_to(ROOT)).replace("\\", "/")
        config["acceptance"] = {
            "mode": "formal-separation-dual-gate",
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "normalization": "absolute LIA-RGI length separation",
            "lia_anchor": "forward_peak",
            "corridor_buffer_fraction": 0.2,
            "corridor_is_hard_gate": False,
            "minimum_at_2000_is_hard_gate": False,
            "modern_mb_is_hard_gate": False,
        }
        config["search"]["blocks"] = [{
            "name": "B_state_transition_midpoint_v2",
            "enabled": True,
            "purpose": "Midpoint test at the stable-short versus colder-runaway spin-up transition.",
            "auto_advance": False,
            "design": {
                "delta_t": [float(spec["delta_t"])],
                "melt_f": {"source": "stage2", "column": "melt_f"},
                "prcp_fac_multiplier": 1.0,
                "glen_a": {"source": "stage2", "column": "glen_a_pa_3_s_1"},
                "temp_melt": -1.0,
            },
        }]
        config["existing_decisions"] = {}
        config["execution"]["output_root"] = str(group_output.relative_to(ROOT)).replace("\\", "/")
        config["execution"]["work_root"] = f"w191_state_transition_midpoint_v2/{name}"
        config["execution"]["max_workers"] = len(ids)
        config_path = CONFIG_DIR / f"{EXPERIMENT}_{name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        configs.append(str(config_path))

    contract = {
        "schema_version": 2,
        "declared_at": "2026-08-16",
        "experiment": EXPERIMENT,
        "mechanism": "full-chain spin-up delta-T stability-boundary midpoint",
        "groups": GROUPS,
        "frozen": {
            "melt_f": "Stage-2",
            "prcp_fac": "Stage-2",
            "glen_a": "Stage-2",
            "temp_melt_c": -1.0,
            "precipitation_anomalies": "off",
            "spinup_window": "1441-1470",
            "forward_period": "1470-2000",
        },
        "promotion_rule": (
            "At least two representatives must be stable, observation-valid, and improve "
            "the formal objective, or at least one representative must formally pass."
        ),
    }
    (OUTPUT / "experiment_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "design_summary.json").write_text(
        json.dumps({"configs": configs, "tasks": 5, "launched_oggm": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"configs": configs, "tasks": 5}, indent=2))


if __name__ == "__main__":
    main()
