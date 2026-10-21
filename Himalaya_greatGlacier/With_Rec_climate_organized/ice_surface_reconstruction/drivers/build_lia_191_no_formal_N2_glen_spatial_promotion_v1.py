"""Build the promoted N2 RGI-reachable spatial glen_a batch."""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
DRIVER_ROOT = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
)
SOURCE_CONFIG = (
    DRIVER_ROOT
    / "config"
    / "lia_191_no_formal_N2_glen_spatial_pilot_v1"
    / "lia_191_no_formal_N2_glen_spatial_pilot_v1.json"
)
INVENTORY = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_class_round2_v1"
    / "chain_dual_gate_inventory.csv"
)
EXPERIMENT = "lia_191_no_formal_N2_glen_spatial_promotion_v1"
BLOCK = "B_nf_N2_ga_prom_v1"
SYSTEM_IDS = [
    890,
    1026,
    1631,
    1856,
    2005,
    2534,
    2576,
    2645,
    2827,
    3137,
    3175,
    3523,
    3897,
    3899,
    4565,
    4607,
    5014,
    5062,
    5127,
    5142,
    5146,
    5510,
    5524,
    5598,
    5770,
]
PREFLIGHT_SYSTEM = 890


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(row: dict[str, str], key: str, default: float | None = None) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError(f"missing {key}")
        return float(default)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {key}: {value}")
    return result


def _select_anchors() -> dict[int, dict[str, object]]:
    with INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    selected: dict[int, dict[str, object]] = {}
    for system_id in SYSTEM_IDS:
        candidates = []
        for row in rows:
            if int(row["lia_unique_id"]) != system_id:
                continue
            if not _truth(row.get("stable_chain")):
                continue
            if not _truth(row.get("audit_length_observation_valid")):
                continue
            if _number(row, "e_rgi_total") > 0.10:
                continue
            candidates.append(row)
        if not candidates:
            raise RuntimeError(f"no stable RGI-reachable anchor for {system_id}")
        candidates.sort(
            key=lambda row: (
                _number(row, "e_lia_sep_formal"),
                _number(row, "objective_sep_formal"),
                _number(row, "e_rgi_total"),
            )
        )
        row = candidates[0]
        selected[system_id] = {
            "lia_unique_id": system_id,
            "source_task_id": row["task_id"],
            "source_fingerprint": row["fingerprint"],
            "source_label": row["score_source"],
            "delta_t": _number(row, "delta_t"),
            "melt_f_stage2_multiplier": _number(
                row, "melt_f_stage2_multiplier", 1.0
            ),
            "prcp_fac_multiplier": _number(row, "prcp_fac_multiplier", 1.0),
            "source_glen_a_stage2_multiplier": _number(
                row, "glen_a_stage2_multiplier", 1.0
            ),
            "temp_melt": _number(row, "temp_melt", -1.0),
            "anchor_e_lia_sep": _number(row, "e_lia_sep_formal"),
            "anchor_e_rgi_sep": _number(row, "e_rgi_sep_formal"),
            "anchor_e_lia_total": _number(row, "e_lia_total"),
            "anchor_e_rgi_total": _number(row, "e_rgi_total"),
        }
    return selected


def _point(anchor: dict[str, object]) -> dict[str, float]:
    return {
        "delta_t": float(anchor["delta_t"]),
        "melt_f_stage2_multiplier": float(
            anchor["melt_f_stage2_multiplier"]
        ),
        "glen_a_stage2_multiplier": 0.75,
        "prcp_fac_multiplier": float(anchor["prcp_fac_multiplier"]),
        "temp_melt": float(anchor["temp_melt"]),
    }


def main() -> None:
    anchors = _select_anchors()
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    full = deepcopy(source)
    purpose = (
        "Promote the representative N2 direction: apply Stage-2 glen_a x0.75 "
        "once at each best stable RGI-reachable anchor for the 25 outlet-pass "
        "systems."
    )
    full["inputs"]["elevation_band_root"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_no_formal_N2_promotion_v1"
    )
    full["inputs"]["geometry_probe"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_downstream_geometry_probe_no_formal_N2_promotion_v1/"
        "downstream_geometry_probe_summary.csv"
    )
    full["inputs"]["geometry_preflight"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_no_formal_N2_promotion_v1/"
        "spatial_downstream_preflight_summary.json"
    )
    full["pilot_design"] = {
        "block": BLOCK,
        "purpose": purpose,
        "systems": {
            str(system_id): {"parameter_pairs": [_point(anchors[system_id])]}
            for system_id in SYSTEM_IDS
        },
        "prcp_fac_multiplier": 1.0,
        "temp_melt": -1.0,
        "glen_a_stage2_multiplier": 0.75,
        "anchor_policy": (
            "best stable observation-valid saved chain with e_rgi_total<=0.10, "
            "ranked by e_lia_sep_formal"
        ),
    }
    full["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + EXPERIMENT
        ),
        "work_root": "w191_nf_n2_prom",
        "allow_runs": True,
        "max_workers": 12,
        "timeout_seconds": 1200,
    }

    preflight = deepcopy(full)
    preflight["pilot_design"]["purpose"] = (
        "Mandatory system-890 dynamics preflight for the promoted N2 batch."
    )
    preflight["pilot_design"]["systems"] = {
        str(PREFLIGHT_SYSTEM): {
            "parameter_pairs": [_point(anchors[PREFLIGHT_SYSTEM])]
        }
    }
    preflight["execution"]["max_workers"] = 1

    config_dir = DRIVER_ROOT / "config" / EXPERIMENT
    result_dir = (
        ROOT
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    full_path = config_dir / f"{EXPERIMENT}.json"
    preflight_path = config_dir / f"{EXPERIMENT}_preflight.json"
    full_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    anchor_fields = list(next(iter(anchors.values())).keys())
    with (result_dir / "anchor_selection.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=anchor_fields)
        writer.writeheader()
        writer.writerows(anchors[system_id] for system_id in SYSTEM_IDS)
    (result_dir / "anchor_selection.json").write_text(
        json.dumps(list(anchors.values()), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "reachability_class": "N2_rgi_only_reachable",
        "question": purpose,
        "system_ids": SYSTEM_IDS,
        "planned_task_count": len(SYSTEM_IDS),
        "glen_a_stage2_multiplier": 0.75,
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "mandatory_preflight": {
            "lia_unique_id": PREFLIGHT_SYSTEM,
            "config": str(preflight_path),
        },
        "full_config": str(full_path),
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+"
            "20% separation endpoints"
        ),
        "automatic_freeze": True,
        "excluded_geometry_gate_systems": [
            2898,
            512,
            1499,
            2473,
            3082,
            3495,
            3808,
            3904,
            3910,
            3914,
            4637,
        ],
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
