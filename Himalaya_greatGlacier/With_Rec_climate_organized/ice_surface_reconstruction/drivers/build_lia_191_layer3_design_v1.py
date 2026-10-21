"""Build bounded Layer-3 single-family designs from the saved routing diagnosis.

The design deliberately keeps one physical family per config.  It writes a
glen_a+delta-T block for systems classified as response-time/amplitude cases
and a separate prcp_fac+delta-T block for persistent level cases.  No OGGM task
is launched here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd


DELTA_RECENTER_OFFSETS = (-0.125, 0.0, 0.125)
GLEN_MULTIPLIERS = (0.75, 1.25)
PRCP_MULTIPLIERS = (1.05, 1.10)


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _config(
    base: dict,
    *,
    family: str,
    systems: dict[str, dict],
    output_name: str,
    work_root: str,
    purpose: str,
) -> dict:
    config = copy.deepcopy(base)
    config["pilot_design"] = {
        "block": f"B_layer3_{family}_delta_v1",
        "purpose": purpose,
        "parameter_family": family,
        "systems": systems,
        "temp_melt": -1.0,
    }
    config["execution"] = {
        **config["execution"],
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + output_name
        ),
        "work_root": work_root,
        "allow_runs": True,
        "max_workers": 12,
    }
    return config


def _system_design(row: pd.Series, family: str, *, preflight: bool) -> dict:
    center = float(row.delta_t if hasattr(row, "delta_t") else row["delta_t"])
    if preflight:
        deltas = [center]
    else:
        deltas = sorted({round(center + offset, 6) for offset in DELTA_RECENTER_OFFSETS})
    if family == "glen_a":
        values = [1.25] if preflight else list(GLEN_MULTIPLIERS)
    elif family == "prcp_fac":
        values = [1.10] if preflight else list(PRCP_MULTIPLIERS)
    else:
        raise ValueError(f"unsupported family {family}")
    return {"delta_t": deltas, "family_values": values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnosis-csv",
        type=Path,
        default=(
            project_root()
            / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/iterative_loop_v3_after_layer2/layer3_family_diagnosis_v2.csv"
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=(
            project_root()
            / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/lia_191_layer2_melt_delta_after_extension_v1.json"
        ),
    )
    args = parser.parse_args()

    diagnosis = pd.read_csv(args.diagnosis_csv)
    required = {"lia_unique_id", "family", "delta_t", "reason"}
    missing = sorted(required.difference(diagnosis.columns))
    if missing:
        raise ValueError(f"Layer-3 diagnosis missing columns: {missing}")
    if diagnosis["lia_unique_id"].astype(int).nunique() != len(diagnosis):
        raise ValueError("Layer-3 diagnosis is not a unique system table")
    unknown = sorted(set(diagnosis["family"]).difference({"glen_a", "prcp_fac"}))
    if unknown:
        raise ValueError(f"unsupported Layer-3 family labels: {unknown}")

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    config_dir = args.base_config.parent
    result_root = (
        project_root()
        / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/iterative_loop_v3_after_layer2"
    )
    rows: list[dict] = []
    for family in ("glen_a", "prcp_fac"):
        selected = diagnosis[diagnosis["family"].eq(family)].sort_values("lia_unique_id")
        if selected.empty:
            continue
        systems = {
            str(int(row.lia_unique_id)): _system_design(row, family, preflight=False)
            for row in selected.itertuples(index=False)
        }
        first = selected.iloc[0]
        preflight_systems = {
            str(int(first.lia_unique_id)): _system_design(first, family, preflight=True)
        }
        output_name = f"lia_191_layer3_{family}_delta_v1"
        preflight_name = f"lia_191_layer3_{family}_delta_preflight_v1"
        full = _config(
            base,
            family=family,
            systems=systems,
            output_name=output_name,
            work_root=f"w191_layer3_{family}_v1",
            purpose=(
                f"Layer-3 {family}+delta-T block: test exactly one parameter family after completed Layer-2; "
                "precipitation anomalies remain off."
            ),
        )
        preflight = _config(
            base,
            family=family,
            systems=preflight_systems,
            output_name=preflight_name,
            work_root=f"w191_layer3_{family}_preflight_v1",
            purpose=(
                f"Mandatory Layer-3 {family}+delta-T center perturbation preflight before releasing its bounded block."
            ),
        )
        (config_dir / f"lia_191_layer3_{family}_delta_v1.json").write_text(
            json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (config_dir / f"lia_191_layer3_{family}_delta_preflight_v1.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for row in selected.itertuples(index=False):
            full_design = _system_design(row, family, preflight=False)
            for delta_t in full_design["delta_t"]:
                for family_value in full_design["family_values"]:
                    rows.append(
                        {
                            "lia_unique_id": int(row.lia_unique_id),
                            "family": family,
                            "delta_t": float(delta_t),
                            "family_value": float(family_value),
                            "diagnostic_reason": str(row.reason),
                            "signed_peak_error": float(row.signed_peak_error),
                            "signed_2000_error": float(row.signed_2000_error),
                            "amplitude_ratio_to_target": float(row.amplitude_ratio_to_target),
                        }
                    )
    design = pd.DataFrame(rows).sort_values(["family", "lia_unique_id", "family_value", "delta_t"])
    design_path = result_root / "layer3_single_family_design_v1.csv"
    design.to_csv(design_path, index=False)
    summary = {
        "schema_version": 1,
        "diagnosis_csv": str(args.diagnosis_csv.resolve()),
        "base_config": str(args.base_config.resolve()),
        "delta_recenter_offsets_k": list(DELTA_RECENTER_OFFSETS),
        "glen_a_multipliers": list(GLEN_MULTIPLIERS),
        "prcp_fac_multipliers": list(PRCP_MULTIPLIERS),
        "task_count_by_family": design.groupby("family").size().to_dict(),
        "total_task_count": int(len(design)),
        "preflight": {
            "glen_a": "first glen_a system, center delta-T and glen_a x1.25",
            "prcp_fac": "first prcp_fac system, center delta-T and prcp_fac x1.10",
        },
        "rule": "Each configuration varies exactly one declared physical family plus a local delta-T recenter; precipitation anomalies remain off.",
    }
    (result_root / "layer3_single_family_design_summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(design.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
