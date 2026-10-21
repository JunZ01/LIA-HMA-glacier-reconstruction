"""Merge completed class-based pilots into the formal dual-gate inventory."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import build_lia_191_dual_gate_inventory_after_batch2_v1 as _builder
from . import build_lia_191_iterative_loop as _base


INPUT_INVENTORY = "dual_gate_inventory_after_layer3_2266_v1"
OUTPUT_NAME = "dual_gate_inventory_after_class_batches_v1"

EXPERIMENTS = [
    ("lia_191_class_C1_glen_pilot_v1", "class_C1_glen_regular_preflight_v1", "class_representative", 201, "core191-length-corrected", "glen"),
    ("lia_191_class_C2_glen_pilot_v1", "class_C2_glen_pilot_v1", "class_representative", 202, "core191-length-corrected", "glen"),
    ("lia_191_class_C3_prcp_pilot_v1", "class_C3_prcp_regular_preflight_v1", "class_representative", 203, "core191-length-corrected", "prcp"),
    ("lia_191_class_C2_glen_promotion_v1", "class_C2_glen_promotion_v1", "class_promotion", 204, "core191-length-corrected", "glen"),
    ("lia_191_class_C2_delta_recenter_v1", "class_C2_delta_recenter_v1", "class_delta_recenter", 205, "core191-length-corrected", "delta"),
    ("lia_191_class_C1_glen_spatial_pilot_v1", "class_C1_glen_spatial_pilot_v1", "class_representative", 206, "spatial-downstream-v2", "glen"),
    ("lia_191_class_C3_prcp_spatial_pilot_v1", "class_C3_prcp_spatial_pilot_v1", "class_representative", 207, "spatial-downstream-v2", "prcp"),
    ("lia_191_class_C1_glen_spatial_promotion_v1", "class_C1_glen_spatial_promotion_v1", "class_promotion", 208, "spatial-downstream-v2", "glen"),
    ("lia_191_class_C4_glen_regular_pilot_v1", "class_C4_glen_regular_pilot_v1", "class_representative", 209, "core191-length-corrected", "glen"),
    ("lia_191_class_C4_glen_spatial_pilot_v1", "class_C4_glen_spatial_pilot_v1", "class_representative", 210, "spatial-downstream-v2", "glen"),
    ("lia_191_class_C5_glen_spatial_pilot_v1", "class_C5_glen_spatial_pilot_v1", "class_representative", 211, "spatial-downstream-v2", "glen"),
]


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _number(value: object, default: float = np.nan) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if np.isfinite(output) else default


def _new_rows() -> pd.DataFrame:
    root = _base.project_root()
    previous = pd.read_csv(
        _base.controller_root()
        / "diagnostics"
        / INPUT_INVENTORY
        / "chain_dual_gate_inventory.csv"
    )
    target_rows = (
        previous.sort_values("score_priority")
        .dropna(subset=["lia_target_m", "rgi_target_m"])
        .drop_duplicates("lia_unique_id", keep="last")
        .set_index("lia_unique_id")[["lia_target_m", "rgi_target_m", "rgi_id"]]
    )
    rows: list[dict[str, object]] = []
    for experiment, label, layer, priority, geometry, family in EXPERIMENTS:
        path = (
            root
            / "validation_data"
            / "results"
            / "paleo_surface_validation_v0"
            / experiment
            / "scores.csv"
        )
        frame = pd.read_csv(path)
        for item in frame.to_dict(orient="records"):
            system_id = int(item["lia_unique_id"])
            target = target_rows.loc[system_id]
            lia = _number(item.get("lia_target_m"), float(target["lia_target_m"]))
            rgi = _number(item.get("rgi_target_m"), float(target["rgi_target_m"]))
            separation = abs(lia - rgi)
            peak = _number(item.get("forward_peak_m"))
            endpoint = _number(item.get("forward_2000_m"))
            minimum = _number(item.get("forward_minimum_m"))
            technical = _as_bool(item.get("technical_failure", False))
            converged = _as_bool(item.get("spinup_converged", False))
            stability = _number(item.get("spinup_stability_fraction"))
            overflow = _as_bool(item.get("observation_domain_overflow", False))
            valid = _as_bool(item.get("length_observation_valid", False))
            stable = bool(
                not technical
                and converged
                and np.isfinite(stability)
                and stability <= 0.02
                and valid
                and not overflow
            )
            e_lia_sep = abs(peak - lia) / separation if np.isfinite(peak) else np.nan
            e_rgi_sep = abs(endpoint - rgi) / separation if np.isfinite(endpoint) else np.nan
            e_lia_total = abs(peak - lia) / lia if np.isfinite(peak) else np.nan
            e_rgi_total = abs(endpoint - rgi) / rgi if np.isfinite(endpoint) else np.nan
            amplitude = peak - endpoint if np.isfinite(peak) and np.isfinite(endpoint) else np.nan
            if family == "glen":
                family_value = _number(item.get("glen_a_stage2_multiplier"))
            elif family == "prcp":
                family_value = _number(item.get("prcp_fac_multiplier"))
            else:
                family_value = _number(item.get("delta_t"))
            rows.append(
                {
                    "task_id": item["task_id"],
                    "fingerprint": item["fingerprint"],
                    "lia_unique_id": system_id,
                    "rgi_id": item.get("rgi_id", target["rgi_id"]),
                    "block": item["block"],
                    "delta_t": _number(item.get("delta_t")),
                    "melt_f": _number(item.get("melt_f")),
                    "prcp_fac_multiplier": _number(item.get("prcp_fac_multiplier"), 1.0),
                    "glen_a": _number(item.get("glen_a")),
                    "temp_melt": _number(item.get("temp_melt"), -1.0),
                    "task_state": item.get("task_state", "completed"),
                    "output_dir": item.get("output_dir", ""),
                    "workdir": item.get("workdir", ""),
                    "returncode": item.get("returncode", np.nan),
                    "runtime_s_controller": item.get("runtime_s_controller", np.nan),
                    "summary_status": item.get("summary_status", ""),
                    "technical_failure": technical,
                    "spinup_converged": converged,
                    "spinup_stability_fraction": stability,
                    "lia_target_m": lia,
                    "rgi_target_m": rgi,
                    "lia_rgi_separation_m": separation,
                    "forward_peak_m": peak,
                    "forward_2000_m": endpoint,
                    "forward_minimum_m": minimum,
                    "e_lia_sep": e_lia_sep,
                    "e_rgi_sep": e_rgi_sep,
                    "objective_sep": e_lia_sep + e_rgi_sep,
                    "peak_to_2000_amplitude_m": amplitude,
                    "amplitude_ratio_to_target": amplitude / separation if np.isfinite(amplitude) else np.nan,
                    "accepted": stable and max(e_lia_sep, e_rgi_sep) <= 0.20 and max(e_lia_total, e_rgi_total) <= 0.10,
                    "acceptance_state": item.get("acceptance_state", "not-accepted"),
                    "score_source": label,
                    "score_layer": layer,
                    "score_priority": priority,
                    "melt_f_stage2_multiplier": _number(item.get("melt_f_stage2_multiplier"), 1.0),
                    "glen_a_stage2_multiplier": _number(item.get("glen_a_stage2_multiplier"), 1.0),
                    "family_value": family_value,
                    "geometry_variant": geometry,
                    "forward_peak_year": item.get("forward_peak_year", np.nan),
                    "observation_domain_node_count": item.get("observation_domain_node_count", np.nan),
                    "observation_domain_overflow": overflow,
                    "observation_domain_assessed": _as_bool(item.get("observation_domain_assessed", not technical)),
                    "observation_domain_unresolved": _as_bool(item.get("observation_domain_unresolved", False)),
                    "length_observation_valid": valid,
                    "technical_failure_bool": technical,
                    "spinup_converged_bool": converged,
                    "stable_chain": stable,
                    "e_lia_total": e_lia_total,
                    "e_rgi_total": e_rgi_total,
                    "direct_candidate": stable and max(e_lia_total, e_rgi_total) <= 0.10,
                    "audit_observation_domain_state": "overflow" if overflow else ("within-domain" if valid else "unassessed"),
                    "audit_length_observation_valid": valid,
                    "audit_observation_domain_overflow": overflow,
                    "audit_observation_domain_unresolved": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    original = (
        _builder.INPUT_INVENTORY,
        _builder.OUTPUT_NAME,
        _builder._batch2_rows,
    )
    try:
        _builder.INPUT_INVENTORY = INPUT_INVENTORY
        _builder.OUTPUT_NAME = OUTPUT_NAME
        _builder._batch2_rows = _new_rows
        _builder.main()
    finally:
        (
            _builder.INPUT_INVENTORY,
            _builder.OUTPUT_NAME,
            _builder._batch2_rows,
        ) = original
    summary_path = (
        _base.controller_root()
        / "diagnostics"
        / OUTPUT_NAME
        / "formal_inventory_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["new_class_batch_row_count"] = int(summary.pop("new_batch2_row_count"))
    summary["class_experiments"] = [item[0] for item in EXPERIMENTS]
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
