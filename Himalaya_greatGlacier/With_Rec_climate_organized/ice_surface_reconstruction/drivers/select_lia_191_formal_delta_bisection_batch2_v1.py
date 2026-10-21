"""Select the next formal delta-T interpolation candidates without running OGGM.

The selector uses only stable, observation-valid chains at each system's
Stage-2 physical parameters.  It searches adjacent saved delta-T points for a
linearized interval in which both signed separation-normalized endpoint errors
can lie inside +/-0.20.  The result is a reachability diagnostic, not an
acceptance decision; every selected system still requires geometry and
single-chain preflight gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
INVENTORY_ROOT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_bisection_v3"
)
CHAIN_PATH = INVENTORY_ROOT / "chain_dual_gate_inventory.csv"
SYSTEM_PATH = INVENTORY_ROOT / "system_dual_gate_inventory.csv"
OUTPUT_ROOT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_formal_delta_bisection_batch2_selection_v1"
)

TOL = 0.20
MAX_BRACKET_WIDTH_K = 0.50
FROZEN = {991, 1077, 1616, 2815, 2902, 2928}
CLOSED = {
    214,
    2569,
    2622,
    4631,
    2645,
    2259,
    4959,
    1545,
    3584,
    5143,
}
GEOMETRY_REVIEW = {1546, 2806}
BLOCKED = {2387, 5508}


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _inside_interval(x0: float, x1: float, y0: float, y1: float) -> tuple[float, float] | None:
    """Return the x subinterval where linear y is within [-TOL, TOL]."""
    lo, hi = sorted((x0, x1))
    if not all(np.isfinite([x0, x1, y0, y1])):
        return None
    if np.isclose(x0, x1):
        return (lo, hi) if abs(y0) <= TOL else None
    slope = (y1 - y0) / (x1 - x0)
    if np.isclose(slope, 0.0):
        return (lo, hi) if abs(y0) <= TOL else None
    xa = x0 + (-TOL - y0) / slope
    xb = x0 + (TOL - y0) / slope
    valid_lo = max(lo, min(xa, xb))
    valid_hi = min(hi, max(xa, xb))
    if valid_lo <= valid_hi + 1e-12:
        return valid_lo, valid_hi
    return None


def _same_physics(frame: pd.DataFrame, reference: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in ["melt_f", "glen_a", "prcp_fac_multiplier", "temp_melt"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        target = float(reference[column])
        mask &= np.isclose(values, target, rtol=1e-9, atol=1e-12)
    return mask


def _geometry_class(row: pd.Series) -> str:
    text = " ".join(
        [str(row.get("score_source", "")), str(row.get("geometry_variant", ""))]
    ).lower()
    return "spatial-downstream-v2" if "spatial" in text else "core-191"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    chains = pd.read_csv(CHAIN_PATH, low_memory=False)
    systems = pd.read_csv(SYSTEM_PATH)
    chains["stable_bool"] = chains["stable_chain"].map(_as_bool)
    chains["valid_bool"] = chains["audit_length_observation_valid"].map(_as_bool)
    chains["formal_dual_bool"] = chains["formal_dual_pass"].map(_as_bool)
    chains["geometry_class"] = chains.apply(_geometry_class, axis=1)

    rejection_rows: list[dict[str, object]] = []
    bracket_rows: list[dict[str, object]] = []
    for sys_row in systems.itertuples(index=False):
        system_id = int(sys_row.lia_unique_id)
        route = str(sys_row.formal_inventory_route)
        reason = None
        if system_id in FROZEN:
            reason = "already_frozen"
        elif system_id in CLOSED:
            reason = "named_parameter_family_closed"
        elif system_id in GEOMETRY_REVIEW:
            reason = "existing_outlet_geometry_gate_failed"
        elif system_id in BLOCKED:
            reason = "blocked_dynamic_review"
        elif route == "small_separation_review":
            reason = "small_separation_review"
        elif route == "manual_provisional_outside_controller_sources":
            reason = "manual_provisional_outside_controller_sources"
        elif route == "no_stable_valid_chain":
            reason = "no_stable_valid_chain"
        if reason:
            rejection_rows.append(
                {"lia_unique_id": system_id, "route": route, "reason": reason}
            )
            continue

        group = chains.loc[chains["lia_unique_id"].astype(int).eq(system_id)].copy()
        reference_pool = group.loc[group["score_source"].eq("controller_10pt_delta")]
        if reference_pool.empty:
            rejection_rows.append(
                {"lia_unique_id": system_id, "route": route, "reason": "missing_stage2_reference"}
            )
            continue
        reference = reference_pool.iloc[0]
        base = group.loc[
            group["stable_bool"]
            & group["valid_bool"]
            & _same_physics(group, reference)
        ].copy()
        if base.empty:
            rejection_rows.append(
                {"lia_unique_id": system_id, "route": route, "reason": "no_stage2_stable_valid_chain"}
            )
            continue

        for geometry_class, geometry in base.groupby("geometry_class"):
            # Retain the highest-priority record for repeated delta-T values.
            geometry = geometry.sort_values(["score_priority", "task_id"]).drop_duplicates(
                "delta_t", keep="last"
            )
            geometry = geometry.sort_values("delta_t")
            separation = float(geometry.iloc[0]["lia_rgi_separation_m_formal"])
            geometry["signed_lia_sep"] = (
                geometry["forward_peak_m"] - geometry["lia_target_m"]
            ) / separation
            geometry["signed_rgi_sep"] = (
                geometry["forward_2000_m"] - geometry["rgi_target_m"]
            ) / separation
            records = list(geometry.to_dict(orient="records"))
            for left, right in zip(records[:-1], records[1:]):
                x0, x1 = float(left["delta_t"]), float(right["delta_t"])
                if x1 - x0 > MAX_BRACKET_WIDTH_K + 1e-12:
                    continue
                lia_interval = _inside_interval(
                    x0, x1, float(left["signed_lia_sep"]), float(right["signed_lia_sep"])
                )
                rgi_interval = _inside_interval(
                    x0, x1, float(left["signed_rgi_sep"]), float(right["signed_rgi_sep"])
                )
                if lia_interval is None or rgi_interval is None:
                    continue
                overlap_lo = max(lia_interval[0], rgi_interval[0])
                overlap_hi = min(lia_interval[1], rgi_interval[1])
                if overlap_lo > overlap_hi + 1e-12:
                    continue
                mid = 0.5 * (overlap_lo + overlap_hi)
                weight = (mid - x0) / (x1 - x0)
                pred_lia = float(left["signed_lia_sep"]) + weight * (
                    float(right["signed_lia_sep"]) - float(left["signed_lia_sep"])
                )
                pred_rgi = float(left["signed_rgi_sep"]) + weight * (
                    float(right["signed_rgi_sep"]) - float(left["signed_rgi_sep"])
                )
                bracket_rows.append(
                    {
                        "lia_unique_id": system_id,
                        "formal_inventory_route": route,
                        "geometry_class": geometry_class,
                        "left_task_id": left["task_id"],
                        "right_task_id": right["task_id"],
                        "left_delta_t": x0,
                        "right_delta_t": x1,
                        "left_signed_lia_sep": float(left["signed_lia_sep"]),
                        "left_signed_rgi_sep": float(left["signed_rgi_sep"]),
                        "right_signed_lia_sep": float(right["signed_lia_sep"]),
                        "right_signed_rgi_sep": float(right["signed_rgi_sep"]),
                        "overlap_delta_t_low": overlap_lo,
                        "overlap_delta_t_high": overlap_hi,
                        "overlap_width_k": max(0.0, overlap_hi - overlap_lo),
                        "predicted_mid_delta_t": mid,
                        "predicted_mid_signed_lia_sep": pred_lia,
                        "predicted_mid_signed_rgi_sep": pred_rgi,
                        "predicted_mid_max_abs_error": max(abs(pred_lia), abs(pred_rgi)),
                        "melt_f": float(reference["melt_f"]),
                        "prcp_fac_multiplier": float(reference["prcp_fac_multiplier"]),
                        "glen_a": float(reference["glen_a"]),
                        "temp_melt": float(reference["temp_melt"]),
                    }
                )

    brackets = pd.DataFrame(bracket_rows)
    if not brackets.empty:
        brackets = brackets.sort_values(
            ["predicted_mid_max_abs_error", "overlap_width_k"],
            ascending=[True, False],
        )
        best = brackets.drop_duplicates("lia_unique_id", keep="first").copy()
        best["proposed_delta_t_values"] = best.apply(
            lambda row: json.dumps(
                [
                    round(float(row.overlap_delta_t_low), 8),
                    round(float(row.predicted_mid_delta_t), 8),
                    round(float(row.overlap_delta_t_high), 8),
                ]
            ),
            axis=1,
        )
    else:
        best = pd.DataFrame()

    brackets.to_csv(OUTPUT_ROOT / "all_linearized_overlap_brackets.csv", index=False)
    best.to_csv(OUTPUT_ROOT / "best_overlap_per_system.csv", index=False)
    pd.DataFrame(rejection_rows).to_csv(OUTPUT_ROOT / "excluded_systems.csv", index=False)
    summary = {
        "schema_version": 1,
        "launched_oggm": False,
        "inventory": str(INVENTORY_ROOT),
        "formal_tolerance": TOL,
        "max_adjacent_bracket_width_k": MAX_BRACKET_WIDTH_K,
        "candidate_system_count": int(len(best)),
        "ranked_candidate_systems": (
            best["lia_unique_id"].astype(int).tolist() if not best.empty else []
        ),
        "next_action": (
            "Apply outlet/observation-domain geometry gates before releasing any OGGM chain."
        ),
    }
    (OUTPUT_ROOT / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not best.empty:
        print(
            best[
                [
                    "lia_unique_id",
                    "geometry_class",
                    "left_delta_t",
                    "right_delta_t",
                    "overlap_delta_t_low",
                    "overlap_delta_t_high",
                    "predicted_mid_max_abs_error",
                    "proposed_delta_t_values",
                ]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()
