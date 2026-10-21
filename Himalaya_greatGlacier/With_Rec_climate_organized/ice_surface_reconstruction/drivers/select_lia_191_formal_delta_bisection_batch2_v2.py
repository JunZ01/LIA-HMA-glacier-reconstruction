"""Add bounded edge extrapolation to the batch-2 delta-T reachability audit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import select_lia_191_formal_delta_bisection_batch2_v1 as _v1


OUTPUT_ROOT = (
    _v1.ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_formal_delta_bisection_batch2_selection_v2"
)
MAX_EDGE_EXTRAPOLATION_K = 0.125


def _excluded(system_id: int, route: str) -> str | None:
    if system_id in _v1.FROZEN:
        return "already_frozen"
    if system_id in _v1.CLOSED:
        return "named_parameter_family_closed"
    if system_id in _v1.GEOMETRY_REVIEW:
        return "existing_outlet_geometry_gate_failed"
    if system_id in _v1.BLOCKED:
        return "blocked_dynamic_review"
    if route == "small_separation_review":
        return "small_separation_review"
    if route == "manual_provisional_outside_controller_sources":
        return "manual_provisional_outside_controller_sources"
    if route == "no_stable_valid_chain":
        return "no_stable_valid_chain"
    return None


def _linear_value(
    x: float, x0: float, x1: float, y0: float, y1: float
) -> float:
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    chains = pd.read_csv(_v1.CHAIN_PATH, low_memory=False)
    systems = pd.read_csv(_v1.SYSTEM_PATH)
    chains["stable_bool"] = chains["stable_chain"].map(_v1._as_bool)
    chains["valid_bool"] = chains["audit_length_observation_valid"].map(_v1._as_bool)
    chains["geometry_class"] = chains.apply(_v1._geometry_class, axis=1)

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for sys_row in systems.itertuples(index=False):
        system_id = int(sys_row.lia_unique_id)
        route = str(sys_row.formal_inventory_route)
        reason = _excluded(system_id, route)
        if reason:
            rejected.append({"lia_unique_id": system_id, "route": route, "reason": reason})
            continue

        group = chains.loc[chains["lia_unique_id"].astype(int).eq(system_id)].copy()
        refs = group.loc[group["score_source"].eq("controller_10pt_delta")]
        if refs.empty:
            rejected.append(
                {"lia_unique_id": system_id, "route": route, "reason": "missing_stage2_reference"}
            )
            continue
        reference = refs.iloc[0]
        base = group.loc[
            group["stable_bool"]
            & group["valid_bool"]
            & _v1._same_physics(group, reference)
        ].copy()
        if base.empty:
            rejected.append(
                {"lia_unique_id": system_id, "route": route, "reason": "no_stage2_stable_valid_chain"}
            )
            continue

        for geometry_class, geometry in base.groupby("geometry_class"):
            geometry = geometry.sort_values(["score_priority", "task_id"]).drop_duplicates(
                "delta_t", keep="last"
            )
            geometry = geometry.sort_values("delta_t")
            if len(geometry) < 2:
                continue
            separation = float(geometry.iloc[0]["lia_rgi_separation_m_formal"])
            geometry["signed_lia_sep"] = (
                geometry["forward_peak_m"] - geometry["lia_target_m"]
            ) / separation
            geometry["signed_rgi_sep"] = (
                geometry["forward_2000_m"] - geometry["rgi_target_m"]
            ) / separation
            records = list(geometry.to_dict(orient="records"))

            edge_specs = [
                (records[0], records[1], "left"),
                (records[-2], records[-1], "right"),
            ]
            for near0, near1, side in edge_specs:
                x0, x1 = float(near0["delta_t"]), float(near1["delta_t"])
                step = min(MAX_EDGE_EXTRAPOLATION_K, x1 - x0)
                if step <= 0:
                    continue
                if side == "left":
                    xa, xb = x0 - step, x0
                else:
                    xa, xb = x1, x1 + step
                lia_a = _linear_value(
                    xa, x0, x1, float(near0["signed_lia_sep"]), float(near1["signed_lia_sep"])
                )
                lia_b = _linear_value(
                    xb, x0, x1, float(near0["signed_lia_sep"]), float(near1["signed_lia_sep"])
                )
                rgi_a = _linear_value(
                    xa, x0, x1, float(near0["signed_rgi_sep"]), float(near1["signed_rgi_sep"])
                )
                rgi_b = _linear_value(
                    xb, x0, x1, float(near0["signed_rgi_sep"]), float(near1["signed_rgi_sep"])
                )
                lia_interval = _v1._inside_interval(xa, xb, lia_a, lia_b)
                rgi_interval = _v1._inside_interval(xa, xb, rgi_a, rgi_b)
                if lia_interval is None or rgi_interval is None:
                    continue
                lo = max(lia_interval[0], rgi_interval[0])
                hi = min(lia_interval[1], rgi_interval[1])
                if lo > hi + 1e-12:
                    continue
                mid = 0.5 * (lo + hi)
                pred_lia = _linear_value(mid, xa, xb, lia_a, lia_b)
                pred_rgi = _linear_value(mid, xa, xb, rgi_a, rgi_b)
                candidates.append(
                    {
                        "lia_unique_id": system_id,
                        "formal_inventory_route": route,
                        "geometry_class": geometry_class,
                        "bracket_kind": f"{side}_edge_extrapolation",
                        "slope_anchor_task_0": near0["task_id"],
                        "slope_anchor_task_1": near1["task_id"],
                        "anchor_delta_t_0": x0,
                        "anchor_delta_t_1": x1,
                        "extrapolation_delta_t_low": xa,
                        "extrapolation_delta_t_high": xb,
                        "overlap_delta_t_low": lo,
                        "overlap_delta_t_high": hi,
                        "overlap_width_k": max(0.0, hi - lo),
                        "predicted_mid_delta_t": mid,
                        "predicted_mid_signed_lia_sep": pred_lia,
                        "predicted_mid_signed_rgi_sep": pred_rgi,
                        "predicted_mid_max_abs_error": max(abs(pred_lia), abs(pred_rgi)),
                        "proposed_delta_t_values": json.dumps(
                            [round(lo, 8), round(mid, 8), round(hi, 8)]
                        ),
                        "melt_f": float(reference["melt_f"]),
                        "prcp_fac_multiplier": float(reference["prcp_fac_multiplier"]),
                        "glen_a": float(reference["glen_a"]),
                        "temp_melt": float(reference["temp_melt"]),
                    }
                )

    frame = pd.DataFrame(candidates)
    if not frame.empty:
        frame = frame.sort_values(
            ["predicted_mid_max_abs_error", "overlap_width_k"],
            ascending=[True, False],
        )
        best = frame.drop_duplicates("lia_unique_id", keep="first").copy()
    else:
        best = pd.DataFrame()
    frame.to_csv(OUTPUT_ROOT / "all_edge_extrapolation_candidates.csv", index=False)
    best.to_csv(OUTPUT_ROOT / "best_edge_extrapolation_per_system.csv", index=False)
    pd.DataFrame(rejected).to_csv(OUTPUT_ROOT / "excluded_systems.csv", index=False)
    summary = {
        "schema_version": 2,
        "launched_oggm": False,
        "formal_tolerance": _v1.TOL,
        "max_edge_extrapolation_k": MAX_EDGE_EXTRAPOLATION_K,
        "candidate_system_count": int(len(best)),
        "ranked_candidate_systems": (
            best["lia_unique_id"].astype(int).tolist() if not best.empty else []
        ),
        "next_action": "Apply geometry/outlet gates, then run one preflight before any batch release.",
    }
    (OUTPUT_ROOT / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not best.empty:
        print(best.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
