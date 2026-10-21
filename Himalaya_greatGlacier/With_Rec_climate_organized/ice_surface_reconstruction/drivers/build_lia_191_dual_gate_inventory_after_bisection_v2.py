"""Rebuild the read-only formal inventory after delta bisection batch 1."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import build_lia_191_dual_gate_inventory_after_analog_v1 as _previous
from . import build_lia_191_iterative_loop as _base


EXPERIMENT = "lia_191_formal_delta_bisection_batch1_v1"
OUTPUT_NAME = "dual_gate_inventory_after_bisection_v2"


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _new_score_rows() -> pd.DataFrame:
    path = _previous._result_root(EXPERIMENT) / "formal_scores.csv"
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for item in frame.to_dict(orient="records"):
        technical_clean = _as_bool(item.get("technical_clean", False))
        converged = _as_bool(item.get("spinup_converged", False))
        overflow = _as_bool(item.get("observation_domain_overflow", True))
        stable = _as_bool(item.get("stable_valid", False))
        lia = float(item["lia_target_m"])
        rgi = float(item["rgi_target_m"])
        separation = abs(lia - rgi)
        peak = float(item["forward_peak_m"]) if pd.notna(item["forward_peak_m"]) else np.nan
        endpoint = float(item["forward_2000_m"]) if pd.notna(item["forward_2000_m"]) else np.nan
        amplitude = peak - endpoint if np.isfinite(peak) and np.isfinite(endpoint) else np.nan
        rows.append(
            {
                "task_id": item["task_id"],
                "fingerprint": item["fingerprint"],
                "lia_unique_id": int(item["lia_unique_id"]),
                "rgi_id": item["rgi_id"],
                "block": item["block"],
                "delta_t": float(item["delta_t"]),
                "melt_f": float(item["melt_f"]),
                "prcp_fac_multiplier": float(item["prcp_fac_multiplier"]),
                "glen_a": float(item["glen_a"]),
                "temp_melt": float(item["temp_melt"]),
                "task_state": item["task_state"],
                "output_dir": item["output_dir"],
                "workdir": item["workdir"],
                "returncode": item.get("returncode", 0),
                "runtime_s_controller": item.get("runtime_s_controller", np.nan),
                "summary_status": item.get("summary_status", ""),
                "technical_failure": not technical_clean and not overflow,
                "spinup_converged": converged,
                "spinup_stability_fraction": float(item["spinup_stability_fraction"]),
                "lia_target_m": lia,
                "rgi_target_m": rgi,
                "lia_rgi_separation_m": separation,
                "forward_peak_m": peak,
                "forward_2000_m": endpoint,
                "e_lia_sep": float(item["e_lia_sep"]) if pd.notna(item["e_lia_sep"]) else np.nan,
                "e_rgi_sep": float(item["e_rgi_sep"]) if pd.notna(item["e_rgi_sep"]) else np.nan,
                "objective_sep": float(item["objective_sep"]) if pd.notna(item["objective_sep"]) else np.nan,
                "peak_to_2000_amplitude_m": amplitude,
                "amplitude_ratio_to_target": amplitude / separation if np.isfinite(amplitude) else np.nan,
                "accepted": _as_bool(item.get("formal_separation_pass", False)),
                "acceptance_state": "accepted" if _as_bool(item.get("formal_separation_pass", False)) else "not-accepted",
                "score_source": "formal_delta_bisection_batch1_v1",
                "score_layer": "formal_delta_bisection",
                "score_priority": 150,
                "melt_f_stage2_multiplier": float(item["melt_f_stage2_multiplier"]),
                "glen_a_stage2_multiplier": float(item["glen_a_stage2_multiplier"]),
                "family_value": float(item["delta_t"]),
                "geometry_variant": str(item["geometry_group"]),
                "observation_domain_node_count": item.get("observation_domain_node_count", np.nan),
                "observation_domain_overflow": overflow,
                "observation_domain_assessed": True,
                "observation_domain_unresolved": False,
                "length_observation_valid": stable,
                "technical_failure_bool": not technical_clean and not overflow,
                "spinup_converged_bool": converged,
                "stable_chain": stable,
                "e_lia_total": float(item["e_lia_total"]) if pd.notna(item["e_lia_total"]) else np.nan,
                "e_rgi_total": float(item["e_rgi_total"]) if pd.notna(item["e_rgi_total"]) else np.nan,
                "direct_candidate": _as_bool(item.get("total_length_pass", False)),
                "audit_observation_domain_state": "overflow" if overflow else "within-domain",
                "audit_length_observation_valid": stable,
                "audit_observation_domain_overflow": overflow,
                "audit_observation_domain_unresolved": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    output = _base.controller_root() / "diagnostics" / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    previous = _previous._load_scores()
    new = _new_score_rows()
    scores = pd.concat([previous, new], ignore_index=True, sort=False)
    scores = scores.sort_values(["score_priority", "task_id"]).drop_duplicates(
        "fingerprint", keep="last"
    )
    separation = (scores["lia_target_m"] - scores["rgi_target_m"]).abs()
    scores["lia_rgi_separation_m_formal"] = separation
    scores["e_lia_sep_formal"] = (
        scores["forward_peak_m"] - scores["lia_target_m"]
    ).abs() / separation
    scores["e_rgi_sep_formal"] = (
        scores["forward_2000_m"] - scores["rgi_target_m"]
    ).abs() / separation
    scores.loc[separation.le(0), ["e_lia_sep_formal", "e_rgi_sep_formal"]] = np.nan
    scores["formal_separation_pass"] = (
        scores["stable_chain"].map(_as_bool)
        & scores["e_lia_sep_formal"].le(0.20)
        & scores["e_rgi_sep_formal"].le(0.20)
    )
    scores["total_length_pass"] = (
        scores["stable_chain"].map(_as_bool)
        & scores["e_lia_total"].le(0.10)
        & scores["e_rgi_total"].le(0.10)
    )
    scores["formal_dual_pass"] = scores["formal_separation_pass"] & scores["total_length_pass"]
    scores["objective_sep_formal"] = scores["e_lia_sep_formal"] + scores["e_rgi_sep_formal"]
    scores.to_csv(output / "chain_dual_gate_inventory.csv", index=False)

    master = _base.load_master()[["lia_unique_id", "lia_rgi_separation_m"]].copy()
    rows: list[dict[str, object]] = []
    best_dual_rows: list[pd.Series] = []
    for master_row in master.itertuples(index=False):
        system_id = int(master_row.lia_unique_id)
        group = scores[scores["lia_unique_id"].astype(int).eq(system_id)]
        stable = group[group["stable_chain"].map(_as_bool)].sort_values(
            ["objective_sep_formal", "e_lia_total", "e_rgi_total", "score_priority"],
            na_position="last",
        )
        dual = group[group["formal_dual_pass"].map(_as_bool)].sort_values(
            ["objective_sep_formal", "e_lia_total", "e_rgi_total", "score_priority"],
            na_position="last",
        )
        if not dual.empty:
            best_dual_rows.append(dual.iloc[0])
        if system_id in _base.MANUAL_PROVISIONAL_IDS:
            route = "manual_provisional_outside_controller_sources"
        elif system_id in _base.BLOCKED_DYNAMIC_IDS:
            route = "blocked_dynamic_review"
        elif abs(float(master_row.lia_rgi_separation_m)) < _base.SMALL_SEPARATION_THRESHOLD_M:
            route = "small_separation_review"
        elif not dual.empty:
            route = "formal_dual_pass"
        elif group["total_length_pass"].map(_as_bool).any():
            route = "total_length_only"
        elif group["formal_separation_pass"].map(_as_bool).any():
            route = "formal_separation_only"
        elif stable.empty:
            route = "no_stable_valid_chain"
        else:
            route = "no_formal_pass"
        row: dict[str, object] = {
            "lia_unique_id": system_id,
            "saved_chain_count": int(len(group)),
            "stable_valid_chain_count": int(group["stable_chain"].map(_as_bool).sum()),
            "total_length_pass_count": int(group["total_length_pass"].map(_as_bool).sum()),
            "formal_separation_pass_count": int(group["formal_separation_pass"].map(_as_bool).sum()),
            "formal_dual_pass_count": int(group["formal_dual_pass"].map(_as_bool).sum()),
            "formal_inventory_route": route,
        }
        if not stable.empty:
            best = stable.iloc[0]
            row.update(
                {
                    "best_stable_task_id": best["task_id"],
                    "best_stable_source": best["score_source"],
                    "best_stable_e_lia_sep": float(best["e_lia_sep_formal"]),
                    "best_stable_e_rgi_sep": float(best["e_rgi_sep_formal"]),
                    "best_stable_objective_sep": float(best["objective_sep_formal"]),
                    "best_stable_e_lia_total": float(best["e_lia_total"]),
                    "best_stable_e_rgi_total": float(best["e_rgi_total"]),
                }
            )
        rows.append(row)
    systems = pd.DataFrame(rows).sort_values("lia_unique_id")
    systems.to_csv(output / "system_dual_gate_inventory.csv", index=False)
    best_dual = pd.DataFrame(best_dual_rows)
    if not best_dual.empty:
        best_dual = best_dual.sort_values("lia_unique_id")
    best_dual.to_csv(output / "best_formal_dual_pass_per_system.csv", index=False)
    route_counts = systems["formal_inventory_route"].value_counts().rename_axis("route").reset_index(name="system_count")
    route_counts.to_csv(output / "formal_inventory_route_counts.csv", index=False)
    summary = {
        "schema_version": 2,
        "launched_oggm": False,
        "score_row_count": int(len(scores)),
        "new_bisection_row_count": int(len(new)),
        "system_count": int(len(systems)),
        "systems_with_saved_chains": int((systems["saved_chain_count"] > 0).sum()),
        "systems_with_total_length_pass": int((systems["total_length_pass_count"] > 0).sum()),
        "systems_with_formal_separation_pass": int((systems["formal_separation_pass_count"] > 0).sum()),
        "systems_with_formal_dual_pass": int((systems["formal_dual_pass_count"] > 0).sum()),
        "formal_dual_pass_systems": sorted(
            systems.loc[systems["formal_dual_pass_count"] > 0, "lia_unique_id"].astype(int).tolist()
        ),
        "route_counts": {
            str(row.route): int(row.system_count) for row in route_counts.itertuples(index=False)
        },
    }
    (output / "formal_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
