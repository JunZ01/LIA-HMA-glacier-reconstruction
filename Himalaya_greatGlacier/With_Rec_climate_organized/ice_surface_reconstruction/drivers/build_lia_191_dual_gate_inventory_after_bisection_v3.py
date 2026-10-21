"""Rebuild the formal dual-gate inventory after bisection batch 1.

Version 3 fixes the empty-system boolean-indexing edge case in v2.  It is a
read-only score consolidation and never advances an OGGM model.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import build_lia_191_dual_gate_inventory_after_analog_v1 as _previous
from . import build_lia_191_dual_gate_inventory_after_bisection_v2 as _v2
from . import build_lia_191_iterative_loop as _base


OUTPUT_NAME = "dual_gate_inventory_after_bisection_v3"


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    output = _base.controller_root() / "diagnostics" / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)

    previous = _previous._load_scores()
    new = _v2._new_score_rows()
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
    scores.loc[
        separation.le(0), ["e_lia_sep_formal", "e_rgi_sep_formal"]
    ] = np.nan
    stable_mask = scores["stable_chain"].map(_as_bool).astype(bool)
    scores["formal_separation_pass"] = (
        stable_mask
        & scores["e_lia_sep_formal"].le(0.20)
        & scores["e_rgi_sep_formal"].le(0.20)
    )
    scores["total_length_pass"] = (
        stable_mask
        & scores["e_lia_total"].le(0.10)
        & scores["e_rgi_total"].le(0.10)
    )
    scores["formal_dual_pass"] = (
        scores["formal_separation_pass"] & scores["total_length_pass"]
    )
    scores["objective_sep_formal"] = (
        scores["e_lia_sep_formal"] + scores["e_rgi_sep_formal"]
    )
    scores.to_csv(output / "chain_dual_gate_inventory.csv", index=False)

    master = _base.load_master()[["lia_unique_id", "lia_rgi_separation_m"]].copy()
    rows: list[dict[str, object]] = []
    best_dual_rows: list[pd.Series] = []
    sort_columns = [
        "objective_sep_formal",
        "e_lia_total",
        "e_rgi_total",
        "score_priority",
    ]
    for master_row in master.itertuples(index=False):
        system_id = int(master_row.lia_unique_id)
        group = scores.loc[scores["lia_unique_id"].astype(int).eq(system_id)]
        stable = group.loc[
            group["stable_chain"].map(_as_bool).astype(bool)
        ].sort_values(sort_columns, na_position="last")
        dual = group.loc[
            group["formal_dual_pass"].map(_as_bool).astype(bool)
        ].sort_values(sort_columns, na_position="last")
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
            "stable_valid_chain_count": int(
                group["stable_chain"].map(_as_bool).sum()
            ),
            "total_length_pass_count": int(
                group["total_length_pass"].map(_as_bool).sum()
            ),
            "formal_separation_pass_count": int(
                group["formal_separation_pass"].map(_as_bool).sum()
            ),
            "formal_dual_pass_count": int(
                group["formal_dual_pass"].map(_as_bool).sum()
            ),
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

    route_counts = (
        systems["formal_inventory_route"]
        .value_counts()
        .rename_axis("route")
        .reset_index(name="system_count")
    )
    route_counts.to_csv(output / "formal_inventory_route_counts.csv", index=False)
    formal_systems = sorted(
        systems.loc[
            systems["formal_dual_pass_count"].gt(0), "lia_unique_id"
        ].astype(int).tolist()
    )
    summary = {
        "schema_version": 3,
        "launched_oggm": False,
        "score_row_count": int(len(scores)),
        "new_bisection_row_count": int(len(new)),
        "system_count": int(len(systems)),
        "systems_with_saved_chains": int(systems["saved_chain_count"].gt(0).sum()),
        "systems_with_total_length_pass": int(
            systems["total_length_pass_count"].gt(0).sum()
        ),
        "systems_with_formal_separation_pass": int(
            systems["formal_separation_pass_count"].gt(0).sum()
        ),
        "systems_with_formal_dual_pass": len(formal_systems),
        "formal_dual_pass_systems": formal_systems,
        "route_counts": {
            str(row.route): int(row.system_count)
            for row in route_counts.itertuples(index=False)
        },
    }
    (output / "formal_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
