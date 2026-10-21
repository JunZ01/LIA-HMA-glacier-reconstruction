"""Build a read-only formal dual-gate inventory after the 991 analog blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_lia_191_iterative_loop as _base
from . import build_lia_191_iterative_loop_after_991_v1 as _after_991


def _result_root(name: str) -> Path:
    return (
        _base.project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / name
    )


ANALOG_SOURCES = (
    (
        "lia_191_analog_melt_delta_after_991_v1",
        "analog_melt_delta_after_991_v1",
        "layer2_melt_delta",
        130,
    ),
    (
        "lia_191_analog_glen_after_991_v1",
        "analog_glen_after_991_v1",
        "layer3_single_parameter",
        140,
    ),
)


def _score_sources() -> list[_base.ScoreSource]:
    sources = list(_after_991._score_sources())
    for experiment, label, layer, priority in ANALOG_SOURCES:
        sources.append(
            _base.ScoreSource(
                _result_root(experiment) / "scores.csv",
                label,
                layer,
                priority,
            )
        )
    return sources


def _combined_audit() -> pd.DataFrame:
    frames = [_after_991._combined_audit()]
    for experiment, _, _, _ in ANALOG_SOURCES:
        score = pd.read_csv(_result_root(experiment) / "scores.csv")
        audit = score[
            [
                "fingerprint",
                "length_observation_valid",
                "observation_domain_overflow",
                "observation_domain_unresolved",
            ]
        ].copy()
        for column in (
            "length_observation_valid",
            "observation_domain_overflow",
            "observation_domain_unresolved",
        ):
            audit[column] = audit[column].map(_base._as_bool)
        audit["observation_domain_state"] = "within-domain"
        audit.loc[
            audit["observation_domain_overflow"], "observation_domain_state"
        ] = "overflow"
        audit.loc[
            audit["observation_domain_unresolved"], "observation_domain_state"
        ] = "unresolved"
        frames.append(
            audit.rename(
                columns={
                    "observation_domain_state": "audit_observation_domain_state",
                    "length_observation_valid": "audit_length_observation_valid",
                    "observation_domain_overflow": "audit_observation_domain_overflow",
                    "observation_domain_unresolved": "audit_observation_domain_unresolved",
                }
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("fingerprint", keep="last")
    if combined["fingerprint"].duplicated().any():
        raise ValueError("combined observation-domain fingerprints are not unique")
    return combined


def _load_scores() -> pd.DataFrame:
    old_sources = _base.score_sources
    old_audit = _base.load_observation_domain_audit
    _base.score_sources = _score_sources
    _base.load_observation_domain_audit = _combined_audit
    try:
        return _base.load_scores()
    finally:
        _base.score_sources = old_sources
        _base.load_observation_domain_audit = old_audit


def main() -> None:
    output = (
        _base.controller_root()
        / "diagnostics"
        / "dual_gate_inventory_after_analog_v1"
    )
    output.mkdir(parents=True, exist_ok=True)
    scores = _load_scores()
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
        scores["stable_chain"]
        & scores["e_lia_sep_formal"].le(0.20)
        & scores["e_rgi_sep_formal"].le(0.20)
    )
    scores["total_length_pass"] = (
        scores["stable_chain"]
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
    for master_row in master.itertuples(index=False):
        system_id = int(master_row.lia_unique_id)
        group = scores[scores["lia_unique_id"].eq(system_id)]
        stable = group[group["stable_chain"]].sort_values(
            ["objective_sep_formal", "e_lia_total", "e_rgi_total", "score_priority"],
            na_position="last",
        )
        dual = group[group["formal_dual_pass"]].sort_values(
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
        elif group["total_length_pass"].any():
            route = "total_length_only"
        elif group["formal_separation_pass"].any():
            route = "formal_separation_only"
        elif stable.empty:
            route = "no_stable_valid_chain"
        else:
            route = "no_formal_pass"
        row: dict[str, object] = {
            "lia_unique_id": system_id,
            "saved_chain_count": int(len(group)),
            "stable_valid_chain_count": int(group["stable_chain"].sum()),
            "total_length_pass_count": int(group["total_length_pass"].sum()),
            "formal_separation_pass_count": int(group["formal_separation_pass"].sum()),
            "formal_dual_pass_count": int(group["formal_dual_pass"].sum()),
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
        systems["formal_inventory_route"].value_counts().rename_axis("route").reset_index(name="system_count")
    )
    route_counts.to_csv(output / "formal_inventory_route_counts.csv", index=False)
    summary = {
        "schema_version": 1,
        "launched_oggm": False,
        "score_row_count": int(len(scores)),
        "system_count": int(len(systems)),
        "systems_with_saved_chains": int((systems["saved_chain_count"] > 0).sum()),
        "systems_with_total_length_pass": int(
            (systems["total_length_pass_count"] > 0).sum()
        ),
        "systems_with_formal_separation_pass": int(
            (systems["formal_separation_pass_count"] > 0).sum()
        ),
        "systems_with_formal_dual_pass": int(
            (systems["formal_dual_pass_count"] > 0).sum()
        ),
        "formal_dual_pass_systems": sorted(
            systems.loc[
                systems["formal_dual_pass_count"] > 0, "lia_unique_id"
            ].astype(int).tolist()
        ),
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
