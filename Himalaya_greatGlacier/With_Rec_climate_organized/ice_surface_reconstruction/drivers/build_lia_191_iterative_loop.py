"""Build the bounded three-layer iterative routing table for the LIA-191 run.

This script launches no OGGM task. It merges the completed layer-1 delta-T
scores with any completed local follow-up score files, applies the current
hard gates and provisional historical endpoint rule, and writes the next-step
route for each LIA system.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LIA_ENDPOINT_TOL = 0.10
RGI_ENDPOINT_TOL = 0.10
SPINUP_STABILITY_LIMIT = 0.02
SMALL_SEPARATION_THRESHOLD_M = 500.0
BLOCKED_DYNAMIC_IDS = {2387, 5508}
NAMED_CLOSED_IDS = {1545, 3584}
MANUAL_PROVISIONAL_IDS = {1044}


@dataclass(frozen=True)
class ScoreSource:
    path: Path
    label: str
    layer: str
    priority: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def controller_root() -> Path:
    return (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_controller_v1"
    )


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_observation_domain_audit() -> pd.DataFrame:
    path = (
        controller_root()
        / "diagnostics"
        / "observation_domain_gate_v1"
        / "chain_observation_domain_audit.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            "observation-domain audit is required before Core-191 routing: "
            f"{path}"
        )
    audit = pd.read_csv(path)
    if audit["fingerprint"].duplicated().any():
        raise ValueError("observation-domain audit fingerprints are not unique")
    audit = audit[
        [
            "fingerprint",
            "observation_domain_state",
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
        audit[column] = audit[column].map(_as_bool)
    return audit.rename(
        columns={
            "observation_domain_state": "audit_observation_domain_state",
            "length_observation_valid": "audit_length_observation_valid",
            "observation_domain_overflow": "audit_observation_domain_overflow",
            "observation_domain_unresolved": "audit_observation_domain_unresolved",
        }
    )


def score_sources() -> list[ScoreSource]:
    base = (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
    )
    return [
        ScoreSource(
            base / "lia_191_controller_v1" / "scores.csv",
            "controller_10pt_delta",
            "layer1_delta",
            0,
        ),
        ScoreSource(
            base / "lia_191_curve_loop_delta_boundary_batch2_v1" / "scores.csv",
            "delta_boundary_batch2",
            "layer1_delta",
            1,
        ),
        ScoreSource(
            base / "lia_191_curve_loop_delta_refine_1077_r1_v1" / "scores.csv",
            "delta_refine_1077_r1",
            "layer1_delta",
            2,
        ),
        ScoreSource(
            base / "lia_191_delta_boundary_extension_v1" / "scores.csv",
            "delta_boundary_extension_v1",
            "layer1_delta",
            3,
        ),
        ScoreSource(
            base / "lia_191_tradeoff_pilot_v1" / "scores.csv",
            "melt_delta_tradeoff_pilot",
            "layer2_melt_delta",
            10,
        ),
        ScoreSource(
            base / "lia_191_tradeoff_refine_2259_v1" / "scores.csv",
            "melt_delta_tradeoff_refine_2259",
            "layer2_melt_delta",
            11,
        ),
        ScoreSource(
            base / "lia_191_tradeoff_refine_2259_warm_v1" / "scores.csv",
            "melt_delta_tradeoff_refine_2259_warm",
            "layer2_melt_delta",
            12,
        ),
        ScoreSource(
            base / "lia_191_curve_loop_melt_delta_directed_v1" / "scores.csv",
            "melt_delta_directed",
            "layer2_melt_delta",
            13,
        ),
        ScoreSource(
            base / "lia_191_curve_loop_melt_delta_strip_2697_v1" / "scores.csv",
            "melt_delta_strip_2697",
            "layer2_melt_delta",
            14,
        ),
        ScoreSource(
            base / "lia_191_curve_loop_melt_delta_2928_v1" / "scores.csv",
            "melt_delta_2928",
            "layer2_melt_delta",
            15,
        ),
        ScoreSource(
            base / "lia_191_layer2_melt_delta_after_extension_v1" / "scores.csv",
            "melt_delta_after_extension_v1",
            "layer2_melt_delta",
            16,
        ),
        ScoreSource(
            base / "lia_191_layer3_glen_a_delta_v1" / "scores.csv",
            "glen_a_delta_layer3_v1",
            "layer3_single_parameter",
            17,
        ),
        ScoreSource(
            base / "lia_191_layer3_prcp_fac_delta_preflight_r1_v1" / "scores.csv",
            "prcp_fac_delta_preflight_r1_v1",
            "layer3_single_parameter",
            18,
        ),
    ]


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "lia_unique_id",
        "delta_t",
        "melt_f",
        "melt_f_stage2_multiplier",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "lia_rgi_separation_m",
        "forward_peak_m",
        "forward_2000_m",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
        "peak_to_2000_amplitude_m",
        "amplitude_ratio_to_target",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_scores() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in score_sources():
        if not source.path.exists():
            continue
        frame = numeric(pd.read_csv(source.path))
        frame["score_source"] = source.label
        frame["score_layer"] = source.layer
        frame["score_priority"] = source.priority
        if "melt_f_stage2_multiplier" not in frame.columns:
            frame["melt_f_stage2_multiplier"] = 1.0
        frame["melt_f_stage2_multiplier"] = pd.to_numeric(
            frame["melt_f_stage2_multiplier"], errors="coerce"
        ).fillna(1.0)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no score sources found")
    scores = pd.concat(frames, ignore_index=True, sort=False)
    scores["lia_unique_id"] = pd.to_numeric(
        scores["lia_unique_id"], errors="raise"
    ).astype(int)
    scores["technical_failure_bool"] = scores["technical_failure"].map(_as_bool)
    scores["spinup_converged_bool"] = scores["spinup_converged"].map(_as_bool)
    scores["stable_chain"] = (
        ~scores["technical_failure_bool"]
        & scores["summary_status"].astype(str).eq("ok")
        & scores["spinup_converged_bool"]
        & scores["spinup_stability_fraction"].le(SPINUP_STABILITY_LIMIT)
    )
    scores["e_lia_total"] = (
        scores["forward_peak_m"] - scores["lia_target_m"]
    ).abs() / scores["lia_target_m"]
    scores["e_rgi_total"] = (
        scores["forward_2000_m"] - scores["rgi_target_m"]
    ).abs() / scores["rgi_target_m"]
    scores["direct_candidate"] = (
        scores["stable_chain"]
        & scores["e_lia_total"].le(LIA_ENDPOINT_TOL)
        & scores["e_rgi_total"].le(RGI_ENDPOINT_TOL)
    )
    scores["lia_reachable_total"] = (
        scores["stable_chain"] & scores["e_lia_total"].le(LIA_ENDPOINT_TOL)
    )
    scores["rgi_reachable_total"] = (
        scores["stable_chain"] & scores["e_rgi_total"].le(RGI_ENDPOINT_TOL)
    )
    scores = scores.sort_values(["score_priority"]).drop_duplicates(
        ["fingerprint"], keep="last"
    )
    audit = load_observation_domain_audit()
    scores = scores.merge(audit, on="fingerprint", how="left", validate="one_to_one")
    if scores["audit_observation_domain_state"].isna().any():
        missing = int(scores["audit_observation_domain_state"].isna().sum())
        raise ValueError(f"observation-domain audit missing {missing} score fingerprints")
    scores["stable_chain"] &= scores["audit_length_observation_valid"]
    scores["direct_candidate"] = (
        scores["stable_chain"]
        & scores["e_lia_total"].le(LIA_ENDPOINT_TOL)
        & scores["e_rgi_total"].le(RGI_ENDPOINT_TOL)
    )
    scores["lia_reachable_total"] = (
        scores["stable_chain"] & scores["e_lia_total"].le(LIA_ENDPOINT_TOL)
    )
    scores["rgi_reachable_total"] = (
        scores["stable_chain"] & scores["e_rgi_total"].le(RGI_ENDPOINT_TOL)
    )
    return scores.reset_index(drop=True)


def load_master() -> pd.DataFrame:
    path = controller_root() / "lia_191_master_status.csv"
    master = pd.read_csv(path)
    master["lia_unique_id"] = pd.to_numeric(
        master["lia_unique_id"], errors="raise"
    ).astype(int)
    if len(master) != 191 or master["lia_unique_id"].nunique() != 191:
        raise ValueError("controller master status is not a unique 191-system table")
    return master.sort_values("lia_unique_id").reset_index(drop=True)


def route_action(route: str) -> str:
    actions = {
        "freeze_provisional_candidate": (
            "retain the current best chain as a provisional historical curve; do not open a new parameter block"
        ),
        "blocked_dynamic_review": (
            "resolve the zero-MB dynamic gate before any further historical search"
        ),
        "named_configuration_closed": (
            "retain the named configuration diagnosis; do not resume physical micro-search under it"
        ),
        "target_direction_review": (
            "review LIA/RGI target ordering before any new parameter block"
        ),
        "small_separation_review": (
            "define an absolute tolerance first; separation-scaled routing is weak here"
        ),
        "no_stable_chain_review": (
            "review geometry, climate window, and spin-up behaviour before opening a new physical parameter"
        ),
        "layer1_delta_extension": (
            "run one bounded delta-T boundary extension, then stop layer-1 tuning"
        ),
        "layer2_melt_f_delta_t": (
            "run one declared six-point melt_f-delta_t block with one center preflight"
        ),
        "layer3_single_parameter": (
            "choose one of glen_a, temp_melt, or prcp_fac from diagnosis; run one bounded block and one delta-T recenter only"
        ),
        "layer3_closed": (
            "retain the Layer-3 diagnosis; the one permitted single-family block is complete"
        ),
        "hold_after_layer1": (
            "do not open a new parameter family yet; use representative systems to diagnose the missing curve structure"
        ),
    }
    return actions[route]


def _best_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    ranked = frame.sort_values(
        ["e_lia_total", "e_rgi_total", "score_priority", "objective_sep"],
        na_position="last",
    )
    return ranked.iloc[0]


def _best_objective_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    ranked = frame.assign(
        objective_total=frame["e_lia_total"].fillna(np.inf)
        + frame["e_rgi_total"].fillna(np.inf)
    ).sort_values(
        ["objective_total", "score_priority", "objective_sep"],
        na_position="last",
    )
    return ranked.iloc[0]


def classify_system(master_row: pd.Series, system_scores: pd.DataFrame) -> dict:
    system_id = int(master_row["lia_unique_id"])
    target_direction = str(master_row["target_direction"])
    separation = float(master_row["lia_rgi_separation_m"])
    layer1 = system_scores[system_scores["score_layer"].eq("layer1_delta")].copy()
    layer2 = system_scores[system_scores["score_layer"].eq("layer2_melt_delta")].copy()
    layer3 = system_scores[
        system_scores["score_layer"].eq("layer3_single_parameter")
    ].copy()
    stable = system_scores[system_scores["stable_chain"]].copy()
    direct = system_scores[system_scores["direct_candidate"]].copy()
    best_direct = _best_row(direct)
    best_stable = _best_objective_row(stable)
    best_layer1 = _best_objective_row(layer1[layer1["stable_chain"]])
    lia_reach_layer1 = bool(layer1["lia_reachable_total"].any())
    rgi_reach_layer1 = bool(layer1["rgi_reachable_total"].any())
    layer1_stable = layer1[layer1["stable_chain"]].copy()
    layer1_extension_complete = bool(
        layer1["score_source"].astype(str).eq("delta_boundary_extension_v1").any()
    )
    delta_boundary_hit = False
    if not layer1_stable.empty and best_layer1 is not None:
        min_dt = layer1_stable["delta_t"].min()
        max_dt = layer1_stable["delta_t"].max()
        delta_boundary_hit = np.isclose(float(best_layer1["delta_t"]), min_dt) or np.isclose(
            float(best_layer1["delta_t"]), max_dt
        )

    if system_id in MANUAL_PROVISIONAL_IDS:
        route = "freeze_provisional_candidate"
        reason = "manual accepted/provisional decision already retained in controller master status"
    elif system_id in BLOCKED_DYNAMIC_IDS:
        route = "blocked_dynamic_review"
        reason = "system is explicitly blocked by the zero-MB dynamic gate"
    elif system_id in NAMED_CLOSED_IDS:
        route = "named_configuration_closed"
        reason = "named configuration already closed after bounded physical-parameter blocks"
    elif target_direction != "retreat":
        route = "target_direction_review"
        reason = "LIA and RGI do not currently define a retreat target"
    elif separation < SMALL_SEPARATION_THRESHOLD_M:
        route = "small_separation_review"
        reason = "LIA-RGI separation is below the declared 500 m routing threshold"
    elif stable.empty:
        route = "no_stable_chain_review"
        reason = "no stable, numerically clean chain exists in the currently merged results"
    elif not direct.empty:
        route = "freeze_provisional_candidate"
        reason = "at least one stable chain places both peak and year-2000 within 10% of total lengths"
    elif not layer3.empty:
        route = "layer3_closed"
        reason = "the one permitted Layer-3 single-family block is complete and still has no direct 10% endpoint pass"
    elif lia_reach_layer1 and rgi_reach_layer1:
        if not layer2.empty:
            route = "layer3_single_parameter"
            reason = "layer-2 melt_f-delta_t follow-up is complete and still has no direct 10% endpoint pass"
        else:
            route = "layer2_melt_f_delta_t"
            reason = "layer-1 reaches the LIA and RGI endpoints on different stable delta-T chains"
    elif delta_boundary_hit and not layer1_extension_complete:
        route = "layer1_delta_extension"
        reason = "best stable layer-1 delta-T sits on the tested boundary and the one bounded extension is not yet used"
    elif layer1_extension_complete:
        route = "hold_after_layer1"
        reason = "the one bounded layer-1 delta-T extension is complete with no direct pass or explicit two-endpoint bracket"
    else:
        route = "hold_after_layer1"
        reason = "current layer-1 results show no direct pass, no explicit two-endpoint bracket, and no boundary case"

    result = {
        "lia_unique_id": system_id,
        "rgi_id": master_row["rgi_id"],
        "target_direction": target_direction,
        "lia_length_m": float(master_row["lia_length_m"]),
        "rgi_length_m": float(master_row["rgi_length_m"]),
        "lia_rgi_separation_m": separation,
        "existing_decision": master_row.get("existing_decision", ""),
        "existing_decision_level": master_row.get("existing_decision_level", ""),
        "existing_decision_evidence": master_row.get("existing_decision_evidence", ""),
        "tested_chain_count_all_layers": int(len(system_scores)),
        "stable_chain_count_all_layers": int(len(stable)),
        "tested_chain_count_layer1": int(len(layer1)),
        "stable_chain_count_layer1": int(layer1["stable_chain"].sum()),
        "tested_chain_count_layer2": int(len(layer2)),
        "stable_chain_count_layer2": int(layer2["stable_chain"].sum()),
        "tested_chain_count_layer3": int(len(layer3)),
        "stable_chain_count_layer3": int(layer3["stable_chain"].sum()),
        "lia_reachable_layer1": lia_reach_layer1,
        "rgi_reachable_layer1": rgi_reach_layer1,
        "delta_boundary_hit": bool(delta_boundary_hit),
        "layer1_extension_complete": bool(layer1_extension_complete),
        "direct_candidate_exists": bool(not direct.empty),
        "next_route": route,
        "next_action": route_action(route),
        "route_reason": reason,
    }

    if best_direct is not None:
        result.update(
            {
                "best_direct_task_id": best_direct["task_id"],
                "best_direct_source": best_direct["score_source"],
                "best_direct_layer": best_direct["score_layer"],
                "best_direct_delta_t": float(best_direct["delta_t"]),
                "best_direct_melt_f_stage2_multiplier": float(
                    best_direct["melt_f_stage2_multiplier"]
                ),
                "best_direct_e_lia_total": float(best_direct["e_lia_total"]),
                "best_direct_e_rgi_total": float(best_direct["e_rgi_total"]),
                "best_direct_amplitude_ratio_to_target": float(
                    best_direct["amplitude_ratio_to_target"]
                ),
            }
        )
    else:
        result.update(
            {
                "best_direct_task_id": "",
                "best_direct_source": "",
                "best_direct_layer": "",
                "best_direct_delta_t": np.nan,
                "best_direct_melt_f_stage2_multiplier": np.nan,
                "best_direct_e_lia_total": np.nan,
                "best_direct_e_rgi_total": np.nan,
                "best_direct_amplitude_ratio_to_target": np.nan,
            }
        )
    if best_stable is not None:
        result.update(
            {
                "best_stable_task_id": best_stable["task_id"],
                "best_stable_source": best_stable["score_source"],
                "best_stable_layer": best_stable["score_layer"],
                "best_stable_delta_t": float(best_stable["delta_t"]),
                "best_stable_melt_f_stage2_multiplier": float(
                    best_stable["melt_f_stage2_multiplier"]
                ),
                "best_stable_e_lia_total": float(best_stable["e_lia_total"]),
                "best_stable_e_rgi_total": float(best_stable["e_rgi_total"]),
                "best_stable_amplitude_ratio_to_target": float(
                    best_stable["amplitude_ratio_to_target"]
                ),
            }
        )
    else:
        result.update(
            {
                "best_stable_task_id": "",
                "best_stable_source": "",
                "best_stable_layer": "",
                "best_stable_delta_t": np.nan,
                "best_stable_melt_f_stage2_multiplier": np.nan,
                "best_stable_e_lia_total": np.nan,
                "best_stable_e_rgi_total": np.nan,
                "best_stable_amplitude_ratio_to_target": np.nan,
            }
        )
    return result


def build_loop_table(master: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for master_row in master.itertuples(index=False):
        system_scores = scores[scores["lia_unique_id"].eq(int(master_row.lia_unique_id))].copy()
        rows.append(classify_system(pd.Series(master_row._asdict()), system_scores))
    routing = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    if len(routing) != 191 or routing["lia_unique_id"].nunique() != 191:
        raise ValueError("iterative loop table is not a unique 191-system table")
    return routing


def build_summary(routing: pd.DataFrame, scores: pd.DataFrame) -> dict:
    route_counts = routing["next_route"].value_counts().sort_index()
    return {
        "schema_version": 1,
        "launched_oggm": False,
        "current_loop": "three-layer-bounded-iteration",
        "endpoint_tolerance_total_length": {
            "lia_peak": LIA_ENDPOINT_TOL,
            "rgi_2000": RGI_ENDPOINT_TOL,
        },
        "routing_tolerance_separation_only": 0.20,
        "spinup_stability_limit": SPINUP_STABILITY_LIMIT,
        "small_separation_threshold_m": SMALL_SEPARATION_THRESHOLD_M,
        "merged_score_rows": int(len(scores)),
        "merged_score_sources": sorted(scores["score_source"].dropna().unique().tolist()),
        "system_count": int(len(routing)),
        "route_counts": {key: int(value) for key, value in route_counts.items()},
        "direct_candidate_system_count": int(routing["direct_candidate_exists"].sum()),
        "layer2_needed_system_count": int(routing["next_route"].eq("layer2_melt_f_delta_t").sum()),
        "layer3_needed_system_count": int(routing["next_route"].eq("layer3_single_parameter").sum()),
        "layer1_extension_system_count": int(routing["next_route"].eq("layer1_delta_extension").sum()),
        "note": (
            "This table is a bounded next-step router. It is not a final posterior "
            "and does not authorize opening multiple parameter families at once."
        ),
    }


def write_outputs(routing: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    routing.to_csv(output_dir / "core191_iterative_loop_table.csv", index=False)
    (
        routing.groupby(["next_route", "next_action"], as_index=False)
        .size()
        .rename(columns={"size": "system_count"})
        .sort_values(["system_count", "next_route"], ascending=[False, True])
        .to_csv(output_dir / "core191_iterative_loop_counts.csv", index=False)
    )
    routing[routing["direct_candidate_exists"]].to_csv(
        output_dir / "route_freeze_provisional_candidate.csv", index=False
    )
    for route, group in routing.groupby("next_route", sort=True):
        group.to_csv(output_dir / f"route_{route}.csv", index=False)
    (output_dir / "iterative_loop_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=controller_root() / "diagnostics" / "iterative_loop_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = load_master()
    scores = load_scores()
    routing = build_loop_table(master, scores)
    summary = build_summary(routing, scores)
    write_outputs(routing, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote iterative loop outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
