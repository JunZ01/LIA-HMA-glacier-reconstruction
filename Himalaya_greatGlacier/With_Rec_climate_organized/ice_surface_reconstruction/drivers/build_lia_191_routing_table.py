"""Build the read-only Core-191 routing table from completed diagnostics.

This script launches no OGGM task and does not mutate controller state.  It
combines the completed 186-system delta-T diagnosis with the five systems that
already have explicit retained decisions in the controller master table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRADEOFF_CLOSED_IDS = {2259, 4959}
SPECIAL_ROUTE = {
    1044: "provisional_historical_candidate",
    1545: "unresolved_named_configuration",
    3584: "unresolved_named_configuration",
    2387: "blocked_dynamic_review",
    5508: "blocked_dynamic_review",
}

ROUTE_ACTION = {
    "provisional_historical_candidate": (
        "retain the named candidate; do not retune physical parameters"
    ),
    "unresolved_named_configuration": (
        "retain the diagnosed failure mode; do not resume parameter micro-search"
    ),
    "blocked_dynamic_review": (
        "resolve the zero-MB dynamic gate before any historical parameter search"
    ),
    "no_stable_chain": (
        "review adaptive spin-up, window, and geometry before endpoint scoring"
    ),
    "target_direction_review": (
        "review LIA/RGI target direction and interpretation before parameter search"
    ),
    "small_separation_review": (
        "define an absolute observational tolerance; separation-scaled routing is weak"
    ),
    "endpoint_tradeoff_melt_f_delta_t_closed": (
        "do not continue melt_f-delta_t refinement; declare a secondary experiment question"
    ),
    "lia_only": (
        "design a modern-endpoint and retreat-amplitude experiment"
    ),
    "rgi_only": (
        "design an LIA-state and early-curve-level experiment"
    ),
    "neither": (
        "use representative systems to design a structured state-parameter block"
    ),
    "joint": "promote the chain to candidate validation",
}


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


def _as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def route_searched(row: pd.Series) -> str:
    system_id = int(row["lia_unique_id"])
    if int(row["stable_chain_count"]) == 0:
        return "no_stable_chain"
    if row["target_direction"] != "retreat":
        return "target_direction_review"
    if _as_bool(row["small_separation_lt_threshold"]):
        return "small_separation_review"
    reachability = str(row["reachability_at_tolerance"])
    if system_id in TRADEOFF_CLOSED_IDS:
        if reachability != "tradeoff_different_delta_t":
            raise ValueError(
                f"System {system_id} no longer has the verified tradeoff route: "
                f"{reachability}"
            )
        return "endpoint_tradeoff_melt_f_delta_t_closed"
    if reachability == "tradeoff_different_delta_t":
        raise ValueError(
            f"Unexpected unreviewed tradeoff system {system_id}; routing must be explicit"
        )
    return reachability


def build_routing_table(
    diagnostics: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = diagnostics.copy()
    master = master.copy()
    diagnostics["lia_unique_id"] = pd.to_numeric(
        diagnostics["lia_unique_id"], errors="raise"
    ).astype(int)
    master["lia_unique_id"] = pd.to_numeric(
        master["lia_unique_id"], errors="raise"
    ).astype(int)

    if len(master) != 191 or master["lia_unique_id"].nunique() != 191:
        raise ValueError("Controller master status is not a unique 191-system table")
    if len(diagnostics) != 186 or diagnostics["lia_unique_id"].nunique() != 186:
        raise ValueError("Delta-T diagnosis is not the verified unique 186-system table")

    rows: list[dict] = []
    master_by_id = master.set_index("lia_unique_id", drop=False)
    for _, diag in diagnostics.sort_values("lia_unique_id").iterrows():
        system_id = int(diag["lia_unique_id"])
        master_row = master_by_id.loc[system_id]
        route = route_searched(diag)
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": diag["rgi_id"],
                "lia_target_m": diag["lia_target_m"],
                "rgi_target_m": diag["rgi_target_m"],
                "lia_rgi_separation_m": diag["lia_rgi_separation_m"],
                "target_direction": diag["target_direction"],
                "primary_route": route,
                "next_action": ROUTE_ACTION[route],
                "routing_source": "completed_10_point_delta_t_block",
                "tested_chain_count": diag["tested_chain_count"],
                "stable_chain_count": diag["stable_chain_count"],
                "reachability_at_20pct_separation": diag[
                    "reachability_at_tolerance"
                ],
                "min_e_lia_sep": diag["min_e_lia_sep"],
                "min_e_rgi_sep": diag["min_e_rgi_sep"],
                "best_delta_t": diag["best_delta_t"],
                "best_delta_t_boundary": diag["best_delta_t_boundary"],
                "best_amplitude_ratio_to_target": diag[
                    "best_amplitude_ratio_to_target"
                ],
                "best_task_id": diag["best_task_id"],
                "existing_decision": master_row.get("existing_decision", ""),
                "existing_decision_evidence": master_row.get(
                    "existing_decision_evidence", ""
                ),
            }
        )

    searched_ids = set(diagnostics["lia_unique_id"].astype(int))
    missing_ids = set(master["lia_unique_id"].astype(int)) - searched_ids
    if missing_ids != set(SPECIAL_ROUTE):
        raise ValueError(
            f"Unexpected systems outside the delta-T block: {sorted(missing_ids)}"
        )

    for system_id in sorted(missing_ids):
        master_row = master_by_id.loc[system_id]
        route = SPECIAL_ROUTE[system_id]
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": master_row["rgi_id"],
                "lia_target_m": master_row["lia_length_m"],
                "rgi_target_m": master_row["rgi_length_m"],
                "lia_rgi_separation_m": master_row["lia_rgi_separation_m"],
                "target_direction": master_row["target_direction"],
                "primary_route": route,
                "next_action": ROUTE_ACTION[route],
                "routing_source": "retained_controller_decision",
                "tested_chain_count": np.nan,
                "stable_chain_count": np.nan,
                "reachability_at_20pct_separation": "not_in_initial_delta_t_block",
                "min_e_lia_sep": np.nan,
                "min_e_rgi_sep": np.nan,
                "best_delta_t": np.nan,
                "best_delta_t_boundary": "",
                "best_amplitude_ratio_to_target": np.nan,
                "best_task_id": "",
                "existing_decision": master_row.get("existing_decision", ""),
                "existing_decision_evidence": master_row.get(
                    "existing_decision_evidence", ""
                ),
            }
        )

    routing = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    if len(routing) != 191 or routing["lia_unique_id"].nunique() != 191:
        raise ValueError("Final routing table is not a unique 191-system table")
    if routing["primary_route"].isna().any():
        raise ValueError("Final routing table contains an empty route")
    return routing


def build_summary(routing: pd.DataFrame) -> dict:
    counts = routing["primary_route"].value_counts().sort_index()
    return {
        "schema_version": 1,
        "created_from_saved_results_only": True,
        "launched_oggm": False,
        "full_system_count": int(len(routing)),
        "route_counts": {key: int(value) for key, value in counts.items()},
        "searched_delta_t_system_count": int(
            routing["routing_source"].eq("completed_10_point_delta_t_block").sum()
        ),
        "retained_decision_system_count": int(
            routing["routing_source"].eq("retained_controller_decision").sum()
        ),
        "endpoint_tolerance": "20% of absolute LIA-RGI length separation",
        "note": (
            "This is a routing diagnosis, not a final observational uncertainty "
            "model or an accepted parameter posterior."
        ),
    }


def write_outputs(routing: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    routing.to_csv(output_dir / "core191_routing_table.csv", index=False)
    counts = (
        routing.groupby(["primary_route", "next_action"], as_index=False)
        .size()
        .rename(columns={"size": "system_count"})
        .sort_values(["system_count", "primary_route"], ascending=[False, True])
    )
    counts.to_csv(output_dir / "core191_route_counts.csv", index=False)
    for route, group in routing.groupby("primary_route", sort=True):
        group.to_csv(output_dir / f"route_{route}.csv", index=False)
    (output_dir / "routing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = controller_root()
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=root
        / "diagnostics"
        / "delta_t_block_v1"
        / "delta_t_block_system_diagnostics.csv",
    )
    parser.add_argument(
        "--master-status",
        type=Path,
        default=root / "lia_191_master_status.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "diagnostics" / "core191_routing_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = pd.read_csv(args.diagnostics)
    master = pd.read_csv(args.master_status)
    routing = build_routing_table(diagnostics, master)
    summary = build_summary(routing)
    write_outputs(routing, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote routing outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
