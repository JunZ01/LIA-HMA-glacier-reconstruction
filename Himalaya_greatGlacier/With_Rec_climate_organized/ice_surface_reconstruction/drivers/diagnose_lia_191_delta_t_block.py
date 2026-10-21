"""Build the read-only first-layer diagnosis for the Core-191 delta-T block.

This script does not launch OGGM and does not modify controller state.  It
summarizes the completed B-window Stage-2-physics delta-T sweep at system level
using only chains that pass the declared numerical and adaptive spin-up gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TOLERANCE = 0.20
DEFAULT_MIN_SEPARATION_M = 500.0
SPINUP_STABILITY_LIMIT = 0.02
DELTA_T_LOWER_BOUND = -3.0
DELTA_T_UPPER_BOUND = 1.5


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


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _row_value(row: pd.Series | None, key: str, default=np.nan):
    if row is None:
        return default
    return row.get(key, default)


def diagnose_systems(
    scores: pd.DataFrame,
    tolerance: float,
    min_separation_m: float,
) -> pd.DataFrame:
    required = {
        "lia_unique_id",
        "rgi_id",
        "delta_t",
        "technical_failure",
        "summary_status",
        "spinup_converged",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "lia_rgi_separation_m",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
        "forward_peak_m",
        "forward_2000_m",
        "peak_to_2000_amplitude_m",
        "amplitude_ratio_to_target",
        "max_corridor_exceedance_m",
        "task_id",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"scores.csv is missing required columns: {missing}")

    scores = scores.copy()
    _numeric(
        scores,
        [
            "lia_unique_id",
            "delta_t",
            "spinup_stability_fraction",
            "lia_target_m",
            "rgi_target_m",
            "lia_rgi_separation_m",
            "e_lia_sep",
            "e_rgi_sep",
            "objective_sep",
            "forward_peak_m",
            "forward_2000_m",
            "peak_to_2000_amplitude_m",
            "amplitude_ratio_to_target",
            "max_corridor_exceedance_m",
        ],
    )
    scores["technical_failure_bool"] = _as_bool(scores["technical_failure"])
    scores["spinup_converged_bool"] = _as_bool(scores["spinup_converged"])
    scores["stable_chain"] = (
        ~scores["technical_failure_bool"]
        & scores["summary_status"].eq("ok")
        & scores["spinup_converged_bool"]
        & scores["spinup_stability_fraction"].le(SPINUP_STABILITY_LIMIT)
    )

    rows: list[dict] = []
    for system_id, group in scores.groupby("lia_unique_id", sort=True):
        group = group.sort_values("delta_t")
        stable = group[group["stable_chain"]].copy()
        first = group.iloc[0]
        lia = float(first["lia_target_m"])
        rgi = float(first["rgi_target_m"])
        separation = float(first["lia_rgi_separation_m"])
        target_direction = "retreat" if lia > rgi else "growth_or_target_review"
        small_separation = bool(separation < min_separation_m)

        best = None
        best_lia = None
        best_rgi = None
        joint_count = 0
        if not stable.empty:
            best = stable.sort_values(["objective_sep", "delta_t"]).iloc[0]
            best_lia = stable.sort_values(["e_lia_sep", "objective_sep"]).iloc[0]
            best_rgi = stable.sort_values(["e_rgi_sep", "objective_sep"]).iloc[0]
            joint_count = int(
                (stable["e_lia_sep"].le(tolerance) & stable["e_rgi_sep"].le(tolerance)).sum()
            )

        min_lia = float(_row_value(best_lia, "e_lia_sep"))
        min_rgi = float(_row_value(best_rgi, "e_rgi_sep"))
        lia_reachable = bool(np.isfinite(min_lia) and min_lia <= tolerance)
        rgi_reachable = bool(np.isfinite(min_rgi) and min_rgi <= tolerance)

        if stable.empty:
            reachability = "no_stable_chain"
        elif joint_count:
            reachability = "joint"
        elif lia_reachable and rgi_reachable:
            reachability = "tradeoff_different_delta_t"
        elif lia_reachable:
            reachability = "lia_only"
        elif rgi_reachable:
            reachability = "rgi_only"
        else:
            reachability = "neither"

        best_delta_t = float(_row_value(best, "delta_t"))
        boundary_side = ""
        if np.isfinite(best_delta_t):
            if np.isclose(best_delta_t, DELTA_T_LOWER_BOUND):
                boundary_side = "cold"
            elif np.isclose(best_delta_t, DELTA_T_UPPER_BOUND):
                boundary_side = "warm"

        clean_retreat = target_direction == "retreat" and not small_separation
        rows.append(
            {
                "lia_unique_id": int(system_id),
                "rgi_id": first["rgi_id"],
                "lia_target_m": lia,
                "rgi_target_m": rgi,
                "lia_rgi_separation_m": separation,
                "target_direction": target_direction,
                "small_separation_lt_threshold": small_separation,
                "clean_retreat_sep_ge_threshold": clean_retreat,
                "tested_chain_count": int(len(group)),
                "stable_chain_count": int(len(stable)),
                "reachability_at_tolerance": reachability,
                "joint_chain_count_at_tolerance": joint_count,
                "lia_reachable_at_tolerance": lia_reachable,
                "rgi_reachable_at_tolerance": rgi_reachable,
                "min_e_lia_sep": min_lia,
                "delta_t_at_min_e_lia": _row_value(best_lia, "delta_t"),
                "task_at_min_e_lia": _row_value(best_lia, "task_id", ""),
                "min_e_rgi_sep": min_rgi,
                "delta_t_at_min_e_rgi": _row_value(best_rgi, "delta_t"),
                "task_at_min_e_rgi": _row_value(best_rgi, "task_id", ""),
                "best_task_id": _row_value(best, "task_id", ""),
                "best_delta_t": best_delta_t,
                "best_objective_sep": _row_value(best, "objective_sep"),
                "best_e_lia_sep": _row_value(best, "e_lia_sep"),
                "best_e_rgi_sep": _row_value(best, "e_rgi_sep"),
                "best_spinup_stability_fraction": _row_value(
                    best, "spinup_stability_fraction"
                ),
                "best_forward_peak_m": _row_value(best, "forward_peak_m"),
                "best_forward_2000_m": _row_value(best, "forward_2000_m"),
                "best_peak_to_2000_amplitude_m": _row_value(
                    best, "peak_to_2000_amplitude_m"
                ),
                "best_amplitude_ratio_to_target": _row_value(
                    best, "amplitude_ratio_to_target"
                ),
                "best_max_corridor_exceedance_m": _row_value(
                    best, "max_corridor_exceedance_m"
                ),
                "best_delta_t_boundary": boundary_side,
                "eligible_tradeoff_pilot": bool(
                    clean_retreat and reachability == "tradeoff_different_delta_t"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)


def build_summary(
    systems: pd.DataFrame,
    tolerance: float,
    min_separation_m: float,
) -> dict:
    clean = systems[systems["clean_retreat_sep_ge_threshold"]]
    reachability = clean["reachability_at_tolerance"].value_counts().to_dict()
    return {
        "schema_version": 1,
        "source": "lia_191_controller_v1/scores.csv",
        "endpoint_tolerance_fraction_of_separation": tolerance,
        "minimum_separation_m": min_separation_m,
        "spinup_stability_limit_fraction": SPINUP_STABILITY_LIMIT,
        "searched_system_count": int(len(systems)),
        "stable_chain_count": int(systems["stable_chain_count"].sum()),
        "systems_with_stable_chain": int(systems["stable_chain_count"].gt(0).sum()),
        "growth_or_target_review_count": int(
            systems["target_direction"].eq("growth_or_target_review").sum()
        ),
        "small_separation_count": int(systems["small_separation_lt_threshold"].sum()),
        "clean_retreat_sep_ge_threshold_count": int(len(clean)),
        "clean_reachability_counts": {key: int(value) for key, value in reachability.items()},
        "warm_boundary_best_count": int(systems["best_delta_t_boundary"].eq("warm").sum()),
        "cold_boundary_best_count": int(systems["best_delta_t_boundary"].eq("cold").sum()),
        "tradeoff_pilot_ids": systems.loc[
            systems["eligible_tradeoff_pilot"], "lia_unique_id"
        ].astype(int).tolist(),
    }


def write_outputs(systems: pd.DataFrame, output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    systems.to_csv(output_dir / "delta_t_block_system_diagnostics.csv", index=False)
    systems[systems["eligible_tradeoff_pilot"]].to_csv(
        output_dir / "tradeoff_pilot_systems.csv", index=False
    )
    systems[systems["target_direction"].eq("growth_or_target_review")].to_csv(
        output_dir / "growth_or_target_review_systems.csv", index=False
    )
    systems[systems["small_separation_lt_threshold"]].to_csv(
        output_dir / "small_separation_systems.csv", index=False
    )
    systems[systems["stable_chain_count"].eq(0)].to_csv(
        output_dir / "no_stable_chain_systems.csv", index=False
    )
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=controller_root() / "scores.csv",
        help="Completed controller score table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=controller_root() / "diagnostics" / "delta_t_block_v1",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--min-separation-m", type=float, default=DEFAULT_MIN_SEPARATION_M
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.scores)
    systems = diagnose_systems(scores, args.tolerance, args.min_separation_m)
    summary = build_summary(systems, args.tolerance, args.min_separation_m)
    write_outputs(systems, args.output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
