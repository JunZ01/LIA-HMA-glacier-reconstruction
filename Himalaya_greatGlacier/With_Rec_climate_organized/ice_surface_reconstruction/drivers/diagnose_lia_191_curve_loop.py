"""Read-only curve diagnosis for the Core-191 adaptive parameter loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOLERANCE = 0.20
MIN_SEPARATION_M = 500.0
STABILITY_LIMIT = 0.02


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def controller_root() -> Path:
    return project_root() / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1"


def _bools(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _classify(row: pd.Series, tolerance: float) -> str:
    if row["technical_failure_bool"] or row["summary_status"] != "ok":
        return "technical_failure"
    if not row["stable_chain"]:
        return "spinup_not_converged"
    if row["target_direction"] != "retreat":
        return "target_direction_review"
    if row["small_separation"]:
        return "small_separation_review"

    peak = row["signed_peak_error_sep"]
    end = row["signed_endpoint_error_sep"]
    peak_ok = abs(peak) <= tolerance
    end_ok = abs(end) <= tolerance
    if peak_ok and end_ok:
        return (
            "joint_endpoints_with_corridor_excursion"
            if row["corridor_penalty_sep"] > 0
            else "joint_endpoint_candidate"
        )
    if peak_ok:
        return (
            "insufficient_retreat_endpoint_too_long"
            if end > tolerance
            else "excessive_retreat_endpoint_too_short"
        )
    if end_ok:
        return "lia_peak_too_long" if peak > tolerance else "lia_peak_too_short"
    if peak > tolerance and end > tolerance:
        return "curve_level_too_long"
    if peak < -tolerance and end < -tolerance:
        return "curve_level_too_short"
    if peak < -tolerance and end > tolerance:
        return "amplitude_deficit"
    if peak > tolerance and end < -tolerance:
        return "amplitude_excess"
    return "mixed_endpoint_miss"


NEXT_ACTION = {
    "joint_endpoint_candidate": "retain_candidate",
    "joint_endpoints_with_corridor_excursion": "review_corridor_excursion",
    "curve_level_too_long": "iterate_delta_t_then_melt_f_toward_shorter_state",
    "curve_level_too_short": "iterate_delta_t_then_melt_f_toward_longer_state",
    "insufficient_retreat_endpoint_too_long": "diagnose_forward_amplitude_or_timing",
    "excessive_retreat_endpoint_too_short": "reduce_forward_retreat_response",
    "lia_peak_too_short": "increase_early_growth_or_shift_state_longer",
    "lia_peak_too_long": "reduce_early_growth_or_shift_state_shorter",
    "amplitude_deficit": "joint_state_and_forward_amplitude_iteration",
    "amplitude_excess": "reduce_curve_amplitude",
    "spinup_not_converged": "repair_or_reestimate_stable_initial_state",
    "technical_failure": "technical_diagnosis_before_parameter_search",
    "target_direction_review": "review_lia_rgi_target_direction",
    "small_separation_review": "review_small_signal_before_parameter_search",
    "mixed_endpoint_miss": "inspect_signed_endpoint_response",
}


def diagnose_chains(
    scores: pd.DataFrame,
    tolerance: float = TOLERANCE,
    min_separation_m: float = MIN_SEPARATION_M,
) -> pd.DataFrame:
    required = {
        "task_id", "lia_unique_id", "rgi_id", "delta_t", "summary_status",
        "technical_failure", "spinup_converged", "spinup_stability_fraction",
        "lia_target_m", "rgi_target_m", "lia_rgi_separation_m",
        "forward_peak_m", "forward_2000_m", "e_lia_sep", "e_rgi_sep",
        "amplitude_ratio_to_target", "max_corridor_exceedance_m",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"scores.csv is missing required columns: {missing}")

    out = scores.copy()
    numeric = [
        "lia_unique_id", "delta_t", "spinup_stability_fraction", "lia_target_m",
        "rgi_target_m", "lia_rgi_separation_m", "forward_peak_m",
        "forward_2000_m", "e_lia_sep", "e_rgi_sep",
        "amplitude_ratio_to_target", "max_corridor_exceedance_m",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["technical_failure_bool"] = _bools(out["technical_failure"])
    out["spinup_converged_bool"] = _bools(out["spinup_converged"])
    out["stable_chain"] = (
        ~out["technical_failure_bool"]
        & out["summary_status"].astype(str).eq("ok")
        & out["spinup_converged_bool"]
        & out["spinup_stability_fraction"].le(STABILITY_LIMIT)
    )
    out["target_direction"] = np.where(
        out["lia_target_m"] > out["rgi_target_m"], "retreat", "growth_or_target_review"
    )
    out["small_separation"] = out["lia_rgi_separation_m"].lt(min_separation_m)
    separation = out["lia_rgi_separation_m"].where(out["lia_rgi_separation_m"].gt(0))
    out["signed_peak_error_sep"] = (out["forward_peak_m"] - out["lia_target_m"]) / separation
    out["signed_endpoint_error_sep"] = (out["forward_2000_m"] - out["rgi_target_m"]) / separation
    out["corridor_penalty_sep"] = out["max_corridor_exceedance_m"].clip(lower=0) / separation
    out["curve_objective_loop_v1"] = (
        out["e_lia_sep"] + out["e_rgi_sep"] + out["corridor_penalty_sep"].fillna(0)
    )
    out["curve_class"] = out.apply(_classify, axis=1, tolerance=tolerance)
    out["next_action"] = out["curve_class"].map(NEXT_ACTION)
    out["spinup_length_used_as_lia_constraint"] = False
    return out


def summarize_systems(chains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system_id, group in chains.groupby("lia_unique_id", sort=True):
        stable = group[group["stable_chain"]]
        eligible = stable[
            stable["target_direction"].eq("retreat") & ~stable["small_separation"]
        ]
        pool = eligible if not eligible.empty else stable
        best = None if pool.empty else pool.sort_values(
            ["curve_objective_loop_v1", "e_lia_sep", "e_rgi_sep", "delta_t"]
        ).iloc[0]
        first = group.iloc[0]
        rows.append({
            "lia_unique_id": int(system_id),
            "rgi_id": first["rgi_id"],
            "target_direction": first["target_direction"],
            "lia_rgi_separation_m": first["lia_rgi_separation_m"],
            "tested_chain_count": len(group),
            "stable_chain_count": len(stable),
            "best_task_id": "" if best is None else best["task_id"],
            "best_delta_t": np.nan if best is None else best["delta_t"],
            "best_curve_class": "no_stable_chain" if best is None else best["curve_class"],
            "best_next_action": (
                "repair_or_reestimate_stable_initial_state" if best is None else best["next_action"]
            ),
            "best_curve_objective_loop_v1": np.nan if best is None else best["curve_objective_loop_v1"],
            "best_e_lia_sep": np.nan if best is None else best["e_lia_sep"],
            "best_e_rgi_sep": np.nan if best is None else best["e_rgi_sep"],
            "best_signed_peak_error_sep": np.nan if best is None else best["signed_peak_error_sep"],
            "best_signed_endpoint_error_sep": np.nan if best is None else best["signed_endpoint_error_sep"],
            "best_amplitude_ratio_to_target": np.nan if best is None else best["amplitude_ratio_to_target"],
            "best_corridor_penalty_sep": np.nan if best is None else best["corridor_penalty_sep"],
        })
    return pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=controller_root() / "scores.csv")
    parser.add_argument("--output-dir", type=Path, default=controller_root() / "diagnostics/curve_loop_v1")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--min-separation-m", type=float, default=MIN_SEPARATION_M)
    args = parser.parse_args()

    chains = diagnose_chains(pd.read_csv(args.scores), args.tolerance, args.min_separation_m)
    systems = summarize_systems(chains)
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "new_oggm_runs": 0,
        "spinup_length_used_as_lia_constraint": False,
        "lia_constraint": "maximum forward length",
        "modern_constraint": "year-2000 forward length",
        "endpoint_tolerance_fraction_of_lia_rgi_separation": args.tolerance,
        "spinup_stability_limit_fraction": STABILITY_LIMIT,
        "minimum_separation_m": args.min_separation_m,
        "chain_count": len(chains),
        "system_count": len(systems),
        "stable_chain_count": int(chains["stable_chain"].sum()),
        "systems_with_stable_chain": int(systems["stable_chain_count"].gt(0).sum()),
        "chain_class_counts": {str(k): int(v) for k, v in chains["curve_class"].value_counts().items()},
        "best_system_class_counts": {str(k): int(v) for k, v in systems["best_curve_class"].value_counts().items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chains.to_csv(args.output_dir / "curve_loop_chain_diagnostics.csv", index=False)
    systems.to_csv(args.output_dir / "curve_loop_system_best.csv", index=False)
    chains[chains["curve_class"].str.startswith("joint_endpoint")].to_csv(
        args.output_dir / "curve_loop_candidate_chains.csv", index=False
    )
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
