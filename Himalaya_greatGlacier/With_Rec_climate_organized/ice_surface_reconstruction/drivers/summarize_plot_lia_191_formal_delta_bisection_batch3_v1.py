"""Formally score and plot the three batch-3 midpoint chains."""

from __future__ import annotations

import json

import pandas as pd

from . import summarize_plot_lia_191_formal_delta_bisection_batch2_v1 as _common
from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_formal_delta_bisection_batch3_v1"


def main() -> None:
    result_root = (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT
    )
    old_files = _common.SCORE_FILES
    try:
        _common.SCORE_FILES = [result_root / "runs" / "scores.csv"]
        scores = _common._formal_scores()
    finally:
        _common.SCORE_FILES = old_files
    scores.to_csv(result_root / "formal_scores.csv", index=False)
    accepted = scores.loc[scores["formal_dual_pass"]].sort_values(
        ["formal_objective", "lia_unique_id"]
    )
    accepted.to_csv(result_root / "accepted_candidates.csv", index=False)
    plot_dir = result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for _, row in scores.sort_values("lia_unique_id").iterrows():
        _common._plot(
            row,
            plot_dir / f"system_{int(row['lia_unique_id'])}_spinup_forward_length.png",
        )
    stopped: dict[str, str] = {
        "2748": "existing outlet geometry gate failed before dynamics"
    }
    for row in scores.itertuples(index=False):
        system_id = str(int(row.lia_unique_id))
        if bool(row.formal_dual_pass):
            continue
        if not bool(row.technical_clean):
            stopped[system_id] = "technical failure"
        elif not bool(row.spinup_converged_bool) or float(row.spinup_stability_fraction) > 0.02:
            stopped[system_id] = "adaptive spin-up stability gate failed"
        elif not bool(row.observation_valid):
            stopped[system_id] = "observation-domain length invalid"
        else:
            stopped[system_id] = "stable valid midpoint missed one or both formal endpoints"
    summary = {
        "schema_version": 1,
        "executed_chain_count": int(len(scores)),
        "formal_dual_pass_count": int(len(accepted)),
        "formal_dual_pass_systems": accepted["lia_unique_id"].astype(int).tolist(),
        "stopped_systems": stopped,
        "side_points_released": False,
        "reason_side_points_not_run": (
            "This batch releases one midpoint per remaining candidate; a failed midpoint "
            "closes the bounded edge branch and a formal pass is frozen without side points."
        ),
    }
    (result_root / "formal_batch3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    display_columns = [
        "lia_unique_id",
        "delta_t",
        "forward_peak_m",
        "forward_2000_m",
        "e_lia_sep_formal",
        "e_rgi_sep_formal",
        "e_lia_total",
        "e_rgi_total",
        "spinup_stability_fraction",
        "formal_dual_pass",
    ]
    print(scores[display_columns].sort_values("lia_unique_id").to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
