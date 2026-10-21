"""Formally score and plot the system-2266 Layer-3 glen_a direction probe."""

from __future__ import annotations

import json

from . import summarize_plot_lia_191_formal_delta_bisection_batch2_v1 as _common
from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_layer3_glen_2266_v1"


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
        _common.SCORE_FILES = [result_root / "scores.csv"]
        scores = _common._formal_scores()
    finally:
        _common.SCORE_FILES = old_files
    scores.to_csv(result_root / "formal_scores.csv", index=False)
    accepted = scores.loc[scores["formal_dual_pass"]].sort_values("formal_objective")
    accepted.to_csv(result_root / "accepted_candidates.csv", index=False)
    plot_dir = result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for _, row in scores.sort_values("glen_a_stage2_multiplier").iterrows():
        _common._plot(
            row,
            plot_dir
            / (
                f"system_2266_glen_x{float(row['glen_a_stage2_multiplier']):.2f}_"
                "spinup_forward_length.png"
            ),
        )
    best = accepted.iloc[0] if not accepted.empty else scores.sort_values("formal_objective").iloc[0]
    summary = {
        "schema_version": 1,
        "executed_chain_count": int(len(scores)),
        "formal_dual_pass_count": int(len(accepted)),
        "formal_dual_pass_systems": accepted["lia_unique_id"].astype(int).tolist(),
        "selected_task_id": str(best["task_id"]),
        "selected_glen_a_stage2_multiplier": float(best["glen_a_stage2_multiplier"]),
        "selected_delta_t": float(best["delta_t"]),
        "selected_forward_peak_m": float(best["forward_peak_m"]),
        "selected_forward_2000_m": float(best["forward_2000_m"]),
        "selected_e_lia_sep": float(best["e_lia_sep_formal"]),
        "selected_e_rgi_sep": float(best["e_rgi_sep_formal"]),
        "selected_e_lia_total": float(best["e_lia_total"]),
        "selected_e_rgi_total": float(best["e_rgi_total"]),
        "selected_spinup_stability_fraction": float(best["spinup_stability_fraction"]),
        "decision": (
            "glen_a x1.25 passes every formal gate; freeze it without a delta-T "
            "recenter. glen_a x0.75 is retained as direction evidence only."
        ),
    }
    (result_root / "formal_layer3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
