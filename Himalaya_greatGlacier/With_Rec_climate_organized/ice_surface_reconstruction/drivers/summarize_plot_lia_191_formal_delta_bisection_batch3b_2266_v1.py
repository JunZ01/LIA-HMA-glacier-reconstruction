"""Formally score and plot the single system-2266 batch-3b side point."""

from __future__ import annotations

import json

from . import summarize_plot_lia_191_formal_delta_bisection_batch2_v1 as _common
from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_formal_delta_bisection_batch3b_2266_v1"


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
    accepted = scores.loc[scores["formal_dual_pass"]]
    accepted.to_csv(result_root / "accepted_candidates.csv", index=False)
    plot_dir = result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row = scores.iloc[0]
    _common._plot(row, plot_dir / "system_2266_spinup_forward_length.png")
    summary = {
        "schema_version": 1,
        "executed_chain_count": 1,
        "formal_dual_pass_count": int(len(accepted)),
        "formal_dual_pass_systems": accepted["lia_unique_id"].astype(int).tolist(),
    }
    (result_root / "formal_batch3b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    columns = [
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
    print(scores[columns].to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
