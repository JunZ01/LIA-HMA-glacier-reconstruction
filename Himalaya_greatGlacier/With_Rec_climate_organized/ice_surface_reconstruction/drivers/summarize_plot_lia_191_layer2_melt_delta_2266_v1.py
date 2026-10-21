"""Formally score and plot both completed system-2266 Layer-2 blocks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import summarize_plot_lia_191_formal_delta_bisection_batch2_v1 as _common
from .build_lia_191_iterative_loop import project_root


EXPERIMENTS = (
    "lia_191_layer2_melt_delta_2266_v1",
    "lia_191_layer2_melt_delta_2266_refine_v1",
)
OUTPUT_NAME = "lia_191_layer2_melt_delta_2266_consolidated_v1"


def _score(path: Path) -> pd.DataFrame:
    old_files = _common.SCORE_FILES
    try:
        _common.SCORE_FILES = [path / "scores.csv"]
        frame = _common._formal_scores()
    finally:
        _common.SCORE_FILES = old_files
    frame.to_csv(path / "formal_scores.csv", index=False)
    return frame


def _plot(scores: pd.DataFrame, output: Path) -> None:
    lia = float(scores.iloc[0]["lia_target_m"])
    rgi = float(scores.iloc[0]["rgi_target_m"])
    separation = abs(lia - rgi)
    lower = min(lia, rgi) - 0.20 * separation
    upper = max(lia, rgi) + 0.20 * separation
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
    colors = {0.95: "#2a9d8f", 1.05: "#e76f51", 1.10: "#9b5de5"}
    y_values: list[float] = [lower / 1000.0, upper / 1000.0]
    for _, row in scores.sort_values(["melt_f_stage2_multiplier", "delta_t"]).iterrows():
        series = pd.read_csv(Path(str(row["output_dir"])) / "timeseries.csv")
        series["length_km"] = pd.to_numeric(series["observation_length_m"]) / 1000.0
        spin = series.loc[series["phase"].eq("spinup")]
        forward = series.loc[series["phase"].eq("forward")]
        multiplier = round(float(row["melt_f_stage2_multiplier"]), 2)
        label = f"melt x{multiplier:.2f}, dT={float(row['delta_t']):+.3f}"
        color = colors[multiplier]
        axes[0].plot(spin["year"], spin["length_km"], color=color, alpha=0.78, linewidth=1.6)
        axes[1].plot(
            forward["year"], forward["length_km"], color=color, alpha=0.82,
            linewidth=1.7, label=label,
        )
        y_values.extend(series["length_km"].dropna().tolist())
    for axis in axes:
        axis.axhspan(lower / 1000.0, upper / 1000.0, color="#dceeff", alpha=0.50)
        axis.axhline(lia / 1000.0, color="#c73e1d", linestyle="--", linewidth=1.5)
        axis.axhline(rgi / 1000.0, color="#2b6cb0", linestyle="--", linewidth=1.5)
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[0].set_title("Adaptive spin-up")
    axes[0].set_xlabel("Spin-up model year")
    axes[0].set_ylabel("Observation-domain length (km)")
    axes[1].set_title("Forward reconstruction")
    axes[1].set_xlabel("Calendar year")
    axes[1].legend(loc="best", fontsize=7.3, ncol=2, frameon=False)
    axes[0].set_ylim(min(y_values) - 0.25, max(y_values) + 0.25)
    fig.suptitle("LIA system 2266 | Layer-2 melt_f + delta-T response", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = project_root() / "validation_data" / "results" / "paleo_surface_validation_v0"
    frames = [_score(root / experiment) for experiment in EXPERIMENTS]
    scores = pd.concat(frames, ignore_index=True, sort=False)
    scores = scores.sort_values(["formal_objective", "melt_f_stage2_multiplier", "delta_t"])
    output = root / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "formal_scores_combined.csv", index=False)
    best = scores.iloc[0]
    scores.head(1).to_csv(output / "best_new_point.csv", index=False)
    plot_path = output / "system_2266_layer2_melt_delta_curves.png"
    _plot(scores, plot_path)
    summary = {
        "schema_version": 1,
        "executed_chain_count": int(len(scores)),
        "technical_clean_count": int(scores["technical_clean"].sum()),
        "stable_valid_count": int(scores["stable_valid"].sum()),
        "formal_dual_pass_count": int(scores["formal_dual_pass"].sum()),
        "best_new_point": {
            "task_id": str(best["task_id"]),
            "delta_t": float(best["delta_t"]),
            "melt_f_stage2_multiplier": float(best["melt_f_stage2_multiplier"]),
            "e_lia_sep": float(best["e_lia_sep_formal"]),
            "e_rgi_sep": float(best["e_rgi_sep_formal"]),
            "formal_objective": float(best["formal_objective"]),
        },
        "decision": (
            "No formal pass. Increasing melt_f reduced the response amplitude; the "
            "single opposite-direction refinement returned to the same RGI length "
            "plateau. Close Layer-2 melt_f + delta-T for system 2266."
        ),
        "plot": str(plot_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
