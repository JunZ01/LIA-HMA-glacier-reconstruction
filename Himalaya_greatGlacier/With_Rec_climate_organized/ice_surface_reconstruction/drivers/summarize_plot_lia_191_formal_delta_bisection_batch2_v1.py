"""Formally score batch-2 preflights and plot the accepted curves separately."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_lia_191_formal_delta_bisection_batch2_v1 import RESULT_ROOT


SCORE_FILES = [
    RESULT_ROOT / "runs" / "scores.csv",
    RESULT_ROOT / "system_preflights_v2" / "scores.csv",
]


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _formal_scores() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in SCORE_FILES if path.exists()]
    scores = pd.concat(frames, ignore_index=True, sort=False)
    scores = scores.sort_values("task_id").drop_duplicates("fingerprint", keep="last")
    scores["technical_clean"] = ~scores["technical_failure"].map(_as_bool)
    scores["spinup_converged_bool"] = scores["spinup_converged"].map(_as_bool)
    scores["observation_valid"] = (
        scores["length_observation_valid"].map(_as_bool)
        & ~scores["observation_domain_overflow"].map(_as_bool)
        & ~scores["observation_domain_unresolved"].map(_as_bool)
    )
    scores["stable_valid"] = (
        scores["technical_clean"]
        & scores["spinup_converged_bool"]
        & pd.to_numeric(scores["spinup_stability_fraction"], errors="coerce").le(0.02)
        & scores["observation_valid"]
    )
    lia = pd.to_numeric(scores["lia_target_m"], errors="coerce")
    rgi = pd.to_numeric(scores["rgi_target_m"], errors="coerce")
    peak = pd.to_numeric(scores["forward_peak_m"], errors="coerce")
    endpoint = pd.to_numeric(scores["forward_2000_m"], errors="coerce")
    separation = (lia - rgi).abs()
    scores["e_lia_sep_formal"] = (peak - lia).abs() / separation
    scores["e_rgi_sep_formal"] = (endpoint - rgi).abs() / separation
    scores["e_lia_total"] = (peak - lia).abs() / lia
    scores["e_rgi_total"] = (endpoint - rgi).abs() / rgi
    scores["formal_dual_pass"] = (
        scores["stable_valid"]
        & scores["e_lia_sep_formal"].le(0.20)
        & scores["e_rgi_sep_formal"].le(0.20)
        & scores["e_lia_total"].le(0.10)
        & scores["e_rgi_total"].le(0.10)
    )
    scores["formal_objective"] = (
        scores["e_lia_sep_formal"] + scores["e_rgi_sep_formal"]
    )
    return scores


def _plot(row: pd.Series, output: Path) -> None:
    series = pd.read_csv(Path(str(row["output_dir"])) / "timeseries.csv")
    length_column = "observation_length_m" if "observation_length_m" in series else "length_m"
    series["plot_length_km"] = pd.to_numeric(series[length_column], errors="coerce") / 1000.0
    spin = series.loc[series["phase"].eq("spinup")].copy()
    forward = series.loc[series["phase"].eq("forward")].copy()
    lia = float(row["lia_target_m"])
    rgi = float(row["rgi_target_m"])
    separation = abs(lia - rgi)
    lower = min(lia, rgi) - 0.20 * separation
    upper = max(lia, rgi) + 0.20 * separation
    y_min = min(float(series["plot_length_km"].min()), lower / 1000.0) - 0.20
    y_max = max(float(series["plot_length_km"].max()), upper / 1000.0) + 0.20

    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 5.0), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.55]}
    )
    for axis in axes:
        axis.axhspan(lower / 1000.0, upper / 1000.0, color="#dceeff", alpha=0.55)
        axis.axhline(lia / 1000.0, color="#c73e1d", linewidth=1.5, linestyle="--", label="LIA target")
        axis.axhline(rgi / 1000.0, color="#2b6cb0", linewidth=1.5, linestyle="--", label="RGI target")
        axis.set_ylim(y_min, y_max)
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[0].plot(spin["year"], spin["plot_length_km"], color="#444444", marker="o", linewidth=2.0)
    axes[0].set_xlabel("Spin-up model year")
    axes[0].set_ylabel("Observation-domain length (km)")
    axes[0].set_title("Adaptive spin-up")

    axes[1].plot(
        forward["year"], forward["plot_length_km"], color="#168aad", marker="o", markersize=3.0, linewidth=2.0
    )
    peak_index = forward["plot_length_km"].idxmax()
    peak = forward.loc[peak_index]
    end = forward.loc[forward["year"].eq(2000)].iloc[-1]
    axes[1].scatter([peak["year"]], [peak["plot_length_km"]], color="#c73e1d", zorder=4)
    axes[1].scatter([end["year"]], [end["plot_length_km"]], color="#2b6cb0", zorder=4)
    axes[1].annotate(
        f"peak {peak['plot_length_km']:.2f} km ({int(peak['year'])})",
        (peak["year"], peak["plot_length_km"]), xytext=(8, 10), textcoords="offset points", fontsize=9,
    )
    axes[1].annotate(
        f"2000 {end['plot_length_km']:.2f} km",
        (end["year"], end["plot_length_km"]), xytext=(-8, -18), ha="right", textcoords="offset points", fontsize=9,
    )
    axes[1].set_xlabel("Calendar year")
    axes[1].set_title("Forward reconstruction")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"LIA system {int(row['lia_unique_id'])} | ΔT={float(row['delta_t']):+.5f} K | "
        f"Esep(LIA/RGI)={float(row['e_lia_sep_formal']):.3f}/{float(row['e_rgi_sep_formal']):.3f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scores = _formal_scores()
    scores.to_csv(RESULT_ROOT / "formal_scores.csv", index=False)
    accepted = scores.loc[scores["formal_dual_pass"]].sort_values(
        ["formal_objective", "lia_unique_id"]
    )
    accepted.to_csv(RESULT_ROOT / "accepted_candidates.csv", index=False)
    plot_dir = RESULT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for _, row in accepted.iterrows():
        _plot(row, plot_dir / f"system_{int(row['lia_unique_id'])}_spinup_forward_length.png")
    summary = {
        "schema_version": 1,
        "executed_chain_count": int(len(scores)),
        "formal_dual_pass_count": int(len(accepted)),
        "formal_dual_pass_systems": accepted["lia_unique_id"].astype(int).tolist(),
        "stopped_systems": {
            "79": "stable and observation-valid but both separation endpoints far outside tolerance",
            "660": "stable and observation-valid but both separation endpoints far outside tolerance",
            "2827": "spin-up did not converge; last-200-year volume change exceeded the 2% gate",
        },
        "unrun_declared_side_points": 10,
        "reason_side_points_not_run": "Adaptive release stopped systems after decisive midpoint/preflight results.",
    }
    (RESULT_ROOT / "formal_batch2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
