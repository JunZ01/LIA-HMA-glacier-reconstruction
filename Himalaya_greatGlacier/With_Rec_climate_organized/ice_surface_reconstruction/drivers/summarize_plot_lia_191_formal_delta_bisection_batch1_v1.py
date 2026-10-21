"""Formally score and plot the first fine delta-T bisection batch."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_formal_delta_bisection_batch1_v1"
)
GROUPS = ("regular", "spatial_2928")


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _score_task(task: dict) -> dict:
    output = Path(str(task["output_dir"]))
    summary_path = output / "summary.json"
    series_path = output / "timeseries.csv"
    if not summary_path.exists() or not series_path.exists():
        return {**task, "technical_clean": False, "formal_dual_pass": False}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    series = pd.read_csv(series_path)
    observation = summary.get("observation_domain") or {}
    overflow = _bool(observation.get("overflow", True))
    length_column = "observation_length_m" if "observation_length_m" in series else None
    forward = series[series["phase"].astype(str).eq("forward")].copy()
    if length_column is None:
        peak = endpoint = math.nan
    else:
        forward[length_column] = pd.to_numeric(forward[length_column], errors="coerce")
        finite = forward[forward[length_column].notna()]
        peak = float(finite[length_column].max()) if not finite.empty else math.nan
        endpoint_rows = finite[pd.to_numeric(finite["year"], errors="coerce").eq(2000)]
        endpoint = float(endpoint_rows.iloc[-1][length_column]) if not endpoint_rows.empty else math.nan
    lia = float(summary["targets"]["lia_length_m"])
    rgi = float(summary["targets"]["rgi_length_m"])
    separation = abs(lia - rgi)
    e_lia_sep = abs(peak - lia) / separation if math.isfinite(peak) else math.nan
    e_rgi_sep = abs(endpoint - rgi) / separation if math.isfinite(endpoint) else math.nan
    e_lia_total = abs(peak - lia) / lia if math.isfinite(peak) else math.nan
    e_rgi_total = abs(endpoint - rgi) / rgi if math.isfinite(endpoint) else math.nan
    spinup = summary.get("spinup_control") or {}
    stability = float(spinup.get("final_stability_fraction", math.nan))
    converged = _bool(spinup.get("converged", False))
    technical_clean = str(summary.get("status")) == "ok"
    stable_valid = bool(
        technical_clean
        and converged
        and math.isfinite(stability)
        and stability <= 0.02
        and not overflow
        and math.isfinite(e_lia_sep)
        and math.isfinite(e_rgi_sep)
    )
    total_pass = bool(stable_valid and e_lia_total <= 0.10 and e_rgi_total <= 0.10)
    separation_pass = bool(stable_valid and e_lia_sep <= 0.20 and e_rgi_sep <= 0.20)
    return {
        **task,
        "summary_status": summary.get("status"),
        "technical_clean": technical_clean,
        "spinup_converged": converged,
        "spinup_years": spinup.get("actual_years"),
        "spinup_stability_fraction": stability,
        "observation_domain_node_count": observation.get("node_count"),
        "observation_domain_overflow": overflow,
        "first_overflow_phase": observation.get("first_overflow_phase"),
        "first_overflow_year": observation.get("first_overflow_year"),
        "maximum_overflow_thickness_m": observation.get("maximum_overflow_thickness_m"),
        "lia_target_m": lia,
        "rgi_target_m": rgi,
        "forward_peak_m": peak,
        "forward_2000_m": endpoint,
        "e_lia_sep": e_lia_sep,
        "e_rgi_sep": e_rgi_sep,
        "objective_sep": e_lia_sep + e_rgi_sep if math.isfinite(e_lia_sep + e_rgi_sep) else math.nan,
        "e_lia_total": e_lia_total,
        "e_rgi_total": e_rgi_total,
        "stable_valid": stable_valid,
        "total_length_pass": total_pass,
        "formal_separation_pass": separation_pass,
        "formal_dual_pass": bool(total_pass and separation_pass),
        "timeseries_path": str(series_path),
    }


def _plot_best(row: pd.Series, plot_dir: Path) -> Path:
    series = pd.read_csv(row["timeseries_path"])
    length_col = "observation_length_m"
    spin = series[series["phase"].astype(str).eq("spinup")].copy()
    forward = series[series["phase"].astype(str).eq("forward")].copy()
    lia = float(row["lia_target_m"])
    rgi = float(row["rgi_target_m"])
    separation = abs(lia - rgi)
    low = min(lia, rgi) - 0.20 * separation
    high = max(lia, rgi) + 0.20 * separation
    values = pd.to_numeric(series[length_col], errors="coerce") / 1000.0
    y_min = min(float(values.min()), low / 1000.0) - 0.25
    y_max = max(float(values.max()), high / 1000.0) + 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, gridspec_kw={"width_ratios": [1, 1.55]})
    for axis in axes:
        axis.axhspan(low / 1000.0, high / 1000.0, color="#dceeff", alpha=0.55)
        axis.axhline(lia / 1000.0, color="#c73e1d", linestyle="--", linewidth=1.5, label="LIA target")
        axis.axhline(rgi / 1000.0, color="#2b6cb0", linestyle="--", linewidth=1.5, label="RGI target")
        axis.grid(alpha=0.22)
        axis.set_ylim(y_min, y_max)
    axes[0].plot(spin["year"], spin[length_col] / 1000.0, color="#444444", marker="o", linewidth=2)
    axes[0].set_title("Adaptive spin-up")
    axes[0].set_xlabel("Spin-up model year")
    axes[0].set_ylabel("Observation-domain length (km)")
    axes[1].plot(forward["year"], forward[length_col] / 1000.0, color="#168aad", marker="o", markersize=3, linewidth=2)
    peak_i = pd.to_numeric(forward[length_col], errors="coerce").idxmax()
    peak = forward.loc[peak_i]
    end = forward[pd.to_numeric(forward["year"], errors="coerce").eq(2000)].iloc[-1]
    axes[1].scatter([peak["year"]], [peak[length_col] / 1000.0], color="#c73e1d", zorder=4)
    axes[1].scatter([end["year"]], [end[length_col] / 1000.0], color="#2b6cb0", zorder=4)
    axes[1].set_title("Forward reconstruction")
    axes[1].set_xlabel("Calendar year")
    axes[1].annotate(f"peak {peak[length_col]/1000.0:.2f} km", (peak["year"], peak[length_col]/1000.0), xytext=(8, 10), textcoords="offset points")
    axes[1].annotate(f"2000 {end[length_col]/1000.0:.2f} km", (end["year"], end[length_col]/1000.0), xytext=(-8, -18), ha="right", textcoords="offset points")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"LIA system {int(row['lia_unique_id'])} | ΔT={float(row['delta_t']):+.6f} K | "
        f"Esep(LIA/RGI)={float(row['e_lia_sep']):.3f}/{float(row['e_rgi_sep']):.3f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    output = plot_dir / f"system_{int(row['lia_unique_id'])}_best_spinup_forward_length.png"
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    rows: list[dict] = []
    for group in GROUPS:
        manifest_path = ROOT / group / "task_manifest.csv"
        if not manifest_path.exists():
            continue
        for task in pd.read_csv(manifest_path).to_dict(orient="records"):
            rows.append({"geometry_group": group, **_score_task(task)})
    scores = pd.DataFrame(rows).sort_values(["lia_unique_id", "delta_t"])
    scores.to_csv(ROOT / "formal_scores.csv", index=False)
    valid = scores[scores["stable_valid"].fillna(False)].copy()
    best = (
        valid.sort_values(["lia_unique_id", "objective_sep", "delta_t"])
        .groupby("lia_unique_id", as_index=False)
        .head(1)
    )
    best.to_csv(ROOT / "best_per_system_formal.csv", index=False)
    plot_dir = ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_rows = []
    for _, row in best.iterrows():
        output = _plot_best(row, plot_dir)
        plot_rows.append({"lia_unique_id": int(row["lia_unique_id"]), "plot": str(output)})
    pd.DataFrame(plot_rows).to_csv(plot_dir / "plot_index.csv", index=False)
    summary = {
        "task_count": int(len(scores)),
        "completed_clean_count": int(scores["technical_clean"].fillna(False).sum()),
        "stable_valid_count": int(scores["stable_valid"].fillna(False).sum()),
        "observation_domain_overflow_count": int(scores["observation_domain_overflow"].fillna(False).sum()),
        "formal_dual_pass_count": int(scores["formal_dual_pass"].fillna(False).sum()),
        "formal_dual_pass_systems": sorted(
            scores.loc[scores["formal_dual_pass"].fillna(False), "lia_unique_id"].astype(int).unique().tolist()
        ),
        "regular_batch_released": False,
        "regular_stop_reason": "2815 mandatory preflight observation-domain overflow",
        "output_root": str(ROOT),
    }
    (ROOT / "formal_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
