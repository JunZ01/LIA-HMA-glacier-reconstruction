"""Consolidate the directed melt_f-delta_T experiment and refinements.

This is a read-only diagnostic. It combines the Stage-2-melt delta-T baseline,
the 18-point directed melt-factor experiment, and six local delta-T bisection
chains for system 2697. It writes one figure per LIA system and a compact
machine-readable summary. No OGGM task is launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOLERANCE = 0.20
STABILITY_LIMIT = 0.02


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_scores(root: Path) -> pd.DataFrame:
    base = root / "validation_data" / "results" / "paleo_surface_validation_v0"
    sources = [
        (
            "stage2_delta_baseline",
            base / "lia_191_curve_loop_delta_bisection_r3_v1" / "diagnostics"
            / "consolidated_v1" / "combined_21_chain_diagnostics.csv",
        ),
        (
            "directed_grid",
            base / "lia_191_curve_loop_melt_delta_directed_v1" / "scores.csv",
        ),
    ]
    for round_number in range(1, 7):
        sources.append(
            (
                f"2697_refine_r{round_number}",
                base / f"lia_191_curve_loop_melt_delta_refine_2697_r{round_number}_v1"
                / "scores.csv",
            )
        )
    frames = []
    for source, path in sources:
        frame = pd.read_csv(path)
        frame["experiment_source"] = source
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False)
    numeric = [
        "lia_unique_id", "delta_t", "melt_f", "spinup_stability_fraction",
        "lia_target_m", "rgi_target_m", "lia_rgi_separation_m",
        "forward_peak_m", "forward_2000_m", "e_lia_sep", "e_rgi_sep",
        "objective_sep", "amplitude_ratio_to_target",
        "max_corridor_exceedance_m",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["lia_unique_id"] = out["lia_unique_id"].astype(int)
    out["technical_failure_bool"] = out["technical_failure"].map(_bool)
    out["spinup_converged_bool"] = out["spinup_converged"].map(_bool)
    out["stable_chain"] = (
        ~out["technical_failure_bool"]
        & out["summary_status"].eq("ok")
        & out["spinup_converged_bool"]
        & out["spinup_stability_fraction"].le(STABILITY_LIMIT)
    )
    out["joint_endpoint_pass"] = (
        out["stable_chain"]
        & out["e_lia_sep"].le(TOLERANCE)
        & out["e_rgi_sep"].le(TOLERANCE)
    )
    stage2 = (
        out[out["experiment_source"].eq("stage2_delta_baseline")]
        .groupby("lia_unique_id")["melt_f"].first().to_dict()
    )
    out["melt_f_multiplier"] = [
        float(row.melt_f) / float(stage2[int(row.lia_unique_id)])
        for row in out.itertuples(index=False)
    ]
    out["melt_f_multiplier"] = out["melt_f_multiplier"].round(6)
    out = out.drop_duplicates(
        ["lia_unique_id", "melt_f_multiplier", "delta_t"], keep="last"
    )
    return out.sort_values(
        ["lia_unique_id", "melt_f_multiplier", "delta_t"]
    ).reset_index(drop=True)


def load_forward(output_dir: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(output_dir) / "timeseries.csv")
    frame = frame[frame["phase"].eq("forward")].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["length_km"] = pd.to_numeric(frame["length_m"], errors="coerce") / 1000.0
    return frame.dropna(subset=["year", "length_km"]).sort_values("year")


def plot_system(group: pd.DataFrame, output_dir: Path) -> Path:
    uid = int(group.iloc[0]["lia_unique_id"])
    stable = group[group["stable_chain"]].copy()
    best = stable.sort_values(["objective_sep", "delta_t"]).iloc[0]
    lia = float(group.iloc[0]["lia_target_m"]) / 1000.0
    rgi = float(group.iloc[0]["rgi_target_m"]) / 1000.0
    separation = abs(lia - rgi)
    tolerance = TOLERANCE * separation
    multipliers = sorted(stable["melt_f_multiplier"].unique())
    colors = plt.cm.tab10(np.linspace(0.0, 0.7, len(multipliers)))
    color_by_multiplier = dict(zip(multipliers, colors))

    fig = plt.figure(figsize=(12.4, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0])
    curve_ax = fig.add_subplot(grid[0, :])
    error_ax = fig.add_subplot(grid[1, 0])
    amplitude_ax = fig.add_subplot(grid[1, 1])

    selected = (
        stable.sort_values(["objective_sep", "delta_t"])
        .groupby("melt_f_multiplier", as_index=False).first()
    )
    for row in selected.itertuples(index=False):
        forward = load_forward(str(row.output_dir))
        multiplier = float(row.melt_f_multiplier)
        curve_ax.plot(
            forward["year"], forward["length_km"], lw=2.3,
            color=color_by_multiplier[multiplier],
            label=(
                f"melt_f x{multiplier:.1f}, dT={float(row.delta_t):+.6f} K "
                f"(E={float(row.objective_sep):.3f})"
            ),
        )
    curve_ax.axhspan(
        min(lia, rgi) - tolerance, max(lia, rgi) + tolerance,
        color="0.6", alpha=0.13,
    )
    curve_ax.axhline(lia, color="#d95f02", lw=1.8, label=f"LIA {lia:.3f} km")
    curve_ax.axhline(rgi, color="#1b9e77", lw=1.8, label=f"RGI {rgi:.3f} km")
    curve_ax.set_xlim(1470, 2000)
    curve_ax.set_xlabel("Year")
    curve_ax.set_ylabel("Length (km)")
    curve_ax.set_title("Best stable curve at each melt_f multiplier")
    curve_ax.grid(alpha=0.22)
    curve_ax.legend(frameon=False, fontsize=8, loc="best")

    for multiplier in multipliers:
        subset = stable[stable["melt_f_multiplier"].eq(multiplier)]
        error_ax.scatter(
            subset["e_lia_sep"], subset["e_rgi_sep"], s=48,
            color=color_by_multiplier[multiplier], alpha=0.86,
            label=f"melt_f x{multiplier:.1f}",
        )
        amplitude_ax.plot(
            subset["delta_t"], subset["amplitude_ratio_to_target"],
            marker="o", lw=1.7, color=color_by_multiplier[multiplier],
            label=f"melt_f x{multiplier:.1f}",
        )
    error_ax.axvspan(0, TOLERANCE, color="0.6", alpha=0.10)
    error_ax.axhspan(0, TOLERANCE, color="0.6", alpha=0.10)
    error_ax.axvline(TOLERANCE, color="0.25", ls="--", lw=1.2)
    error_ax.axhline(TOLERANCE, color="0.25", ls="--", lw=1.2)
    error_ax.scatter(
        [float(best["e_lia_sep"])], [float(best["e_rgi_sep"])],
        marker="*", s=170, color="black", zorder=6, label="Best overall",
    )
    error_ax.set_xlim(left=0)
    error_ax.set_ylim(bottom=0)
    error_ax.set_xlabel("LIA peak error / separation")
    error_ax.set_ylabel("RGI endpoint error / separation")
    error_ax.set_title("Joint endpoint error; lower-left box passes")
    error_ax.grid(alpha=0.22)
    error_ax.legend(frameon=False, fontsize=8, loc="best")

    amplitude_ax.axhline(1.0, color="0.25", ls="--", lw=1.2, label="Target amplitude")
    amplitude_ax.set_xlabel("Delta T (K)")
    amplitude_ax.set_ylabel("Peak-to-2000 amplitude / target")
    amplitude_ax.set_title("Amplitude response")
    amplitude_ax.grid(alpha=0.22)
    amplitude_ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        f"LIA system {uid} | best melt_f x{float(best['melt_f_multiplier']):.1f}, "
        f"dT={float(best['delta_t']):+.9f} K | "
        f"E_LIA={float(best['e_lia_sep']):.3f}, "
        f"E_RGI={float(best['e_rgi_sep']):.3f} | "
        f"joint passes={int(group['joint_endpoint_pass'].sum())}",
        fontsize=13,
    )
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"system_{uid:04d}_melt_delta_directed.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def summarize(scores: pd.DataFrame, paths: dict[int, Path]) -> pd.DataFrame:
    rows = []
    for uid, group in scores.groupby("lia_unique_id", sort=True):
        stable = group[group["stable_chain"]]
        best = stable.sort_values(["objective_sep", "delta_t"]).iloc[0]
        baseline = stable[stable["melt_f_multiplier"].eq(1.0)]
        baseline_best = baseline.sort_values(["objective_sep", "delta_t"]).iloc[0]
        rows.append(
            {
                "lia_unique_id": int(uid),
                "chain_count": int(len(group)),
                "stable_chain_count": int(group["stable_chain"].sum()),
                "technical_failure_count": int(group["technical_failure_bool"].sum()),
                "joint_endpoint_pass_count": int(group["joint_endpoint_pass"].sum()),
                "baseline_best_objective": float(baseline_best["objective_sep"]),
                "best_melt_f_multiplier": float(best["melt_f_multiplier"]),
                "best_delta_t": float(best["delta_t"]),
                "best_e_lia_sep": float(best["e_lia_sep"]),
                "best_e_rgi_sep": float(best["e_rgi_sep"]),
                "best_objective_sep": float(best["objective_sep"]),
                "best_amplitude_ratio": float(best["amplitude_ratio_to_target"]),
                "plot_path": str(paths[int(uid)]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    root = project_root()
    output = (
        root / "validation_data" / "results" / "paleo_surface_validation_v0"
        / "lia_191_curve_loop_melt_delta_directed_v1" / "diagnostics"
        / "consolidated_v1"
    )
    output.mkdir(parents=True, exist_ok=True)
    scores = load_scores(root)
    scores.to_csv(output / "combined_melt_delta_diagnostics.csv", index=False)
    paths = {
        int(uid): plot_system(group, output)
        for uid, group in scores.groupby("lia_unique_id", sort=True)
    }
    summary = summarize(scores, paths)
    summary.to_csv(output / "system_summary.csv", index=False)
    record = {
        "launched_oggm": False,
        "combined_chain_count": int(len(scores)),
        "directed_grid_chain_count": 18,
        "2697_refinement_chain_count": 6,
        "technical_failure_count": int(scores["technical_failure_bool"].sum()),
        "stable_chain_count": int(scores["stable_chain"].sum()),
        "joint_endpoint_pass_count": int(scores["joint_endpoint_pass"].sum()),
        "routing_tolerance": TOLERANCE,
        "system_summary": summary.to_dict(orient="records"),
    }
    (output / "diagnosis.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
