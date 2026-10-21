"""Consolidate and plot the declared 2697 melt_f-delta_t strip.

This diagnostic launches no OGGM task. It combines the four newly sampled
melt_f rows with the existing melt_f x1.1 local evidence, writes one curve
figure per multiplier, and records separation-scaled endpoint diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEM_ID = 2697
TOLERANCE = 0.20
STABILITY_LIMIT = 0.02
STRIP_MULTIPLIERS = (1.05, 1.075, 1.10, 1.125, 1.15)


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "lia_unique_id",
        "delta_t",
        "melt_f",
        "melt_f_stage2_multiplier",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "lia_rgi_separation_m",
        "forward_peak_m",
        "forward_2000_m",
        "forward_minimum_m",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
        "peak_to_2000_amplitude_m",
        "amplitude_ratio_to_target",
        "max_corridor_exceedance_m",
    ]
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_strip(root: Path) -> pd.DataFrame:
    base = root / "validation_data" / "results" / "paleo_surface_validation_v0"
    old_path = (
        base / "lia_191_curve_loop_melt_delta_directed_v1" / "diagnostics"
        / "consolidated_v1" / "combined_melt_delta_diagnostics.csv"
    )
    new_path = base / "lia_191_curve_loop_melt_delta_strip_2697_v1" / "scores.csv"

    old = _numeric(pd.read_csv(old_path))
    old = old[
        old["lia_unique_id"].eq(SYSTEM_ID)
        & np.isclose(old["melt_f_multiplier"], 1.10)
    ].copy()
    old["melt_f_stage2_multiplier"] = 1.10
    old["strip_source"] = "existing_x1.1"

    new = _numeric(pd.read_csv(new_path))
    new = new[new["lia_unique_id"].eq(SYSTEM_ID)].copy()
    new["strip_source"] = "new_strip"

    frame = pd.concat([old, new], ignore_index=True, sort=False)
    frame["technical_failure_bool"] = frame["technical_failure"].map(_bool)
    frame["spinup_converged_bool"] = frame["spinup_converged"].map(_bool)
    frame["stable_chain"] = (
        ~frame["technical_failure_bool"]
        & frame["summary_status"].eq("ok")
        & frame["spinup_converged_bool"]
        & frame["spinup_stability_fraction"].le(STABILITY_LIMIT)
    )
    frame["joint_endpoint_pass"] = (
        frame["stable_chain"]
        & frame["e_lia_sep"].le(TOLERANCE)
        & frame["e_rgi_sep"].le(TOLERANCE)
    )
    frame = frame.drop_duplicates(
        ["melt_f_stage2_multiplier", "delta_t"], keep="last"
    )
    return frame.sort_values(
        ["melt_f_stage2_multiplier", "delta_t"]
    ).reset_index(drop=True)


def load_forward(output_dir: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(output_dir) / "timeseries.csv")
    frame = frame[frame["phase"].eq("forward")].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["length_km"] = pd.to_numeric(frame["length_m"], errors="coerce") / 1000
    return frame.dropna(subset=["year", "length_km"]).sort_values("year")


def multiplier_tag(multiplier: float) -> str:
    return str(multiplier).replace(".", "p")


def plot_multiplier(group: pd.DataFrame, plot_dir: Path) -> Path:
    multiplier = float(group.iloc[0]["melt_f_stage2_multiplier"])
    stable = group[group["stable_chain"]].copy()
    lia = float(group.iloc[0]["lia_target_m"]) / 1000
    rgi = float(group.iloc[0]["rgi_target_m"]) / 1000
    separation = abs(lia - rgi)
    tolerance_km = TOLERANCE * separation

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    colors = plt.cm.viridis(np.linspace(0.10, 0.88, max(len(stable), 2)))
    for color, row in zip(colors, stable.itertuples(index=False)):
        forward = load_forward(str(row.output_dir))
        passed = bool(row.joint_endpoint_pass)
        ax.plot(
            forward["year"],
            forward["length_km"],
            color=color,
            lw=3.0 if passed else 2.0,
            label=(
                f"dT={float(row.delta_t):+.6f} K | "
                f"E_LIA={float(row.e_lia_sep):.3f}, "
                f"E_RGI={float(row.e_rgi_sep):.3f}"
            ),
        )

    ax.axhspan(
        min(lia, rgi) - tolerance_km,
        max(lia, rgi) + tolerance_km,
        color="0.7",
        alpha=0.14,
        label="20% separation routing corridor",
    )
    ax.axhline(lia, color="#d95f02", lw=1.8, label=f"LIA target {lia:.3f} km")
    ax.axhline(rgi, color="#1b9e77", lw=1.8, label=f"RGI target {rgi:.375f} km")
    ax.set_xlim(1470, 2000)
    ax.set_xlabel("Year")
    ax.set_ylabel("Main-flowline length (km)")
    ax.set_title(
        f"LIA system {SYSTEM_ID} | melt_f = Stage-2 x{multiplier:g} | "
        f"stable curves {len(stable)}/{len(group)}"
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2 if len(stable) > 5 else 1, loc="best")

    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"system_{SYSTEM_ID}_melt_f_x{multiplier_tag(multiplier)}.png"
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_error_plane(frame: pd.DataFrame, plot_dir: Path) -> Path:
    stable = frame[frame["stable_chain"]].copy()
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    colors = plt.cm.plasma(np.linspace(0.08, 0.88, len(STRIP_MULTIPLIERS)))
    for color, multiplier in zip(colors, STRIP_MULTIPLIERS):
        group = stable[np.isclose(stable["melt_f_stage2_multiplier"], multiplier)]
        if group.empty:
            continue
        ax.plot(
            group["e_lia_sep"],
            group["e_rgi_sep"],
            marker="o",
            lw=1.4,
            color=color,
            label=f"melt_f x{multiplier:g}",
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                f"{float(row.delta_t):+.3f}",
                (float(row.e_lia_sep), float(row.e_rgi_sep)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axvspan(0, TOLERANCE, color="0.7", alpha=0.12)
    ax.axhspan(0, TOLERANCE, color="0.7", alpha=0.12)
    ax.axvline(TOLERANCE, color="0.25", ls="--", lw=1.2)
    ax.axhline(TOLERANCE, color="0.25", ls="--", lw=1.2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("LIA peak error / |LIA - RGI|")
    ax.set_ylabel("2000 endpoint error / |LIA - RGI|")
    ax.set_title("System 2697 endpoint-error plane; lower-left box passes")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="best")

    path = plot_dir / f"system_{SYSTEM_ID}_endpoint_error_plane.png"
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.90)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def main() -> int:
    root = project_root()
    output = (
        root / "validation_data" / "results" / "paleo_surface_validation_v0"
        / "lia_191_curve_loop_melt_delta_strip_2697_v1" / "diagnostics"
        / "consolidated_v1"
    )
    plot_dir = output / "plots"
    output.mkdir(parents=True, exist_ok=True)

    frame = load_strip(root)
    frame.to_csv(output / "combined_strip_diagnostics.csv", index=False)

    plot_paths = {}
    for multiplier, group in frame.groupby("melt_f_stage2_multiplier", sort=True):
        plot_paths[str(float(multiplier))] = str(plot_multiplier(group, plot_dir))
    error_plot = plot_error_plane(frame, plot_dir)

    stable = frame[frame["stable_chain"]].copy()
    best = stable.sort_values(["objective_sep", "e_lia_sep", "e_rgi_sep"]).iloc[0]
    summary = {
        "launched_oggm": False,
        "lia_unique_id": SYSTEM_ID,
        "new_chain_count": 12,
        "combined_strip_chain_count": int(len(frame)),
        "technical_failure_count": int(frame["technical_failure_bool"].sum()),
        "stable_chain_count": int(frame["stable_chain"].sum()),
        "joint_endpoint_pass_count": int(frame["joint_endpoint_pass"].sum()),
        "routing_tolerance": TOLERANCE,
        "best_melt_f_stage2_multiplier": float(best["melt_f_stage2_multiplier"]),
        "best_delta_t": float(best["delta_t"]),
        "best_e_lia_sep": float(best["e_lia_sep"]),
        "best_e_rgi_sep": float(best["e_rgi_sep"]),
        "best_objective_sep": float(best["objective_sep"]),
        "best_amplitude_ratio_to_target": float(best["amplitude_ratio_to_target"]),
        "plot_paths": plot_paths,
        "error_plot_path": str(error_plot),
    }
    (output / "diagnosis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
