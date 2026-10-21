"""Consolidate the declared B-window melt_f--delta_t amplitude block for 2928.

This is read-only with respect to OGGM states: it combines already completed
chains, writes an auditable diagnostics table, and plots the individual
forward length curves needed to decide the next parameter direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEM_ID = 2928
TOLERANCE = 0.20
STABILITY_LIMIT = 0.02


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    for column in (
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
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def read_source(path: Path, source: str, source_order: int) -> pd.DataFrame:
    frame = numeric(pd.read_csv(path))
    frame = frame[frame["lia_unique_id"].eq(SYSTEM_ID)].copy()
    frame["source"] = source
    frame["source_order"] = source_order
    if "melt_f_stage2_multiplier" not in frame:
        frame["melt_f_stage2_multiplier"] = 1.0
    frame["melt_f_stage2_multiplier"] = frame["melt_f_stage2_multiplier"].fillna(1.0)
    return frame


def load_all(root: Path) -> pd.DataFrame:
    base = root / "validation_data" / "results" / "paleo_surface_validation_v0"
    controller = read_source(
        base / "lia_191_controller_v1" / "scores.csv",
        "stage2_delta_profile",
        0,
    )
    boundary = read_source(
        base / "lia_191_curve_loop_delta_boundary_batch2_v1" / "scores.csv",
        "delta_boundary_batch2",
        1,
    )
    melt_delta = read_source(
        base / "lia_191_curve_loop_melt_delta_2928_v1" / "scores.csv",
        "melt_delta_2928",
        2,
    )
    frame = pd.concat([controller, boundary, melt_delta], ignore_index=True, sort=False)
    frame["technical_failure_bool"] = frame["technical_failure"].map(as_bool)
    frame["spinup_converged_bool"] = frame["spinup_converged"].map(as_bool)
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
    frame = frame.sort_values("source_order").drop_duplicates(
        ["melt_f_stage2_multiplier", "delta_t"], keep="last"
    )
    return frame.sort_values(["melt_f_stage2_multiplier", "delta_t"]).reset_index(drop=True)


def load_forward(output_dir: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(output_dir) / "timeseries.csv")
    frame = frame[frame["phase"].eq("forward")].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["length_km"] = pd.to_numeric(frame["length_m"], errors="coerce") / 1000.0
    return frame.dropna(subset=["year", "length_km"]).sort_values("year")


def multiplier_label(row: pd.Series) -> str:
    return (
        f"melt_f×{float(row.melt_f_stage2_multiplier):.2f}, "
        f"ΔT={float(row.delta_t):+.3f} K"
    )


def style_for_multiplier(multiplier: float) -> tuple[str, str]:
    palette = {
        0.90: ("#6a3d9a", "--"),
        0.95: ("#e66101", "-"),
        1.00: ("#1b9e77", "-"),
    }
    return palette.get(round(multiplier, 2), ("#4d4d4d", "-"))


def draw_targets(ax: plt.Axes, lia: float, rgi: float) -> None:
    separation = abs(lia - rgi)
    buffer_km = TOLERANCE * separation
    ax.axhspan(
        min(lia, rgi) - buffer_km,
        max(lia, rgi) + buffer_km,
        color="0.75",
        alpha=0.15,
        label="20% separation routing corridor",
        zorder=0,
    )
    ax.axhline(lia, color="#d95f02", lw=1.8, label=f"LIA target {lia:.3f} km")
    ax.axhline(rgi, color="#1b9e77", lw=1.8, label=f"RGI target {rgi:.3f} km")


def draw_curves(
    frame: pd.DataFrame,
    plot_dir: Path,
    filename: str,
    title: str,
    y_limits: tuple[float, float] | None,
) -> Path:
    lia = float(frame.iloc[0]["lia_target_m"]) / 1000.0
    rgi = float(frame.iloc[0]["rgi_target_m"]) / 1000.0
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    draw_targets(ax, lia, rgi)
    for row in frame.itertuples(index=False):
        curve = load_forward(str(row.output_dir))
        color, linestyle = style_for_multiplier(float(row.melt_f_stage2_multiplier))
        ax.plot(
            curve["year"],
            curve["length_km"],
            color=color,
            linestyle=linestyle,
            alpha=0.92,
            lw=2.0,
            label=(
                f"melt_f×{float(row.melt_f_stage2_multiplier):.2f}, "
                f"ΔT={float(row.delta_t):+.3f} K | "
                f"A={float(row.amplitude_ratio_to_target):.2f}"
            ),
        )
    ax.set_xlim(1470, 2000)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xlabel("Year")
    ax.set_ylabel("Main-flowline length (km)")
    ax.set_title(title)
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.90)
    path = plot_dir / filename
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_error_amplitude(frame: pd.DataFrame, plot_dir: Path) -> Path:
    stable = frame[frame["stable_chain"]].copy()
    fig, ax = plt.subplots(figsize=(8.1, 6.8))
    for multiplier, group in stable.groupby("melt_f_stage2_multiplier", sort=True):
        color, _ = style_for_multiplier(float(multiplier))
        ax.scatter(
            group["e_lia_sep"],
            group["e_rgi_sep"],
            s=54,
            color=color,
            label=f"melt_f×{float(multiplier):.2f}",
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                f"{float(row.delta_t):+.3f}; A={float(row.amplitude_ratio_to_target):.2f}",
                (float(row.e_lia_sep), float(row.e_rgi_sep)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
    ax.axvspan(0, TOLERANCE, color="0.75", alpha=0.15)
    ax.axhspan(0, TOLERANCE, color="0.75", alpha=0.15)
    ax.axvline(TOLERANCE, color="0.25", ls="--", lw=1.1)
    ax.axhline(TOLERANCE, color="0.25", ls="--", lw=1.1)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Forward-peak error / |LIA − RGI|")
    ax.set_ylabel("2000-endpoint error / |LIA − RGI|")
    ax.set_title("System 2928: endpoint trade-off; annotation A = amplitude ratio")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.90)
    path = plot_dir / "system_2928_endpoint_amplitude_plane.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def main() -> int:
    root = project_root()
    output = (
        root / "validation_data" / "results" / "paleo_surface_validation_v0"
        / "lia_191_curve_loop_melt_delta_2928_v1" / "diagnostics" / "consolidated_v1"
    )
    plot_dir = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    frame = load_all(root)
    frame.to_csv(output / "combined_melt_delta_diagnostics.csv", index=False)
    stable = frame[frame["stable_chain"]].copy()

    full = draw_curves(
        stable,
        plot_dir,
        "system_2928_melt_delta_all_stable_curves.png",
        "LIA system 2928: all stable local melt_f–ΔT curves",
        None,
    )
    zoom = stable[
        stable["e_lia_sep"].le(1.5) & stable["e_rgi_sep"].le(1.5)
    ].copy()
    target_zoom = draw_curves(
        zoom,
        plot_dir,
        "system_2928_melt_delta_target_zoom.png",
        "LIA system 2928: target-scale stable curves",
        (9.0, 20.0),
    )
    plane = plot_error_amplitude(frame, plot_dir)

    new_block = frame[frame["source"].eq("melt_delta_2928")].copy()
    lower_melt_stable = new_block[new_block["stable_chain"]]
    summary = {
        "launched_oggm": False,
        "lia_unique_id": SYSTEM_ID,
        "question": (
            "Can lowering melt_f below Stage-2 reduce the excessive peak-to-2000 "
            "length amplitude while delta_t re-centers the endpoints?"
        ),
        "sources": {
            "stage2_delta_profile": int(frame["source"].eq("stage2_delta_profile").sum()),
            "delta_boundary_batch2": int(frame["source"].eq("delta_boundary_batch2").sum()),
            "melt_delta_2928": int(new_block.shape[0]),
        },
        "technical_failure_count": int(frame["technical_failure_bool"].sum()),
        "stable_chain_count": int(stable.shape[0]),
        "joint_endpoint_pass_count": int(frame["joint_endpoint_pass"].sum()),
        "new_block_stable_count": int(lower_melt_stable.shape[0]),
        "new_block_out_of_range_count": int(new_block["summary_status"].eq("out_of_range").sum()),
        "lower_melt_direction_result": (
            "closed: all stable lower-melt cases have amplitude ratios above 1.0; "
            "the 0.90 x 0.50 point is out of range, so lowering melt_f does not "
            "provide the requested amplitude reduction under this configuration."
        ),
        "candidate_for_next_direction": {
            "existing_stage2_delta_t": 0.4375,
            "melt_f_stage2_multiplier": 1.0,
            "amplitude_ratio": float(
                frame.loc[
                    np.isclose(frame["melt_f_stage2_multiplier"], 1.0)
                    & np.isclose(frame["delta_t"], 0.4375),
                    "amplitude_ratio_to_target",
                ].iloc[0]
            ),
        },
        "next_experiment_not_declared": (
            "If the user approves continuation, test the opposite, explicitly named "
            "melt_f-increase direction with delta_t re-centering; do not open a new "
            "parameter family automatically."
        ),
        "plots": {
            "all_stable_curves": str(full),
            "target_zoom": str(target_zoom),
            "endpoint_amplitude_plane": str(plane),
        },
    }
    (output / "diagnosis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
