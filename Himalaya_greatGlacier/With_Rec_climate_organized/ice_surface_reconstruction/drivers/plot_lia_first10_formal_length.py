"""Plot formal first-10 spin-up and forward length trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_references(
    chain: pd.DataFrame,
    spinup: pd.DataFrame,
    lia_targets_path: Path,
    modern_tau_path: Path,
) -> pd.DataFrame:
    lia_targets = pd.read_csv(lia_targets_path)[
        ["lia_unique_id", "target_main_flowline_length_m"]
    ].rename(columns={"target_main_flowline_length_m": "lia_axis_length_m"})
    modern = pd.read_csv(modern_tau_path)
    modern = modern[modern["bed_scenario"].eq("farinotti2019")][
        ["rgi_id", "modern_main_centerline_length_km"]
    ].drop_duplicates("rgi_id")
    terminal = (
        spinup.sort_values("year")
        .groupby("lia_unique_id", as_index=False)
        .tail(1)[["lia_unique_id", "length_m"]]
        .rename(columns={"length_m": "spinup_500_length_m"})
    )
    refs = chain.merge(terminal, on="lia_unique_id", validate="one_to_one")
    refs = refs.merge(lia_targets, on="lia_unique_id", validate="one_to_one")
    refs = refs.merge(modern, on="rgi_id", validate="many_to_one")
    refs["lia_model_reference_length_m"] = refs["spinup_500_length_m"] / (
        1.0 + refs["length_relative_error"]
    )
    refs["rgi_to_lia_length_ratio"] = (
        refs["modern_main_centerline_length_km"] * 1000.0
        / refs["lia_axis_length_m"]
    )
    refs["rgi_model_equivalent_length_m"] = (
        refs["lia_model_reference_length_m"] * refs["rgi_to_lia_length_ratio"]
    )
    return refs.sort_values("lia_unique_id")


def _plot_system(
    unique_id: int,
    ref: pd.Series,
    spinup: pd.DataFrame,
    forward: pd.DataFrame,
    output_path: Path,
) -> None:
    lia_ref = float(ref["lia_model_reference_length_m"])
    rgi_ratio = float(ref["rgi_to_lia_length_ratio"])
    spin = spinup[spinup["lia_unique_id"].eq(unique_id)].sort_values("year")
    run = forward[forward["lia_unique_id"].eq(unique_id)].sort_values("year")
    spin_ratio = spin["length_m"].to_numpy(dtype=float) / lia_ref
    run_ratio = run["length_m"].to_numpy(dtype=float) / lia_ref
    end_ratio = float(run_ratio[-1])
    modern_error = (end_ratio - rgi_ratio) / rgi_ratio

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    fig.suptitle(
        f"LIA system {unique_id} | deltaT={float(ref['delta_t_c']):+.4g} C | "
        f"spin-up error={float(ref['length_relative_error']):+.1%}",
        fontsize=14,
    )

    axes[0].plot(
        spin["year"], spin_ratio, color="#2563a6", marker="o", linewidth=2.2,
        label="Spin-up length",
    )
    axes[0].axhline(1.0, color="#222222", linewidth=1.5, label="LIA reference")
    axes[0].set_xlabel("Spin-up year")
    axes[0].set_ylabel("Length / LIA reference")
    axes[0].set_title("1471-1500 climate repeated")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(
        run["year"], run_ratio, color="#c44e52", linewidth=2.4,
        label="Simulated length",
    )
    axes[1].axhline(1.0, color="#222222", linewidth=1.5, label="LIA reference")
    axes[1].axhline(
        rgi_ratio, color="#198754", linewidth=1.8, linestyle="--",
        label="RGI modern reference",
    )
    axes[1].scatter(
        [2000], [end_ratio], color="#c44e52", edgecolor="white", zorder=5,
    )
    axes[1].annotate(
        f"2000 vs RGI: {modern_error:+.1%}",
        xy=(2000, end_ratio), xytext=(-8, 10), textcoords="offset points",
        ha="right", fontsize=10,
    )
    axes[1].set_xlabel("Calendar year")
    axes[1].set_title("Forward simulation")
    axes[1].legend(frameon=False, loc="best")

    values = np.concatenate([spin_ratio, run_ratio, [1.0, rgi_ratio]])
    pad = max(0.04, 0.08 * float(np.ptp(values)))
    axes[0].set_ylim(float(values.min() - pad), float(values.max() + pad))
    for axis in axes:
        axis.grid(True, color="#d5d8dc", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(
    refs: pd.DataFrame,
    forward: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True)
    for axis, (_, ref) in zip(axes.flat, refs.iterrows()):
        unique_id = int(ref["lia_unique_id"])
        run = forward[forward["lia_unique_id"].eq(unique_id)].sort_values("year")
        lia_ref = float(ref["lia_model_reference_length_m"])
        run_ratio = run["length_m"].to_numpy(dtype=float) / lia_ref
        rgi_ratio = float(ref["rgi_to_lia_length_ratio"])
        axis.plot(run["year"], run_ratio, color="#c44e52", linewidth=1.8)
        axis.axhline(1.0, color="#222222", linewidth=1.0)
        axis.axhline(rgi_ratio, color="#198754", linewidth=1.2, linestyle="--")
        axis.set_title(
            f"LIA {unique_id} | dT={float(ref['delta_t_c']):+.4g} C",
            fontsize=11,
        )
        axis.set_ylabel("Length / LIA")
        axis.grid(True, color="#d5d8dc", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Calendar year")
    fig.suptitle(
        "Formal 1500-2000 length trajectories (black: LIA, green dashed: RGI)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir", required=True, type=Path)
    parser.add_argument("--lia-targets", required=True, type=Path)
    parser.add_argument("--modern-tau", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chain = pd.read_csv(args.chain_dir / "selected_chain_summary.csv")
    spinup = pd.read_csv(args.chain_dir / "selected_spinup_timeseries.csv")
    forward = pd.read_csv(args.chain_dir / "forward_1500_2000_timeseries.csv")
    refs = _load_references(chain, spinup, args.lia_targets, args.modern_tau)
    refs.to_csv(args.output_dir / "length_reference_audit.csv", index=False)

    for _, ref in refs.iterrows():
        unique_id = int(ref["lia_unique_id"])
        _plot_system(
            unique_id,
            ref,
            spinup,
            forward,
            args.output_dir / f"lia_{unique_id}_formal_length_run.png",
        )
    _plot_overview(refs, forward, args.output_dir / "first10_formal_length_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
