"""Temporarily screen nearly monotonic declining curves from accepted-58."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
CURVES = (
    RESULTS
    / "lia_191_accepted_58_curves_visualization_v1/accepted_58_annual_length_curves.csv"
)
OUTPUT = RESULTS / "lia_191_accepted_58_curve_shape_screen_v1"

SMOOTHING_YEARS = 11
MAX_UPWARD_VARIATION_SHARE = 0.15
MAX_REBOUND_RANGE_FRACTION = 0.20

ROUTE_COLORS = {
    "frozen-legacy": "#287271",
    "synthetic-system": "#d77a61",
    "formal-historical-reproduction-pass": "#3f5f8f",
}


def shape_metrics(group: pd.DataFrame) -> dict[str, object]:
    group = group.sort_values("year")
    raw = group["length_km"].to_numpy(dtype=float)
    smooth = (
        pd.Series(raw)
        .rolling(SMOOTHING_YEARS, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    differences = np.diff(smooth)
    upward = float(np.clip(differences, 0, None).sum())
    downward = float(np.clip(-differences, 0, None).sum())
    total_variation = upward + downward
    upward_share = upward / total_variation if total_variation else 0.0

    running_minimum = np.minimum.accumulate(smooth)
    maximum_rebound_km = float(np.max(smooth - running_minimum))
    curve_range_km = float(np.max(smooth) - np.min(smooth))
    rebound_fraction = maximum_rebound_km / curve_range_km if curve_range_km else 0.0
    net_change_km = float(smooth[-1] - smooth[0])
    temporary_exclusion = (
        net_change_km < 0
        and upward_share <= MAX_UPWARD_VARIATION_SHARE
        and rebound_fraction <= MAX_REBOUND_RANGE_FRACTION
    )
    first = group.iloc[0]
    return {
        "system_id": int(first["system_id"]),
        "rgi_id": first["rgi_id"],
        "route": first["route"],
        "upward_variation_share": upward_share,
        "maximum_rebound_km": maximum_rebound_km,
        "smoothed_curve_range_km": curve_range_km,
        "maximum_rebound_range_fraction": rebound_fraction,
        "net_change_km": net_change_km,
        "temporary_exclusion_nearly_monotonic_decline": temporary_exclusion,
    }


def plot_group_grid(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    system_ids: list[int],
    title: str,
    filename: str,
    columns: int,
) -> Path:
    rows = int(np.ceil(len(system_ids) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.35 * rows), sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    metrics_by_id = metrics.set_index("system_id")
    for axis, system_id in zip(axes_array, system_ids, strict=False):
        group = curves[curves["system_id"] == system_id].sort_values("year")
        metric = metrics_by_id.loc[system_id]
        axis.plot(
            group["year"],
            group["length_km"],
            color=ROUTE_COLORS[str(metric["route"])],
            linewidth=1.25,
        )
        axis.set_title(
            f"{system_id} | up {metric['upward_variation_share']:.2f} | "
            f"rebound {metric['maximum_rebound_range_fraction']:.2f}",
            fontsize=8,
            pad=3,
        )
        axis.set_xlim(1500, 2000)
        axis.grid(axis="y", color="#e1e1e1", linewidth=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=6, length=2)
    for axis in axes_array[len(system_ids) :]:
        axis.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="semibold", y=0.997)
    fig.supxlabel("Year", fontsize=11)
    fig.supylabel("Glacier length (km)", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.025, 1, 0.975), h_pad=1.0, w_pad=0.8)
    path = OUTPUT / filename
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_comparison(
    curves: pd.DataFrame, retained_ids: list[int], excluded_ids: list[int]
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.5), sharex=True)
    groups = [
        (retained_ids, "Retained: 32 curves with meaningful non-monotonic structure"),
        (excluded_ids, "Temporarily excluded: 26 nearly monotonic declining curves"),
    ]
    for axis, (system_ids, title) in zip(axes, groups, strict=True):
        for system_id in system_ids:
            group = curves[curves["system_id"] == system_id].sort_values("year")
            route = str(group.iloc[0]["route"])
            normalized = group["length_km"] / float(group.iloc[0]["lia_target_km"])
            axis.plot(
                group["year"],
                normalized,
                color=ROUTE_COLORS[route],
                linewidth=0.85,
                alpha=0.48,
            )
        axis.axhline(1.0, color="#8a3b3b", linewidth=1.0, linestyle="--")
        axis.set_title(title)
        axis.set_ylabel("Length / LIA target")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Year")
    axes[-1].set_xlim(1500, 2000)
    fig.suptitle(
        "Accepted-58 temporary curve-shape screen",
        fontsize=16,
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUTPUT / "retained_32_vs_excluded_26_overview.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    curves = pd.read_csv(CURVES)
    if curves["system_id"].nunique() != 58:
        raise RuntimeError("annual curve table does not contain exactly 58 systems")
    metrics = pd.DataFrame(
        [shape_metrics(group) for _, group in curves.groupby("system_id", sort=True)]
    ).sort_values("system_id")

    excluded = metrics[
        metrics["temporary_exclusion_nearly_monotonic_decline"]
    ].copy()
    retained = metrics[
        ~metrics["temporary_exclusion_nearly_monotonic_decline"]
    ].copy()
    excluded_ids = excluded["system_id"].astype(int).tolist()
    retained_ids = retained["system_id"].astype(int).tolist()
    if len(excluded_ids) != 26 or len(retained_ids) != 32:
        raise RuntimeError(
            f"unexpected screen count: excluded={len(excluded_ids)}, retained={len(retained_ids)}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "accepted_58_curve_shape_metrics.csv", index=False)
    retained.to_csv(OUTPUT / "retained_32_systems.csv", index=False)
    excluded.to_csv(OUTPUT / "temporarily_excluded_26_systems.csv", index=False)
    comparison = plot_comparison(curves, retained_ids, excluded_ids)
    retained_grid = plot_group_grid(
        curves,
        metrics,
        retained_ids,
        "Retained 32 accepted curves after temporary shape screen",
        "retained_32_length_curves_grid.png",
        columns=8,
    )
    excluded_grid = plot_group_grid(
        curves,
        metrics,
        excluded_ids,
        "Temporarily excluded 26 nearly monotonic declining curves",
        "temporarily_excluded_26_length_curves_grid.png",
        columns=7,
    )

    summary = {
        "status": "temporary-curve-shape-screen-complete",
        "formal_accepted_count_unchanged": 58,
        "temporary_excluded_count": len(excluded_ids),
        "temporary_retained_count": len(retained_ids),
        "criterion": {
            "smoothing_window_years": SMOOTHING_YEARS,
            "net_change_must_be_negative": True,
            "maximum_upward_variation_share": MAX_UPWARD_VARIATION_SHARE,
            "maximum_rebound_range_fraction": MAX_REBOUND_RANGE_FRACTION,
        },
        "temporarily_excluded_system_ids": excluded_ids,
        "retained_system_ids": retained_ids,
        "registry_or_controller_modified": False,
        "outputs": {
            "metrics_csv": "accepted_58_curve_shape_metrics.csv",
            "retained_csv": "retained_32_systems.csv",
            "excluded_csv": "temporarily_excluded_26_systems.csv",
            "comparison_png": comparison.name,
            "retained_grid_png": retained_grid.name,
            "excluded_grid_png": excluded_grid.name,
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
