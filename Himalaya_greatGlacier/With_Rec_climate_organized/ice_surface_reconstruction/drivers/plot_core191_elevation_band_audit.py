"""Summarize the corrected 191-system elevation-band length contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results)
    if len(frame) != 191 or frame["lia_unique_id"].nunique() != 191:
        raise ValueError("core191-elevation-band-result-count-mismatch")
    if not frame["status"].eq("ok").all():
        raise ValueError("core191-elevation-band-has-build-failures")

    frame["unscaled_length_error_fraction"] = (
        frame["unscaled_pseudo_flowline_length_m"]
        - frame["target_main_flowline_length_m"]
    ) / frame["target_main_flowline_length_m"]
    frame["corrected_length_error_fraction"] = (
        frame["pseudo_flowline_length_m"]
        - frame["target_main_flowline_length_m"]
    ) / frame["target_main_flowline_length_m"]
    frame["length_contract_pass"] = (
        frame["corrected_length_error_fraction"].abs()
        <= float(args.length_tolerance_fraction)
    )
    frame["static_geometry_pass"] = (
        frame["area_error_fraction"].abs().le(1e-12)
        & frame["volume_error_fraction"].abs().le(1e-12)
        & frame["hypsometry_maximum_cumulative_area_error_fraction"].le(0.02)
    )
    frame["zero_mb_pass"] = (
        frame["final_zero_mb_volume_change_fraction"].abs().le(0.001)
        & frame["terminal_thickness_m"].le(10.0)
    )
    frame["all_geometry_gates_pass"] = (
        frame["length_contract_pass"]
        & frame["static_geometry_pass"]
        & frame["zero_mb_pass"]
    )
    frame = frame.sort_values("lia_unique_id")
    frame.to_csv(args.output_dir / "core191_elevation_band_audit.csv", index=False)
    failures = frame[~frame["all_geometry_gates_pass"]].copy()
    failures.to_csv(args.output_dir / "core191_elevation_band_failures.csv", index=False)

    target_km = frame["target_main_flowline_length_m"] / 1000.0
    unscaled_km = frame["unscaled_pseudo_flowline_length_m"] / 1000.0
    corrected_km = frame["pseudo_flowline_length_m"] / 1000.0
    maximum = float(max(target_km.max(), unscaled_km.max(), corrected_km.max()))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    ax = axes[0]
    ax.scatter(target_km, unscaled_km, s=22, alpha=0.58, color="#8b8b8b", label="Before: unscaled hypsometric length")
    ax.scatter(target_km, corrected_km, s=18, alpha=0.8, color="#2878a5", label="After: main-flowline target applied")
    ax.plot([0, maximum], [0, maximum], color="#333333", linewidth=1.0, linestyle="--", label="1:1")
    ax.set_xlim(0, maximum * 1.03)
    ax.set_ylim(0, maximum * 1.03)
    ax.set_xlabel("Spatial LIA main-flowline target (km)")
    ax.set_ylabel("Elevation-band initial ice length (km)")
    ax.set_title("Length definition alignment")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color="#dddddd", linewidth=0.7)

    ax = axes[1]
    old = np.sort(frame["unscaled_length_error_fraction"].abs().to_numpy()) * 100.0
    new = np.sort(frame["corrected_length_error_fraction"].abs().to_numpy()) * 100.0
    rank = np.arange(1, len(frame) + 1)
    ax.plot(rank, old, color="#8b8b8b", linewidth=1.8, label="Before")
    ax.plot(rank, new, color="#2878a5", linewidth=2.0, label="After")
    ax.axhline(args.length_tolerance_fraction * 100.0, color="#b33b32", linewidth=1.2, linestyle="--", label="10% gate")
    ax.set_yscale("log")
    ax.set_xlabel("System rank")
    ax.set_ylabel("Absolute length error (%)")
    ax.set_title("All 191 systems")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.7, which="both")

    ax = axes[2]
    zero = frame.sort_values("final_zero_mb_volume_change_fraction").reset_index(drop=True)
    values = zero["final_zero_mb_volume_change_fraction"].to_numpy() * 100.0
    passed = zero["zero_mb_pass"].to_numpy()
    colors = np.where(passed, "#2878a5", "#b33b32")
    ax.scatter(np.arange(1, len(zero) + 1), values, s=24, color=colors, alpha=0.85)
    ax.axhline(0.1, color="#b33b32", linewidth=1.1, linestyle="--")
    ax.axhline(-0.1, color="#b33b32", linewidth=1.1, linestyle="--")
    for index, row in zero.loc[~zero["zero_mb_pass"]].iterrows():
        ax.annotate(
            str(int(row["lia_unique_id"])),
            (index + 1, float(row["final_zero_mb_volume_change_fraction"]) * 100.0),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Systems sorted by 5-year volume change")
    ax.set_ylabel("Zero-MB volume change after 5 years (%)")
    ax.set_title("Dynamic compatibility check")
    ax.grid(color="#dddddd", linewidth=0.7)

    fig.suptitle("Core-191 corrected elevation-band geometry audit", fontsize=14)
    fig.tight_layout()
    figure_path = args.output_dir / "core191_elevation_band_audit.png"
    fig.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "system_count": int(len(frame)),
        "build_success_count": int(frame["status"].eq("ok").sum()),
        "length_contract_pass_count": int(frame["length_contract_pass"].sum()),
        "static_geometry_pass_count": int(frame["static_geometry_pass"].sum()),
        "zero_mb_pass_count": int(frame["zero_mb_pass"].sum()),
        "all_geometry_gates_pass_count": int(frame["all_geometry_gates_pass"].sum()),
        "maximum_corrected_length_error_fraction": float(
            frame["corrected_length_error_fraction"].abs().max()
        ),
        "median_unscaled_length_error_fraction": float(
            frame["unscaled_length_error_fraction"].median()
        ),
        "zero_mb_failed_system_ids": [
            int(value) for value in frame.loc[~frame["zero_mb_pass"], "lia_unique_id"]
        ],
        "figure": str(figure_path),
    }
    (args.output_dir / "core191_elevation_band_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--length-tolerance-fraction", type=float, default=0.10)
    build(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
