"""Summarise and plot the frozen fixed-bed geometry audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import resolve


GATE_COLORS = {"green": "#237a57", "yellow": "#d39518", "red": "#b33a3a"}


def build_report(config: dict) -> dict:
    audit = config["lia_dynamic_fixed_bed_audit"]
    output_dir = resolve(config["output_dir"]) / str(audit["output_subdir"])
    systems = pd.read_csv(output_dir / "fixed_bed_geometry_systems_v03.csv")
    nodes = []
    for unique_id in systems["lia_unique_id"]:
        frame = pd.read_csv(output_dir / f"fixed_bed_nodes_{unique_id}_v03.csv")
        nodes.append(frame)
    node_frame = pd.concat(nodes, ignore_index=True)

    node_frame["observed_area_m2"] = (
        node_frame["observed_width_m"] * node_frame["dx_meter"]
    )
    node_frame["effective_area_m2"] = (
        node_frame["effective_width_m"] * node_frame["dx_meter"]
    )
    node_frame["contract_area_change_m2"] = (
        node_frame["effective_area_m2"] - node_frame["observed_area_m2"]
    )
    regime = (
        node_frame.groupby(["lia_unique_id", "regime_code"], as_index=False)
        .agg(
            node_count=("node_index", "size"),
            observed_area_m2=("observed_area_m2", "sum"),
            effective_area_m2=("effective_area_m2", "sum"),
            contract_area_change_m2=("contract_area_change_m2", "sum"),
            median_shape_factor=("shape_factor", "median"),
            maximum_shape_factor=("shape_factor", "max"),
        )
    )
    regime["regime"] = regime["regime_code"].map(
        {1: "F<=0.5-near-triangular", 2: "0.5<F<=1-exact-width", 3: "F>1-rectangular"}
    )
    regime.to_csv(output_dir / "fixed_bed_regime_attribution_v03.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for gate, subset in systems.groupby("mb_gate"):
        color = GATE_COLORS.get(gate, "#666666")
        axes[0].scatter(
            subset["lia_area_km2_sample_manifest"],
            100.0 * subset["contract_area_error_fraction"],
            label=gate,
            color=color,
            s=42,
            alpha=0.9,
        )
        axes[1].scatter(
            subset["shape_factor_above_1_fraction"],
            subset["maximum_absolute_width_induced_net_mb_bias_mwea"],
            color=color,
            s=42,
            alpha=0.9,
        )
        axes[2].scatter(
            subset["width_weighted_mean_surface_elevation_shift_m"],
            subset["maximum_absolute_width_induced_net_mb_bias_mwea"],
            color=color,
            s=42,
            alpha=0.9,
        )
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("symlog", linthresh=5.0)
    axes[0].set_xlabel("LIA system area (km2)")
    axes[0].set_ylabel("Contract area error (%)")
    axes[0].legend(frameon=False, title="MB gate")
    axes[1].set_xlabel("Fraction of nodes with F > 1")
    axes[1].set_ylabel("Maximum net MB bias (m w.e. yr-1)")
    axes[2].axvline(0, color="#777777", linewidth=0.8)
    axes[2].set_xlabel("Width-weighted elevation shift (m)")
    axes[2].set_ylabel("Maximum net MB bias (m w.e. yr-1)")
    for row in systems[systems["mb_gate"].eq("red")].itertuples(index=False):
        label = str(int(row.lia_unique_id))
        axes[0].annotate(
            label,
            (row.lia_area_km2_sample_manifest, 100.0 * row.contract_area_error_fraction),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
        axes[1].annotate(
            label,
            (
                row.shape_factor_above_1_fraction,
                row.maximum_absolute_width_induced_net_mb_bias_mwea,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)
    fig.suptitle("Fixed-bed FluxBased contract audit (75 m, n=22 successful)")
    figure_path = output_dir / "fixed_bed_geometry_audit_v03.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    red = systems[systems["mb_gate"].eq("red")]
    summary = {
        "status": "stop-before-dynamics",
        "successful_system_count": int(len(systems)),
        "failed_mapping_system_count": 2,
        "gate_counts": {
            str(key): int(value)
            for key, value in systems["mb_gate"].value_counts().to_dict().items()
        },
        "red_system_ids": [int(value) for value in red["lia_unique_id"]],
        "median_contract_area_error_fraction": float(
            systems["contract_area_error_fraction"].median()
        ),
        "p90_contract_area_error_fraction": float(
            systems["contract_area_error_fraction"].quantile(0.9)
        ),
        "maximum_contract_area_error_fraction": float(
            systems["contract_area_error_fraction"].max()
        ),
        "median_maximum_net_mb_bias_mwea": float(
            systems["maximum_absolute_width_induced_net_mb_bias_mwea"].median()
        ),
        "p90_maximum_net_mb_bias_mwea": float(
            systems["maximum_absolute_width_induced_net_mb_bias_mwea"].quantile(0.9)
        ),
        "maximum_net_mb_bias_mwea": float(
            systems["maximum_absolute_width_induced_net_mb_bias_mwea"].max()
        ),
        "fixed_bed_invariants": {
            "maximum_absolute_bed_adjustment_m": float(
                systems["maximum_absolute_bed_adjustment_m"].max()
            ),
            "maximum_absolute_thickness_adjustment_m": float(
                systems["maximum_absolute_thickness_adjustment_m"].max()
            ),
            "maximum_absolute_section_preservation_error_fraction": float(
                systems["contract_section_preservation_error_fraction"].abs().max()
            ),
        },
        "decision": "The fixed-bed trapezoid formula is numerically valid but not scientifically transferable across the frozen sample; do not run junction or zero-MB dynamics with this contract.",
        "next_method_candidate": "replace representative centerline thickness with a section-consistent characteristic thickness or switch to an elevation-band representation; both require a new pre-registered audit",
    }
    (output_dir / "fixed_bed_geometry_decision_v03.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build_report(config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
