"""Attribute fixed-bed red-gate failures to thin, off-axis and catchment effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import resolve


def run(config: dict) -> dict:
    audit = config["lia_dynamic_fixed_bed_audit"]
    validation_root = resolve(config["output_dir"])
    source_dir = validation_root / str(audit["output_subdir"])
    diagnostic_dir = validation_root / "lia_dynamic_fixed_bed_diagnosis_v04"
    systems = pd.read_csv(source_dir / "fixed_bed_geometry_systems_v03.csv")
    red_ids = [
        int(value)
        for value in systems.loc[systems["mb_gate"].eq("red"), "lia_unique_id"]
    ]
    rows = []
    plot_data = []
    for unique_id in red_ids:
        nodes = pd.read_csv(
            diagnostic_dir / f"fixed_bed_nodes_{unique_id}_v03.csv"
        )
        high = nodes[nodes["shape_factor"].gt(1.0)].copy()
        high["contract_area_change_m2"] = (
            (high["effective_width_m"] - high["observed_width_m"])
            * high["dx_meter"]
        )
        high["thin_node"] = high["thickness_m"].lt(5.0)
        high["off_thickness_axis"] = (
            high["maximum_thickness_offset_m"].gt(37.5)
            & (
                high["cross_section_maximum_thickness_m"]
                / high["thickness_m"]
            ).gt(1.5)
        )
        high["catchment_cut"] = high["catchment_cut_side_count"].gt(0)
        total_change = float(high["contract_area_change_m2"].sum())

        def contribution(mask) -> float:
            if total_change == 0:
                return 0.0
            return float(high.loc[mask, "contract_area_change_m2"].sum() / total_change)

        off_axis_fraction = contribution(high["off_thickness_axis"])
        catchment_fraction = contribution(high["catchment_cut"])
        thin_fraction = contribution(high["thin_node"])
        secondary = []
        if thin_fraction >= 0.5:
            secondary.append("thin-node-amplification")
        if catchment_fraction >= 0.5:
            secondary.append("catchment-boundary-coupling")
        rows.append(
            {
                "lia_unique_id": unique_id,
                "f_above_1_node_count": int(len(high)),
                "f_above_1p5_node_count": int(high["shape_factor"].gt(1.5).sum()),
                "maximum_shape_factor": float(high["shape_factor"].max()),
                "median_centerline_thickness_m": float(high["thickness_m"].median()),
                "median_thickness_axis_offset_m": float(
                    high["maximum_thickness_offset_m"].median()
                ),
                "p90_thickness_axis_offset_m": float(
                    high["maximum_thickness_offset_m"].quantile(0.9)
                ),
                "off_axis_area_change_fraction": off_axis_fraction,
                "thin_node_area_change_fraction": thin_fraction,
                "catchment_cut_area_change_fraction": catchment_fraction,
                "primary_class": (
                    "centerline-thickness-axis-mismatch"
                    if off_axis_fraction >= 0.9
                    else "mixed-cross-section-mismatch"
                ),
                "secondary_classes": ";".join(secondary) or "none",
            }
        )
        plot_data.append((unique_id, nodes, high))
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    frame.to_csv(
        diagnostic_dir / "fixed_bed_red_attribution_v04.csv", index=False
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), constrained_layout=True)
    for axis, (unique_id, nodes, high) in zip(axes.flat, plot_data):
        x0 = float(nodes["station_x_m"].min())
        y0 = float(nodes["station_y_m"].min())
        axis.scatter(
            (nodes["station_x_m"] - x0) / 1000.0,
            (nodes["station_y_m"] - y0) / 1000.0,
            color="#b8b8b8",
            s=5,
            alpha=0.55,
        )
        points = axis.scatter(
            (high["station_x_m"] - x0) / 1000.0,
            (high["station_y_m"] - y0) / 1000.0,
            c=np.log10(high["shape_factor"]),
            cmap="magma",
            s=14 + 8 * np.clip(np.log10(high["shape_factor"]), 0, 2),
            alpha=0.9,
        )
        axis.set_title(str(unique_id))
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("Local easting (km)")
        axis.set_ylabel("Local northing (km)")
        axis.grid(True, color="#e2e2e2", linewidth=0.5)
    fig.colorbar(points, ax=axes, label="log10(F)", shrink=0.72)
    fig.suptitle("Locations of F > 1 nodes in red-gate systems")
    fig.savefig(
        diagnostic_dir / "fixed_bed_red_nodes_v04.png", dpi=180
    )
    plt.close(fig)

    failures = pd.read_csv(
        diagnostic_dir / "fixed_bed_geometry_failures_v03.csv"
    )
    summary = {
        "status": "root-cause-classified-stop-before-dynamics",
        "red_system_count": int(len(frame)),
        "mapping_failure_count": int(len(failures)),
        "mapping_failure_system_ids": [
            int(value) for value in failures["lia_unique_id"]
        ],
        "common_primary_class": "centerline-thickness-axis-mismatch",
        "minimum_off_axis_area_change_fraction": float(
            frame["off_axis_area_change_fraction"].min()
        ),
        "thickness_axis_offset_median_range_m": [
            float(frame["median_thickness_axis_offset_m"].min()),
            float(frame["median_thickness_axis_offset_m"].max()),
        ],
        "decision": "Do not modify the fixed-bed formula and do not run dynamics. Test an automatically bed-routed or thickness-axis-aligned centerline on these eight systems under the same frozen gates.",
        "elevation_band_fallback": "Use if the alternative centerline audit still produces any red system or mapping failure.",
    }
    (diagnostic_dir / "fixed_bed_red_diagnosis_v04.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
