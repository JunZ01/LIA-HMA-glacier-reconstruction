"""Plot compact review atlases from the core-191 topology audit."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def plot_atlas(
    *,
    system_ids: list[int],
    systems: pd.DataFrame,
    outlines: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
    issues: gpd.GeoDataFrame,
    output_path: Path,
    title_suffix: str,
) -> None:
    columns = 3
    rows = max(1, math.ceil(len(system_ids) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.8 * rows), squeeze=False)
    for axis, system_id in zip(axes.flat, system_ids):
        outline = outlines.loc[outlines["UniqueID"].astype(int).eq(system_id)]
        lines = flowlines.loc[flowlines["lia_unique_id"].astype(int).eq(system_id)]
        problem = issues.loc[
            issues["lia_unique_id"].astype(int).eq(system_id)
            & issues["event_type"].eq("outside-lia")
        ]
        outline.plot(ax=axis, facecolor="none", edgecolor="black", linewidth=1.2)
        lines.loc[~lines["is_main_branch"].astype(bool)].plot(
            ax=axis, color="#5e81ac", linewidth=0.8, alpha=0.8
        )
        lines.loc[lines["is_main_branch"].astype(bool)].plot(
            ax=axis, color="#2e3440", linewidth=1.5, alpha=0.9
        )
        if not problem.empty:
            problem.plot(ax=axis, color="#bf616a", linewidth=3.0)
        row = systems.loc[systems["lia_unique_id"].astype(int).eq(system_id)].iloc[0]
        validity = "valid" if bool(row["source_lia_valid"]) else "invalid SHP"
        axis.set_title(f"UID {system_id} | {row['main_rgi_id']} | {validity}", fontsize=9)
        axis.set_aspect("equal")
        axis.axis("off")
    for axis in axes.flat[len(system_ids) :]:
        axis.axis("off")
    figure.suptitle(title_suffix, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--lia-shp", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    systems = pd.read_csv(audit_dir / "core191_flowline_topology_systems_v1.csv")
    outlines = gpd.read_file(args.lia_shp)
    flowlines = gpd.read_file(audit_dir / "core191_member_flowlines_v1.geojson")
    issues = gpd.read_file(audit_dir / "core191_flowline_topology_issues_v1.geojson")
    hard_outside_ids = systems.loc[
        systems["source_outside_lia_beyond_tolerance_branch_count"] > 0,
        "lia_unique_id",
    ].astype(int).tolist()
    invalid_ids = systems.loc[~systems["source_lia_valid"].astype(bool), "lia_unique_id"].astype(int).tolist()
    plot_atlas(
        system_ids=hard_outside_ids,
        systems=systems,
        outlines=outlines,
        flowlines=flowlines,
        issues=issues,
        output_path=audit_dir / "core191_hard_outside_flowlines_atlas_v1.png",
        title_suffix="Core-191 source flowlines beyond one-grid-cell LIA tolerance",
    )
    plot_atlas(
        system_ids=invalid_ids,
        systems=systems,
        outlines=outlines,
        flowlines=flowlines,
        issues=issues,
        output_path=audit_dir / "core191_invalid_lia_polygons_atlas_v1.png",
        title_suffix="Core-191 source LIA polygons requiring make_valid",
    )


if __name__ == "__main__":
    main()
