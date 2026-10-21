"""Plot selected LIA system boundaries without altering the source geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_IDS = (7, 79, 1044, 1545, 3584)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lia-shp", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", default=list(DEFAULT_IDS))
    args = parser.parse_args()

    outlines = gpd.read_file(args.lia_shp)
    selected = outlines[outlines["UniqueID"].astype(int).isin(args.ids)].copy()
    if set(selected["UniqueID"].astype(int)) != set(args.ids):
        missing = sorted(set(args.ids) - set(selected["UniqueID"].astype(int)))
        raise ValueError(f"missing-lia-ids:{missing}")
    targets = pd.read_csv(args.targets).set_index("lia_unique_id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, len(args.ids), height_ratios=[1, 1.05])
    overview = fig.add_subplot(grid[0, :])
    selected.plot(ax=overview, facecolor="#8ecae6", edgecolor="#023047", linewidth=1.2)
    for row in selected.itertuples():
        point = row.geometry.representative_point()
        overview.annotate(str(int(row.UniqueID)), (point.x, point.y), ha="center", va="center", fontsize=10,
                          bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#023047", "alpha": 0.85})
    overview.set_title("Selected LIA systems: spatial overview")
    overview.set_aspect("equal")
    overview.set_axis_off()

    detail_axes = [fig.add_subplot(grid[1, index]) for index in range(len(args.ids))]
    for axis, system_id in zip(detail_axes, args.ids):
        row = selected[selected["UniqueID"].astype(int).eq(system_id)].iloc[0]
        gpd.GeoSeries([row.geometry], crs=selected.crs).plot(
            ax=axis, facecolor="#8ecae6", edgecolor="#023047", linewidth=1.3
        )
        target = targets.loc[system_id]
        axis.set_title(
            f"LIA system {system_id}\n"
            f"area={target['target_area_km2']:.1f} km², "
            f"main length={target['target_main_flowline_length_m']/1000:.2f} km"
        )
        axis.set_aspect("equal")
        axis.set_axis_off()

    fig.suptitle("Five LIA boundaries used for the next calibration pilot", fontsize=15)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
