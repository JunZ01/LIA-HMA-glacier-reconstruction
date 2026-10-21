"""Plot spatial and numeric length definitions for one LIA glacier system."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import tarfile
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from shapely import affinity


def _load_pickle(path: Path):
    opener = gzip.open if path.suffix == ".gz" or path.read_bytes()[:2] == b"\x1f\x8b" else open
    with opener(path, "rb") as stream:
        return pickle.load(stream)


def _grid_xy(grid: dict, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = (float(value) for value in grid["x0y0"])
    dx, dy = (float(value) for value in grid["dxdy"])
    return x0 + np.asarray(col) * dx, y0 + np.asarray(row) * dy


def _line_length(coords: np.ndarray) -> float:
    return float(np.hypot(*np.diff(coords, axis=0).T).sum())


def build_plot(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.grid_json.open("r", encoding="utf-8") as stream:
        grid = json.load(stream)
    with xr.open_dataset(args.lia_grid) as ds:
        mask = ds["lia_mask"].values.astype(bool)

    network = _load_pickle(args.lia_network)
    main_index = int(network["main_branch_index"])
    lia_main = np.asarray(network["branches"][main_index]["coords_projected_m"], dtype=float)
    lia_main_length_m = _line_length(lia_main)

    modern_centerlines = _load_pickle(args.modern_gdir / "centerlines.pkl")
    root_indices = [i for i, line in enumerate(modern_centerlines) if line.flows_to is None]
    if len(root_indices) != 1:
        raise ValueError(f"modern-root-centerline-count:{len(root_indices)}")
    modern_main_index = root_indices[0]
    modern_main = modern_centerlines[modern_main_index]
    modern_pixels = np.asarray(modern_main.line.coords, dtype=float)
    mx, my = _grid_xy(grid, modern_pixels[:, 0], modern_pixels[:, 1])
    modern_main_xy = np.column_stack([mx, my])
    modern_main_length_m = _line_length(modern_main_xy)

    references = pd.read_csv(args.references)
    ref = references[references["lia_unique_id"].astype(int).eq(args.system_id)]
    if len(ref) != 1:
        raise ValueError(f"reference-row-count:{args.system_id}:{len(ref)}")
    ref = ref.iloc[0]

    band_results = pd.read_csv(args.elevation_band_results)
    band = band_results[band_results["lia_unique_id"].astype(int).eq(args.system_id)]
    if len(band) != 1:
        raise ValueError(f"elevation-band-row-count:{args.system_id}:{len(band)}")
    band = band.iloc[0]

    with tempfile.TemporaryDirectory(prefix="lia-length-audit-") as tmp:
        with tarfile.open(args.modern_gdir / "outlines.tar.gz", "r:gz") as archive:
            archive.extractall(tmp, filter="data")
        outline = gpd.read_file(Path(tmp) / "outlines.shp")
        if outline.crs is not None:
            outline = outline.to_crs(network["crs"])
        outline = outline.copy()
        outline.geometry = outline.geometry.apply(
            lambda geometry: affinity.scale(
                geometry, xfact=0.001, yfact=0.001, origin=(0.0, 0.0)
            )
        )

    rows = pd.DataFrame(
        [
            ("LIA registered target", float(ref["lia_registered_length_m"]), "GIS/reference"),
            ("LIA spatial main centerline", lia_main_length_m, f"network branch {main_index}"),
            ("RGI registered target", float(ref["rgi_registered_length_m"]), "GIS/reference"),
            ("OGGM modern root centerline", modern_main_length_m, f"centerlines index {modern_main_index}"),
            ("Elevation-band initial ice", float(band["pseudo_flowline_length_m"]), "1-D pseudo-flowline"),
        ],
        columns=["definition", "length_m", "source"],
    )
    rows["length_km"] = rows["length_m"] / 1000.0
    rows.to_csv(args.output_dir / f"lia_{args.system_id}_length_definitions.csv", index=False)

    x0, y0 = (float(value) for value in grid["x0y0"])
    dx, dy = (float(value) for value in grid["dxdy"])
    ny, nx = mask.shape
    x_centers = x0 + np.arange(nx) * dx
    y_centers = y0 + np.arange(ny) * dy

    fig, (map_ax, bar_ax) = plt.subplots(1, 2, figsize=(14.5, 7.2), gridspec_kw={"width_ratios": [1.35, 1]})
    map_ax.contourf(x_centers / 1000.0, y_centers / 1000.0, mask, levels=[0.5, 1.5], colors=["#b9d8e8"], alpha=0.72)
    outline.boundary.plot(ax=map_ax, color="#333333", linewidth=1.2, linestyle="--")
    map_ax.plot(lia_main[:, 0] / 1000.0, lia_main[:, 1] / 1000.0, color="#c43c35", linewidth=2.8)
    map_ax.plot(modern_main_xy[:, 0] / 1000.0, modern_main_xy[:, 1] / 1000.0, color="#2c6e9b", linewidth=2.0)
    map_ax.scatter(lia_main[0, 0] / 1000.0, lia_main[0, 1] / 1000.0, marker="^", s=55, color="#c43c35", zorder=5)
    map_ax.scatter(lia_main[-1, 0] / 1000.0, lia_main[-1, 1] / 1000.0, marker="o", s=40, color="#c43c35", zorder=5)
    map_ax.set_aspect("equal")
    map_ax.set_xlabel("Projected x (km)")
    map_ax.set_ylabel("Projected y (km)")
    map_ax.set_title(f"LIA system {args.system_id}: spatial flowline selection")
    map_ax.legend(
        handles=[
            Line2D([0], [0], color="#c43c35", lw=2.8, label=f"LIA main branch {main_index}: {lia_main_length_m/1000:.2f} km"),
            Line2D([0], [0], color="#2c6e9b", lw=2.0, label=f"OGGM modern root index {modern_main_index}: {modern_main_length_m/1000:.2f} km"),
            Line2D([0], [0], color="#333333", lw=1.2, ls="--", label="RGI outline"),
            Line2D([0], [0], color="#b9d8e8", lw=7, label="LIA outline/mask"),
        ],
        loc="best",
        frameon=True,
        fontsize=9,
    )

    colors = ["#c43c35", "#c43c35", "#2c6e9b", "#2c6e9b", "#777777"]
    positions = np.arange(len(rows))[::-1]
    bar_ax.barh(positions, rows["length_km"], color=colors, alpha=0.88)
    bar_ax.set_yticks(positions, rows["definition"])
    bar_ax.set_xlabel("Length (km)")
    bar_ax.set_title("Direct comparison of length definitions")
    bar_ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    for y, value in zip(positions, rows["length_km"]):
        bar_ax.text(value + 0.25, y, f"{value:.2f}", va="center", fontsize=10)
    bar_ax.set_xlim(0, max(rows["length_km"]) * 1.18)
    bar_ax.set_ylim(-1.05, positions.max() + 0.65)
    bar_ax.text(
        0.5,
        -0.72,
        "Elevation-band target was its unscaled hypsometric length "
        f"({float(band['target_main_flowline_length_m'])/1000:.2f} km), not the mapped LIA main line.",
        va="center",
        fontsize=9,
        color="#444444",
    )
    fig.suptitle("Why the simulated OGGM length differs from the mapped main-flowline length", fontsize=14)
    fig.tight_layout()
    output_png = args.output_dir / f"lia_{args.system_id}_length_definition_audit.png"
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output_png)
    print(rows.to_string(index=False))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", type=int, default=3584)
    parser.add_argument("--grid-json", required=True, type=Path)
    parser.add_argument("--modern-gdir", required=True, type=Path)
    parser.add_argument("--lia-grid", required=True, type=Path)
    parser.add_argument("--lia-network", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--elevation-band-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    build_plot(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
