"""Plot a LIA thickness raster with model flowlines and connectivity markers."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap


FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT_PATH.exists():
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


def load_flowlines(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if hasattr(payload, "fls"):
        payload = payload.fls
    if isinstance(payload, dict):
        for key in ("flowlines", "model_flowlines", "fls"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"unsupported-state-payload:{type(payload).__name__}")
    return list(payload)


def projected_coordinates(flowline, grid: dict) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(flowline.line.coords, dtype=float)
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    return x0 + coords[:, 0] * dx, y0 + coords[:, 1] * dy


def components(thick: np.ndarray, tolerance: float) -> list[np.ndarray]:
    active = np.flatnonzero(np.isfinite(thick) & (thick > tolerance))
    if not len(active):
        return []
    cuts = np.flatnonzero(np.diff(active) > 1) + 1
    return [chunk for chunk in np.split(active, cuts) if len(chunk)]


def node_rows(flowlines, grid: dict, tolerance: float) -> list[dict]:
    roots = [i for i, flowline in enumerate(flowlines) if flowline.flows_to is None]
    if len(roots) != 1:
        raise ValueError(f"root-count:{len(roots)}")
    root = roots[0]
    rows = []
    for branch, flowline in enumerate(flowlines):
        thick = np.asarray(flowline.thick, dtype=float)
        x, y = projected_coordinates(flowline, grid)
        branch_components = components(thick, tolerance)
        component_id = np.full(len(thick), -1, dtype=int)
        for index, nodes in enumerate(branch_components):
            component_id[nodes] = index
        internal_gap = np.zeros(len(thick), dtype=bool)
        active = np.flatnonzero(thick > tolerance)
        if len(active):
            internal_gap[active[0] : active[-1] + 1] = thick[active[0] : active[-1] + 1] <= tolerance
        for node in range(len(thick)):
            rows.append({
                "branch": branch,
                "node": node,
                "is_root": branch == root,
                "x_m": float(x[node]),
                "y_m": float(y[node]),
                "thickness_m": float(thick[node]),
                "active": bool(thick[node] > tolerance),
                "component": int(component_id[node]),
                "internal_gap_node": bool(internal_gap[node]),
            })
    return rows


def plot(
    system_id: int,
    netcdf: Path,
    state: Path,
    grid_json: Path,
    output_root: Path,
    state_label: str,
    highlight_branch: int | None,
    tolerance: float,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    grid = json.loads(grid_json.read_text(encoding="utf-8"))
    flowlines = load_flowlines(state)
    rows = node_rows(flowlines, grid, tolerance)
    roots = sorted({row["branch"] for row in rows if row["is_root"]})
    root = roots[0]

    with xr.open_dataset(netcdf) as ds:
        h = np.asarray(ds["thickness_m"].isel(member=0), dtype=float)
        valid = np.asarray(ds["valid_mask"].isel(member=0), dtype=bool)
        lia = np.asarray(ds["lia_mask"], dtype=bool)
        mask = valid & lia & np.isfinite(h) & (h > tolerance)
        shown = np.where(mask, h, np.nan)
        x = np.asarray(ds["x"], dtype=float) / 1000.0
        y = np.asarray(ds["y"], dtype=float) / 1000.0

    bounds = np.array([0, 5, 10, 20, 50, 100, 200, 350, 550], dtype=float)
    cmap = ListedColormap(["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"])
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(bounds, cmap.N)

    by_branch = {}
    for row in rows:
        by_branch.setdefault(row["branch"], []).append(row)
    gap_rows = [row for row in rows if row["internal_gap_node"]]

    for theme, colors in {
        "light": {"bg": "#ffffff", "fg": "#111827", "flow": "#6b7280", "root": "#111827", "highlight": "#b45309", "gap": "#be123c"},
        "dark": {"bg": "#111827", "fg": "#f9fafb", "flow": "#9ca3af", "root": "#f9fafb", "highlight": "#fbbf24", "gap": "#fb7185"},
    }.items():
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.1), constrained_layout=True)
        fig.patch.set_facecolor(colors["bg"])
        for ax in axes:
            ax.set_facecolor(colors["bg"])
            ax.tick_params(colors=colors["fg"], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(colors["flow"])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("投影东向距离 (km)", color=colors["fg"])
            ax.set_ylabel("投影北向距离 (km)", color=colors["fg"])
        im = axes[0].pcolormesh(x, y, shown, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        axes[0].contour(x, y, mask.astype(float), levels=[0.5], colors=[colors["fg"]], linewidths=0.45, alpha=0.7)
        for branch, branch_rows in by_branch.items():
            xx = np.array([row["x_m"] for row in branch_rows]) / 1000.0
            yy = np.array([row["y_m"] for row in branch_rows]) / 1000.0
            color = colors["root"] if branch == root else colors["flow"]
            width = 1.5 if branch == root else 0.65
            if branch == highlight_branch:
                color, width = colors["highlight"], 2.3
            axes[0].plot(xx, yy, color=color, lw=width, alpha=0.92)
        if gap_rows:
            axes[0].scatter(
                [row["x_m"] / 1000.0 for row in gap_rows],
                [row["y_m"] / 1000.0 for row in gap_rows],
                marker="X", s=75, color=colors["gap"], edgecolor=colors["bg"], linewidth=0.9, zorder=6,
                label="内部断冰节点",
            )
        axes[0].set_title(f"系统{system_id}：二维冰厚与全部流线", color=colors["fg"], weight="medium")
        if gap_rows:
            axes[0].legend(loc="lower left", frameon=False, labelcolor=colors["fg"], fontsize=8)

        focus = highlight_branch if highlight_branch is not None else root
        focus_rows = by_branch[focus]
        fx = np.array([row["x_m"] for row in focus_rows]) / 1000.0
        fy = np.array([row["y_m"] for row in focus_rows]) / 1000.0
        axes[1].pcolormesh(x, y, shown, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        axes[1].contour(x, y, mask.astype(float), levels=[0.5], colors=[colors["fg"]], linewidths=0.5, alpha=0.75)
        axes[1].plot(fx, fy, color=colors["highlight"], lw=3.0, marker="o", ms=3.5)
        focus_gaps = [row for row in focus_rows if row["internal_gap_node"]]
        if focus_gaps:
            axes[1].scatter(
                [row["x_m"] / 1000.0 for row in focus_gaps],
                [row["y_m"] / 1000.0 for row in focus_gaps],
                marker="X", s=110, color=colors["gap"], edgecolor=colors["bg"], linewidth=1.1, zorder=6,
            )
        pad = max(0.35, 0.4 * max(float(np.ptp(fx)), float(np.ptp(fy)), 0.5))
        axes[1].set_xlim(float(np.min(fx) - pad), float(np.max(fx) + pad))
        axes[1].set_ylim(float(np.min(fy) - pad), float(np.max(fy) + pad))
        axes[1].set_title(f"支流{focus}局部与连通组件（{state_label}）", color=colors["fg"], weight="medium")

        cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.055, pad=0.10, ticks=bounds)
        cbar.set_label("权威原始二维冰厚 H (m)", color=colors["fg"])
        cbar.ax.tick_params(colors=colors["fg"], labelsize=8)
        cbar.outline.set_edgecolor(colors["flow"])
        fig.savefig(output_root / f"system_{system_id}_thickness_connectivity_{theme}.png", dpi=150, facecolor=colors["bg"], bbox_inches="tight")
        plt.close(fig)

    fields = list(rows[0])
    with (output_root / "flowline_nodes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "lia-thickness-connectivity-2d-v1",
        "status": "plot-complete-no-model-run",
        "system_id": system_id,
        "state_label": state_label,
        "root_branch": root,
        "highlight_branch": highlight_branch,
        "internal_gap_node_count": len(gap_rows),
        "internal_gap_branches": sorted({row["branch"] for row in gap_rows}),
        "paired_grid": str(netcdf.resolve()),
        "state_pickle": str(state.resolve()),
        "grid_json": str(grid_json.resolve()),
        "active_thickness_tolerance_m": tolerance,
        "outputs": {
            "light_png": f"system_{system_id}_thickness_connectivity_light.png",
            "dark_png": f"system_{system_id}_thickness_connectivity_dark.png",
            "nodes_csv": "flowline_nodes.csv",
        },
        "note": "The raster is the paired V3/F19 source. Flowline overlays are the selected saved model state and are not a second raster reconstruction.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", type=int, required=True)
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--grid-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-label", default="saved-state")
    parser.add_argument("--highlight-branch", type=int)
    parser.add_argument("--active-thickness-tolerance-m", type=float, default=1e-6)
    args = parser.parse_args()
    print(json.dumps(plot(
        args.system_id, args.netcdf, args.state, args.grid_json, args.output_root,
        args.state_label, args.highlight_branch, args.active_thickness_tolerance_m,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
