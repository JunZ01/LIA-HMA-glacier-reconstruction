"""Plot three evidence-only geometry diagnostics; never run an OGGM model."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


CASES = {
    84: {
        "geometry_root": "w404geom84v1",
        "regularization_root": "w405reggeom84_batch26_v1",
        "gdir_rel": "synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084",
        "expected_junction_reverse": 10,
        "role": "maximum-adjustment failure",
    },
    1856: {
        "geometry_root": "w447geom1856v1",
        "regularization_root": "w449reggeom1856_batch31_v1",
        "gdir_rel": "synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91856",
        "expected_junction_reverse": 2,
        "role": "adjustment + hypsometry failure",
    },
    1931: {
        "geometry_root": "w400geom1931v1",
        "regularization_root": "w403reggeom1931_batch25_v1",
        "gdir_rel": "synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91931",
        "expected_junction_reverse": 3,
        "role": "hypsometry-CDF failure",
    },
}
REFS = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_controller_v1/core191_length_references.csv"
)
OUTPUT = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_problem_visualization_v1/diagnostic_figures/geometry"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _grid_to_map(coords: np.ndarray, grid: dict[str, Any]) -> np.ndarray:
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    out = np.asarray(coords, dtype=float).copy()
    out[:, 0] = x0 + out[:, 0] * dx
    out[:, 1] = y0 + out[:, 1] * dy
    return out


def _select_reg_row(summary: dict[str, Any], system_id: int) -> dict[str, Any]:
    rows = [row for row in summary["systems"] if int(row["system_id"]) == system_id]
    _require(len(rows) == 1, f"regularization-row-count:{system_id}:{len(rows)}")
    return rows[0]


def _plot_case(root: Path, output: Path, refs: pd.DataFrame, system_id: int, spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    geometry_root = root / str(spec["geometry_root"])
    regularization_root = root / str(spec["regularization_root"])
    gdir = geometry_root / str(spec["gdir_rel"])
    outline_path = gdir / "outlines.shp"
    grid_path = gdir / "glacier_grid.json"
    flowline_path = gdir / "inversion_flowlines.pkl"
    geometry_summary_path = geometry_root / "geometry_summary.json"
    regularization_summary_path = regularization_root / "summary.json"
    nodes_path = regularization_root / "node_adjustments.csv"
    for path in (
        outline_path,
        grid_path,
        flowline_path,
        geometry_summary_path,
        regularization_summary_path,
        nodes_path,
    ):
        _require(path.is_file(), f"required-evidence-missing:{path}")

    outline = gpd.read_file(outline_path)
    grid = _read_json(grid_path)
    with gzip.open(flowline_path, "rb") as handle:
        flowlines = pickle.load(handle)
    geometry_summary = _read_json(geometry_summary_path)
    reg_summary = _read_json(regularization_summary_path)
    reg_row = _select_reg_row(reg_summary, system_id)
    nodes = pd.read_csv(nodes_path)
    nodes = nodes.loc[nodes["system_id"].astype(int) == system_id].copy()
    ref = refs.loc[system_id]
    root_indexes = [i for i, flowline in enumerate(flowlines) if flowline.flows_to is None]
    _require(root_indexes == [int(geometry_summary["root_index"])], f"root-index-conflict:{system_id}")
    root_index = root_indexes[0]
    by_identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    grid_dx = abs(float(grid["dxdy"][0]))

    spatial_lines: list[np.ndarray] = []
    junctions: list[dict[str, Any]] = []
    for branch_index, flowline in enumerate(flowlines):
        coords = _grid_to_map(np.asarray(flowline.line.coords, dtype=float), grid)
        spatial_lines.append(coords)
        branch_nodes = nodes.loc[nodes["branch_index"].astype(int) == branch_index].sort_values("node_index")
        _require(len(branch_nodes) == len(flowline.surface_h), f"node-count-conflict:{system_id}:{branch_index}")
        _require(
            np.max(np.abs(branch_nodes["surface_before_m"].to_numpy(float) - np.asarray(flowline.surface_h, dtype=float))) < 1e-8,
            f"surface-source-conflict:{system_id}:{branch_index}",
        )
        if flowline.flows_to is not None:
            receiver_index = int(flowline.flows_to_indice)
            receiver_branch = by_identity[id(flowline.flows_to)]
            receiver_surface = float(flowline.flows_to.surface_h[receiver_index])
            tributary_surface = float(flowline.surface_h[-1])
            point_grid = np.asarray(flowline.flows_to.line.coords[receiver_index], dtype=float).reshape(1, 2)
            point_map = _grid_to_map(point_grid, grid)[0]
            junctions.append({
                "branch_index": branch_index,
                "receiver_branch": receiver_branch,
                "x": float(point_map[0]),
                "y": float(point_map[1]),
                "rise_m": receiver_surface - tributary_surface,
                "reverse": receiver_surface > tributary_surface + 1e-7,
            })
    reverse_junctions = [item for item in junctions if item["reverse"]]
    _require(len(reverse_junctions) == int(spec["expected_junction_reverse"]), f"junction-count-conflict:{system_id}")
    _require(len(reverse_junctions) == int(geometry_summary["junction_reverse_slope_count"]), f"summary-junction-count-conflict:{system_id}")

    fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(15.2, 7.6), constrained_layout=True)
    fig.suptitle(
        f"System {system_id}  |  {ref['rgi_id']}  |  {spec['role']}",
        fontsize=16,
        fontweight="semibold",
    )

    # Spatial panel: true outline, every native flowline and exact receiver junctions.
    outline.plot(ax=ax_map, facecolor="#dcecf4", edgecolor="#2f4858", linewidth=1.3, zorder=1)
    reverse_branches = {int(item["branch_index"]) for item in reverse_junctions}
    for branch_index, coords in enumerate(spatial_lines):
        if branch_index == root_index:
            color, width, zorder = "#172a3a", 2.8, 4
        elif branch_index in reverse_branches:
            color, width, zorder = "#d97706", 2.1, 4
        else:
            color, width, zorder = "#6b8796", 1.15, 3
        ax_map.plot(coords[:, 0], coords[:, 1], color=color, linewidth=width, zorder=zorder)
    for item in junctions:
        if item["reverse"]:
            ax_map.scatter(item["x"], item["y"], marker="X", s=75, color="#c81e1e", edgecolor="white", linewidth=0.7, zorder=7)
            ax_map.annotate(
                f"b{item['branch_index']} +{item['rise_m']:.1f} m",
                (item["x"], item["y"]), xytext=(5, 5), textcoords="offset points",
                fontsize=7.5, color="#8b1111", zorder=8,
            )
        else:
            ax_map.scatter(item["x"], item["y"], marker="o", s=15, facecolor="white", edgecolor="#6b8796", linewidth=0.7, zorder=5)
    xmin, ymin, xmax, ymax = outline.total_bounds
    padx, pady = max((xmax - xmin) * 0.06, 300.0), max((ymax - ymin) * 0.06, 300.0)
    ax_map.set_xlim(xmin - padx, xmax + padx)
    ax_map.set_ylim(ymin - pady, ymax + pady)
    ax_map.set_aspect("equal")
    ax_map.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{(value - xmin) / 1000:.1f}"))
    ax_map.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{(value - ymin) / 1000:.1f}"))
    ax_map.set_xlabel("Relative easting (km)")
    ax_map.set_ylabel("Relative northing (km)")
    ax_map.set_title(f"LIA outline + all {len(flowlines)} native flowlines")
    ax_map.grid(color="#d0d7dc", linewidth=0.5, alpha=0.55)
    map_handles = [
        Line2D([0], [0], color="#172a3a", lw=2.8, label=f"Root branch b{root_index}"),
        Line2D([0], [0], color="#6b8796", lw=1.2, label="Other native branches"),
        Line2D([0], [0], color="#d97706", lw=2.1, label="Branch entering reverse junction"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#c81e1e", markeredgecolor="white", markersize=8, label="Formal reverse junction"),
    ]
    ax_map.legend(handles=map_handles, loc="lower left", frameon=True, framealpha=0.94, fontsize=8)

    # Profile panel: native and deterministic regularized surface on every branch.
    max_adjustment = 0.0
    for branch_index, flowline in enumerate(flowlines):
        branch_nodes = nodes.loc[nodes["branch_index"].astype(int) == branch_index].sort_values("node_index")
        x_km = branch_nodes["node_index"].to_numpy(float) * grid_dx * float(geometry_summary.get("flowline_dx", 2.0)) / 1000.0
        # Geometry configs fix flowline_dx=2; the inventory's physical spacing is 50 m.
        if len(x_km) > 1:
            observed_dx = float(spatial_lines[branch_index].shape[0] > 1) * 0.0
        x_km = branch_nodes["node_index"].to_numpy(float) * 50.0 / 1000.0
        native = branch_nodes["surface_before_m"].to_numpy(float)
        regularized = branch_nodes["surface_after_m"].to_numpy(float)
        adjustments = branch_nodes["adjustment_m"].to_numpy(float)
        max_adjustment = max(max_adjustment, float(np.max(np.abs(adjustments))))
        is_root = branch_index == root_index
        is_reverse_branch = branch_index in reverse_branches
        native_color = "#172a3a" if is_root else ("#d97706" if is_reverse_branch else "#a5b4bc")
        native_width = 2.5 if is_root else (1.45 if is_reverse_branch else 0.75)
        native_alpha = 1.0 if (is_root or is_reverse_branch) else 0.62
        ax_profile.plot(x_km, native, color=native_color, linewidth=native_width, alpha=native_alpha, zorder=3 if is_root else 2)
        ax_profile.plot(x_km, regularized, color="#147d92", linewidth=1.5 if is_root else 0.75, linestyle="--", alpha=0.95 if is_root else 0.55, zorder=4)
        changed = np.abs(adjustments) > 1e-8
        if np.any(changed):
            ax_profile.scatter(x_km[changed], regularized[changed], s=9, color="#c81e1e", alpha=0.68, zorder=5)
    _require(abs(max_adjustment - float(reg_row["maximum_adjustment_m"])) < 1e-8, f"max-adjustment-conflict:{system_id}")
    ax_profile.set_xlabel("Distance from branch head (km)")
    ax_profile.set_ylabel("Surface elevation (m a.s.l.)")
    ax_profile.set_title("Native vs deterministic regularized surface profiles")
    ax_profile.grid(color="#d0d7dc", linewidth=0.5, alpha=0.55)
    profile_handles = [
        Line2D([0], [0], color="#172a3a", lw=2.5, label="Native root surface"),
        Line2D([0], [0], color="#d97706", lw=1.5, label="Native problem tributary"),
        Line2D([0], [0], color="#147d92", lw=1.5, ls="--", label="Regularized surface"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c81e1e", markersize=5, label="Adjusted node"),
    ]
    ax_profile.legend(handles=profile_handles, loc="upper right", frameon=True, framealpha=0.94, fontsize=8)
    diagnostic_text = (
        f"Native junction reversals: {len(reverse_junctions)}\n"
        f"Maximum surface adjustment: {float(reg_row['maximum_adjustment_m']):.2f} m\n"
        f"100 m hypsometry CDF error: {100 * float(reg_row['cdf_error_100m']):.3f}%\n"
        f"Root-length error: {100 * float(geometry_summary['root_length_relative_error']):.3f}%\n"
        "BED UNAVAILABLE — pre-inversion geometry contains surface only"
    )
    ax_profile.text(
        0.02, 0.02, diagnostic_text, transform=ax_profile.transAxes,
        ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff7ed", "edgecolor": "#c2410c", "alpha": 0.96},
    )
    fig.text(
        0.5, 0.005,
        "Red X marks the exact formal receiver-above-tributary junction test. Bed/thickness is not inferred because inversion was never run.",
        ha="center", va="bottom", fontsize=9, color="#4b5563",
    )
    png_path = output / f"geometry_diagnostic_{system_id}.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, {
        "system_id": system_id,
        "rgi_id": str(ref["rgi_id"]),
        "png": str(png_path.relative_to(root)),
        "root_branch_index": root_index,
        "branch_count": len(flowlines),
        "junction_reverse_count": len(reverse_junctions),
        "junction_reverse_rises_m": [float(item["rise_m"]) for item in reverse_junctions],
        "maximum_surface_adjustment_m": float(reg_row["maximum_adjustment_m"]),
        "hypsometry_cdf_error_100m": float(reg_row["cdf_error_100m"]),
        "bed_available": False,
        "source_hashes": {
            str(path.relative_to(root)): _sha(path)
            for path in (
                outline_path, grid_path, flowline_path, geometry_summary_path,
                regularization_summary_path, nodes_path,
            )
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    output = root / OUTPUT
    _require(not output.exists(), f"refusing-to-overwrite:{output}")
    refs_path = root / REFS
    refs = pd.read_csv(refs_path).set_index("lia_unique_id")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for system_id, spec in CASES.items():
        png_path, record = _plot_case(root, output, refs, system_id, spec)
        _require(png_path.is_file() and png_path.stat().st_size > 100_000, f"png-missing-or-small:{system_id}")
        record["png_sha256"] = _sha(png_path)
        records.append(record)
    manifest = {
        "status": "three-geometry-diagnostics-rendered",
        "figure_count": 3,
        "figures": records,
        "bed_note": "Unavailable and not inferred because all selected systems stopped before inversion.",
        "oggm_model_run": False,
        "facts_modified": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(output), "figures": [item["png"] for item in records]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
