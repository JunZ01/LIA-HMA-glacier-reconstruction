"""Read-only comparison of two saved OGGM flowline branches.

This utility never advances an OGGM model.  It reads a common Core flowline
and the saved ``spinup_state.pkl.gz`` / ``state_2000.pkl.gz`` files from two
completed chains, verifies the fixed geometry contract, and writes reusable
node-, distance-band-, and elevation-band diagnostics plus separate figures.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_FILES = {
    "spinup": "spinup_state.pkl.gz",
    "year2000": "state_2000.pkl.gz",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--core-flowline", type=Path, required=True)
    parser.add_argument("--branch-a-dir", type=Path, required=True)
    parser.add_argument("--branch-b-dir", type=Path, required=True)
    parser.add_argument("--branch-a-label", default="branch_a")
    parser.add_argument("--branch-b-label", default="branch_b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-limit-km", type=float, default=25.0)
    parser.add_argument("--elevation-bin-m", type=float, default=100.0)
    return parser.parse_args()


def _load_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        value = pickle.load(stream)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one flowline in {path}, found {len(value)}")
        return value[0]
    return value


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_").lower()


def _state_arrays(flowline, dx_meter: float) -> dict[str, np.ndarray]:
    thick = np.asarray(flowline.thick, dtype=float)
    bed_h = np.asarray(flowline.bed_h, dtype=float)
    surface_h = np.asarray(flowline.surface_h, dtype=float)
    widths_m = np.asarray(flowline.widths_m, dtype=float)
    section_m2 = np.asarray(flowline.section, dtype=float)
    ice = thick > 0
    return {
        "thickness_m": thick,
        "surface_h_m": surface_h,
        "width_m": widths_m,
        "section_m2": section_m2,
        "node_area_m2": np.where(ice, widths_m * dx_meter, 0.0),
        "node_volume_m3": section_m2 * dx_meter,
        "ice": ice,
        "bed_h_m": bed_h,
    }


def _last_ice(flowline) -> tuple[int | None, float]:
    indices = np.flatnonzero(np.asarray(flowline.thick, dtype=float) > 0)
    if not indices.size:
        return None, 0.0
    return int(indices[-1]), float(flowline.length_m)


def _flowline_metrics(flowline) -> dict:
    last_index, length_m = _last_ice(flowline)
    thick = np.asarray(flowline.thick, dtype=float)
    return {
        "flowline_type": type(flowline).__name__,
        "nx": int(flowline.nx),
        "dx": float(flowline.dx),
        "map_dx_m": float(flowline.map_dx),
        "dx_meter": float(flowline.dx * flowline.map_dx),
        "length_m": length_m,
        "area_m2": float(flowline.area_m2),
        "volume_m3": float(flowline.volume_m3),
        "mean_thickness_m": float(flowline.volume_m3 / flowline.area_m2)
        if flowline.area_m2 > 0
        else math.nan,
        "last_ice_index": last_index,
        "last_ice_node_distance_m": float(last_index * flowline.dx * flowline.map_dx)
        if last_index is not None
        else math.nan,
        "max_thickness_m": float(np.nanmax(thick)),
        "length_gt_0p1m_m": float(np.count_nonzero(thick > 0.1) * flowline.dx_meter),
        "length_gt_1m_m": float(np.count_nonzero(thick > 1.0) * flowline.dx_meter),
        "length_gt_5m_m": float(np.count_nonzero(thick > 5.0) * flowline.dx_meter),
        "length_gt_10m_m": float(np.count_nonzero(thick > 10.0) * flowline.dx_meter),
        "length_gt_20m_m": float(np.count_nonzero(thick > 20.0) * flowline.dx_meter),
    }


def _buffer_metrics(flowline, core_ice_node_count: int) -> dict:
    dx_meter = float(flowline.dx_meter)
    thick = np.asarray(flowline.thick, dtype=float)
    widths_m = np.asarray(flowline.widths_m, dtype=float)
    section_m2 = np.asarray(flowline.section, dtype=float)
    buffer = np.arange(flowline.nx) >= core_ice_node_count
    buffer_ice = buffer & (thick > 0)
    return {
        "buffer_ice_length_m": float(np.count_nonzero(buffer_ice) * dx_meter),
        "buffer_area_m2": float(np.sum(widths_m[buffer_ice]) * dx_meter),
        "buffer_volume_m3": float(np.sum(section_m2[buffer]) * dx_meter),
        "buffer_max_thickness_m": float(np.max(thick[buffer], initial=0.0)),
        "core_area_m2": float(np.sum(widths_m[(~buffer) & (thick > 0)]) * dx_meter),
        "core_volume_m3": float(np.sum(section_m2[~buffer]) * dx_meter),
    }


def _assert_common_geometry(core, states: dict[str, object]) -> dict:
    bed = np.asarray(core.bed_h, dtype=float)
    width = np.asarray(core.widths_m, dtype=float)
    checks = {}
    for name, flowline in states.items():
        checks[name] = {
            "same_nx": bool(flowline.nx == core.nx),
            "same_dx": bool(np.isclose(flowline.dx, core.dx, rtol=0, atol=0)),
            "same_map_dx": bool(np.isclose(flowline.map_dx, core.map_dx, rtol=0, atol=0)),
            "bed_exact": bool(np.array_equal(np.asarray(flowline.bed_h), bed)),
            "width_exact": bool(np.array_equal(np.asarray(flowline.widths_m), width)),
            "section_rectangular_equivalent": bool(
                np.allclose(
                    np.asarray(flowline.section, dtype=float),
                    np.asarray(flowline.widths_m, dtype=float)
                    * np.asarray(flowline.thick, dtype=float),
                    rtol=1e-12,
                    atol=1e-8,
                )
            ),
            "lambda_zero": bool(
                hasattr(flowline, "_lambdas")
                and np.all(np.asarray(flowline._lambdas, dtype=float) == 0)
            ),
        }
    failures = [
        f"{name}:{key}"
        for name, values in checks.items()
        for key, passed in values.items()
        if key not in {"lambda_zero"} and not passed
    ]
    if failures:
        raise ValueError("Fixed geometry contract failed: " + ", ".join(failures))
    return checks


def _build_node_table(core, states: dict[str, object]) -> pd.DataFrame:
    dx_meter = float(core.dx * core.map_dx)
    data: dict[str, np.ndarray] = {
        "node_index": np.arange(core.nx, dtype=int),
        "distance_m": np.arange(core.nx, dtype=float) * dx_meter,
        "distance_km": np.arange(core.nx, dtype=float) * dx_meter / 1000.0,
        "bed_h_m": np.asarray(core.bed_h, dtype=float),
        "fixed_width_m": np.asarray(core.widths_m, dtype=float),
    }
    for name, flowline in states.items():
        arrays = _state_arrays(flowline, dx_meter)
        for key in (
            "thickness_m",
            "surface_h_m",
            "section_m2",
            "node_area_m2",
            "node_volume_m3",
        ):
            data[f"{name}_{key}"] = arrays[key]
        data[f"{name}_ice"] = arrays["ice"]
    return pd.DataFrame(data)


def _band_table(
    nodes: pd.DataFrame,
    states: list[str],
    branch_a: str,
    branch_b: str,
    *,
    dimension: str,
    edges: np.ndarray,
) -> pd.DataFrame:
    values = nodes[dimension].to_numpy(dtype=float)
    band_index = np.digitize(values, edges, right=False) - 1
    rows = []
    for index in range(len(edges) - 1):
        select = band_index == index
        if not np.any(select):
            continue
        row = {
            "band_lower": float(edges[index]),
            "band_upper": float(edges[index + 1]),
            "node_count": int(np.sum(select)),
        }
        for state in states:
            row[f"{state}_area_m2"] = float(nodes.loc[select, f"{state}_node_area_m2"].sum())
            row[f"{state}_volume_m3"] = float(
                nodes.loc[select, f"{state}_node_volume_m3"].sum()
            )
        for phase in ("spinup", "year2000"):
            a = f"{branch_a}_{phase}"
            b = f"{branch_b}_{phase}"
            row[f"{phase}_area_a_minus_b_m2"] = row[f"{a}_area_m2"] - row[f"{b}_area_m2"]
            row[f"{phase}_volume_a_minus_b_m3"] = (
                row[f"{a}_volume_m3"] - row[f"{b}_volume_m3"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_surface(
    nodes: pd.DataFrame,
    branch_a: str,
    branch_b: str,
    label_a: str,
    label_b: str,
    phase: str,
    output: Path,
    distance_limit_km: float,
    system_id: str,
    core_end_km: float,
) -> None:
    select = nodes["distance_km"] <= distance_limit_km
    x = nodes.loc[select, "distance_km"]
    bed = nodes.loc[select, "bed_h_m"]
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.plot(x, bed, color="0.18", linewidth=1.3, label="Fixed bed")
    for name, label, color in ((branch_a, label_a, "#1f77b4"), (branch_b, label_b, "#d95f02")):
        thick = nodes.loc[select, f"{name}_{phase}_thickness_m"].to_numpy()
        surface = nodes.loc[select, f"{name}_{phase}_surface_h_m"].to_numpy()
        surface = np.where(thick > 0, surface, np.nan)
        ax.plot(x, surface, color=color, linewidth=2.0, label=f"{label} surface")
    ax.axvline(core_end_km, color="0.35", linestyle=":", linewidth=1.2, label="Core/buffer boundary")
    ax.set_title(f"System {system_id} {phase}: fixed bed and glacier surface")
    ax.set_xlabel("Distance along elevation-band flowline (km)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_thickness(
    nodes: pd.DataFrame,
    branch_a: str,
    branch_b: str,
    label_a: str,
    label_b: str,
    phase: str,
    output: Path,
    distance_limit_km: float,
    system_id: str,
    core_end_km: float,
) -> None:
    select = nodes["distance_km"] <= distance_limit_km
    x = nodes.loc[select, "distance_km"]
    a = nodes.loc[select, f"{branch_a}_{phase}_thickness_m"]
    b = nodes.loc[select, f"{branch_b}_{phase}_thickness_m"]
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.plot(x, a, color="#1f77b4", linewidth=2.0, label=label_a)
    ax.plot(x, b, color="#d95f02", linewidth=2.0, label=label_b)
    ax.fill_between(x, b, a, color="#7b3294", alpha=0.12, label="Branch difference")
    ax.axvline(core_end_km, color="0.35", linestyle=":", linewidth=1.2, label="Core/buffer boundary")
    ax.set_title(f"System {system_id} {phase}: ice-thickness profile")
    ax.set_xlabel("Distance along elevation-band flowline (km)")
    ax.set_ylabel("Ice thickness (m)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_cumulative(
    nodes: pd.DataFrame,
    branch_a: str,
    branch_b: str,
    label_a: str,
    label_b: str,
    measure: str,
    divisor: float,
    unit: str,
    output: Path,
    distance_limit_km: float,
    system_id: str,
    core_end_km: float,
) -> None:
    select = nodes["distance_km"] <= distance_limit_km
    x = nodes.loc[select, "distance_km"].to_numpy()
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    styles = {"spinup": "--", "year2000": "-"}
    for name, label, color in ((branch_a, label_a, "#1f77b4"), (branch_b, label_b, "#d95f02")):
        for phase in ("spinup", "year2000"):
            values = nodes.loc[select, f"{name}_{phase}_node_{measure}"].to_numpy()
            ax.plot(
                x,
                np.cumsum(values) / divisor,
                color=color,
                linestyle=styles[phase],
                linewidth=2.0,
                label=f"{label}, {phase}",
            )
    ax.axvline(core_end_km, color="0.35", linestyle=":", linewidth=1.2, label="Core/buffer boundary")
    ax.set_title(f"System {system_id}: cumulative {measure.replace('_', ' ')} along flowline")
    ax.set_xlabel("Distance along elevation-band flowline (km)")
    ax.set_ylabel(f"Cumulative {measure.replace('_', ' ')} ({unit})")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_bottleneck(
    nodes: pd.DataFrame,
    branch_a: str,
    branch_b: str,
    label_a: str,
    label_b: str,
    output: Path,
    distance_limit_km: float,
    system_id: str,
    core_end_km: float,
) -> None:
    select = nodes["distance_km"] <= distance_limit_km
    x = nodes.loc[select, "distance_km"]
    width = nodes.loc[select, "fixed_width_m"]
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.plot(x, width, color="0.18", linewidth=2.0, label="Fixed width")
    for name, label, color in ((branch_a, label_a, "#1f77b4"), (branch_b, label_b, "#d95f02")):
        for phase, marker in (("spinup", "o"), ("year2000", "s")):
            ice = nodes.loc[select, f"{name}_{phase}_ice"].to_numpy(dtype=bool)
            if np.any(ice):
                last = np.flatnonzero(ice)[-1]
                ax.scatter(
                    [x.iloc[last]],
                    [width.iloc[last]],
                    s=55,
                    marker=marker,
                    color=color,
                    label=f"{label} {phase} terminus",
                    zorder=4,
                )
    ax.set_yscale("log")
    ax.axvline(core_end_km, color="0.35", linestyle=":", linewidth=1.2, label="Core/buffer boundary")
    ax.set_title(f"System {system_id}: fixed-width contraction and branch termini")
    ax.set_xlabel("Distance along elevation-band flowline (km)")
    ax.set_ylabel("Fixed flowline width (m, log scale)")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_amplification(
    nodes: pd.DataFrame,
    branch_a: str,
    branch_b: str,
    label_a: str,
    label_b: str,
    output: Path,
    distance_limit_km: float,
    system_id: str,
    core_end_km: float,
) -> None:
    select = nodes["distance_km"] <= distance_limit_km
    x = nodes.loc[select, "distance_km"]
    spin_diff = (
        nodes.loc[select, f"{branch_a}_spinup_thickness_m"]
        - nodes.loc[select, f"{branch_b}_spinup_thickness_m"]
    )
    final_diff = (
        nodes.loc[select, f"{branch_a}_year2000_thickness_m"]
        - nodes.loc[select, f"{branch_b}_year2000_thickness_m"]
    )
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.plot(x, spin_diff, color="#1b9e77", linewidth=2.0, label="A - B at spin-up end")
    ax.plot(x, final_diff, color="#7b3294", linewidth=2.0, label="A - B in 2000")
    ax.axhline(0, color="0.35", linewidth=0.9)
    ax.axvline(core_end_km, color="0.35", linestyle=":", linewidth=1.2, label="Core/buffer boundary")
    ax.set_title(f"System {system_id}: amplification of {label_a} minus {label_b}")
    ax.set_xlabel("Distance along elevation-band flowline (km)")
    ax.set_ylabel("Ice-thickness difference (m)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core = _load_flowline(args.core_flowline)
    branch_a = _safe_label(args.branch_a_label)
    branch_b = _safe_label(args.branch_b_label)
    if branch_a == branch_b:
        raise ValueError("Branch labels must be distinct")

    flowlines = {"core": core}
    for branch_name, branch_dir in ((branch_a, args.branch_a_dir), (branch_b, args.branch_b_dir)):
        for phase, filename in STATE_FILES.items():
            flowlines[f"{branch_name}_{phase}"] = _load_flowline(branch_dir / filename)

    state_flowlines = {key: value for key, value in flowlines.items() if key != "core"}
    geometry_checks = _assert_common_geometry(core, state_flowlines)
    nodes = _build_node_table(core, flowlines)
    nodes.to_csv(args.output_dir / "node_profiles.csv", index=False)

    state_names = list(flowlines)
    max_distance_m = float(nodes["distance_m"].max())
    distance_edges = np.arange(0.0, max_distance_m + 1000.0, 1000.0)
    distance_bands = _band_table(
        nodes,
        state_names,
        branch_a,
        branch_b,
        dimension="distance_m",
        edges=distance_edges,
    )
    distance_bands.rename(
        columns={"band_lower": "distance_lower_m", "band_upper": "distance_upper_m"}
    ).to_csv(args.output_dir / "distance_bands_1km.csv", index=False)

    union_ice = np.zeros(core.nx, dtype=bool)
    for name in state_names:
        union_ice |= nodes[f"{name}_ice"].to_numpy(dtype=bool)
    relevant_bed = nodes.loc[union_ice, "bed_h_m"].to_numpy(dtype=float)
    step = float(args.elevation_bin_m)
    elevation_edges = np.arange(
        math.floor(relevant_bed.min() / step) * step,
        math.ceil(relevant_bed.max() / step) * step + step,
        step,
    )
    elevation_bands = _band_table(
        nodes.loc[union_ice].reset_index(drop=True),
        state_names,
        branch_a,
        branch_b,
        dimension="bed_h_m",
        edges=elevation_edges,
    )
    elevation_bands.rename(
        columns={"band_lower": "bed_elevation_lower_m", "band_upper": "bed_elevation_upper_m"}
    ).to_csv(args.output_dir / "bed_elevation_bands.csv", index=False)

    core_ice_node_count = int(np.count_nonzero(np.asarray(core.thick, dtype=float) > 0))
    dx_meter = float(core.dx_meter)
    buffer_contract = {
        "core_ice_node_count": core_ice_node_count,
        "core_target_domain_length_m": float(core_ice_node_count * dx_meter),
        "total_domain_node_count": int(core.nx),
        "total_domain_length_m": float(core.nx * dx_meter),
        "buffer_node_count": int(core.nx - core_ice_node_count),
        "buffer_length_m": float((core.nx - core_ice_node_count) * dx_meter),
        "buffer_fixed_width_m": float(np.asarray(core.widths_m)[core_ice_node_count]),
    }
    summaries = {
        name: {
            **_flowline_metrics(flowline),
            **_buffer_metrics(flowline, core_ice_node_count),
        }
        for name, flowline in flowlines.items()
    }
    comparison_rows = []
    for phase in ("spinup", "year2000"):
        a = f"{branch_a}_{phase}"
        b = f"{branch_b}_{phase}"
        thick_diff = nodes[f"{a}_thickness_m"] - nodes[f"{b}_thickness_m"]
        max_index = int(np.nanargmax(np.abs(thick_diff.to_numpy())))
        comparison_rows.append(
            {
                "phase": phase,
                "branch_a_length_m": summaries[a]["length_m"],
                "branch_b_length_m": summaries[b]["length_m"],
                "length_a_minus_b_m": summaries[a]["length_m"] - summaries[b]["length_m"],
                "branch_a_area_m2": summaries[a]["area_m2"],
                "branch_b_area_m2": summaries[b]["area_m2"],
                "area_a_minus_b_m2": summaries[a]["area_m2"] - summaries[b]["area_m2"],
                "branch_a_volume_m3": summaries[a]["volume_m3"],
                "branch_b_volume_m3": summaries[b]["volume_m3"],
                "volume_a_minus_b_m3": summaries[a]["volume_m3"] - summaries[b]["volume_m3"],
                "max_abs_thickness_difference_m": float(abs(thick_diff.iloc[max_index])),
                "max_abs_thickness_difference_distance_m": float(nodes.iloc[max_index]["distance_m"]),
                "max_abs_thickness_difference_bed_h_m": float(nodes.iloc[max_index]["bed_h_m"]),
                "fixed_width_at_max_difference_m": float(nodes.iloc[max_index]["fixed_width_m"]),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(args.output_dir / "branch_comparison.csv", index=False)

    spin_gap = float(comparisons.loc[comparisons["phase"] == "spinup", "length_a_minus_b_m"].iloc[0])
    final_gap = float(
        comparisons.loc[comparisons["phase"] == "year2000", "length_a_minus_b_m"].iloc[0]
    )
    spin_volume_gap = float(
        comparisons.loc[comparisons["phase"] == "spinup", "volume_a_minus_b_m3"].iloc[0]
    )
    final_volume_gap = float(
        comparisons.loc[comparisons["phase"] == "year2000", "volume_a_minus_b_m3"].iloc[0]
    )

    summary = {
        "system_id": str(args.system_id),
        "mode": "read_only_saved_state_diagnostic",
        "core_flowline": str(args.core_flowline.resolve()),
        "branch_a": {"label": args.branch_a_label, "directory": str(args.branch_a_dir.resolve())},
        "branch_b": {"label": args.branch_b_label, "directory": str(args.branch_b_dir.resolve())},
        "geometry_checks": geometry_checks,
        "buffer_contract": buffer_contract,
        "state_metrics": summaries,
        "branch_gap_amplification": {
            "spinup_length_gap_m": spin_gap,
            "year2000_length_gap_m": final_gap,
            "length_gap_ratio_2000_to_spinup": final_gap / spin_gap if spin_gap else math.nan,
            "spinup_volume_gap_m3": spin_volume_gap,
            "year2000_volume_gap_m3": final_volume_gap,
            "volume_gap_ratio_2000_to_spinup": final_volume_gap / spin_volume_gap
            if spin_volume_gap
            else math.nan,
        },
        "notes": [
            "No OGGM model was advanced by this diagnostic.",
            "RectangularBedFlowline to TrapezoidalBedFlowline with lambda=0 is section-equivalent.",
            "The saved branches share exactly the same bed and fixed width at every node.",
        ],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    pd.DataFrame(
        [{"state": name, **metrics} for name, metrics in summaries.items()]
    ).to_csv(args.output_dir / "state_metrics.csv", index=False)
    core_end_km = buffer_contract["core_target_domain_length_m"] / 1000.0

    _plot_surface(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        "spinup",
        args.output_dir / "01_spinup_surface_profile.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )
    _plot_surface(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        "year2000",
        args.output_dir / "02_year2000_surface_profile.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )
    _plot_thickness(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        "spinup",
        args.output_dir / "03_spinup_thickness_profile.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )
    _plot_thickness(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        "year2000",
        args.output_dir / "04_year2000_thickness_profile.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )
    _plot_cumulative(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        "area_m2",
        1e6,
        "km²",
        args.output_dir / "05_cumulative_area_by_distance.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )
    _plot_cumulative(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        "volume_m3",
        1e9,
        "km³",
        args.output_dir / "06_cumulative_volume_by_distance.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )
    _plot_bottleneck(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        args.output_dir / "07_fixed_width_contraction.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )
    _plot_amplification(
        nodes,
        branch_a,
        branch_b,
        args.branch_a_label,
        args.branch_b_label,
        args.output_dir / "08_branch_amplification.png",
        args.distance_limit_km,
        str(args.system_id),
        core_end_km,
    )

    print(json.dumps(summary["branch_gap_amplification"], indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
