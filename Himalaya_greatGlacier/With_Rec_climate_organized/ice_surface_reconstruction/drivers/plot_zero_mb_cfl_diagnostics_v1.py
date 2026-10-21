"""Plot read-only CFL diagnostics from saved inversion and zero-MB evidence.

This driver never instantiates or advances an OGGM model.  OGGM classes are
imported only as a pickle dependency while reading saved ``model_flowlines``.
"""

from __future__ import annotations

import gzip
import json
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_problem_visualization_v1/diagnostic_figures/zero_mb"
)
PROBLEM_CSV = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_problem_visualization_v1/zero_mb"
    / "zero_mb_problem_diagnostics_plot_ready.csv"
)

SYSTEMS = {
    2177: {
        "role": "representative (median required/configured ratio)",
        "zero_root": "w643zero_conservative_batch35_39_2177_v1",
        "inversion_root": "w621inv2177v2",
    },
    2729: {
        "role": "most severe CFL case",
        "zero_root": "w302zero2729v1",
        "inversion_root": "w301inv2729v1",
    },
    2611: {
        "role": "near-threshold CFL case",
        "zero_root": "w643zero_conservative_batch35_39_2611_v1",
        "inversion_root": "w582inv2611v1",
    },
}

COLORS = {
    "bed": "#4d4d4d",
    "surface": "#277da1",
    "ice": "#a8dadc",
    "section": "#6a4c93",
    "network": "#8d99ae",
    "branch": "#f8961e",
    "cfl": "#d62828",
    "terminus": "#111111",
    "threshold": "#d62828",
    "accepted": "#277da1",
    "creation": "#f8961e",
    "prevented": "#43aa8b",
    "unresolved": "#d62828",
    "closure": "#6a4c93",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path(root: Path) -> Path:
    formal = root / "zero_mb_summary_formal.json"
    return formal if formal.exists() else root / "zero_mb_summary.json"


def _read_flowlines(inversion_root: Path):
    candidates = list(inversion_root.rglob("model_flowlines.pkl"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one model_flowlines.pkl under {inversion_root}")
    with gzip.open(candidates[0], "rb") as handle:
        return pickle.load(handle), candidates[0]


def _parse_cfl(summary: dict) -> dict:
    error = summary.get("runtime_error") or ""
    pattern = re.compile(
        r"required time step smaller than the minimum allowed:\s*"
        r"(?P<required>[0-9.]+)s\s*vs\s*(?P<configured>[0-9.]+)s\.\s*"
        r"Happening at simulation year\s*(?P<year>[0-9.]+),\s*"
        r"fl_id\s*(?P<fl>[0-9]+),\s*bin_id\s*(?P<bin>[0-9]+)\s*"
        r"and max_u\s*(?P<max_u>[0-9.]+)"
    )
    match = pattern.search(error)
    if not match:
        raise RuntimeError(f"could not parse CFL error: {error}")
    out = {key: float(value) for key, value in match.groupdict().items()}
    out["fl"] = int(out["fl"])
    out["bin"] = int(out["bin"])
    return out


def _coords_km(flowlines) -> tuple[float, float]:
    coords = np.concatenate([np.asarray(fl.line.coords, dtype=float) for fl in flowlines])
    map_dx = float(flowlines[0].map_dx)
    return float(coords[:, 0].min() * map_dx), float(coords[:, 1].min() * map_dx)


def _plot_network(ax, flowlines, cfl: dict) -> tuple[float, float]:
    x0, y0 = _coords_km(flowlines)
    cfl_xy = None
    for i, fl in enumerate(flowlines):
        coords = np.asarray(fl.line.coords, dtype=float)
        map_dx = float(fl.map_dx)
        x = (coords[:, 0] * map_dx - x0) / 1000.0
        y = (coords[:, 1] * map_dx - y0) / 1000.0
        thickness = np.asarray(fl.thick, dtype=float)
        positive = thickness[thickness > 0]
        scale = float(np.nanmedian(positive)) if positive.size else 0.0
        width = 0.7 + min(scale / 80.0, 2.5)
        color = COLORS["branch"] if i == cfl["fl"] else COLORS["network"]
        zorder = 3 if i == cfl["fl"] else 1
        ax.plot(x, y, color=color, linewidth=width, alpha=0.92, zorder=zorder)

        terminus = int(np.clip(fl.terminus_index, 0, len(x) - 1))
        ax.scatter(
            x[terminus], y[terminus], marker="|", s=45,
            color=COLORS["terminus"], linewidths=1.3, zorder=4,
        )
        if fl.flows_to is None:
            ax.scatter(
                x[-1], y[-1], marker="s", s=28, facecolors="none",
                edgecolors=COLORS["terminus"], linewidths=1.0, zorder=4,
            )
        if i == cfl["fl"]:
            idx = int(np.clip(cfl["bin"], 0, len(x) - 1))
            cfl_xy = (x[idx], y[idx])
            ax.scatter(
                x[idx], y[idx], marker="X", s=92, color=COLORS["cfl"],
                edgecolors="white", linewidths=0.8, zorder=6,
            )

    if cfl_xy is None:
        raise RuntimeError("CFL branch not found")
    ax.annotate(
        f"CFL fl {cfl['fl']}, bin {cfl['bin']}", cfl_xy,
        xytext=(8, 8), textcoords="offset points", fontsize=8,
        color=COLORS["cfl"],
    )
    ax.set_title("Flowline network and CFL location", fontsize=11)
    ax.set_xlabel("relative projected x (km)")
    ax.set_ylabel("relative projected y (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.18, linewidth=0.5)
    return cfl_xy


def _plot_profile(ax, flowlines, cfl: dict) -> dict:
    fl = flowlines[cfl["fl"]]
    n = int(fl.nx)
    idx = int(np.clip(cfl["bin"], 0, n - 1))
    x = np.arange(n, dtype=float) * float(fl.dx_meter) / 1000.0
    bed = np.asarray(fl.bed_h, dtype=float)
    surface = np.asarray(fl.surface_h, dtype=float)
    thick = np.asarray(fl.thick, dtype=float)
    section = np.asarray(fl.section, dtype=float)
    if idx < n - 1:
        slope = (surface[idx] - surface[idx + 1]) / float(fl.dx_meter)
    elif idx > 0:
        slope = (surface[idx - 1] - surface[idx]) / float(fl.dx_meter)
    else:
        slope = np.nan

    ax.plot(x, bed, color=COLORS["bed"], linewidth=1.2, label="bed")
    ax.plot(x, surface, color=COLORS["surface"], linewidth=1.5, label="surface")
    ax.fill_between(x, bed, surface, where=thick > 0, color=COLORS["ice"], alpha=0.55)
    ax.axvline(x[idx], color=COLORS["cfl"], linewidth=1.2, linestyle="--")
    ax.scatter([x[idx]], [surface[idx]], marker="X", s=70, color=COLORS["cfl"], zorder=5)
    ax.set_title("CFL-branch longitudinal profile", fontsize=11)
    ax.set_xlabel("distance along branch (km)")
    ax.set_ylabel("elevation (m)")
    ax.grid(alpha=0.18, linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(x, section / 1000.0, color=COLORS["section"], linewidth=1.0, alpha=0.8)
    ax2.set_ylabel("section (10³ m²)", color=COLORS["section"])
    ax2.tick_params(axis="y", colors=COLORS["section"])
    suffix = " (reported bin lies at/after branch end)" if cfl["bin"] >= n else ""
    ax.text(
        0.02, 0.03,
        f"local downslope={100*slope:.2f}%  |  thickness={thick[idx]:.1f} m  |  "
        f"section={section[idx]:.0f} m²{suffix}",
        transform=ax.transAxes, fontsize=8, va="bottom",
    )
    return {
        "flowline_index": cfl["fl"],
        "reported_bin_index": cfl["bin"],
        "plotted_bin_index": idx,
        "node_count": n,
        "local_downslope_percent": float(100 * slope),
        "thickness_m": float(thick[idx]),
        "section_m2": float(section[idx]),
        "surface_elevation_m": float(surface[idx]),
        "bed_elevation_m": float(bed[idx]),
        "is_root": fl.flows_to is None,
    }


def _plot_timestep(ax, steps: pd.DataFrame, checkpoints: pd.DataFrame, cfl: dict) -> None:
    if len(steps):
        x = steps["end_year"].to_numpy(float)
        y = steps["dt_seconds"].to_numpy(float)
        ax.plot(x, y, color=COLORS["accepted"], linewidth=0.9, alpha=0.9, label="accepted model step")
    else:
        ax.text(0.5, 0.55, "No accepted step was recorded", ha="center", va="center", transform=ax.transAxes)
    ax.scatter([cfl["year"]], [cfl["required"]], marker="X", s=78, color=COLORS["cfl"], zorder=5, label="reported failing requirement")
    ax.axhline(cfl["configured"], color=COLORS["threshold"], linestyle="--", linewidth=1.0, label="configured minimum")
    for year in checkpoints.get("year", pd.Series(dtype=float)).astype(float):
        if year > 0:
            ax.axvline(year, color=COLORS["terminus"], alpha=0.18, linewidth=0.7)
    ax.set_yscale("log")
    ax.set_title("Recorded adaptive timesteps", fontsize=11)
    ax.set_xlabel("simulation year")
    ax.set_ylabel("step duration (s, log scale)")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="best", fontsize=7, frameon=False)


def _plot_budget(ax, steps: pd.DataFrame) -> dict:
    if not len(steps):
        ax.text(
            0.5, 0.52,
            "No accepted step → no recorded flux-budget time series",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title("Recorded flux budget", fontsize=11)
        ax.set_xlabel("simulation year")
        ax.set_ylabel("cumulative volume (m³)")
        return {
            "accepted_step_count": 0,
            "cumulative_negative_tributary_creation_m3": 0.0,
            "cumulative_prevented_creation_m3": 0.0,
            "cumulative_unresolved_creation_m3": 0.0,
            "cumulative_closure_residual_m3": 0.0,
        }

    x = steps["end_year"].to_numpy(float)
    neg = np.cumsum(steps["negative_tributary_creation_m3"].to_numpy(float))
    prevented = np.cumsum(steps["prevented_negative_tributary_creation_m3"].to_numpy(float))
    unresolved = np.cumsum(steps["unresolved_negative_tributary_creation_m3"].to_numpy(float))
    closure = np.cumsum(steps["closure_residual_m3"].to_numpy(float))
    ax.plot(x, neg, color=COLORS["creation"], linewidth=1.2, label="negative tributary creation")
    ax.plot(x, prevented, color=COLORS["prevented"], linewidth=1.0, linestyle="--", label="prevented creation")
    ax.plot(x, unresolved, color=COLORS["unresolved"], linewidth=1.0, label="unresolved creation")
    ax.plot(x, closure, color=COLORS["closure"], linewidth=1.0, label="closure residual")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_title("Recorded cumulative flux budget", fontsize=11)
    ax.set_xlabel("simulation year")
    ax.set_ylabel("cumulative volume (m³, symlog)")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="best", fontsize=7, frameon=False)
    return {
        "accepted_step_count": int(len(steps)),
        "cumulative_negative_tributary_creation_m3": float(neg[-1]),
        "cumulative_prevented_creation_m3": float(prevented[-1]),
        "cumulative_unresolved_creation_m3": float(unresolved[-1]),
        "cumulative_closure_residual_m3": float(closure[-1]),
    }


def plot_system(system_id: int, config: dict) -> dict:
    zero_root = PROJECT / config["zero_root"]
    inversion_root = PROJECT / config["inversion_root"]
    summary_path = _summary_path(zero_root)
    summary = _read_json(summary_path)
    cfl = _parse_cfl(summary)
    steps_path = zero_root / "zero_mb_steps.csv"
    checkpoints_path = zero_root / "zero_mb_checkpoints.csv"
    try:
        steps = pd.read_csv(steps_path)
    except pd.errors.EmptyDataError:
        steps = pd.DataFrame()
    try:
        checkpoints = pd.read_csv(checkpoints_path)
    except pd.errors.EmptyDataError:
        checkpoints = pd.DataFrame()
    flowlines, flowline_path = _read_flowlines(inversion_root)

    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_network = fig.add_subplot(grid[0, 0])
    ax_profile = fig.add_subplot(grid[0, 1])
    ax_timestep = fig.add_subplot(grid[1, 0])
    ax_budget = fig.add_subplot(grid[1, 1])

    _plot_network(ax_network, flowlines, cfl)
    profile = _plot_profile(ax_profile, flowlines, cfl)
    _plot_timestep(ax_timestep, steps, checkpoints, cfl)
    budget = _plot_budget(ax_budget, steps)
    fig.suptitle(
        f"System {system_id} — {config['role']}\n"
        f"reported CFL requirement {cfl['required']:.1f} s < configured {cfl['configured']:.1f} s; "
        f"failure year {cfl['year']:g}, max velocity {cfl['max_u']:,.0f} m yr⁻¹",
        fontsize=14,
    )
    output_path = OUTPUT / f"system_{system_id}_cfl_diagnostic.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "system_id": system_id,
        "role": config["role"],
        "required_timestep_s": cfl["required"],
        "configured_min_timestep_s": cfl["configured"],
        "failure_year_reported": cfl["year"],
        "max_velocity_m_per_year_reported": cfl["max_u"],
        "profile": profile,
        "budget": budget,
        "zero_mb_summary": str(summary_path.relative_to(PROJECT)),
        "flowline_pickle": str(flowline_path.relative_to(PROJECT)),
        "output_png": str(output_path.relative_to(PROJECT)),
    }


def plot_legacy_rescore_pending() -> dict:
    frame = pd.read_csv(PROBLEM_CSV)
    frame = frame.loc[frame["formal_rescore_needed"].astype(bool)].copy()
    frame["ratio"] = (
        frame["maximum_absolute_step_closure_residual_m3"]
        / frame["per_step_closure_limit_m3"]
    )
    frame = frame.sort_values("lia_unique_id")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(frame))
    ax.scatter(x, frame["ratio"], s=62, color=COLORS["closure"], zorder=3)
    ax.axhline(1.0, color=COLORS["threshold"], linestyle="--", linewidth=1.2, label="formal per-step threshold")
    for xi, row in zip(x, frame.itertuples(index=False)):
        ax.annotate(
            f"{int(row.lia_unique_id)}\n{row.maximum_absolute_step_closure_residual_m3:.2e} / "
            f"{row.per_step_closure_limit_m3:.2e}",
            (xi, row.ratio), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=8,
        )
    ax.set_xticks(x, [str(int(v)) for v in frame["lia_unique_id"]])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("LIA system ID")
    ax.set_ylabel("read-only max-step residual / formal limit")
    ax.set_title("Legacy cumulative-closure labels — formal rescore pending", fontsize=13)
    ax.text(
        0.01, 0.03,
        "These four rows are neither reclassified as formal pass nor retained as formal closure failure here.",
        transform=ax.transAxes, fontsize=8,
    )
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    output_path = OUTPUT / "legacy_closure_rescore_pending.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "system_ids": [int(v) for v in frame["lia_unique_id"]],
        "status": "formal-rescore-pending-not-reclassified",
        "output_png": str(output_path.relative_to(PROJECT)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    system_results = [plot_system(system_id, config) for system_id, config in SYSTEMS.items()]
    pending = plot_legacy_rescore_pending()
    summary = {
        "status": "read-only-diagnostic-figures-complete",
        "oggm_model_run": False,
        "facts_modified": False,
        "scoring_modified": False,
        "systems": system_results,
        "legacy_closure": pending,
    }
    (OUTPUT / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
