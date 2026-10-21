"""Plot the complete authoritative set of 58 accepted LIA length curves."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
REGISTRY = RESULTS / "lia_191_accepted_58_registry_v1/accepted_58_curve_metrics.csv"
INDEX = RESULTS / "lia_191_accepted55_evidence_index_v1/accepted55_evidence_index.csv"
FROZEN = RESULTS / "lia_191_frozen_curve_registry_v1"
OUTPUT = RESULTS / "lia_191_accepted_58_curves_visualization_v1"

YEARS = np.arange(1500, 2001, dtype=int)
ROUTE_LABELS = {
    "frozen-legacy": "Frozen legacy (22)",
    "synthetic-system": "Synthetic system (18)",
    "formal-historical-reproduction-pass": "Formal reproduction (18)",
}
ROUTE_COLORS = {
    "frozen-legacy": "#287271",
    "synthetic-system": "#d77a61",
    "formal-historical-reproduction-pass": "#3f5f8f",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_source(
    metric: dict[str, str], evidence: dict[int, dict[str, str]]
) -> tuple[Path, str]:
    system_id = int(metric["system_id"])
    if metric["route"] == "frozen-legacy":
        return FROZEN / f"system_{system_id}" / "timeseries.csv", "observation_length_m"

    reproduction_root = metric.get("reproduction_root", "").strip()
    if not reproduction_root:
        reproduction_root = evidence[system_id]["reproduction_root"].strip()
    return ROOT / reproduction_root / "historical_1470_2000.csv", "length_m"


def read_curve(source: Path, length_key: str) -> np.ndarray:
    rows = read_csv(source)
    if length_key == "observation_length_m":
        rows = [row for row in rows if row.get("phase") == "forward"]

    points: dict[int, float] = {}
    for row in rows:
        year = int(round(float(row["year"])))
        if 1470 <= year <= 2000:
            raw = row.get(length_key) or row.get("length_m")
            points[year] = float(raw) / 1000.0

    source_years = np.array(sorted(points), dtype=float)
    if source_years.size < 2 or source_years[0] > 1500 or source_years[-1] < 2000:
        raise RuntimeError(f"curve does not cover 1500-2000: {source}")
    source_values = np.array([points[int(year)] for year in source_years], dtype=float)
    return np.interp(YEARS, source_years, source_values)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def plot_overview(curves: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.2), sharex=True)

    for route in ROUTE_LABELS:
        selected = [curve for curve in curves if curve["route"] == route]
        for index, curve in enumerate(selected):
            axes[0].plot(
                YEARS,
                curve["length_km"],
                color=ROUTE_COLORS[route],
                linewidth=0.9,
                alpha=0.5,
                label=ROUTE_LABELS[route] if index == 0 else None,
            )
    axes[0].set_ylabel("Glacier length (km)")
    axes[0].set_title("A. Absolute glacier length trajectories")
    axes[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    style_axis(axes[0])

    normalized = np.vstack(
        [curve["length_km"] / float(curve["lia_km"]) for curve in curves]
    )
    for curve, values in zip(curves, normalized, strict=True):
        axes[1].plot(
            YEARS,
            values,
            color=ROUTE_COLORS[str(curve["route"])],
            linewidth=0.75,
            alpha=0.25,
        )
    q25, median, q75 = np.quantile(normalized, [0.25, 0.5, 0.75], axis=0)
    axes[1].fill_between(
        YEARS, q25, q75, color="#c6a15b", alpha=0.28, label="Interquartile range"
    )
    axes[1].plot(YEARS, median, color="#202020", linewidth=2.2, label="Median")
    axes[1].axhline(1.0, color="#8a3b3b", linewidth=1.1, linestyle="--", label="LIA target")
    axes[1].set_ylabel("Length / LIA target")
    axes[1].set_xlabel("Year")
    axes[1].set_title("B. Normalized trajectories and accepted-set distribution")
    axes[1].legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    axes[1].set_xlim(1500, 2000)
    style_axis(axes[1])

    fig.suptitle(
        "All 58 formally accepted LIA glacier-length curves (1500-2000)",
        fontsize=16,
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = OUTPUT / "accepted_58_length_curves_overview.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_grid(curves: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(8, 8, figsize=(20, 21), sharex=True)
    flat_axes = axes.ravel()
    for axis, curve in zip(flat_axes, curves, strict=False):
        route = str(curve["route"])
        axis.plot(
            YEARS,
            curve["length_km"],
            color=ROUTE_COLORS[route],
            linewidth=1.35,
        )
        axis.axhline(float(curve["lia_km"]), color="#8a3b3b", linewidth=0.75, linestyle="--")
        axis.axhline(float(curve["rgi_km"]), color="#333333", linewidth=0.75, linestyle=":")
        axis.set_title(
            f"System {curve['system_id']} | {curve['rgi_id']}", fontsize=8, pad=3
        )
        axis.set_xlim(1500, 2000)
        axis.grid(axis="y", color="#e2e2e2", linewidth=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=6, length=2)
    for axis in flat_axes[len(curves) :]:
        axis.axis("off")

    fig.suptitle(
        "Individual accepted curves | dashed: LIA target | dotted: RGI target",
        fontsize=16,
        fontweight="semibold",
        y=0.997,
    )
    fig.supxlabel("Year", fontsize=12)
    fig.supylabel("Glacier length (km)", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98), h_pad=1.0, w_pad=0.8)
    path = OUTPUT / "accepted_58_length_curves_grid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_long_csv(curves: list[dict[str, object]]) -> Path:
    path = OUTPUT / "accepted_58_annual_length_curves.csv"
    fields = [
        "system_id",
        "rgi_id",
        "route",
        "year",
        "length_km",
        "length_over_lia_target",
        "lia_target_km",
        "rgi_target_km",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for curve in curves:
            lia_km = float(curve["lia_km"])
            for year, value in zip(YEARS, curve["length_km"], strict=True):
                writer.writerow(
                    {
                        "system_id": curve["system_id"],
                        "rgi_id": curve["rgi_id"],
                        "route": curve["route"],
                        "year": int(year),
                        "length_km": f"{float(value):.6f}",
                        "length_over_lia_target": f"{float(value) / lia_km:.8f}",
                        "lia_target_km": f"{lia_km:.6f}",
                        "rgi_target_km": f"{float(curve['rgi_km']):.6f}",
                        "source": curve["source"],
                    }
                )
    return path


def main() -> None:
    metrics = read_csv(REGISTRY)
    index = {int(row["system_id"]): row for row in read_csv(INDEX)}
    if len(metrics) != 58 or len({row["system_id"] for row in metrics}) != 58:
        raise RuntimeError("authoritative accepted registry is not exactly 58 unique systems")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, object]] = []
    for metric in sorted(metrics, key=lambda row: int(row["system_id"])):
        source, length_key = resolve_source(metric, index)
        if not source.is_file():
            raise FileNotFoundError(source)
        curves.append(
            {
                "system_id": int(metric["system_id"]),
                "rgi_id": metric["rgi_id"],
                "route": metric["route"],
                "lia_km": float(metric["lia_km"]),
                "rgi_km": float(metric["rgi_km"]),
                "length_km": read_curve(source, length_key),
                "source": str(source.relative_to(ROOT)),
            }
        )

    overview = plot_overview(curves)
    grid = plot_grid(curves)
    curve_csv = write_long_csv(curves)
    route_counts = Counter(str(curve["route"]) for curve in curves)
    normalized = np.vstack(
        [curve["length_km"] / float(curve["lia_km"]) for curve in curves]
    )
    summary = {
        "status": "accepted-58-curves-visualized",
        "accepted_system_count": len(curves),
        "year_start": int(YEARS[0]),
        "year_end": int(YEARS[-1]),
        "annual_points_per_curve": len(YEARS),
        "total_curve_points": len(curves) * len(YEARS),
        "route_counts": dict(sorted(route_counts.items())),
        "all_source_files_exist": True,
        "median_length_over_lia_target_1500": float(np.median(normalized[:, 0])),
        "median_length_over_lia_target_2000": float(np.median(normalized[:, -1])),
        "registry": str(REGISTRY.relative_to(ROOT)),
        "outputs": {
            "overview_png": str(overview.relative_to(ROOT)),
            "grid_png": str(grid.relative_to(ROOT)),
            "annual_curve_csv": str(curve_csv.relative_to(ROOT)),
        },
    }
    summary_path = OUTPUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
