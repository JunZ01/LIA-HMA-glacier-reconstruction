"""Build mean geometry and volume-change-equivalent mass-balance curves for retained-32."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
RETAINED = RESULTS / "lia_191_accepted_58_curve_shape_screen_v1/retained_32_systems.csv"
REGISTRY = RESULTS / "lia_191_accepted_58_registry_v1/accepted_58_curve_metrics.csv"
INDEX = RESULTS / "lia_191_accepted55_evidence_index_v1/accepted55_evidence_index.csv"
FROZEN = RESULTS / "lia_191_frozen_curve_registry_v1"
OUTPUT = RESULTS / "lia_191_retained_32_mean_geometry_mass_balance_v1"

SOURCE_YEARS = np.arange(1499, 2001, dtype=int)
OUTPUT_YEARS = np.arange(1500, 2001, dtype=int)
ICE_DENSITY_KG_M3 = 900.0
MB_SMOOTHING_YEARS = 21


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


def interpolate_source(source: Path, preferred_length_key: str) -> dict[str, np.ndarray]:
    rows = read_csv(source)
    if "phase" in rows[0]:
        rows = [row for row in rows if row.get("phase") == "forward"]

    points: dict[int, tuple[float, float, float]] = {}
    for row in rows:
        year = int(round(float(row["year"])))
        if 1470 <= year <= 2000:
            length_raw = row.get(preferred_length_key) or row.get("length_m")
            points[year] = (
                float(length_raw) / 1000.0,
                float(row["area_m2"]) / 1.0e6,
                float(row["volume_m3"]) / 1.0e9,
            )

    years = np.array(sorted(points), dtype=float)
    if years.size < 2 or years[0] > SOURCE_YEARS[0] or years[-1] < SOURCE_YEARS[-1]:
        raise RuntimeError(f"source does not cover 1499-2000: {source}")
    values = np.array([points[int(year)] for year in years], dtype=float)
    return {
        "length_km": np.interp(SOURCE_YEARS, years, values[:, 0]),
        "area_km2": np.interp(SOURCE_YEARS, years, values[:, 1]),
        "volume_km3": np.interp(SOURCE_YEARS, years, values[:, 2]),
    }


def ensemble_statistics(matrix: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"mean_{prefix}": np.mean(matrix, axis=0),
        f"median_{prefix}": np.median(matrix, axis=0),
        f"q25_{prefix}": np.quantile(matrix, 0.25, axis=0),
        f"q75_{prefix}": np.quantile(matrix, 0.75, axis=0),
        f"std_{prefix}": np.std(matrix, axis=0, ddof=1),
    }


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#dddddd", linewidth=0.65, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def plot_absolute(stats: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    geometry = [
        (axes[0, 0], "length_km", "A. Mean length", "Length (km)", "#287271"),
        (axes[0, 1], "area_km2", "B. Mean area", "Area (km2)", "#d77a61"),
        (axes[1, 0], "volume_km3", "C. Mean volume", "Volume (km3)", "#3f5f8f"),
    ]
    for axis, key, title, ylabel, color in geometry:
        axis.fill_between(
            stats["year"],
            stats[f"q25_{key}"],
            stats[f"q75_{key}"],
            color=color,
            alpha=0.20,
            label="Interquartile range",
        )
        axis.plot(stats["year"], stats[f"mean_{key}"], color=color, linewidth=2.2, label="Mean")
        axis.plot(
            stats["year"],
            stats[f"median_{key}"],
            color="#303030",
            linewidth=1.1,
            linestyle="--",
            label="Median",
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=8, ncol=3, loc="best")
        style_axis(axis)

    mb_axis = axes[1, 1]
    mb_axis.axhline(0, color="#555555", linewidth=0.8)
    mb_axis.plot(
        stats["year"],
        stats["mean_net_mb_kg_m2_yr"],
        color="#c65d3a",
        linewidth=0.7,
        alpha=0.35,
        label="Equal-glacier mean (annual)",
    )
    mb_axis.plot(
        stats["year"],
        stats["mean_net_mb_21yr_kg_m2_yr"],
        color="#c65d3a",
        linewidth=2.0,
        label="Equal-glacier mean (21-year)",
    )
    mb_axis.plot(
        stats["year"],
        stats["area_weighted_net_mb_21yr_kg_m2_yr"],
        color="#3f5f8f",
        linewidth=1.8,
        label="Area-weighted (21-year)",
    )
    mb_axis.set_title("D. Volume-change-equivalent net mass balance")
    mb_axis.set_ylabel("Net balance (kg m-2 yr-1)")
    mb_axis.legend(frameon=False, fontsize=8, loc="best")
    style_axis(mb_axis)

    for axis in axes[-1, :]:
        axis.set_xlabel("Year")
    for axis in axes.ravel():
        axis.set_xlim(1500, 2000)
    fig.suptitle(
        "Retained-32 ensemble geometry and net mass-change curves",
        fontsize=16,
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUTPUT / "retained_32_mean_length_area_volume_mass_balance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_normalized(stats: pd.DataFrame) -> Path:
    fig, axis = plt.subplots(figsize=(13.2, 6.2))
    series = [
        ("mean_length_relative_1500", "Length", "#287271"),
        ("mean_area_relative_1500", "Area", "#d77a61"),
        ("mean_volume_relative_1500", "Volume", "#3f5f8f"),
    ]
    for key, label, color in series:
        axis.plot(stats["year"], 100.0 * (stats[key] - 1.0), color=color, linewidth=2.1, label=label)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xlim(1500, 2000)
    axis.set_xlabel("Year")
    axis.set_ylabel("Mean change relative to each glacier's 1500 value (%)")
    axis.set_title("Retained-32 equal-glacier mean normalized geometry")
    axis.legend(frameon=False, ncol=3, loc="best")
    style_axis(axis)
    fig.tight_layout()
    path = OUTPUT / "retained_32_normalized_geometry_change.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    retained_ids = set(pd.read_csv(RETAINED)["system_id"].astype(int))
    metrics = {int(row["system_id"]): row for row in read_csv(REGISTRY)}
    evidence = {int(row["system_id"]): row for row in read_csv(INDEX)}
    if len(retained_ids) != 32:
        raise RuntimeError("retained list is not exactly 32 unique systems")

    records: list[dict[str, object]] = []
    source_manifest: list[dict[str, object]] = []
    for system_id in sorted(retained_ids):
        metric = metrics[system_id]
        source, length_key = resolve_source(metric, evidence)
        if not source.is_file():
            raise FileNotFoundError(source)
        arrays = interpolate_source(source, length_key)

        area_m2 = arrays["area_km2"] * 1.0e6
        volume_m3 = arrays["volume_km3"] * 1.0e9
        mean_area_m2 = 0.5 * (area_m2[1:] + area_m2[:-1])
        delta_volume_m3 = volume_m3[1:] - volume_m3[:-1]
        net_mb = ICE_DENSITY_KG_M3 * delta_volume_m3 / mean_area_m2

        source_manifest.append(
            {
                "system_id": system_id,
                "rgi_id": metric["rgi_id"],
                "route": metric["route"],
                "source": str(source.relative_to(ROOT)),
                "direct_surface_mass_balance_available": False,
            }
        )
        for offset, year in enumerate(OUTPUT_YEARS, start=1):
            records.append(
                {
                    "system_id": system_id,
                    "rgi_id": metric["rgi_id"],
                    "route": metric["route"],
                    "year": int(year),
                    "length_km": arrays["length_km"][offset],
                    "area_km2": arrays["area_km2"][offset],
                    "volume_km3": arrays["volume_km3"][offset],
                    "net_volume_change_equivalent_mb_kg_m2_yr": net_mb[offset - 1],
                    "source": str(source.relative_to(ROOT)),
                }
            )

    long_table = pd.DataFrame(records)
    if len(long_table) != 32 * len(OUTPUT_YEARS):
        raise RuntimeError("unexpected long-table row count")
    if not np.isfinite(
        long_table[
            ["length_km", "area_km2", "volume_km3", "net_volume_change_equivalent_mb_kg_m2_yr"]
        ].to_numpy()
    ).all():
        raise RuntimeError("non-finite geometry or mass balance")
    if (long_table[["length_km", "area_km2", "volume_km3"]] <= 0).any().any():
        raise RuntimeError("non-positive geometry in retained curves")

    first = long_table[long_table["year"] == 1500].set_index("system_id")
    for key in ["length_km", "area_km2", "volume_km3"]:
        long_table[f"{key}_relative_1500"] = long_table.apply(
            lambda row, name=key: row[name] / first.loc[int(row["system_id"]), name], axis=1
        )

    annual_rows: list[dict[str, float | int]] = []
    for year, group in long_table.groupby("year", sort=True):
        row: dict[str, float | int] = {"year": int(year), "system_count": len(group)}
        for key in ["length_km", "area_km2", "volume_km3"]:
            row.update(ensemble_statistics(group[key].to_numpy(float), key))
            row[f"mean_{key.removesuffix('_km').removesuffix('2').removesuffix('3')}_relative_1500"] = float(
                group[f"{key}_relative_1500"].mean()
            )
        mb = group["net_volume_change_equivalent_mb_kg_m2_yr"].to_numpy(float)
        row["mean_net_mb_kg_m2_yr"] = float(np.mean(mb))
        row["median_net_mb_kg_m2_yr"] = float(np.median(mb))
        row["q25_net_mb_kg_m2_yr"] = float(np.quantile(mb, 0.25))
        row["q75_net_mb_kg_m2_yr"] = float(np.quantile(mb, 0.75))
        area_m2 = group["area_km2"].to_numpy(float) * 1.0e6
        row["area_weighted_net_mb_kg_m2_yr"] = float(np.average(mb, weights=area_m2))
        annual_rows.append(row)
    stats = pd.DataFrame(annual_rows)

    expected_relative_columns = {
        "mean_length_relative_1500",
        "mean_area_km_relative_1500",
        "mean_volume_km_relative_1500",
    }
    rename = {
        "mean_area_km_relative_1500": "mean_area_relative_1500",
        "mean_volume_km_relative_1500": "mean_volume_relative_1500",
    }
    if not expected_relative_columns.issubset(stats.columns):
        raise RuntimeError("normalized ensemble columns were not built as expected")
    stats = stats.rename(columns=rename)
    stats["mean_net_mb_21yr_kg_m2_yr"] = stats["mean_net_mb_kg_m2_yr"].rolling(
        MB_SMOOTHING_YEARS, center=True, min_periods=1
    ).mean()
    stats["area_weighted_net_mb_21yr_kg_m2_yr"] = stats[
        "area_weighted_net_mb_kg_m2_yr"
    ].rolling(MB_SMOOTHING_YEARS, center=True, min_periods=1).mean()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    long_path = OUTPUT / "retained_32_annual_geometry_and_net_mass_balance.csv"
    stats_path = OUTPUT / "retained_32_annual_ensemble_statistics.csv"
    manifest_path = OUTPUT / "source_manifest.csv"
    long_table.to_csv(long_path, index=False)
    stats.to_csv(stats_path, index=False)
    pd.DataFrame(source_manifest).to_csv(manifest_path, index=False)
    absolute_plot = plot_absolute(stats)
    normalized_plot = plot_normalized(stats)

    start = stats.iloc[0]
    end = stats.iloc[-1]
    summary = {
        "status": "retained-32-mean-curves-complete",
        "system_count": 32,
        "year_start": 1500,
        "year_end": 2000,
        "ice_density_kg_m3": ICE_DENSITY_KG_M3,
        "mass_balance_definition": (
            "annual net volume-change-equivalent specific mass balance: "
            "rho_ice * (V_t - V_t-1) / mean(A_t, A_t-1)"
        ),
        "direct_climatic_surface_mass_balance_available": False,
        "mass_balance_warning": (
            "This is a net/geodetic-equivalent balance derived from annual geometry, "
            "not a directly saved climatic surface mass-balance diagnostic."
        ),
        "mean_1500": {
            "length_km": float(start["mean_length_km"]),
            "area_km2": float(start["mean_area_km2"]),
            "volume_km3": float(start["mean_volume_km3"]),
        },
        "mean_2000": {
            "length_km": float(end["mean_length_km"]),
            "area_km2": float(end["mean_area_km2"]),
            "volume_km3": float(end["mean_volume_km3"]),
        },
        "equal_glacier_mean_relative_change_1500_2000_percent": {
            "length": float(100.0 * (end["mean_length_relative_1500"] - 1.0)),
            "area": float(100.0 * (end["mean_area_relative_1500"] - 1.0)),
            "volume": float(100.0 * (end["mean_volume_relative_1500"] - 1.0)),
        },
        "mean_net_mb_1500_2000_kg_m2_yr": float(stats["mean_net_mb_kg_m2_yr"].mean()),
        "area_weighted_net_mb_1500_2000_kg_m2_yr": float(
            stats["area_weighted_net_mb_kg_m2_yr"].mean()
        ),
        "formal_accepted_count_unchanged": 58,
        "registry_or_controller_modified": False,
        "outputs": {
            "absolute_plot": absolute_plot.name,
            "normalized_plot": normalized_plot.name,
            "annual_ensemble_statistics": stats_path.name,
            "annual_glacier_table": long_path.name,
            "source_manifest": manifest_path.name,
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
