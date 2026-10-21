"""Plot the audited annual source and support-exchange curves."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
DECOMP_ROOT = BASE / "lia_191_system2005_masked_internal_source_elevation_feedback_v3"
EXCHANGE_ROOT = BASE / "lia_191_system2005_masked_support_exchange_v1"
OUT = BASE / "lia_191_system2005_masked_internal_source_feedback_plot_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(stream)]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    annual_path = DECOMP_ROOT / "annual_source_decomposition.csv"
    exchange_path = EXCHANGE_ROOT / "annual_support_exchange.csv"
    annual = read_csv(annual_path)
    exchange = read_csv(exchange_path)
    if len(annual) != 50 or len(exchange) != 50:
        raise RuntimeError("annual-row-count-not-50")
    years = np.asarray([row["year"] for row in annual])
    million = 1e6
    actual = np.asarray([row["actual_source_m3"] for row in annual])
    initial = np.asarray([row["initial_geometry_source_m3"] for row in annual])
    support = np.asarray([row["active_support_effect_m3"] for row in annual])
    surface = np.asarray([row["surface_elevation_effect_m3"] for row in annual])
    interaction = np.asarray([row["support_surface_interaction_m3"] for row in annual])
    within = np.asarray([row["within_year_geometry_effect_m3"] for row in annual])
    frozen = initial + support + surface + interaction
    added = np.asarray([row["added_support_effect_m3"] for row in exchange])
    removed = np.asarray([row["removed_support_effect_m3"] for row in exchange])
    volume = 100.0 * np.asarray([row["end_volume_change_fraction"] for row in annual])
    area = np.asarray([row["start_active_area_m2"] for row in annual]) / million

    OUT.mkdir(parents=False)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(years, actual / million, label="Actual source", linewidth=2)
    axes[0, 0].plot(years, initial / million, label="Initial geometry", linewidth=1.5)
    axes[0, 0].plot(years, frozen / million, label="Annual-start geometry", linewidth=1.5)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("Annual source (million m3)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(years, np.cumsum(initial) / million, label="Initial climate")
    axes[0, 1].plot(years, np.cumsum(support) / million, label="Support exchange")
    axes[0, 1].plot(years, np.cumsum(surface) / million, label="Surface elevation")
    axes[0, 1].plot(years, np.cumsum(interaction + within) / million, label="Interaction + within-year")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("Cumulative contribution (million m3)")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(years, volume, color="#b33b32", label="Volume")
    axes[1, 0].axhline(2, color="black", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(-2, color="black", linestyle="--", linewidth=0.8)
    axes[1, 0].set_ylabel("Volume change (%)")
    area_axis = axes[1, 0].twinx()
    area_axis.plot(years, area, color="#2d6a4f", label="Active area")
    area_axis.set_ylabel("Active area (km2)")

    axes[1, 1].plot(years, added / million, label="Added high support", color="#4c78a8")
    axes[1, 1].plot(years, removed / million, label="Removed low support", color="#e45756")
    axes[1, 1].plot(years, (added + removed) / million, label="Total support effect", color="black", linewidth=1.8)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("Annual support effect (million m3)")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel("Model year")
        axis.grid(alpha=0.2)
    figure.savefig(OUT / "system2005_masked_internal_source_feedback.png", dpi=180)
    plt.close(figure)

    summary = {
        "schema": "lia-191-system2005-masked-internal-source-feedback-plot-v1",
        "status": "plot-complete",
        "annual_rows": len(annual),
        "final_volume_change_percent": float(volume[-1]),
        "cumulative_actual_source_m3": float(np.sum(actual)),
        "cumulative_support_effect_m3": float(np.sum(support)),
        "cumulative_added_support_effect_m3": float(np.sum(added)),
        "cumulative_removed_support_effect_m3": float(np.sum(removed)),
        "figure": str((OUT / "system2005_masked_internal_source_feedback.png").relative_to(ROOT)),
        "source_hashes": {"annual": sha256(annual_path), "exchange": sha256(exchange_path)},
        "scientific_data_modified": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
