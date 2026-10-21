"""Plot the verified HMA curves with the early period shaded correctly."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_jja_temperature_network_repair_v1"


def main() -> None:
    frame = pd.read_csv(OUTPUT / "hma_regional_series_2deg.csv")
    frame = frame[frame.year.between(1850, 2000)]
    figure, axis = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for name in frame.columns[1:]:
        smooth = frame[name].rolling(15, center=True, min_periods=15).mean()
        linewidth = 2.2 if name in {"baseline_annual", "repair_50_temperature_JJA"} else 1.25
        axis.plot(frame.year, smooth, linewidth=linewidth, label=name)
    axis.axvspan(1901, 1950, color="0.9", zorder=-2)
    axis.axhline(0, color="0.3", linewidth=0.7)
    axis.set(
        xlabel="Year",
        ylabel="Temperature anomaly (degC)",
        title="HMA temperature: annual baseline, 50-record JJA branch, and instruments",
    )
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.savefig(OUTPUT / "figure_hma_temperature_curves_1850_2000_verified.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
