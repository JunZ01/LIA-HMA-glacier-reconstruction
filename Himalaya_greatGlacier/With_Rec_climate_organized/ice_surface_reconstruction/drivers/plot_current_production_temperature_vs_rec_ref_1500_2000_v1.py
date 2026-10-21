"""Plot the retained HMA production-temperature mean against rec_ref."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
SOURCE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_23_corrected_reconstruction_reference_envelope_v1/"
    "corrected_reconstruction_reference_curves.csv"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_current_production_temperature_vs_rec_ref_1500_2000_v1"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
LOCKED = (1921, 1939)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smooth_segments(frame: pd.DataFrame, column: str, window: int = 21) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        selected = frame["year"].between(start, end)
        result.loc[selected] = (
            frame.loc[selected, column]
            .rolling(window, center=True, min_periods=(window + 1) // 2)
            .mean()
            .to_numpy(float)
        )
    return result


def plot_segments(axis, frame: pd.DataFrame, column: str, **kwargs) -> None:
    for start, end in ((1500, 1920), (1940, 2000)):
        selected = frame["year"].between(start, end)
        axis.plot(frame.loc[selected, "year"], frame.loc[selected, column], **kwargs)
        kwargs.pop("label", None)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    before = {"source": sha256(SOURCE), "facts": sha256(FACTS)}
    source = pd.read_csv(SOURCE)
    required = {
        "year",
        "corrected_six_member_hma_degC",
        "reference_member_equal_hma_degC",
    }
    missing = required.difference(source.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    curves = source.loc[
        source["year"].between(1500, 2000),
        ["year", "corrected_six_member_hma_degC", "reference_member_equal_hma_degC"],
    ].copy()
    curves = curves.rename(
        columns={
            "corrected_six_member_hma_degC": "current_production_temperature_degC",
            "reference_member_equal_hma_degC": "rec_ref_degC",
        }
    )
    locked = curves["year"].between(*LOCKED)
    if not curves.loc[locked, ["current_production_temperature_degC", "rec_ref_degC"]].isna().all().all():
        raise RuntimeError("Locked-period values are unexpectedly populated")

    for column in ("current_production_temperature_degC", "rec_ref_degC"):
        curves[f"{column}_smooth21"] = smooth_segments(curves, column)
    curves.to_csv(OUTPUT / "hma_temperature_current_production_vs_rec_ref_1500_2000.csv", index=False)

    colors = {"production": "#D55E00", "rec_ref": "#2166AC"}
    fig, axis = plt.subplots(figsize=(14.5, 6.8), constrained_layout=True)
    plot_segments(
        axis,
        curves,
        "current_production_temperature_degC",
        color=colors["production"],
        linewidth=0.65,
        alpha=0.22,
        label="Current retained production temperature (annual)",
    )
    plot_segments(
        axis,
        curves,
        "rec_ref_degC",
        color=colors["rec_ref"],
        linewidth=0.65,
        alpha=0.20,
        label="rec_ref (annual)",
    )
    plot_segments(
        axis,
        curves,
        "current_production_temperature_degC_smooth21",
        color=colors["production"],
        linewidth=2.7,
        label="Current retained production temperature (21-year mean)",
    )
    plot_segments(
        axis,
        curves,
        "rec_ref_degC_smooth21",
        color=colors["rec_ref"],
        linewidth=2.7,
        label="rec_ref (21-year mean)",
    )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E6E6E6", alpha=0.9, zorder=-5)
    axis.text(1930, -1.03, "sealed", ha="center", va="bottom", color="#666666", fontsize=9)
    axis.set_xlim(1500, 2000)
    axis.set_ylim(-1.08, 1.02)
    axis.set_xticks([1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000])
    axis.set_xlabel("Year")
    axis.set_ylabel("Temperature anomaly (degC, 1961-1990)")
    axis.set_title("HMA temperature: retained production reconstruction vs rec_ref, 1500-2000")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper left")

    figure = OUTPUT / "hma_temperature_current_production_vs_rec_ref_1500_2000.png"
    fig.savefig(figure, dpi=240)
    plt.close(fig)

    after = {"source": sha256(SOURCE), "facts": sha256(FACTS)}
    summary = {
        "status": "complete",
        "period": [1500, 2000],
        "current_production_series": "coordinate-corrected six-member HMA temperature mean retained by Round 223",
        "rec_ref_series": "equal-member mean of the ten rec_ref temperature members",
        "baseline": "1961-1990 grid-cell temperature anomaly",
        "regional_weighting": "HMA cosine-latitude area weighting inherited from the verified Round 23 curves",
        "smoothing": "21-year centered mean calculated independently on 1500-1920 and 1940-2000",
        "locked_1921_1939_values_accessed": 0,
        "sources_unchanged": before == after,
        "figure": str(figure),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
