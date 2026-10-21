"""Locate the late-19th-century HMA temperature transition mismatch."""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import validate_temperature_external_low_frequency_v1 as base


SOURCE = (
    base.RESULTS
    / "climate_reconstruction_temperature_circulation_proxy_regions_1500_2000_v1"
    / "temperature_circulation_proxy_region_curves_1500_2000.csv"
)
OUTPUT = base.RESULTS / "climate_reconstruction_temperature_transition_onset_1800_1900_v1"
REGIONS = {
    "west_lt80": 418,
    "central_80_90": 489,
    "east_ge90": 674,
}
BASELINE = (1800, 1860)
THRESHOLD = 0.15
RUN_LENGTH = 10


def read_safe_prefix(path: Path) -> pd.DataFrame:
    # The source begins at 1500; header plus 401 rows ends exactly at 1900.
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index > 401:
                break
            lines.append(line)
    frame = pd.read_csv(io.StringIO("".join(lines)))
    if frame.year.min() != 1500 or frame.year.max() != 1900 or len(frame) != 401:
        raise RuntimeError("Safe prefix chronology does not close at 1500-1900")
    return frame


def first_persistent_crossing(years: np.ndarray, values: np.ndarray) -> int | None:
    eligible = years >= BASELINE[1] + 1
    y = years[eligible]
    v = values[eligible]
    for index in range(len(v) - RUN_LENGTH + 1):
        window = v[index : index + RUN_LENGTH]
        if np.all(window >= THRESHOLD) or np.all(window <= -THRESHOLD):
            return int(y[index])
    return None


def slope_per_century(years: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    selected = (years >= start) & (years <= end)
    return float(np.polyfit(years[selected], values[selected], 1)[0] * 100)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Read only 1500-1900 from the existing regional curve table.\n"
        "- Use 1800-1860 as the mean-bias baseline.\n"
        "- Define onset before seeing results as 10 consecutive years with absolute "
        "mean-adjusted 21-year residual >= 0.15 degC.\n"
        "- Do not access or score 1901-1920 or 1921-1939.\n",
        encoding="utf-8",
    )
    before = {"source": base.sha256(SOURCE), "contract": base.sha256(contract)}
    frame = read_safe_prefix(SOURCE)
    years = frame.year.to_numpy(int)

    residuals = {}
    for region in REGIONS:
        project = frame[f"{region}_project_smooth21_degC"].to_numpy(float)
        reference = frame[f"{region}_reference_smooth21_degC"].to_numpy(float)
        residuals[region] = project - reference
    total_weight = float(sum(REGIONS.values()))
    residuals["hma_weighted"] = sum(
        residuals[region] * weight for region, weight in REGIONS.items()
    ) / total_weight

    baseline_mask = (years >= BASELINE[0]) & (years <= BASELINE[1])
    adjusted = {
        name: values - float(np.mean(values[baseline_mask]))
        for name, values in residuals.items()
    }
    output_frame = pd.DataFrame({"year": years})
    rows = []
    for name, values in adjusted.items():
        onset = first_persistent_crossing(years, values)
        output_frame[f"{name}_raw_residual_degC"] = residuals[name]
        output_frame[f"{name}_mean_adjusted_residual_degC"] = values
        late = (years >= 1861) & (years <= 1900)
        max_index = np.flatnonzero(late)[np.argmax(np.abs(values[late]))]
        row = {
            "region": name,
            "baseline_mean_residual_1800_1860_degC": float(np.mean(residuals[name][baseline_mask])),
            "persistent_onset_year": onset,
            "slope_1800_1860_degC_per_century": slope_per_century(years, values, 1800, 1860),
            "slope_1861_1880_degC_per_century": slope_per_century(years, values, 1861, 1880),
            "slope_1881_1900_degC_per_century": slope_per_century(years, values, 1881, 1900),
            "max_abs_residual_year_1861_1900": int(years[max_index]),
            "max_abs_residual_1861_1900_degC": float(values[max_index]),
        }
        for target_year in [1850, 1860, 1870, 1880, 1890, 1900]:
            row[f"residual_{target_year}_degC"] = float(values[years == target_year][0])
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "transition_onset_metrics.csv", index=False)
    output_frame.loc[output_frame.year.between(1800, 1900)].to_csv(
        OUTPUT / "transition_residual_curves_1800_1900.csv", index=False
    )

    order = ["hma_weighted", "west_lt80", "central_80_90", "east_ge90"]
    labels = {
        "hma_weighted": "HMA weighted",
        "west_lt80": "West (<80E)",
        "central_80_90": "Central (80-90E)",
        "east_ge90": "East (>=90E)",
    }
    colors = {
        "hma_weighted": "#111111",
        "west_lt80": "#2B6CB0",
        "central_80_90": "#B7791F",
        "east_ge90": "#2F855A",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, name in zip(axes.flat, order):
        selected = years >= 1800
        ax.plot(years[selected], adjusted[name][selected], color=colors[name], linewidth=2.2)
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
        ax.axhline(THRESHOLD, color="#B92D27", linewidth=0.8, linestyle="--")
        ax.axhline(-THRESHOLD, color="#B92D27", linewidth=0.8, linestyle="--")
        onset = metrics.loc[metrics.region.eq(name), "persistent_onset_year"].iloc[0]
        if pd.notna(onset):
            ax.axvline(float(onset), color="#B92D27", linewidth=1.2)
        ax.set_title(labels[name])
        ax.set_xlim(1800, 1900)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].set_ylabel("Residual after mean-bias removal (degC)")
    axes[1, 0].set_ylabel("Residual after mean-bias removal (degC)")
    fig.suptitle("Onset of the late-19th-century reconstruction-reference divergence")
    fig.savefig(OUTPUT / "temperature_transition_onset_1800_1900.png", dpi=260)
    plt.close(fig)

    after = {"source": base.sha256(SOURCE), "contract": base.sha256(contract)}
    summary = {
        "status": "complete",
        "baseline_period": list(BASELINE),
        "onset_definition": {
            "absolute_residual_degC": THRESHOLD,
            "consecutive_years": RUN_LENGTH,
        },
        "onset_years": {
            row.region: (None if pd.isna(row.persistent_onset_year) else int(row.persistent_onset_year))
            for row in metrics.itertuples(index=False)
        },
        "input_hashes_unchanged": before == after,
        "years_read": [1500, 1900],
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
