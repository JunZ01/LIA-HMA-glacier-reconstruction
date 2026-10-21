"""Plot one system-3584 shape block grouped by its fixed parameter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FORWARD_START = 1470


def _display_year(series: pd.DataFrame) -> pd.Series:
    values = series["year"].astype(float).copy()
    spin = series["phase"].astype(str).eq("spinup")
    spin_end = float(series.loc[spin, "year"].max())
    values.loc[spin] = FORWARD_START - spin_end + values.loc[spin]
    return values


def _plot(group: pd.DataFrame, group_column: str, value: float, output_path: Path) -> dict:
    group = group.sort_values("delta_t_c")
    first_dir = Path(str(group.iloc[0]["source_dir"]))
    summary = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
    lia = float(summary["targets"]["lia_length_m"]) / 1000.0
    rgi = float(summary["targets"]["rgi_length_m"]) / 1000.0
    gap = abs(lia - rgi)
    low = min(lia, rgi) - 0.2 * gap
    high = max(lia, rgi) + 0.2 * gap

    figure, axis = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.85, max(len(group), 2)))
    handles: list[Line2D] = []
    for color, (_, row) in zip(colors, group.iterrows()):
        series = pd.read_csv(Path(str(row["source_dir"])) / "timeseries.csv")
        x = _display_year(series)
        y = series["length_m"].astype(float) / 1000.0
        accepted = bool(row["accepted_curve_v3"])
        for phase, style in (("spinup", "--"), ("forward", "-")):
            mask = series["phase"].astype(str).eq(phase)
            axis.plot(
                x.loc[mask],
                y.loc[mask],
                color=color,
                linestyle=style,
                linewidth=2.2 if accepted and phase == "forward" else 1.35,
                alpha=1.0 if accepted else 0.78,
            )
        status = "pass" if accepted else str(row["failure_reasons_v3"])
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.0,
                label=f"dT={float(row['delta_t_c']):+.2f} K | {status}",
            )
        )

    axis.axhspan(low, high, color="0.75", alpha=0.18)
    axis.axhline(lia, color="black", linestyle="--", linewidth=1.5)
    axis.axhline(rgi, color="0.25", linestyle="-.", linewidth=1.4)
    axis.axvline(FORWARD_START, color="0.35", linestyle=":", linewidth=1.2)
    axis.text(axis.get_xlim()[0], lia + 0.03 * (high - low), f"LIA {lia:.2f} km")
    axis.text(axis.get_xlim()[0], rgi - 0.04 * (high - low), f"RGI {rgi:.2f} km")
    label = "temp_melt" if group_column == "temp_melt_c" else "prcp_fac multiplier"
    axis.set_title(f"System 3584 | {label}={value:+g}: adaptive spin-up and 1470-2000 length")
    axis.set_xlabel("Display year")
    axis.set_ylabel("Length (km)")
    axis.grid(True, alpha=0.20)
    axis.legend(handles=handles, frameon=False, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "group_column": group_column,
        "group_value": value,
        "chain_count": len(group),
        "accepted_count": int(group["accepted_curve_v3"].sum()),
        "output_png": str(output_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument(
        "--group-column",
        required=True,
        choices=("temp_melt_c", "prcp_fac_multiplier"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.scores)
    records: list[dict] = []
    for value, group in frame.groupby(args.group_column):
        tag = f"{float(value):g}".replace("-", "m").replace(".", "p")
        records.append(
            _plot(
                group,
                args.group_column,
                float(value),
                args.output_dir / f"{args.prefix}_{tag}.png",
            )
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(args.output_dir / f"{args.prefix}_manifest.csv", index=False)
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
