"""Plot matched qcf station curves for the two Background Candidate v2 fields."""

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
LOOP = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2"
)
RAW = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/"
    "station_year_candidate_values.csv"
)
CONSTRAINED = LOOP / (
    "temperature_background_candidate_v2_constrained_bridge_v1/"
    "station_year_candidate_values.csv"
)
OUTPUT = LOOP / "temperature_background_candidate_v2_curve_comparison_v1"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
PERIODS = [(1880, 1900), (1901, 1920), (1940, 1950)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    sources = {"raw": RAW, "constrained": CONSTRAINED, "facts": FACTS, "driver": DRIVER}
    before = {name: sha256(path) for name, path in sources.items()}

    raw = pd.read_csv(RAW)
    constrained = pd.read_csv(CONSTRAINED)[
        ["station_id", "year", "candidate_posterior_fixed_increment_degC"]
    ].rename(columns={
        "candidate_posterior_fixed_increment_degC": "constrained_candidate_degC"
    })
    values = raw.merge(constrained, on=["station_id", "year"], how="inner", validate="one_to_one")
    values = values.rename(columns={
        "candidate_posterior_fixed_increment_degC": "rolling31_candidate_degC"
    })
    if values["year"].between(1921, 1939).any():
        raise RuntimeError("Locked year present")

    columns = [
        "qcf_station_degC",
        "project_six_member_degC",
        "rolling31_candidate_degC",
        "constrained_candidate_degC",
    ]
    curve_rows = []
    for start, end in PERIODS:
        segment = values[values["year"].between(start, end)].copy()
        matched = segment[np.all(np.isfinite(segment[columns]), axis=1)]
        for year, group in matched.groupby("year"):
            curve_rows.append({
                "period": f"{start}-{end}",
                "year": int(year),
                "station_count": int(len(group)),
                **{column: float(group[column].mean()) for column in columns},
            })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUTPUT / "matched_annual_curves.csv", index=False)

    styles = [
        ("qcf_station_degC", "#111111", "GHCN-M qcf", 2.2, "o"),
        ("project_six_member_degC", "#D55E00", "Original reconstruction", 1.8, None),
        ("rolling31_candidate_degC", "#0072B2", "31-year Candidate v2", 2.0, None),
        ("constrained_candidate_degC", "#009E73", "Constrained Candidate v2", 2.0, None),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    for axis, (start, end) in zip(axes, PERIODS, strict=True):
        rows = curves[curves["period"].eq(f"{start}-{end}")].sort_values("year")
        for column, color, label, width, marker in styles:
            axis.plot(
                rows["year"], rows[column], color=color, label=label,
                linewidth=width, marker=marker, markersize=3.5 if marker else 0,
            )
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_title(f"{start}-{end} matched qcf station mean")
        axis.set_ylabel("Temperature anomaly (degC)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(start, end)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel("Year")
    figure = OUTPUT / "temperature_background_candidate_v2_matched_station_curves.png"
    fig.savefig(figure, dpi=240)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "status": "complete",
        "experiment": "temperature_background_candidate_v2_curve_comparison_v1",
        "periods": [f"{start}-{end}" for start, end in PERIODS],
        "locked_1921_1939_numeric_values_accessed": 0,
        "curve_definition": "annual equal mean over station rows with finite qcf, original, and both candidates",
        "all_declared_sources_unchanged": unchanged,
        "figure": str(figure),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
