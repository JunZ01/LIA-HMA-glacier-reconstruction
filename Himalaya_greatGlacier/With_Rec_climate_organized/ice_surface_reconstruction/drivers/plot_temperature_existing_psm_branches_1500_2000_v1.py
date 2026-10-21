"""Plot the existing HMA temperature reconstruction branches for 1500-2000."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
SOURCE = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_23_corrected_reconstruction_reference_envelope_v1"
    / "corrected_reconstruction_reference_curves.csv"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_temperature_existing_psm_branches_1500_2000_v1"
)

BRANCHES = {
    "AO_6_hma_degC": "AO / proxy6",
    "AO_9_hma_degC": "AO / proxy9",
    "HGAO_6_hma_degC": "HGAO / proxy6",
    "HGAO_9_hma_degC": "HGAO / proxy9",
    "O_6_hma_degC": "O / proxy6",
    "O_9_hma_degC": "O / proxy9",
}
COLORS = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#E69F00", "#D55E00"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source_hash_before = sha256(SOURCE)
    frame = pd.read_csv(SOURCE)
    required = {"year", "corrected_six_member_hma_degC", "reference_family_balanced_hma_degC", *BRANCHES}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    if not np.array_equal(frame["year"].to_numpy(), np.arange(1000, 2001)):
        raise RuntimeError("Unexpected source year axis")

    curves = frame.loc[frame["year"].between(1500, 2000), ["year", *BRANCHES]].copy()
    curves["six_member_mean_degC"] = frame.loc[
        frame["year"].between(1500, 2000), "corrected_six_member_hma_degC"
    ].to_numpy()
    curves["reference_family_central_degC"] = frame.loc[
        frame["year"].between(1500, 2000), "reference_family_balanced_hma_degC"
    ].to_numpy()
    for column in [*BRANCHES, "six_member_mean_degC", "reference_family_central_degC"]:
        curves[f"{column}_smooth21"] = curves[column].rolling(21, center=True, min_periods=11).mean()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    curves.to_csv(OUTPUT / "temperature_psm_branches_1500_2000.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, constrained_layout=True)
    years = curves["year"]

    for (column, label), color in zip(BRANCHES.items(), COLORS):
        axes[0].plot(years, curves[column], color=color, linewidth=0.75, alpha=0.62, label=label)
    axes[0].plot(years, curves["six_member_mean_degC"], color="#111111", linewidth=1.8, label="Six-member mean")
    axes[0].plot(
        years,
        curves["reference_family_central_degC"],
        color="#666666",
        linewidth=1.6,
        linestyle="--",
        label="Public-reference family central",
    )
    axes[0].set_title("Annual HMA temperature anomaly")
    axes[0].set_ylabel("Temperature anomaly (degC)")
    axes[0].legend(ncol=4, frameon=False, loc="upper left")

    for (column, label), color in zip(BRANCHES.items(), COLORS):
        axes[1].plot(years, curves[f"{column}_smooth21"], color=color, linewidth=1.35, label=label)
    axes[1].plot(
        years,
        curves["six_member_mean_degC_smooth21"],
        color="#111111",
        linewidth=2.5,
        label="Six-member mean",
    )
    axes[1].plot(
        years,
        curves["reference_family_central_degC_smooth21"],
        color="#666666",
        linewidth=2.2,
        linestyle="--",
        label="Public-reference family central",
    )
    axes[1].set_title("21-year centered mean")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Temperature anomaly (degC)")

    for axis in axes:
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_xlim(1500, 2000)
        axis.set_xticks([1500, 1600, 1700, 1800, 1900, 1950, 2000])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Existing HMA temperature reconstruction branches, 1500-2000", fontsize=15)
    figure_path = OUTPUT / "temperature_psm_branches_1500_2000.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    smooth_columns = [f"{column}_smooth21" for column in BRANCHES]
    branch_spread = curves[smooth_columns].max(axis=1) - curves[smooth_columns].min(axis=1)
    summary = {
        "status": "complete",
        "period": [1500, 2000],
        "branch_count": len(BRANCHES),
        "branches": list(BRANCHES.values()),
        "smoothing": "21-year centered rolling mean, minimum 11 years at endpoints",
        "median_smoothed_branch_spread_degC": float(branch_spread.median()),
        "maximum_smoothed_branch_spread_degC": float(branch_spread.max()),
        "reference_gap_1921_1939_retained": bool(
            curves.loc[curves["year"].between(1921, 1939), "reference_family_central_degC"].isna().all()
        ),
        "interpretation_limit": (
            "These are the six existing AO/HGAO/O x proxy6/proxy9 reconstruction branches. "
            "Round 14 linear-product and Round 29 isotonic PSM tests did not produce full-period reconstructions."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    source_hash_after = sha256(SOURCE)
    manifest = {
        "source": str(SOURCE),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source file changed")

    print(json.dumps(summary, indent=2))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
