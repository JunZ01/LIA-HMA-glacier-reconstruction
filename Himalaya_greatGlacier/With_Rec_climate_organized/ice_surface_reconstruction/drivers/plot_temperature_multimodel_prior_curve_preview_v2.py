from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
INPUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/"
    "annual_hma_temperature_curves_1500_1899.csv"
)
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "temperature_multimodel_prior_curve_preview_v2"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    return float(np.sqrt(np.mean(np.square(a[valid] - b[valid]))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    source_hash_before = sha256(INPUT)
    data = pd.read_csv(INPUT)
    required = {
        "year", "four_family_anchor_hma_degC", "background_six_member_degC",
        "MPI-ESM1-2-LR", "MIROC-ES2L", "ModE-Sim",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if int(data["year"].min()) != 1500 or int(data["year"].max()) != 1899:
        raise ValueError("This preview must use exactly 1500-1899.")

    external = ["ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]
    all_models = ["background_six_member_degC", *external]
    data["external_three_mean_degC"] = data[external].mean(axis=1)
    data["old_50_50_prior_degC"] = (
        0.5 * data["background_six_member_degC"]
        + 0.5 * data["external_three_mean_degC"]
    )
    data["equal_four_prior_mean_degC"] = data[all_models].mean(axis=1)

    metric_cols = [
        "four_family_anchor_hma_degC", "background_six_member_degC",
        "ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L",
        "external_three_mean_degC", "old_50_50_prior_degC",
        "equal_four_prior_mean_degC",
    ]
    for col in metric_cols:
        data[f"{col}_31yr"] = data[col].rolling(31, center=True, min_periods=16).mean()
    data.to_csv(OUT / "annual_and_derived_prior_curves_1500_1899.csv", index=False)

    rows = []
    for start in range(1500, 1900, 50):
        block = data.loc[data["year"].between(start, start + 49)]
        row = {"start_year": start, "end_year": start + 49}
        row.update({col: float(block[col].mean()) for col in metric_cols})
        rows.append(row)
    windows = pd.DataFrame(rows)
    windows.to_csv(OUT / "fifty_year_prior_means.csv", index=False)

    anchor = data["four_family_anchor_hma_degC"]
    metrics = {}
    for col in metric_cols[1:]:
        metrics[col] = {
            "annual_rmse_vs_anchor_degC": rmse(data[col], anchor),
            "mean_bias_vs_anchor_degC": float((data[col] - anchor).mean()),
            "fifty_year_rmse_vs_anchor_degC": rmse(
                windows[col], windows["four_family_anchor_hma_degC"]
            ),
        }
    baseline_rmse = metrics["background_six_member_degC"]["fifty_year_rmse_vs_anchor_degC"]
    for values in metrics.values():
        values["fifty_year_rmse_improvement_vs_background_pct"] = float(
            100 * (baseline_rmse - values["fifty_year_rmse_vs_anchor_degC"]) / baseline_rmse
        )

    colors = {
        "anchor": "#111111", "background": "#c0392b", "old": "#2166ac",
        "four": "#00876c", "mode": "#7b3294", "mpi": "#d95f02",
        "miroc": "#7570b3",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8.4),
        gridspec_kw={"height_ratios": [2.25, 1], "hspace": 0.30},
    )
    for col, color, label in [
        ("ModE-Sim", colors["mode"], "ModE-Sim"),
        ("MPI-ESM1-2-LR", colors["mpi"], "MPI-ESM1-2-LR"),
        ("MIROC-ES2L", colors["miroc"], "MIROC-ES2L"),
    ]:
        ax1.plot(data.year, data[f"{col}_31yr"], color=color, lw=1.1, alpha=.55, label=label)
    ax1.plot(data.year, data.background_six_member_degC_31yr, color=colors["background"], lw=2, label="Archived CESM-derived background")
    ax1.plot(data.year, data.old_50_50_prior_degC_31yr, color=colors["old"], lw=2.2, label="Old 50:50 prior blend")
    ax1.plot(data.year, data.equal_four_prior_mean_degC_31yr, color=colors["four"], lw=2.5, label="Equal-four prior mean preview")
    ax1.plot(data.year, data.four_family_anchor_hma_degC_31yr, color=colors["anchor"], lw=2.8, label="Four-family reference anchor")
    ax1.axhline(0, color="#777777", lw=.8)
    ax1.set(xlim=(1500, 1899), ylabel="HMA temperature anomaly (degC)")
    ax1.set_title("Prior-level temperature comparison, 1500-1899 (31-year smooth)", loc="left")
    ax1.text(.995, .02, "Diagnostic preview only: no new EnKF posterior", transform=ax1.transAxes, ha="right", color="#555555", fontsize=9)
    ax1.grid(axis="y", color="#dddddd", lw=.7)
    ax1.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.10), frameon=False, fontsize=8.7)

    centers = (windows.start_year + windows.end_year) / 2
    for col, color, marker, label in [
        ("four_family_anchor_hma_degC", colors["anchor"], "o", "Reference"),
        ("background_six_member_degC", colors["background"], "s", "CESM background"),
        ("old_50_50_prior_degC", colors["old"], "D", "Old 50:50"),
        ("equal_four_prior_mean_degC", colors["four"], "^", "Equal-four preview"),
    ]:
        ax2.plot(centers, windows[col], color=color, marker=marker, lw=2, label=label)
    ax2.axhline(0, color="#777777", lw=.8)
    ax2.set(xlim=(1495, 1904), ylabel="50-year mean (degC)")
    ax2.set_title("Fifty-year means", loc="left")
    ax2.set_xticks(centers)
    ax2.set_xticklabels([f"{a}-{b}" for a, b in zip(windows.start_year, windows.end_year)], rotation=25, ha="right")
    ax2.grid(axis="y", color="#dddddd", lw=.7)
    ax2.legend(ncol=4, frameon=False, fontsize=8.7)
    fig.subplots_adjust(top=.94, bottom=.12, left=.075, right=.985)
    fig.savefig(OUT / "temperature_multimodel_prior_curves_1500_1899.png", dpi=180, facecolor="white")
    plt.close(fig)

    source_hash_after = sha256(INPUT)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source CSV changed during the read-only preview.")
    summary = {
        "experiment": "temperature_multimodel_prior_curve_preview_v2",
        "scope": "pre-EnKF prior-level diagnostic only",
        "years": [1500, 1899], "locked_period_accessed": False,
        "production_files_modified": False,
        "source_csv": str(INPUT), "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after, "metrics": metrics,
        "warning": "Equal-four is an arithmetic prior preview; the supported method separately assimilates each prior and averages posterior means.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "PLOT_CONTRACT.md").write_text(
        "# Plot contract\n\nIndependent read-only diagnostic using 1500-1899 only. "
        "No EnKF, locked-period access, OGGM, or production edits.\n", encoding="ascii"
    )
    (OUT / "source_manifest.txt").write_text(f"{source_hash_before}  {INPUT}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
