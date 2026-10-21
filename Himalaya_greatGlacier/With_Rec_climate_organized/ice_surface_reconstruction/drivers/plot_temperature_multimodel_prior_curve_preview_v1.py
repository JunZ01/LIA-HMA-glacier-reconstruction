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
    "temperature_multimodel_prior_curve_preview_v1"
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
        "year",
        "four_family_anchor_hma_degC",
        "background_six_member_degC",
        "MPI-ESM1-2-LR",
        "MIROC-ES2L",
        "ModE-Sim",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if int(data["year"].min()) != 1500 or int(data["year"].max()) != 1899:
        raise ValueError("This preview must use exactly 1500-1899.")

    models = ["background_six_member_degC", "ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]
    data["external_three_mean_degC"] = data[["ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]].mean(axis=1)
    data["old_50_50_prior_degC"] = 0.5 * data["background_six_member_degC"] + 0.5 * data["external_three_mean_degC"]
    data["equal_four_prior_mean_degC"] = data[models].mean(axis=1)

    smooth_cols = [
        "four_family_anchor_hma_degC",
        "background_six_member_degC",
        "ModE-Sim",
        "MPI-ESM1-2-LR",
        "MIROC-ES2L",
        "old_50_50_prior_degC",
        "equal_four_prior_mean_degC",
    ]
    for col in smooth_cols:
        data[f"{col}_31yr"] = data[col].rolling(31, center=True, min_periods=16).mean()

    data.to_csv(OUT / "annual_and_derived_prior_curves_1500_1899.csv", index=False)

    anchor = data["four_family_anchor_hma_degC"]
    window_rows: list[dict[str, float | int]] = []
    for start in range(1500, 1900, 50):
        stop = start + 49
        block = data.loc[data["year"].between(start, stop)]
        row: dict[str, float | int] = {"start_year": start, "end_year": stop}
        for col in smooth_cols:
            row[col] = float(block[col].mean())
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows.to_csv(OUT / "fifty_year_prior_means.csv", index=False)

    metrics = {}
    for col in [
        "background_six_member_degC",
        "ModE-Sim",
        "MPI-ESM1-2-LR",
        "MIROC-ES2L",
        "external_three_mean_degC",
        "old_50_50_prior_degC",
        "equal_four_prior_mean_degC",
    ]:
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
            100.0 * (baseline_rmse - values["fifty_year_rmse_vs_anchor_degC"]) / baseline_rmse
        )

    colors = {
        "anchor": "#111111",
        "background": "#c0392b",
        "old_half": "#2166ac",
        "equal_four": "#00876c",
        "mode": "#7b3294",
        "mpi": "#d95f02",
        "miroc": "#7570b3",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8.4), sharex=False,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.30},
    )

    model_specs = [
        ("ModE-Sim", colors["mode"], "ModE-Sim"),
        ("MPI-ESM1-2-LR", colors["mpi"], "MPI-ESM1-2-LR"),
        ("MIROC-ES2L", colors["miroc"], "MIROC-ES2L"),
    ]
    for col, color, label in model_specs:
        ax1.plot(data["year"], data[f"{col}_31yr"], color=color, lw=1.1, alpha=0.55, label=label)
    ax1.plot(
        data["year"], data["background_six_member_degC_31yr"],
        color=colors["background"], lw=2.0, label="Archived CESM-derived background",
    )
    ax1.plot(
        data["year"], data["old_50_50_prior_degC_31yr"],
        color=colors["old_half"], lw=2.2, label="Old 50:50 prior blend",
    )
    ax1.plot(
        data["year"], data["equal_four_prior_mean_degC_31yr"],
        color=colors["equal_four"], lw=2.5, label="Equal-four prior mean preview",
    )
    ax1.plot(
        data["year"], data["four_family_anchor_hma_degC_31yr"],
        color=colors["anchor"], lw=2.8, label="Four-family reference anchor",
    )
    ax1.axhline(0, color="#777777", lw=0.8)
    ax1.set_xlim(1500, 1899)
    ax1.set_ylabel("HMA temperature anomaly (degC)")
    ax1.set_title("Prior-level temperature comparison, 1500-1899 (31-year smooth)", loc="left")
    ax1.text(
        0.995, 0.02,
        "Diagnostic preview only: no new EnKF posterior",
        transform=ax1.transAxes, ha="right", va="bottom", color="#555555", fontsize=9,
    )
    ax1.grid(axis="y", color="#dddddd", lw=0.7)
    ax1.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=8.7)

    centers = (windows["start_year"] + windows["end_year"]) / 2
    ax2.plot(centers, windows["four_family_anchor_hma_degC"], color=colors["anchor"], marker="o", lw=2.5, label="Reference")
    ax2.plot(centers, windows["background_six_member_degC"], color=colors["background"], marker="s", lw=1.8, label="CESM background")
    ax2.plot(centers, windows["old_50_50_prior_degC"], color=colors["old_half"], marker="D", lw=2.0, label="Old 50:50")
    ax2.plot(centers, windows["equal_four_prior_mean_degC"], color=colors["equal_four"], marker="^", lw=2.2, label="Equal-four preview")
    ax2.axhline(0, color="#777777", lw=0.8)
    ax2.set_xlim(1495, 1904)
    ax2.set_xticks(centers)
    ax2.set_xticklabels([f"{a}-{b}" for a, b in zip(windows["start_year"], windows["end_year"])], rotation=25, ha="right")
    ax2.set_ylabel("50-year mean (degC)")
    ax2.set_title("Fifty-year means", loc="left")
    ax2.grid(axis="y", color="#dddddd", lw=0.7)
    ax2.legend(ncol=4, frameon=False, loc="best", fontsize=8.7)

    fig.subplots_adjust(top=0.94, bottom=0.12, left=0.075, right=0.985)
    png = OUT / "temperature_multimodel_prior_curves_1500_1899.png"
    fig.savefig(png, dpi=180, facecolor="white")
    plt.close(fig)

    source_hash_after = sha256(INPUT)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source CSV changed during the read-only preview.")

    summary = {
        "experiment": "temperature_multimodel_prior_curve_preview_v1",
        "scope": "pre-EnKF prior-level diagnostic only",
        "years": [1500, 1899],
        "locked_period_accessed": False,
        "production_files_modified": False,
        "source_csv": str(INPUT),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "derived_definitions": {
            "external_three_mean_degC": "mean(ModE-Sim, MPI-ESM1-2-LR, MIROC-ES2L)",
            "old_50_50_prior_degC": "0.5 * archived CESM-derived background + 0.5 * external-three mean",
            "equal_four_prior_mean_degC": "equal arithmetic mean of CESM-derived background, ModE-Sim, MPI-ESM1-2-LR, MIROC-ES2L",
        },
        "metrics": metrics,
        "warning": "The equal-four curve is an arithmetic prior preview. The supported method assimilates each prior separately and averages posterior means.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "PLOT_CONTRACT.md").write_text(
        "# Plot contract\n\n"
        "This directory is an independent, read-only diagnostic. It reads the existing "
        "1500-1899 HMA annual prior audit table and derives two arithmetic comparison curves. "
        "It does not run EnKF, access 1921-1939, modify production data, or claim a Candidate v2 posterior.\n",
        encoding="ascii",
    )
    (OUT / "source_manifest.txt").write_text(
        f"{source_hash_before}  {INPUT}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
