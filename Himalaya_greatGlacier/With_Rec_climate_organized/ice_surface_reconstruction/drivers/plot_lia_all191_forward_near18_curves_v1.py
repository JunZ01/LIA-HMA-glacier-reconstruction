"""Plot the 18 endpoint-10%-and-shape near-candidate forward curves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
SCREEN = BASE / "lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_246_v1"
OUTPUT = BASE / "lia_191_all191_forward_curve_screen_v10_247_figures_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def smooth(values: np.ndarray, window: int = 21) -> np.ndarray:
    half = window // 2
    padded = np.pad(np.asarray(values, dtype=float), half, mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing-to-overwrite:{OUTPUT}")
    scores_path = SCREEN / "cell_scores.csv"
    scores = pd.read_csv(scores_path)
    selected = scores[
        (scores["computed_e_lia_total"] <= 0.10)
        & (scores["computed_e_rgi_total"] <= 0.10)
        & scores["lag_gate_pass"].astype(bool)
        & scores["bidirectional_multistage_gate_pass"].astype(bool)
        & scores["artifact_gate_pass"].astype(bool)
    ].sort_values(["system_id", "delta_t_eq_degC"])
    if len(selected) != 18 or selected["system_id"].nunique() != 18:
        raise RuntimeError(f"near-candidate-cardinality:{len(selected)}:{selected['system_id'].nunique()}")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfb",
    })
    fig, axes = plt.subplots(6, 3, figsize=(16, 22), sharex=True)
    for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
        diagnostics = ROOT / str(row["diagnostics_path"])
        frame = pd.read_json(diagnostics, lines=True).sort_values("year")
        years = frame["year"].to_numpy(int)
        length = frame["length_m"].to_numpy(float) / 1000.0
        length_smooth = smooth(length)
        lia = float(row["lia_target_km"])
        rgi = float(row["rgi_target_km"])
        peak_pos = int(np.argmax(length))

        ax.plot(years, length, color="#9aa0a6", lw=0.8, alpha=0.48)
        ax.plot(years, length_smooth, color="#1769aa", lw=2.0)
        ax.axhline(lia, color="#c62828", lw=1.4, ls="--")
        ax.axhline(rgi, color="#2e7d32", lw=1.4, ls=":")
        ax.scatter([years[peak_pos]], [length[peak_pos]], s=26, color="#c62828", marker="^", zorder=5)
        ax.scatter([2000], [length[-1]], s=24, color="#2e7d32", marker="o", zorder=5)
        values = np.r_[length, lia, rgi]
        pad = max(0.35, 0.08 * float(np.ptp(values)))
        ax.set_ylim(float(values.min() - pad), float(values.max() + pad))
        ax.set_xlim(1470, 2000)
        ax.grid(True, alpha=0.18, lw=0.6)
        ax.set_title(
            f"System {int(row['system_id'])}  |  ΔT={float(row['delta_t_eq_degC']):+.2f} °C  |  {row['member_key']}\n"
            f"peak={float(row['source_forward_peak_km_1470_2000']):.2f} km, 2000={float(row['source_length_2000_km']):.2f} km"
        )
        ax.text(0.015, 0.03, f"LIA err {100*float(row['computed_e_lia_total']):.1f}%  ·  RGI err {100*float(row['computed_e_rgi_total']):.1f}%",
                transform=ax.transAxes, fontsize=8, color="#333333")
    for row_axes in axes:
        row_axes[0].set_ylabel("Length (km)")
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    handles = [
        plt.Line2D([0], [0], color="#9aa0a6", lw=1.0, alpha=0.6, label="Annual model length"),
        plt.Line2D([0], [0], color="#1769aa", lw=2.0, label="21-year smooth"),
        plt.Line2D([0], [0], color="#c62828", lw=1.4, ls="--", label="LIA target"),
        plt.Line2D([0], [0], color="#2e7d32", lw=1.4, ls=":", label="RGI 2000 target"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.978))
    fig.suptitle(
        "Forward glacier-length curves: 18 near-candidates\n"
        "Both total-length endpoint errors ≤10% and all v3 dynamic-shape gates pass; diagnostic only, not accepted",
        fontsize=15, y=0.997,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.965), h_pad=2.1, w_pad=1.2)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    figure = OUTPUT / "near18_forward_glacier_length_curves.png"
    fig.savefig(figure, dpi=190, bbox_inches="tight")
    plt.close(fig)
    selected.to_csv(OUTPUT / "near18_members.csv", index=False)
    summary = {
        "schema": "lia-all191-forward-near18-curve-figure-v1",
        "status": "visualization-complete-no-controller-effect",
        "member_count": 18,
        "system_count": 18,
        "selection": "both 10% total-length endpoint components plus lag, bidirectional-phase, and artifact gates; 20%-of-separation gate shown neither applied nor changed",
        "formal_acceptance_effect": 0,
        "figure": figure.name,
        "figure_sha256": sha256(figure),
        "source_scores_sha256": sha256(scores_path),
        "member_keys": selected["member_key"].astype(str).tolist(),
        "system_ids": selected["system_id"].astype(int).tolist(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
