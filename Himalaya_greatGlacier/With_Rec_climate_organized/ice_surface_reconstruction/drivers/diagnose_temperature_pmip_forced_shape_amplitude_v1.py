"""Separate forced-model low-frequency shape from amplitude without tuning a candidate."""

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
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
PARENT = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2"
INPUT = PARENT / "annual_hma_temperature_curves_1500_1899.csv"
PARENT_SUMMARY = PARENT / "summary.json"
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_forced_shape_amplitude_v1"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
MODELS = ["ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]
ANCHOR = "four_family_anchor_hma_degC"
BACKGROUND = "background_six_member_degC"
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left, ddof=1) == 0 or np.std(right, ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# PMIP Forced-Signal Shape/Amplitude Contract\n\n"
        "Use only the completed v2 annual HMA curves for 1500-1899. Freeze eight 50-year windows, "
        "three forced simulations, the current background, and the four-family anchor. Decompose each "
        "model-anchor error into mean amplitude and centered temporal shape. The implied background-to-model "
        "blend is diagnostic only and cannot be used as a tuned production parameter. No climate value from "
        "1901-1939 is read.\n",
        encoding="utf-8",
    )
    paths = {
        "annual_curves": INPUT,
        "parent_summary": PARENT_SUMMARY,
        "facts": FACTS,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}
    frame = pd.read_csv(INPUT)
    if len(frame) != 400 or frame.year.min() != 1500 or frame.year.max() != 1899:
        raise RuntimeError("Frozen annual input alignment failed")

    window_rows: list[dict[str, float | int | str | bool]] = []
    for start, end in WINDOWS:
        selected = frame.year.between(start, end)
        row: dict[str, float | int | str | bool] = {
            "start_year": start,
            "end_year": end,
            "window_center": (start + end) / 2,
        }
        for name in [ANCHOR, BACKGROUND, *MODELS]:
            row[name] = float(frame.loc[selected, name].mean())
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows["forced_mean"] = windows[MODELS].mean(axis=1)
    windows["forced_median"] = windows[MODELS].median(axis=1)

    metrics: dict[str, dict[str, float | bool]] = {}
    detail_rows: list[dict[str, float | str | int | bool]] = []
    anchor = windows[ANCHOR].to_numpy(float)
    background = windows[BACKGROUND].to_numpy(float)
    evaluated = [*MODELS, "forced_mean", "forced_median"]
    for name in evaluated:
        values = windows[name].to_numpy(float)
        centered_values = values - values.mean()
        centered_anchor = anchor - anchor.mean()
        between = ((values <= anchor) & (anchor <= background)) | (
            (background <= anchor) & (anchor <= values)
        )
        denominator = values - background
        required_fraction = np.divide(
            anchor - background,
            denominator,
            out=np.full_like(anchor, np.nan),
            where=np.abs(denominator) > 1e-12,
        )
        valid_fraction = np.isfinite(required_fraction) & (required_fraction >= 0) & (
            required_fraction <= 1
        )
        metrics[name] = {
            "mean_gap_vs_anchor_degC": float(np.mean(values - anchor)),
            "total_window_rmse_degC": rmse(values, anchor),
            "centered_shape_rmse_degC": rmse(centered_values, centered_anchor),
            "centered_shape_correlation": correlation(centered_values, centered_anchor),
            "fraction_windows_anchor_bracketed": float(np.mean(between)),
            "median_required_background_to_model_fraction": float(
                np.nanmedian(required_fraction)
            ),
            "iqr_required_fraction": float(
                np.nanpercentile(required_fraction, 75)
                - np.nanpercentile(required_fraction, 25)
            ),
            "fraction_required_fractions_in_0_1": float(np.mean(valid_fraction)),
            "all_windows_cooler_than_background": bool(np.all(values < background)),
        }
        for index, row in windows.iterrows():
            detail_rows.append(
                {
                    "series": name,
                    "start_year": int(row.start_year),
                    "end_year": int(row.end_year),
                    "series_mean_degC": float(row[name]),
                    "anchor_mean_degC": float(row[ANCHOR]),
                    "background_mean_degC": float(row[BACKGROUND]),
                    "anchor_bracketed": bool(between[index]),
                    "required_background_to_model_fraction": float(
                        required_fraction[index]
                    ),
                }
            )

    all_model_windows_cooler = bool(
        all(bool(metrics[name]["all_windows_cooler_than_background"]) for name in MODELS)
    )
    bracket_fraction = float(
        np.mean(
            [
                float(metrics[name]["fraction_windows_anchor_bracketed"])
                for name in MODELS
            ]
        )
    )
    positive_shape_count = int(
        sum(float(metrics[name]["centered_shape_correlation"]) > 0 for name in MODELS)
    )
    forced_shape_supported = bool(
        all_model_windows_cooler and bracket_fraction >= 0.75 and positive_shape_count >= 2
    )

    pd.DataFrame(detail_rows).to_csv(
        OUTPUT / "forced_shape_amplitude_50year_diagnostics.csv", index=False
    )
    windows.to_csv(OUTPUT / "forced_model_50year_curves.csv", index=False)

    colors = {
        BACKGROUND: "#D55E00",
        ANCHOR: "#111111",
        "ModE-Sim": "#0072B2",
        "MPI-ESM1-2-LR": "#009E73",
        "MIROC-ES2L": "#CC79A7",
        "forced_mean": "#6A3D9A",
    }
    centers = windows.window_center.to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for name in [ANCHOR, BACKGROUND, *MODELS, "forced_mean"]:
        width = 2.7 if name in [ANCHOR, "forced_mean"] else 1.7
        axes[0].plot(
            centers,
            windows[name],
            marker="o",
            linewidth=width,
            color=colors[name],
            label=name.replace("_six_member_degC", "").replace("_hma_degC", ""),
        )
    axes[0].set(
        title="Frozen HMA 50-year temperature means",
        ylabel="Temperature anomaly (degC)",
    )
    axes[0].legend(frameon=False, ncol=3)
    detail = pd.DataFrame(detail_rows)
    for name in [*MODELS, "forced_mean"]:
        selected = detail[detail.series.eq(name)]
        axes[1].plot(
            centers,
            selected.required_background_to_model_fraction,
            marker="o",
            linewidth=1.8,
            color=colors[name],
            label=name,
        )
    axes[1].axhspan(0, 1, color="#EEEEEE", zorder=-1)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].axhline(1, color="#777777", linewidth=0.8)
    axes[1].set(
        title="Diagnostic fraction of forced amplitude needed to reach the anchor",
        xlabel="Window center year",
        ylabel="Background-to-model fraction",
    )
    axes[1].legend(frameon=False, ncol=4)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure = OUTPUT / "temperature_pmip_forced_shape_amplitude.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    decision = (
        "advance_dynamic_forced_amplitude_prior_design"
        if forced_shape_supported
        else "forced_models_support_cold_mean_but_not_stable_temporal_shape"
    )
    summary = {
        "status": "complete",
        "locked_1901_1939_climate_values_read": 0,
        "source_hashes_unchanged": unchanged,
        "series_metrics": metrics,
        "gates": {
            "all_three_models_cooler_than_background_all_windows": all_model_windows_cooler,
            "mean_model_anchor_bracketing_fraction_at_least_0p75": bool(
                bracket_fraction >= 0.75
            ),
            "at_least_two_models_positive_centered_shape_correlation": bool(
                positive_shape_count >= 2
            ),
        },
        "model_anchor_bracketing_fraction": bracket_fraction,
        "positive_centered_shape_model_count": positive_shape_count,
        "forced_temporal_shape_supported": forced_shape_supported,
        "decision": decision,
        "interpretation_limit": (
            "The required amplitude fractions are target-derived diagnostics, not fitted parameters. "
            "No corrected climate series is produced in this experiment."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {"before": before, "after": after, "source_hashes_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )
    consensus = metrics["forced_mean"]
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# PMIP Forced-Signal Shape and Amplitude\n\n"
        f"- All three models cooler than background in all windows: `{str(all_model_windows_cooler).lower()}`.\n"
        f"- Model-window anchor bracketing fraction: `{bracket_fraction:.1%}`.\n"
        f"- Models with positive centered shape correlation: `{positive_shape_count}/3`.\n"
        f"- Forced-mean centered shape correlation: `{float(consensus['centered_shape_correlation']):+.3f}`.\n"
        f"- Forced-mean median required amplitude fraction: `{float(consensus['median_required_background_to_model_fraction']):.3f}` "
        f"(IQR `{float(consensus['iqr_required_fraction']):.3f}`).\n"
        "- No corrected series was produced.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
