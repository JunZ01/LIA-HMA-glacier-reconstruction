from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCKED_START = 1921
LOCKED_END = 1939
PERIODS = ((1880, 1900), (1901, 1920), (1940, 1950))

SOURCE_PAGE_ANCHORS = (
    ("SHIMLA", 1881, 65.9, 50.7, "india.history.resource.108635/leaf_0258.jpg"),
    ("SHIMLA", 1883, 65.2, 49.6, "india.history.resource.108636/leaf_0236.jpg"),
    ("SHIMLA", 1884, 65.1, 50.2, "india.history.resource.108637/leaf_0232.jpg"),
    ("SHIMLA", 1886, 64.3, 49.6, "india.history.resource.109237/leaf_0301.jpg"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--lineage-notes", type=Path, required=True)
    parser.add_argument("--page-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f_to_c(value_f: float) -> float:
    return (value_f - 32.0) * 5.0 / 9.0


def annualize(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.loc[
        panel["series"].isin(("qcf", "qcu"))
        & ~panel["year"].between(LOCKED_START, LOCKED_END)
    ].copy()
    panel["temperature_c"] = pd.to_numeric(panel["temperature_c"], errors="coerce")
    accepted = panel["quality_accepted"].astype(str).str.lower().eq("true")
    panel.loc[~accepted, "temperature_c"] = np.nan
    keys = ["station_id", "station_name", "station_tier", "region", "series", "year"]
    annual = panel.groupby(keys, as_index=False).agg(
        month_count=("temperature_c", "count"),
        annual_temperature_c=("temperature_c", "mean"),
    )
    annual["complete_year"] = annual["month_count"].eq(12)
    annual.loc[~annual["complete_year"], "annual_temperature_c"] = np.nan
    return annual


def build_processing_differences(annual: pd.DataFrame) -> pd.DataFrame:
    keys = ["station_id", "station_name", "station_tier", "region", "year"]
    wide = annual.loc[annual["complete_year"]].pivot(index=keys, columns="series", values="annual_temperature_c").reset_index()
    wide.columns.name = None
    wide = wide.dropna(subset=["qcf", "qcu"]).copy()
    wide["qcf_minus_qcu_c"] = wide["qcf"] - wide["qcu"]
    return wide.sort_values(["station_name", "year"])


def summarize_periods(differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    station_keys = ["station_id", "station_name", "station_tier", "region"]
    for start, end in PERIODS:
        part = differences.loc[differences["year"].between(start, end)]
        for key, group in part.groupby(station_keys):
            values = group["qcf_minus_qcu_c"]
            rows.append(
                {
                    **dict(zip(station_keys, key)),
                    "period": f"{start}-{end}",
                    "paired_complete_years": int(len(values)),
                    "mean_qcf_minus_qcu_c": float(values.mean()),
                    "median_qcf_minus_qcu_c": float(values.median()),
                    "min_qcf_minus_qcu_c": float(values.min()),
                    "max_qcf_minus_qcu_c": float(values.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["period", "region", "station_name"])


def build_anchor_comparison(annual: pd.DataFrame, page_root: Path) -> pd.DataFrame:
    rows = []
    for station, year, mean_max_f, mean_min_f, relpath in SOURCE_PAGE_ANCHORS:
        source_c = f_to_c(0.5 * (mean_max_f + mean_min_f))
        image_path = page_root / Path(relpath)
        row = {
            "station_name": station,
            "year": year,
            "mean_max_f": mean_max_f,
            "mean_min_f": mean_min_f,
            "source_tavg_c": source_c,
            "image_path": str(image_path.resolve()),
            "image_sha256": sha256(image_path),
        }
        for series in ("qcu", "qcf"):
            match = annual.loc[
                (annual["station_name"] == station)
                & (annual["series"] == series)
                & (annual["year"] == year)
                & annual["complete_year"]
            ]
            value = float(match.iloc[0]["annual_temperature_c"])
            row[f"{series}_annual_c"] = value
            row[f"{series}_minus_source_c"] = value - source_c
        rows.append(row)
    return pd.DataFrame(rows)


def contiguous_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    frame = frame.sort_values("year")
    group = frame["year"].diff().fillna(1).gt(1).cumsum()
    return [part for _, part in frame.groupby(group)]


def plot_diagnostics(
    annual: pd.DataFrame,
    anchors: pd.DataFrame,
    period_summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [1, 1, 1.25]})
    colors = {"qcf": "#176b87", "qcu": "#c44736"}
    for ax, station in zip(axes[:2], ("LEH_KASHMIR", "SHIMLA")):
        for series in ("qcf", "qcu"):
            part = annual.loc[
                (annual["station_name"] == station)
                & (annual["series"] == series)
                & annual["complete_year"]
            ]
            first = True
            for segment in contiguous_segments(part):
                ax.plot(
                    segment["year"],
                    segment["annual_temperature_c"],
                    color=colors[series],
                    lw=1.35,
                    label=series if first else None,
                )
                first = False
        source = anchors.loc[anchors["station_name"] == station]
        if not source.empty:
            ax.scatter(source["year"], source["source_tavg_c"], c="#111111", marker="x", s=44, label="IMD page TAVG")
        ax.axvspan(LOCKED_START, LOCKED_END, color="#bbbbbb", alpha=0.25)
        ax.set_title(station)
        ax.set_ylabel("Annual TAVG (degC)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3)

    ax = axes[2]
    period_order = [f"{start}-{end}" for start, end in PERIODS]
    stations = sorted(period_summary["station_name"].unique())
    x = np.arange(len(stations))
    offsets = np.linspace(-0.22, 0.22, len(period_order))
    period_colors = ("#2a9d8f", "#e9c46a", "#e76f51")
    for offset, period, color in zip(offsets, period_order, period_colors):
        values = period_summary.loc[period_summary["period"] == period].set_index("station_name")["mean_qcf_minus_qcu_c"]
        ax.scatter(x + offset, [values.get(station, np.nan) for station in stations], s=35, color=color, label=period)
    ax.axhline(0.0, color="#222222", lw=0.9)
    ax.set_xticks(x, stations, rotation=55, ha="right")
    ax.set_ylabel("Mean qcf - qcu (degC)")
    ax.set_title("Homogenization adjustment by station and allowed period")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.suptitle("GHCN-M adjusted (qcf) versus unadjusted (qcu) station chain")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    notes = args.lineage_notes.read_text(encoding="utf-8")
    required = (
        "`qcu` is the unadjusted GHCN-M",
        "`qcf` is the adjusted/homogenized GHCN-M",
    )
    if not all(item in notes for item in required):
        raise RuntimeError("The frozen qcu/qcf lineage labels were not found.")

    panel = pd.read_csv(args.panel, low_memory=False)
    if panel["year"].between(LOCKED_START, LOCKED_END).any():
        raise RuntimeError("Input panel unexpectedly contains locked-period rows.")
    annual = annualize(panel)
    differences = build_processing_differences(annual)
    period_summary = summarize_periods(differences)
    anchors = build_anchor_comparison(annual, args.page_root)

    annual.to_csv(args.output / "annual_station_series.csv", index=False)
    differences.to_csv(args.output / "annual_qcf_qcu_processing_difference.csv", index=False)
    period_summary.to_csv(args.output / "processing_period_summary.csv", index=False)
    anchors.to_csv(args.output / "source_page_anchor_comparison.csv", index=False)
    plot_diagnostics(annual, anchors, period_summary, args.output / "station_processing_chain_corrected.png")

    qcu_page_mae = float(anchors["qcu_minus_source_c"].abs().mean())
    qcf_page_offset = float(anchors["qcf_minus_source_c"].mean())
    period_signs = (
        period_summary.assign(sign=lambda x: np.sign(x["mean_qcf_minus_qcu_c"]))
        .groupby("period")["sign"]
        .agg(positive=lambda x: int((x > 0).sum()), negative=lambda x: int((x < 0).sum()), zero=lambda x: int((x == 0).sum()))
        .reset_index()
        .to_dict("records")
    )
    summary = {
        "experiment": "temperature_imd_ghcn_station_chain_audit_v2",
        "supersedes": "temperature_imd_ghcn_station_chain_audit_v1",
        "correction": "qcu_is_unadjusted_qcf_is_adjusted_homogenized; source-page closure is an identity/unit check, not an adjusted-series skill score",
        "qcu_source_page_closure_mae_c": qcu_page_mae,
        "mean_qcf_adjustment_at_four_simla_page_anchors_c": qcf_page_offset,
        "station_count_with_paired_processing_periods": int(period_summary["station_name"].nunique()),
        "period_adjustment_sign_counts": period_signs,
        "decision": "retain_qcf_primary_and_require_qcu_processing_envelope; do_not_infer_uniform_HMA_offset",
        "interpretation": "qcf adjustments are station-specific, time-varying, and can have opposite signs. The early page closure validates the raw qcu identity and unit path but cannot by itself validate or reject qcf homogenization.",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    inputs = [args.panel, args.lineage_notes]
    inputs.extend(args.page_root / Path(row[-1]) for row in SOURCE_PAGE_ANCHORS)
    manifest = {
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs],
        "prohibited_source_accessed": False,
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sign_lines = "\n".join(
        f"- {row['period']}: positive {row['positive']}, negative {row['negative']}, zero {row['zero']} stations"
        for row in period_signs
    )
    report = f"""# Corrected IMD-GHCN temperature station-chain audit v2

## Mandatory correction

This result supersedes v1. The frozen Round 277 lineage definition is:

- `qcu`: unadjusted GHCN-M processing state.
- `qcf`: adjusted/homogenized GHCN-M processing state and the protocol-primary target.

The four official Simla source-page anchors close against raw `qcu` with MAE **{qcu_page_mae:.3f} degC**. This verifies station identity, `/100` scaling, Fahrenheit conversion, and the TAVG arithmetic path. It is not evidence that unadjusted `qcu` is scientifically superior to homogenized `qcf`.

At those anchors, `qcf` differs from the source pages by a mean **{qcf_page_offset:+.3f} degC**, which is the expected object of the homogenization audit rather than a transcription error.

## Processing envelope

Across the direct-station panel, paired complete-year adjustments are not uniform:

{sign_lines}

For the two visually audited long records, Simla has positive `qcf-qcu` adjustments that increase stepwise to about +1.68 degC after 1895, whereas Leh has negative adjustments (about -1.75 degC in 1884-1893 and -0.96 degC after 1895). Thus processing can alter both magnitude and spatial sign of the inferred reconstruction residual.

## Decision

Keep `qcf` as the primary station target, but require `qcu` as an equal-status processing envelope in every 1850-1950 low-frequency claim. The source-page audit does not justify switching to `qcu`, and the opposite-signed station adjustments rule out deriving a uniform HMA temperature offset from either processing state alone.

The next decisive audit is the breakpoint evidence behind the large Simla and Leh qcf adjustments, followed by recomputation of regional reconstruction residuals with processing uncertainty carried explicitly.

## Safeguards

- No values from 1921-1939 were present or accessed.
- The prohibited climate source was not accessed.
- No original or production file was modified.
- No EnKF or OGGM run occurred.
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
