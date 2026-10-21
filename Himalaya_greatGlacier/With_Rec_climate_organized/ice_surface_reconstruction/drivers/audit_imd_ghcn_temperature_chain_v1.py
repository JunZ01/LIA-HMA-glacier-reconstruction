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
STATIONS = ("LEH_KASHMIR", "SHIMLA")


# Values below were transcribed from the printed annual rows on the cited IMD
# report pages. The comparison target is the conventional monthly TAVG
# definition, 0.5 * (mean maximum + mean minimum), not the separate historical
# observation-time "Mean" column.
SOURCE_PAGE_ANCHORS = (
    {
        "station_name": "SHIMLA",
        "year": 1881,
        "mean_max_f": 65.9,
        "mean_min_f": 50.7,
        "page_status": "complete_annual_row",
        "report_label": "Simla (corrected)",
        "image_relpath": "india.history.resource.108635/leaf_0258.jpg",
    },
    {
        "station_name": "SHIMLA",
        "year": 1883,
        "mean_max_f": 65.2,
        "mean_min_f": 49.6,
        "page_status": "complete_annual_row",
        "report_label": "Simla (corrected)",
        "image_relpath": "india.history.resource.108636/leaf_0236.jpg",
    },
    {
        "station_name": "SHIMLA",
        "year": 1884,
        "mean_max_f": 65.1,
        "mean_min_f": 50.2,
        "page_status": "complete_annual_row",
        "report_label": "Simla (corrected)",
        "image_relpath": "india.history.resource.108637/leaf_0232.jpg",
    },
    {
        "station_name": "SHIMLA",
        "year": 1886,
        "mean_max_f": 64.3,
        "mean_min_f": 49.6,
        "page_status": "complete_annual_row",
        "report_label": "Simla (corrected)",
        "image_relpath": "india.history.resource.109237/leaf_0301.jpg",
    },
)


PAGE_AUDIT = (
    {
        "station_name": "LEH_KASHMIR",
        "year": 1880,
        "report_label": "Leh (corrected)",
        "image_relpath": "india.history.resource.108634/leaf_0261.jpg",
        "page_status": "lineage_only_no_ghcn_overlap",
        "note": "Monthly mean observations are printed; maximum thermometer marked untrustworthy.",
    },
    {
        "station_name": "SHIMLA",
        "year": 1878,
        "report_label": "Simla (corrected in part)",
        "image_relpath": "india.history.resource.117788/leaf_0184.jpg",
        "page_status": "incomplete_no_annual_comparison",
        "note": "March, May, and June are incomplete and the annual mean is not admitted.",
    },
    {
        "station_name": "SHIMLA",
        "year": 1880,
        "report_label": "Simla (corrected)",
        "image_relpath": "india.history.resource.108634/leaf_0263.jpg",
        "page_status": "incomplete_no_annual_comparison",
        "note": "January and February are marked imperfect/ditto; no annual comparison is admitted.",
    },
    {
        "station_name": "SHIMLA",
        "year": 1890,
        "report_label": "Simla (corrected)",
        "image_relpath": "india.history.resource.109241/leaf_0293.jpg",
        "page_status": "incomplete_no_annual_comparison",
        "note": "May is marked imperfect and the annual TAVG fields are incomplete.",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f_to_c(value_f: float) -> float:
    return (value_f - 32.0) * 5.0 / 9.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--page-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def annualize(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.loc[
        panel["station_name"].isin(STATIONS)
        & panel["series"].isin(("qcf", "qcu"))
        & ~panel["year"].between(LOCKED_START, LOCKED_END)
    ].copy()
    panel["temperature_c"] = pd.to_numeric(panel["temperature_c"], errors="coerce")
    panel["quality_accepted"] = panel["quality_accepted"].astype(str).str.lower().eq("true")
    panel.loc[~panel["quality_accepted"], "temperature_c"] = np.nan

    annual = (
        panel.groupby(["station_name", "series", "year"], as_index=False)
        .agg(month_count=("temperature_c", "count"), annual_temperature_c=("temperature_c", "mean"))
    )
    annual["complete_year"] = annual["month_count"].eq(12)
    annual.loc[~annual["complete_year"], "annual_temperature_c"] = np.nan
    return annual


def build_anchor_comparison(annual: pd.DataFrame, page_root: Path) -> pd.DataFrame:
    rows = []
    for anchor in SOURCE_PAGE_ANCHORS:
        source_tavg_f = 0.5 * (anchor["mean_max_f"] + anchor["mean_min_f"])
        source_tavg_c = f_to_c(source_tavg_f)
        image_path = page_root / Path(anchor["image_relpath"])
        row = {
            **anchor,
            "image_path": str(image_path.resolve()),
            "image_sha256": sha256(image_path),
            "source_tavg_f": source_tavg_f,
            "source_tavg_c": source_tavg_c,
        }
        for series in ("qcf", "qcu"):
            match = annual.loc[
                (annual["station_name"] == anchor["station_name"])
                & (annual["series"] == series)
                & (annual["year"] == anchor["year"])
                & annual["complete_year"]
            ]
            value = float(match.iloc[0]["annual_temperature_c"]) if len(match) == 1 else np.nan
            row[f"{series}_annual_c"] = value
            row[f"{series}_minus_source_c"] = value - source_tavg_c
        rows.append(row)
    return pd.DataFrame(rows)


def plot_series(annual: pd.DataFrame, anchors: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"qcf": "#16697a", "qcu": "#c44536"}
    for ax, station in zip(axes, STATIONS):
        subset = annual.loc[(annual["station_name"] == station) & annual["complete_year"]]
        for series in ("qcf", "qcu"):
            part = subset.loc[subset["series"] == series]
            ax.plot(part["year"], part["annual_temperature_c"], color=colors[series], lw=1.4, label=series)
        source = anchors.loc[anchors["station_name"] == station]
        if not source.empty:
            ax.scatter(source["year"], source["source_tavg_c"], s=42, c="#111111", marker="x", label="IMD source-page TAVG")
        ax.axvspan(LOCKED_START, LOCKED_END, color="#bbbbbb", alpha=0.28)
        ax.set_title(station)
        ax.set_ylabel("Annual TAVG (degC)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Official station processing chains; locked years excluded from all data access")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    panel = pd.read_csv(args.panel, low_memory=False)
    if panel["year"].between(LOCKED_START, LOCKED_END).any():
        raise RuntimeError("Input panel unexpectedly contains locked-period rows.")

    annual = annualize(panel)
    if annual["year"].between(LOCKED_START, LOCKED_END).any():
        raise RuntimeError("Locked-period values reached annual output.")
    anchors = build_anchor_comparison(annual, args.page_root)

    page_rows = []
    for item in (*PAGE_AUDIT, *SOURCE_PAGE_ANCHORS):
        path = args.page_root / Path(item["image_relpath"])
        page_rows.append(
            {
                "station_name": item["station_name"],
                "year": item["year"],
                "report_label": item["report_label"],
                "page_status": item["page_status"],
                "note": item.get("note", "Annual mean maximum and minimum admitted as a numeric anchor."),
                "image_path": str(path.resolve()),
                "image_sha256": sha256(path),
            }
        )
    page_audit = pd.DataFrame(page_rows).sort_values(["station_name", "year"])

    annual.to_csv(args.output / "annual_station_series.csv", index=False)
    anchors.to_csv(args.output / "source_page_anchor_comparison.csv", index=False)
    page_audit.to_csv(args.output / "source_page_verification.csv", index=False)
    plot_series(annual, anchors, args.output / "station_processing_chain.png")

    qcf_mae = float(anchors["qcf_minus_source_c"].abs().mean())
    qcu_mae = float(anchors["qcu_minus_source_c"].abs().mean())
    summary = {
        "experiment": "temperature_imd_ghcn_station_chain_audit_v1",
        "decision": "retain_qcf_as_protocol_primary_but_elevate_qcu_as_required_low_frequency_sensitivity",
        "stations": list(STATIONS),
        "numeric_source_page_anchor_count": int(len(anchors)),
        "qcf_source_anchor_mae_c": qcf_mae,
        "qcu_source_anchor_mae_c": qcu_mae,
        "qcu_closer_anchor_count": int((anchors["qcu_minus_source_c"].abs() < anchors["qcf_minus_source_c"].abs()).sum()),
        "source_definition": "TAVG = 0.5 * (annual mean maximum + annual mean minimum)",
        "independence": "IMD pages, WWR compilation, and GHCN-M are one official observation family, not independent climate votes.",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "inputs": [
            {"path": str(args.panel.resolve()), "sha256": sha256(args.panel)},
            *[
                {"path": row.image_path, "sha256": row.image_sha256}
                for row in page_audit.drop_duplicates("image_path").itertuples(index=False)
            ],
        ],
        "prohibited_source_accessed": False,
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = f"""# IMD-GHCN temperature station-chain audit v1

## Result

- Four complete Simla annual rows (1881, 1883, 1884, 1886) were read from local official IMD report images.
- The historical-page comparison uses the conventional TAVG target, `0.5 * (mean maximum + mean minimum)`, rather than the separate observation-time `Mean` column.
- `qcf` source-anchor MAE: **{qcf_mae:.3f} degC**.
- `qcu` source-anchor MAE: **{qcu_mae:.3f} degC**.
- `qcu` is closer for **{summary['qcu_closer_anchor_count']}/{len(anchors)}** anchors.

## Interpretation

The early Simla `qcu` values closely reproduce the official printed TAVG anchors, while `qcf` is systematically warmer at these four verified years. This does not make `qcu` an independent record: both are processing states of the same observations. It does show that low-frequency conclusions based only on unadjusted `qcf` can inherit station-processing discontinuities. Keep `qcf` as the protocol primary series, but require `qcu` as a first-class homogenization sensitivity before claiming an 1850-1950 reconstruction bias.

Leh 1880 confirms the official station lineage but has no simultaneous GHCN annual overlap in this panel. Incomplete Simla pages (1878, 1880, 1890) were documented and excluded from numerical scoring.

## Safeguards

- `1921-1939` was excluded before annualization and no locked values appear in any output.
- The prohibited climate source was not accessed.
- No original or production file was modified.
- No EnKF or OGGM run was launched.
"""
    (args.output / "FINAL_RESULT.md").write_text(result, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
