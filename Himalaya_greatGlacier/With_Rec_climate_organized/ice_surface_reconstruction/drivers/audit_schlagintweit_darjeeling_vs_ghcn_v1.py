from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schlagintweit", type=Path, required=True)
    parser.add_argument("--ghcn-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.schlagintweit, args.ghcn_panel]
    before = {str(path.resolve()): sha256(path) for path in sources}

    official = pd.read_csv(args.schlagintweit)
    official = official[
        official.station.eq("Darjeeling Observatory Hill")
        & official.year_start.isin([1850, 1852])
    ][["year_start", "month", "reported_temperature_f", "temperature_c", "source_page"]].rename(
        columns={"year_start": "year", "temperature_c": "official_temperature_c"}
    )
    panel = pd.read_csv(args.ghcn_panel)
    panel = panel[
        panel.station_name.eq("DARJEELING")
        & panel.year.isin([1850, 1852])
        & panel.quality_accepted.eq(True)
    ][["series", "year", "month", "analysis_temperature_c"]].rename(
        columns={"analysis_temperature_c": "ghcn_temperature_c"}
    )
    joined = official.merge(panel, on=["year", "month"], how="inner", validate="one_to_many")
    if len(joined) != 48 or set(joined.series) != {"qcf", "qcu"}:
        raise RuntimeError(f"Unexpected comparison coverage: {len(joined)} rows")
    joined["ghcn_minus_official_c"] = joined.ghcn_temperature_c - joined.official_temperature_c
    joined.to_csv(args.output / "darjeeling_1850_1852_official_vs_ghcn_monthly.csv", index=False)

    metric_rows = []
    for series, group in joined.groupby("series"):
        residual = group.ghcn_minus_official_c.to_numpy(float)
        metric_rows.append(
            {
                "series": series,
                "n_months": len(group),
                "correlation": float(np.corrcoef(group.official_temperature_c, group.ghcn_temperature_c)[0, 1]),
                "mean_ghcn_minus_official_c": float(np.mean(residual)),
                "rmse_c": float(np.sqrt(np.mean(residual**2))),
                "max_absolute_difference_c": float(np.max(np.abs(residual))),
                "residual_standard_deviation_c": float(np.std(residual, ddof=1)),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "darjeeling_1850_1852_official_vs_ghcn_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), constrained_layout=True)
    for year, color in [(1850, "#4c72b0"), (1852, "#dd8452")]:
        source = official[official.year.eq(year)]
        axes[0].plot(source.month, source.official_temperature_c, marker="o", linewidth=2.2, color=color, label=f"Official {year}")
        for series, style in [("qcu", "--"), ("qcf", ":")]:
            subset = joined[(joined.year.eq(year)) & (joined.series.eq(series))]
            axes[0].plot(subset.month, subset.ghcn_temperature_c, linestyle=style, marker=".", linewidth=1.5, color=color, label=f"GHCN {series} {year}")
    axes[0].set_ylabel("Monthly mean temperature (deg C)")
    axes[0].set_title("Darjeeling official table versus GHCN-M")
    axes[0].grid(axis="y", color="#dddddd")
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    for series, color in [("qcu", "#55a868"), ("qcf", "#c44e52")]:
        subset = joined[joined.series.eq(series)].copy()
        x = np.arange(len(subset))
        axes[1].plot(x, subset.ghcn_minus_official_c, marker="o", linewidth=1.7, color=color, label=series)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("GHCN minus official (deg C)")
    axes[1].set_xlabel("24 month sequence: 1850 then 1852")
    axes[1].set_title("Processing difference")
    axes[1].grid(axis="y", color="#dddddd")
    axes[1].legend(frameon=False)
    fig.savefig(args.output / "darjeeling_1850_1852_official_vs_ghcn.png", dpi=220)
    plt.close(fig)

    qcu = metrics.loc[metrics.series.eq("qcu")].iloc[0]
    qcf = metrics.loc[metrics.series.eq("qcf")].iloc[0]
    after = {str(path.resolve()): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("Source changed during audit")
    summary = {
        "experiment": "temperature_schlagintweit_darjeeling_vs_ghcn_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "overlap_years": [1850, 1852],
        "overlap_months_per_series": 24,
        "qcu_correlation": float(qcu.correlation),
        "qcu_mean_offset_c": float(qcu.mean_ghcn_minus_official_c),
        "qcu_rmse_c": float(qcu.rmse_c),
        "qcf_correlation": float(qcf.correlation),
        "qcf_mean_offset_c": float(qcf.mean_ghcn_minus_official_c),
        "qcf_rmse_c": float(qcf.rmse_c),
        "independence_decision": "same_underlying_observation_family_not_independent_votes",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "source_manifest.json").write_text(
        json.dumps(
            {
                "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
                "source_hashes_unchanged": True,
                "locked_period_accessed": False,
                "prohibited_source_accessed": False,
                "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = f"""# Darjeeling official table versus GHCN-M

The 1850 and 1852 monthly values in the Heidelberg primary table were compared directly with the Round 277 GHCN-M v4 panel.

- qcu: correlation `{qcu.correlation:.6f}`, mean GHCN-official offset `{qcu.mean_ghcn_minus_official_c:+.3f} deg C`, RMSE `{qcu.rmse_c:.3f} deg C`.
- qcf: correlation `{qcf.correlation:.6f}`, mean GHCN-official offset `{qcf.mean_ghcn_minus_official_c:+.3f} deg C`, RMSE `{qcf.rmse_c:.3f} deg C`.

The comparison audits transcription and homogenization. It is not an independent climate vote because both products trace the same physical station observations.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 accessed or used: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
