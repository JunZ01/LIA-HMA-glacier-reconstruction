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
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--primary-scan", type=Path, required=True)
    parser.add_argument("--ghcn-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(group: pd.DataFrame) -> dict[str, float]:
    residual = group.ghcn_minus_official_c.to_numpy(float)
    return {
        "n_months": int(len(group)),
        "correlation": float(np.corrcoef(group.official_temperature_c, group.ghcn_temperature_c)[0, 1]),
        "mean_offset_c": float(np.mean(residual)),
        "rmse_c": float(np.sqrt(np.mean(residual**2))),
        "max_absolute_difference_c": float(np.max(np.abs(residual))),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.comparison, args.primary_scan, args.ghcn_manifest]
    before = {str(path.resolve()): sha256(path) for path in sources}
    table = pd.read_csv(args.comparison)
    ghcn_manifest = json.loads(args.ghcn_manifest.read_text(encoding="utf-8"))
    if not ghcn_manifest.get("source_hashes_unchanged_during_parse", False):
        raise RuntimeError("Round 277 official-source lineage is not frozen")

    table["suspected_10F_leading_digit_error"] = (
        ((table.year.eq(1850)) & table.month.eq(12))
        | ((table.year.eq(1852)) & table.month.eq(11))
    )
    table["official_minus_ghcn_f"] = table.reported_temperature_f - (table.ghcn_temperature_c * 9 / 5 + 32)
    outliers = table[table.suspected_10F_leading_digit_error].copy()
    if len(outliers) != 4 or not np.allclose(np.abs(outliers.official_minus_ghcn_f), 10.0, atol=0.6):
        raise RuntimeError("Frozen 10F-like discrepancy pattern did not reproduce")
    table.to_csv(args.output / "darjeeling_digit_error_classification.csv", index=False)
    outliers.to_csv(args.output / "two_suspected_leading_digit_errors.csv", index=False)

    rows = []
    for series, group in table.groupby("series"):
        all_metrics = metrics(group)
        core_metrics = metrics(group[~group.suspected_10F_leading_digit_error])
        rows.append({"series": series, **{f"all_{k}": v for k, v in all_metrics.items()}, **{f"core22_{k}": v for k, v in core_metrics.items()}})
    result_metrics = pd.DataFrame(rows)
    result_metrics.to_csv(args.output / "metrics_with_and_without_two_digit_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for series, color in [("qcu", "#4c72b0"), ("qcf", "#c44e52")]:
        group = table[table.series.eq(series)].reset_index(drop=True)
        ax.plot(np.arange(len(group)), group.ghcn_minus_official_c, marker="o", linewidth=1.8, color=color, label=series)
    for index in [11, 22]:
        ax.axvspan(index - 0.35, index + 0.35, color="#f2c14e", alpha=0.35)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("24 month sequence: 1850 then 1852")
    ax.set_ylabel("GHCN minus primary table (deg C)")
    ax.set_title("Darjeeling: two isolated approximately 10 F digit discrepancies")
    ax.grid(axis="y", color="#dddddd")
    ax.legend(frameon=False)
    fig.savefig(args.output / "darjeeling_two_digit_discrepancies.png", dpi=220)
    plt.close(fig)

    qcu = result_metrics[result_metrics.series.eq("qcu")].iloc[0]
    qcf = result_metrics[result_metrics.series.eq("qcf")].iloc[0]
    summary = {
        "experiment": "temperature_schlagintweit_darjeeling_vs_ghcn_v2",
        "supersedes_interpretation": "temperature_schlagintweit_darjeeling_vs_ghcn_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_two_digit_discrepancy_adjudication",
        "suspected_archive_digit_errors": [
            {"year": 1850, "month": 12, "primary_table_f": 45.9, "GHCN_qcu_c": 13.3, "pattern": "GHCN behaves as 55.9 F"},
            {"year": 1852, "month": 11, "primary_table_f": 58.0, "GHCN_qcu_c": 8.9, "pattern": "GHCN behaves as 48.0 F"},
        ],
        "core22_qcu_correlation": float(qcu.core22_correlation),
        "core22_qcu_rmse_c": float(qcu.core22_rmse_c),
        "core22_qcf_correlation": float(qcf.core22_correlation),
        "core22_qcf_rmse_c": float(qcf.core22_rmse_c),
        "decision": "primary_table_controls_two_flagged_months_GHCN_remains_same_family_processing_series",
        "correction_scope": "audit flags only; no source or production value modified",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = {str(path.resolve()): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("Source changed during audit")
    (args.output / "source_manifest.json").write_text(
        json.dumps(
            {
                "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
                "round277_official_archive_hashes": ghcn_manifest["requests"],
                "source_hashes_unchanged": True,
                "locked_period_accessed": False,
                "prohibited_source_accessed": False,
                "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = f"""# Darjeeling primary table and GHCN digit audit

Two isolated monthly discrepancies explain almost all of the apparent 1850/1852 mismatch:

- December 1850 is `45.9 F` in the primary scan; GHCN behaves as `55.9 F`.
- November 1852 is `58.0 F` in the primary scan; GHCN behaves as `48.0 F`.

After flagging those two approximately 10 F leading-digit discrepancies, the remaining 22 months agree closely:

- qcu: `r={qcu.core22_correlation:.6f}`, RMSE `{qcu.core22_rmse_c:.3f} deg C`.
- qcf: `r={qcf.core22_correlation:.6f}`, RMSE `{qcf.core22_rmse_c:.3f} deg C`.

This is concrete evidence that the early GHCN Darjeeling chain is generally faithful but contains at least two likely archival/transcription errors. The primary scan controls those flagged months in validation. No official source file was edited.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 accessed or used: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
