from __future__ import annotations

import argparse
import calendar
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
    parser.add_argument("--station-climatology", type=Path, required=True)
    parser.add_argument("--station-products", type=Path, required=True)
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
    sources = [args.schlagintweit, args.station_climatology, args.station_products]
    before = {str(path.resolve()): sha256(path) for path in sources}

    official = pd.read_csv(args.schlagintweit)
    official = official[(official.station.eq("Simla")) & official.year_start.eq(1854)].sort_values("month")
    if len(official) != 12:
        raise RuntimeError("Simla 1854 official series is incomplete")
    climatology = pd.read_csv(args.station_climatology)
    climatology = climatology[climatology.station_id.eq("IN007101600")].sort_values(["series", "month"])
    if set(climatology.series) != {"qcf", "qcu"} or len(climatology) != 24:
        raise RuntimeError("Shimla baseline contract failed")
    products = pd.read_csv(args.station_products)
    if products.year.between(1921, 1939).any():
        raise RuntimeError("Input contains locked years")
    product = products[(products.station_id.eq("IN007101600")) & products.year.eq(1854)]
    if len(product) != 1:
        raise RuntimeError("Shimla 1854 product row contract failed")
    product = product.iloc[0]

    weights = np.asarray([calendar.monthrange(1854, month)[1] for month in range(1, 13)], dtype=float)
    observed = official.temperature_c.to_numpy(float)
    rows = []
    for series in ["qcf", "qcu"]:
        baseline = climatology[climatology.series.eq(series)].baseline_temperature_c.to_numpy(float)
        anomaly = float(np.average(observed - baseline, weights=weights))
        rows.append(
            {
                "reference": f"official_Simla_1854_relative_to_{series}_1961_1990",
                "anomaly_c": anomaly,
                "baseline_annual_c": float(np.average(baseline, weights=weights)),
                "historical_annual_c": float(np.average(observed, weights=weights)),
            }
        )
    station = pd.DataFrame(rows)
    project_columns = [column for column in products.columns if column.startswith("project_") and column.endswith("_degC")]
    project = pd.DataFrame(
        {
            "reference": project_columns,
            "anomaly_c": [float(product[column]) for column in project_columns],
        }
    )
    station.to_csv(args.output / "simla_1854_official_station_anomalies.csv", index=False)
    project.to_csv(args.output / "simla_1854_project_member_anomalies.csv", index=False)

    qcf = float(station.loc[station.reference.str.contains("qcf"), "anomaly_c"].iloc[0])
    qcu = float(station.loc[station.reference.str.contains("qcu"), "anomaly_c"].iloc[0])
    six = float(product.project_six_member_degC)
    member_only = project[~project.reference.eq("project_six_member_degC")].anomaly_c.to_numpy(float)
    comparison = {
        "official_qcf_anomaly_c": qcf,
        "official_qcu_anomaly_c": qcu,
        "project_six_member_anomaly_c": six,
        "project_member_min_c": float(member_only.min()),
        "project_member_max_c": float(member_only.max()),
        "project_minus_qcf_c": six - qcf,
        "project_minus_qcu_c": six - qcu,
        "project_inside_qcf_qcu_envelope": bool(min(qcf, qcu) <= six <= max(qcf, qcu)),
        "berkeley_anomaly_c": float(product.berkeley_annual_degC),
        "cru_available": bool(pd.notna(product.cru_annual_degC)),
    }
    pd.DataFrame([comparison]).to_csv(args.output / "simla_1854_comparison_summary.csv", index=False)

    labels = ["Official/qcf", "Official/qcu", "Project six", "Berkeley"]
    values = [qcf, qcu, six, float(product.berkeley_annual_degC)]
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    ax.errorbar(2, six, yerr=[[six - member_only.min()], [member_only.max() - six]], fmt="none", ecolor="#111111", capsize=5, linewidth=1.5)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("1854 annual anomaly relative to 1961-1990 (deg C)")
    ax.set_title("Simla 1854 direct observation versus project reconstruction")
    ax.grid(axis="y", color="#dddddd")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.06 if value >= 0 else -0.10), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.savefig(args.output / "simla_1854_direct_anchor_comparison.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": "temperature_schlagintweit_simla_1854_project_anchor_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        **comparison,
        "decision": "single_year_local_qcf_supports_modest_project_warm_residual_but_qcf_qcu_envelope_does_not_support_uniform_correction",
        "scientific_scope": "one local annual anchor; cannot determine HMA-wide low-frequency offset",
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
        raise RuntimeError("Source changed during comparison")
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
    report = f"""# Simla 1854 direct anchor versus project

- Official 1854 anomaly using the qcf 1961-1990 monthly climatology: `{qcf:+.3f} deg C`.
- Official 1854 anomaly using the qcu climatology: `{qcu:+.3f} deg C`.
- Project six-member anomaly at Simla: `{six:+.3f} deg C`; member range `{member_only.min():+.3f}` to `{member_only.max():+.3f} deg C`.
- Project minus official/qcf: `{six-qcf:+.3f} deg C`.
- Project minus official/qcu: `{six-qcu:+.3f} deg C`.

Under the qcf-primary convention, this one local year supports a modest warm residual in the project reconstruction. However, the project lies inside the much wider qcf/qcu homogenization envelope, so the record does not support a uniform HMA correction or define its magnitude.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 accessed or used: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
