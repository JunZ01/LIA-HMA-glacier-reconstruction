from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_noaa_hma_lake_temperature_project_v1 import LINEAGE, PROJECT_MEMBERS, read_project_series


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
INPUT = BASE / "temperature_lake_brgdgt_recalibration_sensitivity_v1"
OUTPUT = BASE / "temperature_lake_brgdgt_recalibration_window_audit_v2"
DRIVER = Path(__file__).resolve()
RECORDS = {
    "Tiancai": {"latitude": 26.63, "longitude": 99.72},
    "Cuoqia": {"latitude": 27.4052, "longitude": 99.7722},
    "Qinghai": {"latitude": 36.8112, "longitude": 100.137},
}
WINDOWS = [
    ("Tiancai", "full_safe", 1500.0, 1920.999999),
    ("Tiancai", "late_LIA_match", 1789.0, 1920.999999),
    ("Cuoqia", "full_safe", 1500.0, 1920.999999),
    ("Cuoqia", "published_long", 1700.0, 1920.999999),
    ("Cuoqia", "late_LIA", 1850.0, 1920.999999),
    ("Qinghai", "full_safe", 1500.0, 1920.999999),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direction(value: float | None) -> str:
    if value is None:
        return "insufficient_points"
    if value > 1e-12:
        return "warming"
    if value < -1e-12:
        return "cooling"
    return "neutral"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    points = pd.read_csv(INPUT / "safe_recalibrated_maf_points.csv")
    lineage = pd.read_csv(LINEAGE)
    lineage = lineage[(lineage.variable == "tas") & lineage.member.isin(PROJECT_MEMBERS)].copy()
    project: dict[str, dict[str, pd.DataFrame]] = {}
    for record, definition in RECORDS.items():
        project[record] = {
            row.member: read_project_series(Path(row.source_tif), definition["latitude"], definition["longitude"])
            for row in lineage.itertuples(index=False)
        }

    rows = []
    for record, window, start, end in WINDOWS:
        selected = points[(points.record == record) & points.year_ce.between(start, end)].sort_values("year_ce")
        n = int(len(selected))
        proxy_slope = None
        member_slopes: list[float] = []
        if n >= 2:
            years = selected.year_ce.to_numpy(float)
            proxy_slope = float(np.polyfit(years, selected.maf_rf_c.to_numpy(float), 1)[0] * 100.0)
            for member in PROJECT_MEMBERS:
                series = project[record][member]
                sampled = np.interp(years, series.year.to_numpy(float), series.temperature.to_numpy(float))
                member_slopes.append(float(np.polyfit(years, sampled, 1)[0] * 100.0))
        project_slope = float(np.median(member_slopes)) if member_slopes else None
        rows.append(
            {
                "record": record,
                "window": window,
                "registered_start_year": start,
                "registered_end_year": end,
                "first_observation_year": float(selected.year_ce.min()) if n else None,
                "last_observation_year": float(selected.year_ce.max()) if n else None,
                "n": n,
                "rf_maf_linear_trend_c_per_century": proxy_slope,
                "project_median_linear_trend_at_proxy_years_c_per_century": project_slope,
                "rf_direction": direction(proxy_slope),
                "project_direction": direction(project_slope),
                "direction_agreement": direction(proxy_slope) == direction(project_slope) if n >= 2 else None,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "window_metrics.csv", index=False)

    usable = metrics[metrics.n >= 2].copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    y = np.arange(len(usable))
    labels = [f"{r.record}: {r.window}" for r in usable.itertuples(index=False)]
    ax.scatter(usable.rf_maf_linear_trend_c_per_century, y + 0.11, color="#111111", s=55, label="2024 RF MAF")
    ax.scatter(
        usable.project_median_linear_trend_at_proxy_years_c_per_century,
        y - 0.11,
        color="#bf3f36",
        s=55,
        label="Project median",
    )
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Linear trend (degC per century)")
    ax.set_title("Window-sensitive common-calibration comparison")
    ax.grid(axis="x", color="#e2e2e2", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "window_trend_comparison.png", dpi=220)
    plt.close(fig)

    records = metrics.astype(object).where(pd.notna(metrics), None).to_dict(orient="records")
    summary = {
        "experiment": "temperature_lake_brgdgt_recalibration_window_audit_v2",
        "status": "complete",
        "records": records,
        "qinghai_resolution": "insufficient_one_safe_point;cannot_adjudicate_1657_1804_conflict",
        "interpretation_rule": "window direction is evaluated only when at least two proxy observations exist",
        "evidence_role": "secondary common-calibration and period-sensitivity audit;not_an_independent_vote",
        "sealed_period_values_used": False,
        "no_metric_crosses_1921_1939": True,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "inputs": [
            {"path": str(INPUT / "safe_recalibrated_maf_points.csv"), "sha256": sha256(INPUT / "safe_recalibrated_maf_points.csv")},
            {"path": str(INPUT / "summary.json"), "sha256": sha256(INPUT / "summary.json")},
            {"path": str(LINEAGE), "sha256": sha256(LINEAGE)},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Common-calibration window audit",
        "",
        "The same 2024 RF-MAF series is evaluated over fixed, scientifically motivated windows. "
        "Project trends are sampled at the exact proxy years. Qinghai has only one safe point and is not scored.",
        "",
        "| Record | Window | n | RF MAF trend | Project trend | Agreement |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in metrics.itertuples(index=False):
        if row.n < 2:
            lines.append(f"| {row.record} | {row.window} | {row.n} | insufficient | insufficient | not evaluated |")
        else:
            lines.append(
                f"| {row.record} | {row.window} | {row.n} | {row.rf_maf_linear_trend_c_per_century:+.3f} | "
                f"{row.project_median_linear_trend_at_proxy_years_c_per_century:+.3f} | "
                f"{'yes' if row.direction_agreement else 'no'} |"
            )
    lines.extend(
        [
            "",
            "No metric crosses the sealed 1921-1939 period. Production was not modified; EnKF and OGGM were not run; "
            "20CRv3 was not accessed.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
