from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_noaa_hma_lake_temperature_project_v1 import (
    LINEAGE,
    PROJECT_MEMBERS,
    read_project_series,
)


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/"
    "temperature_noaa_ngamring_original_v1"
)
SOURCE = OUTPUT / "sun2022-ngamring-co.txt"
DRIVER = Path(__file__).resolve()
LATITUDE = 29.31
LONGITUDE = 87.18
LOCKED_START = 1921.0
LOCKED_END = 1939.999999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_safe() -> tuple[pd.DataFrame, dict[str, int]]:
    header: list[str] | None = None
    rows = []
    audit = {"locked_rows_skipped_before_temperature_decode": 0, "safe_temperature_cells_decoded": 0}
    with SOURCE.open("rb") as handle:
        for raw in handle:
            if raw.startswith(b"#") or not raw.strip():
                continue
            tokens = raw.decode("utf-8").rstrip("\r\n").split("\t")
            while tokens and tokens[-1] == "":
                tokens.pop()
            if header is None:
                header = tokens
                continue
            if len(tokens) < len(header):
                tokens.extend([""] * (len(header) - len(tokens)))
            names = {name: index for index, name in enumerate(header)}
            age_bp = float(tokens[names["age_BP"]])
            year_ce = 1950.0 - age_bp
            if LOCKED_START <= year_ce <= LOCKED_END:
                audit["locked_rows_skipped_before_temperature_decode"] += 1
                continue
            if not (1500.0 <= year_ce <= 1920.999999 or 1940.0 <= year_ce <= 1950.0):
                continue
            audit["safe_temperature_cells_decoded"] += 1
            token = tokens[names["MAAT"]].strip()
            value = float(token) if token and token.lower() != "nan" else math.nan
            if not np.isfinite(value):
                continue
            rows.append({"year_ce": year_ce, "age_bp": age_bp, "maat_c": value})
    if header is None:
        raise RuntimeError("missing-data-header")
    return pd.DataFrame(rows).sort_values("year_ce").reset_index(drop=True), audit


def slope(years: np.ndarray, values: np.ndarray) -> float | None:
    if len(years) < 2:
        return None
    return float(np.polyfit(years, values, 1)[0] * 100.0)


def direction(value: float | None) -> str:
    if value is None:
        return "insufficient_points"
    if value > 1e-12:
        return "warming"
    if value < -1e-12:
        return "cooling"
    return "neutral"


def main() -> None:
    source_hash_before = sha256(SOURCE)
    points, audit = parse_safe()
    points.to_csv(OUTPUT / "safe_ngamring_maat_points.csv", index=False)
    lineage = pd.read_csv(LINEAGE)
    lineage = lineage[(lineage.variable == "tas") & lineage.member.isin(PROJECT_MEMBERS)].copy()
    project = {
        row.member: read_project_series(Path(row.source_tif), LATITUDE, LONGITUDE)
        for row in lineage.itertuples(index=False)
    }

    rows = []
    for label, start, end in (("full_safe", 1500.0, 1920.999999), ("bridge_1850_1920", 1850.0, 1920.999999)):
        selected = points[points.year_ce.between(start, end)].copy()
        years = selected.year_ce.to_numpy(float)
        proxy_slope = slope(years, selected.maat_c.to_numpy(float))
        member_slopes = []
        if len(years) >= 2:
            for member in PROJECT_MEMBERS:
                series = project[member]
                sampled = np.interp(years, series.year.to_numpy(float), series.temperature.to_numpy(float))
                member_slopes.append(slope(years, sampled))
        project_slope = float(np.median(member_slopes)) if member_slopes else None
        rows.append(
            {
                "window": label,
                "n": int(len(selected)),
                "first_year": float(years.min()) if len(years) else None,
                "last_year": float(years.max()) if len(years) else None,
                "ngamring_maat_trend_c_per_century": proxy_slope,
                "project_median_trend_at_proxy_years_c_per_century": project_slope,
                "ngamring_direction": direction(proxy_slope),
                "project_direction": direction(project_slope),
                "direction_agreement": direction(proxy_slope) == direction(project_slope) if proxy_slope is not None else None,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "ngamring_project_metrics.csv", index=False)

    prelock = points[points.year_ce <= 1920.999999].copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(prelock.year_ce, prelock.maat_c - prelock.maat_c.iloc[0], color="#111111", marker="o", lw=1.8, label="NOAA Ngamring MAAT")
    years = np.arange(int(np.ceil(prelock.year_ce.min())), 1921)
    member_values = []
    for member in PROJECT_MEMBERS:
        series = project[member]
        values = np.interp(years, series.year.to_numpy(float), series.temperature.to_numpy(float))
        base = np.interp(prelock.year_ce.min(), series.year.to_numpy(float), series.temperature.to_numpy(float))
        member_values.append(values - base)
        ax.plot(years, values - base, color="#5f8f9d", alpha=0.18, lw=0.8)
    ax.plot(years, np.median(np.vstack(member_values), axis=0), color="#bf3f36", lw=2.0, label="Project median")
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_xlabel("Year CE")
    ax.set_ylabel("Change from first safe point (degC)")
    ax.set_title("Official NOAA Ngamring MAAT versus project temperature")
    ax.grid(color="#e2e2e2", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ngamring_project_safe_curve.png", dpi=220)
    plt.close(fig)

    if source_hash_before != sha256(SOURCE):
        raise RuntimeError("source-changed")
    records = metrics.astype(object).where(pd.notna(metrics), None).to_dict(orient="records")
    summary = {
        "experiment": "temperature_noaa_ngamring_original_v1",
        "status": "complete",
        "source_identity": "NOAA_NCEI_WDS_study_38499_original_author_contribution",
        "dataset_doi": "10.25921/e3rg-vc84",
        "paper_doi": "10.1016/j.quascirev.2022.107690",
        "target": "brGDGT_based_mean_annual_air_temperature_with_possible_winter_season_bias",
        "records": records,
        **audit,
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
        "source": {"path": str(SOURCE), "sha256": source_hash_before},
        "download_url": "https://www.ncei.noaa.gov/pub/data/paleo/paleolimnology/asia/china/sun2022/sun2022-ngamring-co.txt",
        "member_lineage": {"path": str(LINEAGE), "sha256": sha256(LINEAGE)},
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# NOAA Ngamring original MAAT audit",
        "",
        "This is the official NOAA/NCEI author-contributed brGDGT MAAT series, not the 2024 secondary compilation.",
        "",
        "| Window | n | Ngamring trend | Project trend | Agreement |",
        "|---|---:|---:|---:|---|",
    ]
    for row in metrics.itertuples(index=False):
        if row.n < 2:
            lines.append(f"| {row.window} | {row.n} | insufficient | insufficient | not evaluated |")
        else:
            lines.append(
                f"| {row.window} | {row.n} | {row.ngamring_maat_trend_c_per_century:+.3f} | "
                f"{row.project_median_trend_at_proxy_years_c_per_century:+.3f} | "
                f"{'yes' if row.direction_agreement else 'no'} |"
            )
    lines.extend(
        [
            "",
            "Age was decoded before temperature; sealed rows were skipped before MAAT decoding. Production was not "
            "modified, EnKF and OGGM were not run, and 20CRv3 was not accessed.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
