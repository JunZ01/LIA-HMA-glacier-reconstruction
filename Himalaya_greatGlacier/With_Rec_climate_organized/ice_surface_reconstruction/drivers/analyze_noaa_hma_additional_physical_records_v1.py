from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_noaa_hma_additional_physical_records_v1"
)
LINEAGE = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v1"
    / "round_00_production_reproduction/member_lineage.csv"
)
DRIVER = Path(__file__).resolve()
PROJECT_MEMBERS = ("AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9")
LOCKED_START = 1921.0
LOCKED_END = 1939.999999


SOURCES = {
    "AwengCo.Li.2017.txt": {
        "site_id": "aweng_co",
        "latitude": 32.75,
        "longitude": 81.67,
        "kind": "age_bp_temperature",
        "variable": "temperature",
        "native_target": "annual_mean_air_temperature_degC",
        "doi": "10.1016/j.orggeochem.2017.07.017",
    },
    "guliya2018core1annual.txt": {
        "site_id": "guliya_ice_core",
        "latitude": 35.23,
        "longitude": 81.46,
        "kind": "annual_isotope",
        "variable": "d18O",
        "native_target": "ice_core_d18O_temperature_direction_proxy",
        "doi": "10.1029/2018GL078455",
    },
    "puruogangri2006-core2-10yr-noaa.txt": {
        "site_id": "puruogangri_ice_core",
        "latitude": 33.9167,
        "longitude": 89.0833,
        "kind": "decadal_isotope",
        "variable": "d18Osmow",
        "native_target": "decadal_ice_core_d18O_summer_temperature_direction_proxy",
        "doi": "10.3189/172756406781812357",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source(path: Path, definition: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    header: list[str] | None = None
    rows = []
    locked_rows_skipped = 0
    value_cells_decoded = 0
    for raw in path.open("rb"):
        raw = raw.lstrip(b"\xef\xbb\xbf")
        if raw.startswith(b"#") or not raw.strip():
            continue
        tokens = raw.decode("utf-8", errors="strict").rstrip("\r\n").split("\t")
        if header is None:
            header = tokens
            continue
        if len(tokens) != len(header):
            raise RuntimeError(f"malformed-row:{path.name}")
        names = {name: index for index, name in enumerate(header)}
        kind = definition["kind"]
        if kind == "age_bp_temperature":
            start_year = end_year = 1950.0 - float(tokens[names["age"]])
        elif kind == "annual_isotope":
            start_year = end_year = float(tokens[names["Age_CE"]])
        elif kind == "decadal_isotope":
            start_year = float(tokens[names["age_bot_CE"]])
            end_year = float(tokens[names["age_top_CE"]])
        else:
            raise RuntimeError(f"unknown-source-kind:{kind}")
        if start_year <= LOCKED_END and end_year >= LOCKED_START:
            locked_rows_skipped += 1
            continue
        requested = (
            (1500.0 <= end_year <= 1920.999999)
            or (1940.0 <= start_year <= 1950.0)
        )
        if not requested:
            continue
        token = tokens[names[definition["variable"]]].strip()
        value_cells_decoded += 1
        if token.lower() in {"nan", "na", ""}:
            continue
        value = float(token)
        rows.append(
            {
                "site_id": definition["site_id"],
                "latitude": definition["latitude"],
                "longitude": definition["longitude"],
                "start_year_ce": start_year,
                "end_year_ce": end_year,
                "mid_year_ce": (start_year + end_year) / 2.0,
                "proxy_value": value,
                "proxy_variable": definition["variable"],
                "native_target": definition["native_target"],
                "source_file": str(path),
                "source_sha256": sha256(path),
                "doi": definition["doi"],
            }
        )
    if header is None or not rows:
        raise RuntimeError(f"no-safe-data:{path.name}")
    return pd.DataFrame(rows).sort_values("mid_year_ce"), {
        "source_file": str(path),
        "source_sha256": sha256(path),
        "locked_or_cross_locked_rows_skipped_before_value_decode": locked_rows_skipped,
        "safe_value_cells_decoded": value_cells_decoded,
        "safe_finite_points": len(rows),
    }


def bilinear_point(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, y: float, x: float) -> np.ndarray:
    data = np.asarray(field, dtype=float)
    if lat[1] < lat[0]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    if lon[1] < lon[0]:
        lon = lon[::-1]
        data = data[:, :, ::-1]
    fy = (y - lat[0]) / (lat[1] - lat[0])
    fx = (x - lon[0]) / (lon[1] - lon[0])
    y0, x0 = int(np.floor(fy)), int(np.floor(fx))
    if y0 < 0 or y0 >= len(lat) - 1 or x0 < 0 or x0 >= len(lon) - 1:
        raise RuntimeError("site-outside-project-grid")
    wy, wx = fy - y0, fx - x0
    corners = np.stack(
        [data[:, y0, x0], data[:, y0, x0 + 1], data[:, y0 + 1, x0], data[:, y0 + 1, x0 + 1]],
        axis=0,
    )
    weights = np.asarray([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
    valid = np.isfinite(corners)
    denominator = np.sum(weights[:, None] * valid, axis=0)
    numerator = np.nansum(weights[:, None] * corners, axis=0)
    return numerator / np.where(denominator > 0, denominator, np.nan)


def project_series(path: Path, latitude: float, longitude: float) -> pd.DataFrame:
    years = np.arange(1500, 1921, dtype=int)
    indices = years - 1000
    with tifffile.TiffFile(path) as source:
        page = source.pages[0]
        scale = tuple(float(v) for v in page.tags["ModelPixelScaleTag"].value)
        tie = tuple(float(v) for v in page.tags["ModelTiepointTag"].value)
        nodata = float(page.tags["GDAL_NODATA"].value)
        complete = source.asarray()
    raw = np.moveaxis(complete[:, :, indices], 2, 0).astype(float)
    raw[np.isclose(raw, nodata) | (raw < -9000)] = np.nan
    h, w = raw.shape[1:]
    lon = tie[3] + (np.arange(w, dtype=float) + 0.5) * scale[0]
    lat = tie[4] - (np.arange(h, dtype=float) + 0.5) * scale[1]
    values = bilinear_point(raw, lat, lon, latitude, longitude)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"project-series-incomplete:{path.name}")
    return pd.DataFrame({"year": years, "temperature_c": values})


def slope_per_century(years: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(np.asarray(years, float), np.asarray(values, float), 1)[0] * 100.0)


def direction(value: float) -> str:
    return "warming" if value > 0 else "cooling" if value < 0 else "neutral"


def standardize(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std(ddof=0)


def main() -> None:
    source_hashes_before = {name: sha256(OUTPUT / name) for name in SOURCES}
    frames, audits = [], []
    for name, definition in SOURCES.items():
        frame, audit = parse_source(OUTPUT / name, definition)
        frames.append(frame)
        audits.append(audit)
    points = pd.concat(frames, ignore_index=True).sort_values(["site_id", "mid_year_ce"])
    points.to_csv(OUTPUT / "additional_physical_safe_points.csv", index=False)

    lineage = pd.read_csv(LINEAGE)
    lineage = lineage[(lineage.variable == "tas") & lineage.member.isin(PROJECT_MEMBERS)].copy()
    project: dict[str, dict[str, pd.DataFrame]] = {}
    for definition in SOURCES.values():
        site = definition["site_id"]
        project[site] = {
            row.member: project_series(Path(row.source_tif), definition["latitude"], definition["longitude"])
            for row in lineage.itertuples(index=False)
        }

    metric_rows = []
    # Quantitative annual MAAT pair.
    aweng = points[(points.site_id == "aweng_co") & (points.mid_year_ce <= 1920)].tail(2)
    start, end = aweng.iloc[0], aweng.iloc[1]
    aweng_proxy_change = float(end.proxy_value - start.proxy_value)
    member_changes = {
        member: float(np.interp(end.mid_year_ce, series.year, series.temperature_c) - np.interp(start.mid_year_ce, series.year, series.temperature_c))
        for member, series in project["aweng_co"].items()
    }
    project_median = float(np.median(list(member_changes.values())))
    metric_rows.append(
        {
            "site_id": "aweng_co",
            "interval": f"{start.mid_year_ce:.1f}-{end.mid_year_ce:.1f}",
            "proxy_metric": "temperature_change_c",
            "proxy_value": aweng_proxy_change,
            "project_metric": "temperature_change_c",
            "project_six_member_median": project_median,
            "proxy_direction": direction(aweng_proxy_change),
            "project_direction": direction(project_median),
            "direction_agreement": direction(aweng_proxy_change) == direction(project_median),
            "amplitude_comparable": True,
        }
    )

    # Annual Guliya isotope direction over the direct historical interval.
    guliya = points[(points.site_id == "guliya_ice_core") & points.mid_year_ce.between(1850, 1920)].copy()
    proxy_trend = slope_per_century(guliya.mid_year_ce, guliya.proxy_value)
    project_trends = []
    for series in project["guliya_ice_core"].values():
        selected = series[series.year.between(1850, 1920)]
        project_trends.append(slope_per_century(selected.year, selected.temperature_c))
    project_median = float(np.median(project_trends))
    metric_rows.append(
        {
            "site_id": "guliya_ice_core",
            "interval": "1850-1920",
            "proxy_metric": "d18O_trend_per_century_per_mil",
            "proxy_value": proxy_trend,
            "project_metric": "temperature_trend_c_per_century",
            "project_six_member_median": project_median,
            "proxy_direction": direction(proxy_trend),
            "project_direction": direction(project_median),
            "direction_agreement": direction(proxy_trend) == direction(project_median),
            "amplitude_comparable": False,
        }
    )

    # Puruogangri uses only complete safe decades from 1850-1859 to 1910-1919.
    puruo = points[
        (points.site_id == "puruogangri_ice_core")
        & (points.start_year_ce >= 1850)
        & (points.end_year_ce <= 1919)
    ].copy()
    proxy_trend = slope_per_century(puruo.mid_year_ce, puruo.proxy_value)
    project_trends = []
    for series in project["puruogangri_ice_core"].values():
        decade_means = []
        for row in puruo.itertuples(index=False):
            decade_means.append(series[series.year.between(row.start_year_ce, row.end_year_ce)].temperature_c.mean())
        project_trends.append(slope_per_century(puruo.mid_year_ce, np.asarray(decade_means)))
    project_median = float(np.median(project_trends))
    metric_rows.append(
        {
            "site_id": "puruogangri_ice_core",
            "interval": "1850-1859_to_1910-1919",
            "proxy_metric": "decadal_d18O_trend_per_century_per_mil",
            "proxy_value": proxy_trend,
            "project_metric": "decadal_temperature_trend_c_per_century",
            "project_six_member_median": project_median,
            "proxy_direction": direction(proxy_trend),
            "project_direction": direction(project_median),
            "direction_agreement": direction(proxy_trend) == direction(project_median),
            "amplitude_comparable": False,
        }
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "additional_physical_project_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 0.75, len(PROJECT_MEMBERS)))
    # Aweng quantitative pair and project change curves.
    for color, (member, series) in zip(colors, project["aweng_co"].items()):
        segment = series[series.year.between(1758, 1834)]
        base = float(np.interp(1758, series.year, series.temperature_c))
        axes[0].plot(segment.year, segment.temperature_c - base, color=color, alpha=0.7, lw=1.0, label=member)
    axes[0].plot(aweng.mid_year_ce, aweng.proxy_value - aweng.iloc[0].proxy_value, color="black", marker="o", lw=2.2, label="Aweng MAAT")
    axes[0].set_title("Aweng Co: 1758-1834")
    axes[0].set_ylabel("Change from interval start (degC)")
    axes[0].legend(handles=[Line2D([0], [0], color="black", marker="o", lw=2.2, label="Aweng MAAT")], frameon=False, loc="upper left", fontsize=8)

    # Guliya standardized 11-year low-frequency curves.
    gx = guliya.set_index(guliya.mid_year_ce.astype(int)).proxy_value.sort_index().rolling(11, center=True, min_periods=7).mean().dropna()
    axes[1].plot(gx.index, standardize(gx), color="black", lw=2.2, label="Guliya d18O")
    for color, (member, series) in zip(colors, project["guliya_ice_core"].items()):
        segment = series[series.year.between(1850, 1920)].set_index("year").temperature_c.rolling(11, center=True, min_periods=7).mean().dropna()
        axes[1].plot(segment.index, standardize(segment), color=color, alpha=0.65, lw=1.0)
    axes[1].set_title("Guliya: 1850-1920")
    axes[1].set_ylabel("Within-series standardized low frequency")
    axes[1].legend(handles=[Line2D([0], [0], color="black", lw=2.2, label="Guliya d18O")], frameon=False, loc="upper left", fontsize=8)

    # Puruogangri decadal standardized comparison.
    axes[2].plot(puruo.mid_year_ce, standardize(puruo.proxy_value), color="black", marker="o", lw=2.2, label="Puruogangri d18O")
    for color, (member, series) in zip(colors, project["puruogangri_ice_core"].items()):
        decade_means = [series[series.year.between(row.start_year_ce, row.end_year_ce)].temperature_c.mean() for row in puruo.itertuples(index=False)]
        axes[2].plot(puruo.mid_year_ce, standardize(pd.Series(decade_means)), color=color, alpha=0.65, lw=1.0)
    axes[2].set_title("Puruogangri: safe decades")
    axes[2].set_ylabel("Within-series standardized decadal value")
    axes[2].legend(handles=[Line2D([0], [0], color="black", marker="o", lw=2.2, label="Puruogangri d18O")], frameon=False, loc="upper left", fontsize=8)
    for ax in axes:
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set_xlabel("Year CE")
        ax.grid(color="#e2e2e2", lw=0.6)
    member_handles = [
        Line2D([0], [0], color=color, lw=1.5, label=member)
        for color, member in zip(colors, PROJECT_MEMBERS)
    ]
    fig.legend(member_handles, PROJECT_MEMBERS, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Additional independent physical records vs project members", y=0.98)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(OUTPUT / "additional_physical_project_curves.png", dpi=220)
    plt.close(fig)

    source_hashes_after = {name: sha256(OUTPUT / name) for name in SOURCES}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("source-file-changed")
    summary = {
        "experiment": "temperature_noaa_hma_additional_physical_records_v1",
        "status": "complete",
        "metric_rows": metrics.to_dict(orient="records"),
        "direction_agreement_count": int(metrics.direction_agreement.sum()),
        "direction_conflict_count": int((~metrics.direction_agreement).sum()),
        "locked_or_cross_locked_rows_skipped_before_value_decode": int(sum(item["locked_or_cross_locked_rows_skipped_before_value_decode"] for item in audits)),
        "amplitude_rule": "Only Aweng quantitative MAAT is amplitude-comparable; isotope records are direction-only.",
        "source_files_unchanged": True,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "official_archive": "NOAA/NCEI Paleoclimatology",
        "sources": [{"path": str(OUTPUT / name), "sha256": source_hashes_before[name], **definition} for name, definition in SOURCES.items()],
        "lineage": {"path": str(LINEAGE), "sha256": sha256(LINEAGE)},
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_rows = "\n".join(
        f"| {row.site_id} | {row.interval} | {row.proxy_direction} | {row.project_direction} | {'yes' if row.direction_agreement else 'no'} |"
        for row in metrics.itertuples(index=False)
    )
    report = f"""# Additional NOAA HMA physical temperature evidence

| Record | Safe interval | Physical direction | Project direction | Agrees |
|---|---|---|---|---|
{report_rows}

Aweng Co is a quantitative annual-MAAT record and permits rough amplitude comparison. Guliya and Puruogangri isotope records are admitted only as low-frequency direction constraints; their per-mil amplitudes are not converted to degC.

Rows overlapping 1921-1939 were skipped before isotope or temperature values were decoded. No production, EnKF, or OGGM file was modified or run. 20CRv3 was not accessed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
