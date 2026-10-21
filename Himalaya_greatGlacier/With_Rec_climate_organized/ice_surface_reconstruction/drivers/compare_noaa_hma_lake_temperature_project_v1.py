from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_noaa_hma_lake_temperature_recovery_v1"
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
PRELOCK_START = 1500.0
PRELOCK_END = 1920.999999
POSTLOCK_START = 1940.0
POSTLOCK_END = 1950.0

SOURCES = {
    "TiancaiLake.Zhang.2017.txt": {
        "site_id": "tiancai_lake",
        "latitude": 26.63,
        "longitude": 99.72,
        "temperature_column": "temperature",
        "season": "July",
        "proxy": "chironomid assemblage",
        "doi": "10.1016/j.quascirev.2017.04.008",
    },
    "HeihaiLake.Chang.2017.txt": {
        "site_id": "heihai_lake",
        "latitude": 27.36,
        "longitude": 100.07,
        "temperature_column": "temperature",
        "season": "July",
        "proxy": "chironomid assemblage",
        "doi": "10.1177/0959683617708456",
    },
    "qinghai2016uk37temp.txt": {
        "site_id": "qinghai_lake",
        "latitude": 36.8112,
        "longitude": 100.137,
        "temperature_column": "lst-uk37",
        "season": "summer lake-surface temperature",
        "proxy": "alkenone UK37",
        "doi": "10.1002/2015GL067317",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_noaa_source(path: Path, definition: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    locked_rows_skipped = 0
    outside_requested_rows_skipped = 0
    temperature_cells_decoded = 0

    with path.open("rb") as handle:
        for raw in handle:
            raw = raw.lstrip(b"\xef\xbb\xbf")
            if raw.startswith(b"#") or not raw.strip():
                continue
            tokens = raw.decode("utf-8", errors="strict").rstrip("\r\n").split("\t")
            if header is None:
                header = tokens
                continue
            if len(tokens) != len(header):
                raise RuntimeError(f"malformed-data-row:{path.name}")
            names = {name: index for index, name in enumerate(header)}
            age_name = "age" if "age" in names else "age_calBP"
            if age_name not in names or definition["temperature_column"] not in names:
                raise RuntimeError(f"missing-required-column:{path.name}:{header}")
            age_bp = float(tokens[names[age_name]])
            year_ce = 1950.0 - age_bp
            if LOCKED_START <= year_ce <= LOCKED_END:
                locked_rows_skipped += 1
                continue
            requested = (
                PRELOCK_START <= year_ce <= PRELOCK_END
                or POSTLOCK_START <= year_ce <= POSTLOCK_END
            )
            if not requested:
                outside_requested_rows_skipped += 1
                continue
            value_token = tokens[names[definition["temperature_column"]]].strip()
            temperature_cells_decoded += 1
            value = float(value_token) if value_token.lower() != "nan" else math.nan
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "site_id": definition["site_id"],
                    "latitude": definition["latitude"],
                    "longitude": definition["longitude"],
                    "year_ce": year_ce,
                    "age_bp": age_bp,
                    "temperature_c": value,
                    "season": definition["season"],
                    "proxy": definition["proxy"],
                    "source_file": str(path),
                    "source_sha256": sha256(path),
                    "doi": definition["doi"],
                }
            )
    if header is None or not rows:
        raise RuntimeError(f"no-safe-data:{path.name}")
    frame = pd.DataFrame(rows).sort_values("year_ce").reset_index(drop=True)
    audit = {
        "source_file": str(path),
        "source_sha256": sha256(path),
        "locked_rows_skipped_before_temperature_decode": locked_rows_skipped,
        "outside_requested_rows_skipped_before_temperature_decode": outside_requested_rows_skipped,
        "safe_temperature_cells_decoded": temperature_cells_decoded,
        "safe_finite_points_retained": int(len(frame)),
    }
    return frame, audit


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
    y0 = int(np.floor(fy))
    x0 = int(np.floor(fx))
    if y0 < 0 or y0 >= len(lat) - 1 or x0 < 0 or x0 >= len(lon) - 1:
        raise RuntimeError("lake-outside-project-grid")
    wy = fy - y0
    wx = fx - x0
    corners = np.stack(
        [data[:, y0, x0], data[:, y0, x0 + 1], data[:, y0 + 1, x0], data[:, y0 + 1, x0 + 1]],
        axis=0,
    )
    weights = np.asarray([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
    valid = np.isfinite(corners)
    denominator = np.sum(weights[:, None] * valid, axis=0)
    numerator = np.nansum(corners * weights[:, None], axis=0)
    return numerator / np.where(denominator > 0, denominator, np.nan)


def read_project_series(path: Path, latitude: float, longitude: float) -> pd.DataFrame:
    years = np.arange(1500, 1921, dtype=int)
    indices = years - 1000
    with tifffile.TiffFile(path) as source:
        series = source.series[0]
        if series.axes != "YXS" or series.shape[2] < int(indices.max()) + 1:
            raise RuntimeError(f"unexpected-project-tiff-layout:{path}")
        page = source.pages[0]
        scale = tuple(float(value) for value in page.tags["ModelPixelScaleTag"].value)
        tiepoint = tuple(float(value) for value in page.tags["ModelTiepointTag"].value)
        nodata = float(page.tags["GDAL_NODATA"].value)
        complete = source.asarray()
    raw = np.moveaxis(complete[:, :, indices], 2, 0).astype(float)
    raw[np.isclose(raw, nodata) | (raw < -9000)] = np.nan
    height, width = raw.shape[1:]
    lon = tiepoint[3] + (np.arange(width, dtype=float) + 0.5) * scale[0]
    lat = tiepoint[4] - (np.arange(height, dtype=float) + 0.5) * scale[1]
    values = bilinear_point(raw, lat, lon, latitude, longitude)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"project-series-incomplete:{path}")
    return pd.DataFrame({"year": years, "temperature": values})


def interpolate_series(series: pd.DataFrame, year: float) -> float:
    return float(np.interp(year, series.year.to_numpy(float), series.temperature.to_numpy(float)))


def sign_name(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "warming"
    if value < -tolerance:
        return "cooling"
    return "neutral"


def plot_pairs(
    points: pd.DataFrame,
    project: dict[str, dict[str, pd.DataFrame]],
    pairs: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    labels = {"tiancai_lake": "Tiancai Lake", "heihai_lake": "Heihai Lake", "qinghai_lake": "Qinghai Lake"}
    colors = plt.cm.tab10(np.linspace(0, 0.75, len(PROJECT_MEMBERS)))
    for ax, pair in zip(axes, pairs.itertuples(index=False)):
        site = pair.site_id
        start, end = pair.start_year_ce, pair.end_year_ce
        site_points = points[(points.site_id == site) & points.year_ce.between(start, end)].copy()
        proxy_base = float(site_points.iloc[0].temperature_c)
        ax.plot(
            site_points.year_ce,
            site_points.temperature_c - proxy_base,
            color="#111111",
            marker="o",
            linewidth=2.2,
            label="Lake proxy",
            zorder=5,
        )
        for color, member in zip(colors, PROJECT_MEMBERS):
            series = project[site][member]
            segment = series[series.year.between(int(np.floor(start)), int(np.ceil(end)))].copy()
            base = interpolate_series(series, start)
            ax.plot(segment.year, segment.temperature - base, color=color, alpha=0.65, linewidth=1.0, label=member)
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.set_title(labels[site])
        ax.set_xlabel("Year CE")
        ax.grid(color="#e1e1e1", linewidth=0.6)
    axes[0].set_ylabel("Temperature change from pair start (degC)")
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=7, frameon=False, fontsize=8)
    fig.suptitle("Safe pre-1921 lake-temperature pairs vs project members", y=0.98)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    source_hashes_before = {name: sha256(OUTPUT / name) for name in SOURCES}
    point_frames = []
    audits = []
    for name, definition in SOURCES.items():
        frame, audit = parse_noaa_source(OUTPUT / name, definition)
        point_frames.append(frame)
        audits.append(audit)
    points = pd.concat(point_frames, ignore_index=True).sort_values(["site_id", "year_ce"])
    points.to_csv(OUTPUT / "lake_temperature_safe_points.csv", index=False)

    lineage = pd.read_csv(LINEAGE)
    lineage = lineage[(lineage.variable == "tas") & lineage.member.isin(PROJECT_MEMBERS)].copy()
    if len(lineage) != len(PROJECT_MEMBERS):
        raise RuntimeError("project-member-lineage-incomplete")

    project: dict[str, dict[str, pd.DataFrame]] = {}
    source_manifest = []
    for definition in SOURCES.values():
        site = definition["site_id"]
        project[site] = {}
        for row in lineage.sort_values("member").itertuples(index=False):
            path = Path(row.source_tif)
            project[site][row.member] = read_project_series(path, definition["latitude"], definition["longitude"])
            source_manifest.append(
                {
                    "site_id": site,
                    "member": row.member,
                    "path": str(path),
                    "archived_sha256": row.source_tif_sha256,
                    "years_read": [1500, 1920],
                }
            )

    pair_rows = []
    for site, group in points[points.year_ce <= PRELOCK_END].groupby("site_id"):
        ordered = group.sort_values("year_ce")
        if len(ordered) < 2:
            raise RuntimeError(f"insufficient-prelock-points:{site}")
        pair = ordered.iloc[-2:]
        start = float(pair.iloc[0].year_ce)
        end = float(pair.iloc[1].year_ce)
        proxy_change = float(pair.iloc[1].temperature_c - pair.iloc[0].temperature_c)
        member_changes = {
            member: interpolate_series(project[site][member], end) - interpolate_series(project[site][member], start)
            for member in PROJECT_MEMBERS
        }
        project_median = float(np.median(list(member_changes.values())))
        pair_rows.append(
            {
                "site_id": site,
                "start_year_ce": start,
                "end_year_ce": end,
                "interval_years": end - start,
                "proxy_change_c": proxy_change,
                "proxy_change_c_per_century": proxy_change / (end - start) * 100.0,
                **{f"project_{member}_change_c": value for member, value in member_changes.items()},
                "project_six_member_median_change_c": project_median,
                "project_six_member_median_change_c_per_century": project_median / (end - start) * 100.0,
                "proxy_direction": sign_name(proxy_change),
                "project_direction": sign_name(project_median),
                "direction_agreement": sign_name(proxy_change) == sign_name(project_median),
                "target_compatibility": "seasonal lake temperature vs annual project temperature; direction diagnostic only",
            }
        )
    pairs = pd.DataFrame(pair_rows).sort_values("site_id")
    pairs.to_csv(OUTPUT / "latest_safe_prelock_pair_comparison.csv", index=False)
    plot_pairs(points, project, pairs, OUTPUT / "lake_proxy_project_safe_pair_curves.png")

    source_hashes_after = {name: sha256(OUTPUT / name) for name in SOURCES}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("downloaded-source-file-changed")

    summary = {
        "experiment": "temperature_noaa_hma_lake_temperature_recovery_v1",
        "status": "complete",
        "safe_point_count": int(len(points)),
        "site_count": int(points.site_id.nunique()),
        "pair_rows": pairs.to_dict(orient="records"),
        "pair_direction_agreement_count": int(pairs.direction_agreement.sum()),
        "pair_direction_disagreement_count": int((~pairs.direction_agreement).sum()),
        "locked_rows_skipped_before_temperature_decode": int(
            sum(item["locked_rows_skipped_before_temperature_decode"] for item in audits)
        ),
        "locked_period_used_for_selection_or_metrics": False,
        "no_interpolation_across_locked_period": True,
        "source_files_unchanged": True,
        "production_files_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "interpretive_limits": [
            "Tiancai and Heihai reconstruct July temperature; Qinghai reconstructs summer lake-surface temperature.",
            "Project members are annual air temperature, so only low-frequency direction and rough amplitude are compared.",
            "The latest safe pair differs by site because the records have different temporal resolution.",
            "No metric joins a pre-1921 point to a post-1939 point.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    manifest = {
        "official_archive": "NOAA/NCEI Paleoclimatology",
        "downloaded_sources": [
            {"path": str(OUTPUT / name), "sha256": source_hashes_before[name], **definition}
            for name, definition in SOURCES.items()
        ],
        "project_member_sources": source_manifest,
        "member_lineage": {"path": str(LINEAGE), "sha256": sha256(LINEAGE)},
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    table = "\n".join(
        f"| {row.site_id} | {row.start_year_ce:.1f}-{row.end_year_ce:.1f} | {row.proxy_change_c:+.3f} | "
        f"{row.project_six_member_median_change_c:+.3f} | {'yes' if row.direction_agreement else 'no'} |"
        for row in pairs.itertuples(index=False)
    )
    report = f"""# NOAA HMA lake-temperature comparison

## Result

Three official NOAA quantitative lake-temperature records were recovered and compared with the six project members at matched coordinates. All metrics use only pairs wholly before 1921; no pair or interpolation crosses the sealed 1921-1939 interval.

| Site | Latest safe pair | Proxy change (degC) | Project median change (degC) | Direction agrees |
|---|---:|---:|---:|---|
{table}

These records are physically independent of the project's tree-ring PSM network. Tiancai and Heihai reconstruct July temperature, while Qinghai reconstructs summer lake-surface temperature; therefore this is a direction and low-frequency-amplitude diagnostic, not an annual-temperature RMSE benchmark.

## Safety

- Locked rows were skipped after decoding age but before decoding temperature.
- No metric joins pre-lock and post-lock points.
- Original and production files were not modified.
- EnKF and OGGM were not run.
- 20CRv3 was not accessed (`prohibited_source_accessed=false`).
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
