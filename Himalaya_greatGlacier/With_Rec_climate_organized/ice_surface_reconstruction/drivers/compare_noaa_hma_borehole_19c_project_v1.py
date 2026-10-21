from __future__ import annotations

import hashlib
import json
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
    / "temperature_noaa_hma_borehole_19c_recovery_v1"
)
LINEAGE = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v1"
    / "round_00_production_reproduction/member_lineage.csv"
)
DRIVER = Path(__file__).resolve()

PROJECT_MEMBERS = ("AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9")
SAFE_CENTURIES = ("16th", "17th", "18th", "19th")
PROJECT_START_YEAR = 1800
PROJECT_END_YEAR = 1900
LOCKED_START = 1921
LOCKED_END = 1939

CLUSTERS = {
    "CN-XZ-naqu903": "central_naqu",
    "CN-PX202": "eastern_px202",
    "CN-m2": "yunnan_cluster",
    "CN-m4": "yunnan_cluster",
    "CN-m8": "yunnan_cluster",
    "CN-m9": "yunnan_cluster",
    "CN-QD2": "qaidam_north_margin_cluster",
    "CN-QD22": "qaidam_north_margin_cluster",
    "CN-QD26": "qaidam_north_margin_cluster",
    "CN-QD28": "qaidam_north_margin_cluster",
    "CN-QD6": "qaidam_north_margin_cluster",
    "CN-QD637": "qaidam_north_margin_cluster",
    "CN-QDa1": "qaidam_north_margin_cluster",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_borehole_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    site_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    rows: list[dict[str, Any]] = []
    skipped_cross_lock_rows = 0
    lines_seen = 0

    site_pattern = re.compile(r"^#\s*Site_Name:\s*(\S+)\s*$")
    lat_pattern = re.compile(r"^#\s*Northernmost_Latitude:\s*([-+0-9.eE]+)\s*$")
    lon_pattern = re.compile(r"^#\s*Easternmost_Longitude:\s*([-+0-9.eE]+)\s*$")
    safe_pattern = re.compile(r"^#\s*(16th|17th|18th|19th)\s+([-+0-9.eE]+)\s*$")

    # NOAA's legacy files use a single-byte degree symbol in descriptive text.
    with path.open("r", encoding="latin-1", errors="strict") as handle:
        for line in handle:
            lines_seen += 1
            stripped = line.strip()
            # The twentieth-century aggregate intersects the sealed period. Skip the
            # complete row before tokenizing or decoding its numeric field.
            if re.match(r"^#\s*20th(?:\s|$)", stripped):
                skipped_cross_lock_rows += 1
                continue
            match = site_pattern.match(stripped)
            if match:
                site_id = match.group(1)
                continue
            match = lat_pattern.match(stripped)
            if match:
                latitude = float(match.group(1))
                continue
            match = lon_pattern.match(stripped)
            if match:
                longitude = float(match.group(1))
                continue
            match = safe_pattern.match(stripped)
            if match:
                rows.append(
                    {
                        "century": match.group(1),
                        "estimated_gst_change_c": float(match.group(2)),
                    }
                )

    if site_id is None or latitude is None or longitude is None:
        raise RuntimeError(f"missing-site-metadata:{path.name}")
    if site_id not in CLUSTERS:
        raise RuntimeError(f"unmapped-site:{site_id}")
    if {row["century"] for row in rows} != set(SAFE_CENTURIES):
        raise RuntimeError(f"incomplete-safe-century-series:{site_id}")
    if skipped_cross_lock_rows != 1:
        raise RuntimeError(f"unexpected-cross-lock-row-count:{site_id}:{skipped_cross_lock_rows}")

    metadata = {
        "site_id": site_id,
        "latitude": latitude,
        "longitude": longitude,
        "cluster": CLUSTERS[site_id],
        "source_file": str(path),
        "source_sha256": sha256(path),
        "physical_lines_seen": lines_seen,
        "cross_locked_century_rows_skipped_before_value_decode": skipped_cross_lock_rows,
    }
    for row in rows:
        row.update(metadata)
    return metadata, rows


def parse_updated_rate_file(path: Path) -> tuple[str, dict[str, float], int]:
    site_id: str | None = None
    values: dict[str, float] = {}
    skipped_cross_lock_rows = 0
    site_pattern = re.compile(r"^#\s*Site_Name:\s*(\S+)\s*$")
    safe_pattern = re.compile(r"^#\s*(16th|17th|18th|19th)\s+([-+0-9.eE]+)\s*$")
    with path.open("r", encoding="latin-1", errors="strict") as handle:
        for line in handle:
            stripped = line.strip()
            if re.match(r"^#\s*20th(?:\s|$)", stripped):
                skipped_cross_lock_rows += 1
                continue
            match = site_pattern.match(stripped)
            if match:
                site_id = match.group(1)
                continue
            match = safe_pattern.match(stripped)
            if match:
                values[match.group(1)] = float(match.group(2))
    if site_id is None or set(values) != set(SAFE_CENTURIES):
        raise RuntimeError(f"updated-rate-file-incomplete:{path.name}")
    if skipped_cross_lock_rows != 1:
        raise RuntimeError(f"updated-rate-cross-lock-row-count:{path.name}")
    return site_id, values, skipped_cross_lock_rows


def bilinear_points(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    point_lat: np.ndarray,
    point_lon: np.ndarray,
) -> np.ndarray:
    data = np.asarray(field, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat[1] < lat[0]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    if lon[1] < lon[0]:
        lon = lon[::-1]
        data = data[:, :, ::-1]
    fy = (point_lat - lat[0]) / (lat[1] - lat[0])
    fx = (point_lon - lon[0]) / (lon[1] - lon[0])
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    if (
        np.any(y0 < 0)
        or np.any(y0 >= len(lat) - 1)
        or np.any(x0 < 0)
        or np.any(x0 >= len(lon) - 1)
    ):
        raise RuntimeError("borehole-outside-project-grid")
    wy = fy - y0
    wx = fx - x0
    corners = np.stack(
        [
            data[:, y0, x0],
            data[:, y0, x0 + 1],
            data[:, y0 + 1, x0],
            data[:, y0 + 1, x0 + 1],
        ],
        axis=0,
    )
    weights = np.stack(
        [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx],
        axis=0,
    )
    valid = np.isfinite(corners)
    denominator = np.sum(weights[:, None, :] * valid, axis=0)
    numerator = np.nansum(corners * weights[:, None, :], axis=0)
    return numerator / np.where(denominator > 0, denominator, np.nan)


def project_changes(path: Path, sites: pd.DataFrame) -> np.ndarray:
    sample_indices = [PROJECT_START_YEAR - 1000, PROJECT_END_YEAR - 1000]
    with tifffile.TiffFile(path) as source:
        series = source.series[0]
        if series.axes != "YXS" or series.shape[2] < max(sample_indices) + 1:
            raise RuntimeError(f"unexpected-project-tiff-layout:{path}:{series.axes}:{series.shape}")
        page = source.pages[0]
        scale = tuple(float(value) for value in page.tags["ModelPixelScaleTag"].value)
        tiepoint = tuple(float(value) for value in page.tags["ModelTiepointTag"].value)
        geokeys = tuple(int(value) for value in page.tags["GeoKeyDirectoryTag"].value)
        if 4326 not in geokeys:
            raise RuntimeError(f"unexpected-project-crs:{path}")
        nodata = float(page.tags["GDAL_NODATA"].value)
        complete = source.asarray()
    raw = np.moveaxis(complete[:, :, sample_indices], 2, 0).astype(float)
    raw[np.isclose(raw, nodata) | (raw < -9000)] = np.nan
    height, width = raw.shape[1:]
    lon = tiepoint[3] + (np.arange(width, dtype=float) + 0.5) * scale[0]
    lat = tiepoint[4] - (np.arange(height, dtype=float) + 0.5) * scale[1]
    values = bilinear_points(
        raw,
        lat,
        lon,
        sites.latitude.to_numpy(float),
        sites.longitude.to_numpy(float),
    )
    changes = values[1] - values[0]
    return changes


def sign_name(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "warming"
    if value < -tolerance:
        return "cooling"
    return "neutral"


def write_figure(cluster: pd.DataFrame, output: Path) -> None:
    order = [
        "central_naqu",
        "eastern_px202",
        "yunnan_cluster",
        "qaidam_north_margin_cluster",
    ]
    frame = cluster.set_index("cluster").loc[order].reset_index()
    x = np.arange(len(frame), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    ax.bar(
        x - width / 2,
        frame.borehole_19th_change_median_c,
        width,
        label="NOAA borehole GST inversion",
        color="#2f6f8f",
    )
    ax.bar(
        x + width / 2,
        frame.project_six_member_median_change_c,
        width,
        label="Project six-member median",
        color="#d17a22",
    )
    member_low = frame.project_member_min_change_c.to_numpy(float)
    member_high = frame.project_member_max_change_c.to_numpy(float)
    median = frame.project_six_member_median_change_c.to_numpy(float)
    ax.errorbar(
        x + width / 2,
        median,
        yerr=np.vstack([median - member_low, member_high - median]),
        fmt="none",
        ecolor="#643b16",
        capsize=4,
        linewidth=1.3,
        label="Project member range",
    )
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Central Naqu", "Eastern PX202", "Yunnan", "Qaidam margin"],
        rotation=0,
    )
    ax.set_ylabel("1800-1900 temperature change (degC)")
    ax.set_title("Independent borehole evidence vs project reconstruction")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_files = sorted(OUTPUT.glob("huang-2013-CN-*.txt"))
    updated_source_files = sorted(OUTPUT.glob("huang2016-CN-*.txt"))
    if len(source_files) != len(CLUSTERS):
        raise RuntimeError(f"expected-{len(CLUSTERS)}-2013-files-found-{len(source_files)}")

    all_downloaded_sources = source_files + updated_source_files
    source_hashes_before = {str(path): sha256(path) for path in all_downloaded_sources}
    metadata_rows: list[dict[str, Any]] = []
    century_rows: list[dict[str, Any]] = []
    for path in source_files:
        metadata, rows = parse_borehole_file(path)
        metadata_rows.append(metadata)
        century_rows.extend(rows)

    sites = pd.DataFrame(metadata_rows).sort_values("site_id").reset_index(drop=True)
    centuries = pd.DataFrame(century_rows).sort_values(["site_id", "century"])
    centuries.to_csv(OUTPUT / "borehole_16th_19th_safe_trends.csv", index=False)

    crosswalk_rows = []
    updated_cross_lock_rows_skipped = 0
    old_lookup = centuries.set_index(["site_id", "century"]).estimated_gst_change_c
    for path in updated_source_files:
        site_id, values, skipped = parse_updated_rate_file(path)
        updated_cross_lock_rows_skipped += skipped
        for century, updated_value in values.items():
            old_value = float(old_lookup.loc[(site_id, century)])
            crosswalk_rows.append(
                {
                    "site_id": site_id,
                    "century": century,
                    "old_estimated_gst_change_c": old_value,
                    "updated_rate_k_per_100a": updated_value,
                    "absolute_difference": abs(old_value - updated_value),
                    "updated_source_file": str(path),
                    "updated_source_sha256": source_hashes_before[str(path)],
                }
            )
    crosswalk = pd.DataFrame(crosswalk_rows).sort_values(["site_id", "century"])
    if crosswalk.empty or float(crosswalk.absolute_difference.max()) > 1e-12:
        raise RuntimeError("old-updated-safe-coefficient-crosswalk-failed")
    crosswalk.to_csv(OUTPUT / "borehole_2013_2016_safe_coefficient_crosswalk.csv", index=False)

    lineage = pd.read_csv(LINEAGE)
    lineage = lineage[(lineage.variable == "tas") & lineage.member.isin(PROJECT_MEMBERS)].copy()
    if set(lineage.member) != set(PROJECT_MEMBERS) or len(lineage) != len(PROJECT_MEMBERS):
        raise RuntimeError("project-member-lineage-incomplete")

    comparison = sites[
        ["site_id", "cluster", "latitude", "longitude", "source_file", "source_sha256"]
    ].copy()
    borehole_19th = (
        centuries[centuries.century == "19th"]
        .set_index("site_id")
        .estimated_gst_change_c
    )
    comparison["borehole_19th_change_c"] = comparison.site_id.map(borehole_19th)

    tif_manifest: list[dict[str, Any]] = []
    for row in lineage.sort_values("member").itertuples(index=False):
        path = Path(row.source_tif)
        if not path.exists():
            raise FileNotFoundError(path)
        changes = project_changes(path, comparison)
        column = f"project_{row.member}_1800_1900_change_c"
        comparison[column] = changes
        tif_manifest.append(
            {
                "member": row.member,
                "path": str(path),
                "archived_sha256": row.source_tif_sha256,
                "years_read": [PROJECT_START_YEAR, PROJECT_END_YEAR],
                "bands_read": [PROJECT_START_YEAR - 999, PROJECT_END_YEAR - 999],
            }
        )

    member_columns = [f"project_{member}_1800_1900_change_c" for member in PROJECT_MEMBERS]
    comparison["project_comparison_eligible"] = comparison[member_columns].notna().all(axis=1)
    comparison["project_six_member_mean_change_c"] = comparison[member_columns].mean(axis=1)
    comparison["project_six_member_median_change_c"] = comparison[member_columns].median(axis=1)
    comparison.loc[
        ~comparison.project_comparison_eligible,
        ["project_six_member_mean_change_c", "project_six_member_median_change_c"],
    ] = np.nan
    comparison["borehole_direction"] = comparison.borehole_19th_change_c.map(sign_name)
    comparison["project_direction"] = comparison.project_six_member_median_change_c.map(
        lambda value: sign_name(value) if np.isfinite(value) else "unavailable"
    )
    comparison["direction_agreement"] = pd.Series(pd.NA, index=comparison.index, dtype="boolean")
    eligible = comparison.project_comparison_eligible
    comparison.loc[eligible, "direction_agreement"] = (
        comparison.loc[eligible, "borehole_direction"].to_numpy()
        == comparison.loc[eligible, "project_direction"].to_numpy()
    )
    comparison.to_csv(OUTPUT / "borehole_project_1800_1900_comparison.csv", index=False)

    cluster_rows: list[dict[str, Any]] = []
    for cluster_name, group in comparison.groupby("cluster", sort=True):
        matched = group[group.project_comparison_eligible].copy()
        if matched.empty:
            raise RuntimeError(f"cluster-has-no-project-comparison:{cluster_name}")
        member_cluster_values = []
        for column in member_columns:
            member_cluster_values.append(float(matched[column].median()))
        borehole = float(matched.borehole_19th_change_c.median())
        project_median = float(np.median(member_cluster_values))
        cluster_rows.append(
            {
                "cluster": cluster_name,
                "site_count": int(len(group)),
                "project_matched_site_count": int(len(matched)),
                "borehole_19th_change_median_c": borehole,
                "borehole_19th_change_mean_c": float(matched.borehole_19th_change_c.mean()),
                "project_six_member_median_change_c": project_median,
                "project_six_member_mean_change_c": float(np.mean(member_cluster_values)),
                "project_member_min_change_c": float(np.min(member_cluster_values)),
                "project_member_max_change_c": float(np.max(member_cluster_values)),
                "borehole_direction": sign_name(borehole),
                "project_direction": sign_name(project_median),
                "direction_agreement": sign_name(borehole) == sign_name(project_median),
                "site_direction_agreement_fraction": float(matched.direction_agreement.mean()),
                "evidence_family_vote_weight": 1,
            }
        )
    clusters = pd.DataFrame(cluster_rows).sort_values("cluster")
    clusters.to_csv(OUTPUT / "cluster_19th_century_comparison.csv", index=False)
    write_figure(clusters, OUTPUT / "borehole_project_19th_century_comparison.png")

    source_hashes_after = {str(path): sha256(path) for path in all_downloaded_sources}
    source_unchanged = source_hashes_before == source_hashes_after
    if not source_unchanged:
        raise RuntimeError("downloaded-source-file-changed-during-analysis")

    summary = {
        "experiment": "temperature_noaa_hma_borehole_19c_recovery_v1",
        "purpose": (
            "Compare independent NOAA borehole-inversion nineteenth-century GST changes "
            "with project-member annual air-temperature changes at matched coordinates."
        ),
        "status": "complete",
        "comparison_interval": [PROJECT_START_YEAR, PROJECT_END_YEAR],
        "borehole_target": "century-long ground-surface-temperature change",
        "coefficient_semantics": (
            "The legacy file labels Estimated_GST_Change(degC); the updated file labels Rate of GST Change (K/100a). "
            "For the exact 1800-1900 century, the numeric coefficient is also the total century change in K."
        ),
        "updated_archive_crosscheck_site_count": int(crosswalk.site_id.nunique()),
        "updated_archive_safe_coefficient_max_abs_difference": float(crosswalk.absolute_difference.max()),
        "project_target": "annual air-temperature change between interval endpoints",
        "site_count": int(len(sites)),
        "project_matched_site_count": int(comparison.project_comparison_eligible.sum()),
        "project_unmatched_site_ids": comparison.loc[
            ~comparison.project_comparison_eligible, "site_id"
        ].tolist(),
        "independent_regional_family_count": int(len(clusters)),
        "cluster_counting_rule": "Each spatial cluster is one evidence-family vote.",
        "cluster_direction_agreement_count": int(clusters.direction_agreement.sum()),
        "cluster_direction_disagreement_count": int((~clusters.direction_agreement).sum()),
        "cluster_rows": clusters.to_dict(orient="records"),
        "safe_centuries_decoded": list(SAFE_CENTURIES),
        "cross_locked_century_rows_skipped_before_value_decode": int(
            sites.cross_locked_century_rows_skipped_before_value_decode.sum()
        ),
        "updated_archive_cross_locked_rows_skipped_before_value_decode": updated_cross_lock_rows_skipped,
        "locked_actual_annual_values_accessed": False,
        "locked_period_used_for_selection_or_metrics": False,
        "source_files_unchanged": source_unchanged,
        "production_files_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "limitations": [
            "Borehole inversions resolve century-scale ground-surface temperature, not annual air temperature.",
            "The comparison tests direction and approximate century-scale amplitude; it is not an annual RMSE benchmark.",
            "Multiple holes in Yunnan and Qaidam are clustered and are not treated as independent votes.",
            "A nineteenth-century agreement or disagreement alone cannot authorize a uniform 1500-1879 correction.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )

    manifest = {
        "official_archive": "NOAA/NCEI Paleoclimatology Global Database of Borehole Temperatures and Climate Reconstructions",
        "original_publication_doi": "10.1038/35001556",
        "downloaded_sources": [
            {"path": str(path), "sha256": source_hashes_before[str(path)]}
            for path in all_downloaded_sources
        ],
        "project_member_sources": tif_manifest,
        "member_lineage": {"path": str(LINEAGE), "sha256": sha256(LINEAGE)},
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )

    agreement = int(clusters.direction_agreement.sum())
    cluster_lines = "\n".join(
        f"| {row.cluster} | {int(row.project_matched_site_count)}/{int(row.site_count)} | {row.borehole_19th_change_median_c:+.3f} | "
        f"{row.project_six_member_median_change_c:+.3f} | "
        f"{'yes' if row.direction_agreement else 'no'} |"
        for row in clusters.itertuples(index=False)
    )
    report = f"""# NOAA HMA borehole nineteenth-century comparison

## Result

The archive contains {len(sites)} official NOAA boreholes grouped into {len(clusters)} regional evidence families; {int(comparison.project_comparison_eligible.sum())} sites fall on valid project-grid cells and enter the matched comparison. The twentieth-century aggregate row in every source file was skipped before its numeric value was decoded; only the 16th-19th century inversion rows were parsed. The project comparison reads only 1800 and 1900. For {int(crosswalk.site_id.nunique())} sites available in both archive versions, every 16th-19th coefficient is identical (maximum absolute difference {float(crosswalk.absolute_difference.max()):.3g}). The legacy label is estimated GST change in degC and the updated label is rate in K/100a; over the exact 1800-1900 century the numeric coefficient equals the total century change.

| Regional family | Matched/all boreholes | Borehole GST change (degC) | Project median change (degC) | Direction agrees |
|---|---:|---:|---:|---|
{cluster_lines}

Direction agrees in **{agreement}/{len(clusters)} regional families**. These are independent physical low-frequency observations, but their target is century-scale ground-surface temperature. They can test the direction and rough amplitude of the project's nineteenth-century evolution; they cannot support annual RMSE or by themselves authorize a uniform correction of 1500-1879.

## Safety and provenance

- Official source: NOAA/NCEI Global Database of Borehole Temperatures and Climate Reconstructions.
- Original study DOI: `10.1038/35001556`.
- The sealed 1921-1939 period was not used in selection, metrics, or outputs.
- The crossing twentieth-century summary row was skipped before value decoding.
- Original downloaded files and project production files were not modified.
- EnKF and OGGM were not run.
- 20CRv3 was not accessed (`prohibited_source_accessed=false`).

## Files

- `borehole_16th_19th_safe_trends.csv`
- `borehole_project_1800_1900_comparison.csv`
- `borehole_2013_2016_safe_coefficient_crosswalk.csv`
- `cluster_19th_century_comparison.csv`
- `borehole_project_19th_century_comparison.png`
- `summary.json`
- `source_manifest.json`
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
