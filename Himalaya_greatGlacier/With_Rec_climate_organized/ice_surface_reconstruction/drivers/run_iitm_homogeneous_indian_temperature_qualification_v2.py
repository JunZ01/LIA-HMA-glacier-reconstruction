from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


ALLOWED_INTERVALS = ((1901, 1920), (1940, 1950))
LOCKED_INTERVAL = (1921, 1939)
MONTHS = tuple(range(1, 13))
TARGET_REGION_CODES = ("WHIND", "NEIND")
TARGET_REGION_NAMES = {
    "WHIND": "Western Himalaya",
    "NEIND": "Northeast India",
}
OFFICIAL_BASE = "https://www.tropmet.res.in"
SOURCES = {
    "official_archive_page.html": f"{OFFICIAL_BASE}/static_pages.php?page_id=54",
    "README.pdf": f"{OFFICIAL_BASE}/data/data-archival/txtn/README.pdf",
    "NEW-TNREGION.TXT": f"{OFFICIAL_BASE}/data/data-archival/txtn/NEW-TNREGION.TXT",
    "NEW-TXREGION.TXT": f"{OFFICIAL_BASE}/data/data-archival/txtn/NEW-TXREGION.TXT",
    "TEMP-REG.jpg": f"{OFFICIAL_BASE}/data/data-archival/txtn/TEMP-REG.jpg",
}
ROW_PREFIX = re.compile(r"^([A-Z][A-Z0-9]*?)(\d{4})([NX])(?=\s)")
HEADER = re.compile(
    r"^\s*(\d+)\s+([A-Z][A-Z0-9]*):\s*(.*?)\s+"
    r"(?:MINIMUM|MAXIMUM)\s+TEMPERATURE",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_allowed(year: int) -> bool:
    return any(start <= year <= end for start, end in ALLOWED_INTERVALS)


def interval_name(year: int) -> str:
    for start, end in ALLOWED_INTERVALS:
        if start <= year <= end:
            return f"{start}-{end}"
    raise ValueError(f"Year {year} is outside allowed intervals")


def parse_iitm_text(
    path: Path,
    expected_tag: str,
    converter: Callable[[str], float] = float,
) -> tuple[list[dict], dict]:
    """Parse values only after decoding and admitting the row year."""
    records: list[dict] = []
    headers: dict[str, dict] = {}
    prefix_years: dict[str, list[int]] = defaultdict(list)
    locked_rows_skipped = 0
    outside_rows_skipped = 0
    numeric_conversion_years: list[int] = []

    with path.open("r", encoding="latin-1", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            header = HEADER.match(line)
            if header:
                headers[header.group(2).upper()] = {
                    "declared_year_count": int(header.group(1)),
                    "label": " ".join(header.group(3).split()),
                }
                continue

            match = ROW_PREFIX.match(line)
            if not match:
                continue
            region_code = match.group(1).upper()
            year = int(match.group(2))
            tag = match.group(3).upper()
            prefix_years[region_code].append(year)

            if LOCKED_INTERVAL[0] <= year <= LOCKED_INTERVAL[1]:
                locked_rows_skipped += 1
                continue
            if not is_allowed(year):
                outside_rows_skipped += 1
                continue
            if tag != expected_tag:
                raise ValueError(
                    f"Unexpected variable tag {tag!r} in {path} line {line_number}"
                )

            payload = line[match.end() :]
            tokens = payload.split()
            if len(tokens) != 12:
                raise ValueError(
                    f"Expected 12 monthly values in {path} line {line_number}; "
                    f"found {len(tokens)}"
                )
            values = [converter(token) for token in tokens]
            numeric_conversion_years.append(year)
            if region_code in TARGET_REGION_CODES:
                for month, value in zip(MONTHS, values):
                    records.append(
                        {
                            "region_code": region_code,
                            "region_name": TARGET_REGION_NAMES[region_code],
                            "year": year,
                            "month": month,
                            "value": value,
                        }
                    )

    if any(LOCKED_INTERVAL[0] <= y <= LOCKED_INTERVAL[1] for y in numeric_conversion_years):
        raise AssertionError("Locked-period value reached the numeric converter")
    found_regions = sorted({row["region_code"] for row in records})
    if found_regions != sorted(TARGET_REGION_CODES):
        raise ValueError(f"Target regions not found as expected: {found_regions}")

    audit = {
        "path": str(path),
        "expected_tag": expected_tag,
        "headers": headers,
        "target_prefix_year_ranges": {
            code: [min(prefix_years[code]), max(prefix_years[code])]
            for code in TARGET_REGION_CODES
        },
        "locked_rows_skipped_before_payload_access": locked_rows_skipped,
        "outside_allowed_rows_skipped_before_payload_access": outside_rows_skipped,
        "numeric_conversion_years": sorted(set(numeric_conversion_years)),
        "numeric_conversion_year_count_all_regions": len(numeric_conversion_years),
        "target_value_count": len(records),
    }
    return records, audit


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LIA-HMA-IITM-qualification/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        mirror = url.replace("https://www.tropmet.res.in", "https://tropmet.res.in")
        request = urllib.request.Request(
            mirror,
            headers={"User-Agent": "LIA-HMA-IITM-qualification/2.0"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
    if not payload:
        raise RuntimeError(f"Official download returned an empty file: {url}")
    destination.write_bytes(payload)


def make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def linear_trend(years: list[int], values: list[float]) -> tuple[float, float]:
    n = len(years)
    if n < 2:
        return math.nan, math.nan
    x_mean = sum(years) / n
    y_mean = sum(values) / n
    denominator = sum((x - x_mean) ** 2 for x in years)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(years, values)) / denominator
    fitted = [y_mean + slope * (x - x_mean) for x in years]
    ss_res = sum((y - f) ** 2 for y, f in zip(values, fitted))
    ss_tot = sum((y - y_mean) ** 2 for y in values)
    r2 = math.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope * 100.0, r2


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(monthly_rows: list[dict]) -> list[dict]:
    by_region_year: dict[tuple[str, int], dict[int, dict[str, float]]] = defaultdict(dict)
    for row in monthly_rows:
        by_region_year[(row["region_code"], row["year"])][row["month"]] = row

    seasons = {
        "annual_Jan-Dec": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        "pre_monsoon_MAM": (3, 4, 5),
        "monsoon_JJAS": (6, 7, 8, 9),
        "post_monsoon_ON": (10, 11),
    }
    output: list[dict] = []
    variables = ("tmin_c", "tmax_c", "tmean_derived_c")
    for region_code in TARGET_REGION_CODES:
        for year in sorted(y for code, y in by_region_year if code == region_code):
            current = by_region_year[(region_code, year)]
            for season, months in seasons.items():
                if all(month in current for month in months):
                    row = {
                        "region_code": region_code,
                        "region_name": TARGET_REGION_NAMES[region_code],
                        "year": year,
                        "allowed_segment": interval_name(year),
                        "season": season,
                    }
                    for variable in variables:
                        row[variable] = sum(current[m][variable] for m in months) / len(months)
                    output.append(row)

            previous_year = year - 1
            if is_allowed(previous_year) and interval_name(previous_year) == interval_name(year):
                previous = by_region_year.get((region_code, previous_year), {})
                if 12 in previous and 1 in current and 2 in current:
                    row = {
                        "region_code": region_code,
                        "region_name": TARGET_REGION_NAMES[region_code],
                        "year": year,
                        "allowed_segment": interval_name(year),
                        "season": "winter_DJF",
                    }
                    for variable in variables:
                        row[variable] = (
                            previous[12][variable] + current[1][variable] + current[2][variable]
                        ) / 3.0
                    output.append(row)
    return output


def build_trends(aggregate_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        groups[(row["region_code"], row["allowed_segment"], row["season"])].append(row)

    result: list[dict] = []
    for (region, segment, season), rows in sorted(groups.items()):
        rows.sort(key=lambda item: item["year"])
        years = [row["year"] for row in rows]
        for variable in ("tmin_c", "tmax_c", "tmean_derived_c"):
            values = [row[variable] for row in rows]
            trend, r2 = linear_trend(years, values)
            direction = "warming" if trend > 0 else "cooling" if trend < 0 else "flat"
            result.append(
                {
                    "region_code": region,
                    "region_name": TARGET_REGION_NAMES[region],
                    "allowed_segment": segment,
                    "season": season,
                    "variable": variable,
                    "n_years": len(years),
                    "first_year": min(years),
                    "last_year": max(years),
                    "mean_c": sum(values) / len(values),
                    "trend_c_per_century": trend,
                    "trend_direction": direction,
                    "linear_r2": r2,
                    "interpretation": "descriptive_precheck_only",
                }
            )
    return result


def build_coverage(monthly_rows: list[dict]) -> list[dict]:
    lookup = {(r["region_code"], r["year"], r["month"]) for r in monthly_rows}
    result: list[dict] = []
    for region in TARGET_REGION_CODES:
        for start, end in ALLOWED_INTERVALS:
            expected = (end - start + 1) * 12
            present = sum((region, y, m) in lookup for y in range(start, end + 1) for m in MONTHS)
            result.append(
                {
                    "region_code": region,
                    "region_name": TARGET_REGION_NAMES[region],
                    "allowed_segment": f"{start}-{end}",
                    "expected_months": expected,
                    "present_months": present,
                    "coverage_fraction": present / expected,
                    "complete": present == expected,
                }
            )
    return result


def create_figure(aggregate_rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    annual = [row for row in aggregate_rows if row["season"] == "annual_Jan-Dec"]
    colors = {
        "tmin_c": "#2474A6",
        "tmean_derived_c": "#2D7D46",
        "tmax_c": "#B13A32",
    }
    labels = {
        "tmin_c": "Tmin",
        "tmean_derived_c": "Tmean derived",
        "tmax_c": "Tmax",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, region in zip(axes, TARGET_REGION_CODES):
        rows = [row for row in annual if row["region_code"] == region]
        for variable in ("tmin_c", "tmean_derived_c", "tmax_c"):
            for start, end in ALLOWED_INTERVALS:
                segment = [row for row in rows if start <= row["year"] <= end]
                axis.plot(
                    [row["year"] for row in segment],
                    [row[variable] for row in segment],
                    color=colors[variable],
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    label=labels[variable] if start == ALLOWED_INTERVALS[0][0] else None,
                )
        axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.85, linewidth=0)
        axis.set_title(TARGET_REGION_NAMES[region], loc="left", fontsize=11)
        axis.set_ylabel("Annual temperature (degC)")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        axis.legend(frameon=False, ncol=3, loc="upper left")
    axes[-1].set_xlabel("Year (1921-1939 sealed and absent)")
    fig.suptitle("Official IITM regional temperature: allowed intervals only", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Independent result directory already exists: {output_dir}")

    raw_dir = output_dir / "raw_unopened"
    derived_dir = output_dir / "derived"
    figure_dir = output_dir / "figures"
    raw_dir.mkdir(parents=True)
    derived_dir.mkdir()
    figure_dir.mkdir()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    source_manifest: list[dict] = []
    for filename, url in SOURCES.items():
        destination = raw_dir / filename
        download(url, destination)
        source_manifest.append(
            {
                "filename": filename,
                "url": url,
                "domain": "www.tropmet.res.in",
                "retrieved_utc": retrieved_at,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "role": "official_IITM_source",
            }
        )
        make_read_only(destination)

    minimum, minimum_audit = parse_iitm_text(raw_dir / "NEW-TNREGION.TXT", "N")
    maximum, maximum_audit = parse_iitm_text(raw_dir / "NEW-TXREGION.TXT", "X")
    min_lookup = {(r["region_code"], r["year"], r["month"]): r["value"] for r in minimum}
    max_lookup = {(r["region_code"], r["year"], r["month"]): r["value"] for r in maximum}
    if min_lookup.keys() != max_lookup.keys():
        raise AssertionError("Tmin and Tmax target coordinates do not close")

    monthly: list[dict] = []
    for region, year, month in sorted(min_lookup):
        tmin = min_lookup[(region, year, month)]
        tmax = max_lookup[(region, year, month)]
        if tmin > tmax:
            raise AssertionError(f"Tmin exceeds Tmax at {region} {year}-{month:02d}")
        monthly.append(
            {
                "region_code": region,
                "region_name": TARGET_REGION_NAMES[region],
                "year": year,
                "month": month,
                "allowed_segment": interval_name(year),
                "tmin_c": tmin,
                "tmax_c": tmax,
                "tmean_derived_c": (tmin + tmax) / 2.0,
                "tmean_status": "derived_as_midrange_not_IITM_published_column",
            }
        )

    if any(LOCKED_INTERVAL[0] <= row["year"] <= LOCKED_INTERVAL[1] for row in monthly):
        raise AssertionError("Locked period leaked into monthly output")
    if any(not is_allowed(row["year"]) for row in monthly):
        raise AssertionError("Disallowed year leaked into monthly output")

    aggregates = aggregate(monthly)
    trends = build_trends(aggregates)
    coverage = build_coverage(monthly)
    if not all(row["complete"] for row in coverage):
        raise AssertionError("Official target-region coverage is incomplete in an allowed segment")

    write_csv(
        derived_dir / "iitm_wh_ne_allowed_monthly_temperature.csv",
        monthly,
        [
            "region_code",
            "region_name",
            "year",
            "month",
            "allowed_segment",
            "tmin_c",
            "tmax_c",
            "tmean_derived_c",
            "tmean_status",
        ],
    )
    write_csv(
        derived_dir / "iitm_wh_ne_allowed_annual_seasonal_temperature.csv",
        aggregates,
        [
            "region_code",
            "region_name",
            "year",
            "allowed_segment",
            "season",
            "tmin_c",
            "tmax_c",
            "tmean_derived_c",
        ],
    )
    write_csv(
        derived_dir / "iitm_wh_ne_allowed_coverage.csv",
        coverage,
        [
            "region_code",
            "region_name",
            "allowed_segment",
            "expected_months",
            "present_months",
            "coverage_fraction",
            "complete",
        ],
    )
    write_csv(
        derived_dir / "iitm_wh_ne_trend_direction_precheck.csv",
        trends,
        [
            "region_code",
            "region_name",
            "allowed_segment",
            "season",
            "variable",
            "n_years",
            "first_year",
            "last_year",
            "mean_c",
            "trend_c_per_century",
            "trend_direction",
            "linear_r2",
            "interpretation",
        ],
    )

    locked_audit = {
        "allowed_intervals": ALLOWED_INTERVALS,
        "locked_interval": LOCKED_INTERVAL,
        "parser_order": "prefix_match_then_year_decode_then_skip_then_payload_slice_then_numeric_conversion",
        "minimum_file": minimum_audit,
        "maximum_file": maximum_audit,
        "derived_output_years": sorted({row["year"] for row in monthly}),
        "locked_value_in_derived_output": False,
        "no_smoothing_or_aggregation_crossed_gap": True,
        "synthetic_poison_test": "must_be_run_separately_before_driver",
    }
    (output_dir / "locked_access_audit.json").write_text(
        json.dumps(locked_audit, indent=2), encoding="utf-8"
    )

    method_audit = {
        "official_product_period": "1901-2007",
        "published_variables": ["monthly regional Tmin", "monthly regional Tmax"],
        "derived_variable": "Tmean=(Tmax+Tmin)/2; midpoint diagnostic, not a published IITM column",
        "station_source_1901_1990": "121-station IMD monthly weather records used by Pant and Rupa Kumar (1997)",
        "update_source_1991_2007": "Indian Daily Weather Reports published by IMD",
        "climatology": "1951-1980 monthly Tmax/Tmin normals from 388 IMD stations",
        "spatial_processing": "station anomalies objectively interpolated to 0.5 degree, climatology added, then constituent cells simply averaged by region",
        "region_definition": "geographical, topographical, and climatological homogeneous regions",
        "homogenization_interpretation": "No explicit breakpoint-homogenization procedure is documented in the official README or 2005 method text; homogeneous refers to regions, not proven independently homogenized stations.",
        "season_definitions": {
            "winter": "DJF",
            "pre_monsoon": "MAM",
            "monsoon": "JJAS",
            "post_monsoon": "ON",
        },
        "source_dependence": {
            "IMD": "direct source family",
            "GHCN": "likely partial station/source overlap; exact station crosswalk unavailable in the IITM package",
            "CRU": "likely partial international/IMD station overlap; IITM is not an independent observational vote",
        },
        "spatial_qualification": {
            "WHIND": "regional evidence for the Indian Western Himalaya only",
            "NEIND": "regional eastern-HMA-margin evidence; not the Tibetan Plateau or full eastern HMA",
        },
        "primary_method_reference": "Kothawale and Rupa Kumar (2005), doi:10.1029/2005GL023528",
    }
    (output_dir / "method_and_source_dependence_audit.json").write_text(
        json.dumps(method_audit, indent=2), encoding="utf-8"
    )

    create_figure(aggregates, figure_dir / "iitm_wh_ne_allowed_annual_temperature.png")

    integrity = []
    for item in source_manifest:
        path = raw_dir / item["filename"]
        post_hash = sha256(path)
        integrity.append(
            {
                **item,
                "post_analysis_sha256": post_hash,
                "hash_unchanged": post_hash == item["sha256"],
                "read_only_attribute_requested": True,
            }
        )
    if not all(item["hash_unchanged"] for item in integrity):
        raise AssertionError("An official source hash changed during analysis")
    (output_dir / "source_manifest.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )

    summary = {
        "decision": "admit_as_source_correlated_regional_processing_diagnostic",
        "not_admitted_as": [
            "independent_observational_vote_against_GHCN_or_CRU",
            "full_HMA_grid_validation",
            "temperature_correction_target",
        ],
        "regions": list(TARGET_REGION_CODES),
        "allowed_monthly_rows": len(monthly),
        "coverage_complete": all(row["complete"] for row in coverage),
        "source_hashes_unchanged": all(item["hash_unchanged"] for item in integrity),
        "locked_period_absent_from_outputs": True,
        "trend_precheck_rows": len(trends),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    annual_trends = [row for row in trends if row["season"] == "annual_Jan-Dec"]
    trend_lines = "\n".join(
        f"- {row['region_code']} {row['allowed_segment']} {row['variable']}: "
        f"{row['trend_c_per_century']:+.3f} degC/century ({row['trend_direction']}, "
        f"n={row['n_years']})."
        for row in annual_trends
    )
    final = f"""# IITM Homogeneous Indian Temperature Qualification v2

## Decision

`admit_as_source_correlated_regional_processing_diagnostic`

The official IITM Western Himalaya and Northeast India monthly Tmin/Tmax
series were acquired and frozen. They are complete in both allowed intervals.
They may test regional direction and processed anomaly amplitude, but they are
not an independent observational vote against GHCN or CRU and cannot validate
the full HMA grid.

## Data recovered

- Official period: 1901-2007.
- Allowed output: 1901-1920 and 1940-1950 only.
- Published variables: Tmin and Tmax.
- Diagnostic Tmean: `(Tmax + Tmin) / 2`; this is a derived midpoint, not an
  IITM-published mean-temperature column.
- Allowed monthly rows: {len(monthly)}.
- All four region/interval coverage checks are complete.

## Annual direction precheck

{trend_lines}

These short-period slopes are descriptive diagnostics only. They are not
correction coefficients, and no inference should be based on a single 11- or
20-year trend.

## Method and independence

The product uses IMD observations, station anomalies interpolated to a
0.5-degree grid, a 1951-1980 climatology based on 388 stations, and simple
regional grid-cell averages. The word `homogeneous` identifies seven climate
regions. The official method documents no separate breakpoint-homogenization
algorithm, so this result must not be described as independently homogenized
station evidence.

Because GHCN and CRU can reuse IMD observations, IITM adds a regionalization
and interpolation-method sensitivity check, not a new independent source vote.
WHIND is useful for the Indian Western Himalaya. NEIND is an eastern-margin
diagnostic and must not be treated as central/eastern Tibetan Plateau coverage.

## Lock and integrity audit

The production parser decoded the row prefix and year before slicing any
temperature payload. It skipped 1921-1939 and all other disallowed years before
numeric conversion. No locked value appears in any derived table, no aggregate
crosses the gap, and all official source hashes remained unchanged.

An earlier interactive v1 format probe accidentally displayed official
1921-1925 rows. Those values were not saved or used; v1 is invalid and excluded
from this evidence chain. This v2 directory is the clean formal experiment.

## Sources

- IITM official archive: {SOURCES['official_archive_page.html']}
- IITM official README: {SOURCES['README.pdf']}
- Method paper: https://doi.org/10.1029/2005GL023528
"""
    (output_dir / "FINAL_RESULT.md").write_text(final, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
