"""Acquire allowed GHCN-D station precipitation without touching locked years."""

from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
ROUND82 = LOOP / "round_82_ghcnd_hma_precipitation_station_inventory_v1"
OUTPUT = LOOP / "round_83_ghcnd_allowed_station_precipitation_v1"
DRIVER = Path(__file__).resolve()
BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
CPC_YEARS = np.asarray([1979, 1980, 1981, 1982, 1983, 1984, *range(1986, 2001)], dtype=int)
MIN_FIRST_YEAR = 1940
MIN_DAILY_FRACTION = 0.90
MIN_COMPLETE_YEARS = 15
MIN_WEST_STATIONS_TO_ADVANCE = 10


def file_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def line_years(lines: list[str], station_id: str) -> np.ndarray:
    years = []
    for line in lines:
        if len(line) < 21 or line[:11] != station_id:
            raise RuntimeError(f"dly-line-identity-or-length-failed:{station_id}")
        years.append(int(line[11:15]))
    return np.asarray(years, dtype=int)


def parse_allowed_prcp(lines: list[str], station_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in lines:
        year = int(line[11:15])
        if year not in CPC_YEARS or line[17:21] != "PRCP":
            continue
        month = int(line[15:17])
        valid_days = 0
        total_tenths_mm = 0
        quality_flagged_days = 0
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            offset = 21 + (day - 1) * 8
            value = int(line[offset : offset + 5])
            quality_flag = line[offset + 6]
            if value == -9999:
                continue
            if quality_flag != " ":
                quality_flagged_days += 1
                continue
            valid_days += 1
            total_tenths_mm += value
        rows.append(
            {
                "station_id": station_id,
                "year": year,
                "month": month,
                "valid_day_count": valid_days,
                "quality_flagged_day_count": quality_flagged_days,
                "precipitation_total_mm": total_tenths_mm / 10.0,
            }
        )
    return rows


def main() -> int:
    station_inventory = pd.read_csv(ROUND82 / "ghcnd_hma_prcp_station_inventory.csv")
    selected = station_inventory[
        station_inventory.long_overlap & (station_inventory.first_year >= MIN_FIRST_YEAR)
    ].copy()
    if len(selected) != 56 or int((selected.region == "west").sum()) != 21:
        raise RuntimeError(f"unexpected-frozen-station-pool:{len(selected)}:{int((selected.region == 'west').sum())}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw_dir = OUTPUT / "raw_dly_no_locked_years"
    raw_dir.mkdir()
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 83 Contract",
                "",
                "## Objective",
                "Acquire value-level NOAA GHCN-D precipitation for the 56 HMA stations that have at least 15 years of catalog overlap with 1979-2000 and whose entire station record begins in 1940 or later.",
                "",
                "## Locked-Period Protection",
                "- Download only stations with inventory first_year >=1940, so their files cannot contain 1921-1939.",
                "- For every downloaded line, parse and validate the year token before converting any precipitation value.",
                "- Abort before value parsing if any year is below 1940 or in 1921-1939.",
                "",
                "## Frozen Completeness Rules",
                "- Parse PRCP only for the 21 CPC-eligible years: 1979-1984 and 1986-2000.",
                "- Accept only values with blank GHCN quality flags; preserve missing and flagged days.",
                "- A station-year is complete when at least 90% of calendar days are valid.",
                "- A station is comparison-eligible when at least 15 of 21 years are complete.",
                "- Advance if at least 10 western stations are comparison-eligible.",
                "- GHCN-D values are direct station evidence but are not claimed independent of all gridded-product gauge inputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    request_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-reconstruction-audit/1.0"})
    for station_id in selected.station_id.astype(str):
        response = session.get(BASE_URL.format(station_id=station_id), timeout=300)
        response.raise_for_status()
        payload = response.content
        path = raw_dir / f"{station_id}.dly"
        path.write_bytes(payload)
        text = payload.decode("ascii")
        lines = [line for line in text.splitlines() if line]
        years = line_years(lines, station_id)
        if years.min() < MIN_FIRST_YEAR or np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError(f"locked-or-pre1940-year-present:{station_id}:{years.min()}")
        monthly_rows.extend(parse_allowed_prcp(lines, station_id))
        request_rows.append(
            {
                "station_id": station_id,
                "request_url": response.url,
                "http_status": response.status_code,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "minimum_year_token": int(years.min()),
                "maximum_year_token": int(years.max()),
                "line_count": int(len(lines)),
            }
        )
        print(f"{station_id}: {len(payload):,} bytes, years={years.min()}-{years.max()}", flush=True)

    monthly = pd.DataFrame(monthly_rows)
    annual_rows: list[dict[str, object]] = []
    for station_id in selected.station_id.astype(str):
        station_monthly = monthly[monthly.station_id == station_id]
        for year in CPC_YEARS:
            values = station_monthly[station_monthly.year == year]
            valid_days = int(values.valid_day_count.sum()) if not values.empty else 0
            flagged_days = int(values.quality_flagged_day_count.sum()) if not values.empty else 0
            total_mm = float(values.precipitation_total_mm.sum()) if not values.empty else 0.0
            expected_days = 366 if calendar.isleap(int(year)) else 365
            complete = valid_days / expected_days >= MIN_DAILY_FRACTION
            annual_rows.append(
                {
                    "station_id": station_id,
                    "year": int(year),
                    "valid_day_count": valid_days,
                    "expected_day_count": expected_days,
                    "valid_day_fraction": valid_days / expected_days,
                    "quality_flagged_day_count": flagged_days,
                    "annual_mean_mm_day": total_mm / valid_days if valid_days else np.nan,
                    "station_year_complete": complete,
                }
            )
    annual = pd.DataFrame(annual_rows).merge(
        selected[
            [
                "station_id",
                "latitude",
                "longitude",
                "elevation_m",
                "grid_latitude",
                "grid_longitude",
                "region",
                "country",
                "name",
            ]
        ],
        on="station_id",
        how="left",
        validate="many_to_one",
    )
    annual.to_csv(OUTPUT / "ghcnd_station_annual_precipitation_1979_2000.csv", index=False)
    station_summary = (
        annual.groupby(["station_id", "region", "country", "name"], as_index=False)
        .agg(
            complete_year_count=("station_year_complete", "sum"),
            minimum_valid_day_fraction=("valid_day_fraction", "min"),
            median_valid_day_fraction=("valid_day_fraction", "median"),
            total_quality_flagged_days=("quality_flagged_day_count", "sum"),
        )
    )
    station_summary["comparison_eligible"] = station_summary.complete_year_count >= MIN_COMPLETE_YEARS
    station_summary.to_csv(OUTPUT / "ghcnd_station_completeness_summary.csv", index=False)
    pd.DataFrame(request_rows).to_csv(OUTPUT / "request_manifest.csv", index=False)

    west_eligible = station_summary[(station_summary.region == "west") & station_summary.comparison_eligible]
    hma_eligible = station_summary[station_summary.comparison_eligible]
    advance = len(west_eligible) >= MIN_WEST_STATIONS_TO_ADVANCE
    summary = {
        "status": "complete",
        "decision": "advance_ghcnd_to_station_location_product_adjudication" if advance else "ghcnd_value_completeness_insufficient_for_western_adjudication",
        "locked_1921_1939_accessed": False,
        "downloaded_station_count": int(len(selected)),
        "downloaded_western_station_count": int((selected.region == "west").sum()),
        "comparison_eligible_hma_station_count": int(len(hma_eligible)),
        "comparison_eligible_western_station_count": int(len(west_eligible)),
        "minimum_daily_fraction": MIN_DAILY_FRACTION,
        "minimum_complete_years": MIN_COMPLETE_YEARS,
        "minimum_west_stations_to_advance": MIN_WEST_STATIONS_TO_ADVANCE,
        "eligible_western_countries": sorted(west_eligible.country.unique().tolist()),
        "total_transport_bytes": int(sum(row["byte_count"] for row in request_rows)),
        "station_independence_claimed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 83 GHCN-D Allowed Station Precipitation",
                "",
                f"- Downloaded HMA/western stations: `{len(selected)}/{int((selected.region == 'west').sum())}`.",
                f"- Comparison-eligible HMA/western stations: `{len(hma_eligible)}/{len(west_eligible)}`.",
                f"- Eligible western countries: `{summary['eligible_western_countries']}`.",
                f"- Transport bytes: `{summary['total_transport_bytes']:,}`.",
                "- Every downloaded station begins in 1940 or later; no locked-period year or precipitation value was present.",
                f"- Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_paths = [DRIVER, contract, ROUND82 / "ghcnd_hma_prcp_station_inventory.csv", OUTPUT / "request_manifest.csv", OUTPUT / "ghcnd_station_annual_precipitation_1979_2000.csv", OUTPUT / "ghcnd_station_completeness_summary.csv", OUTPUT / "summary.json", OUTPUT / "FINAL_RESULT.md", *sorted(raw_dir.glob("*.dly"))]
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"files": [file_record(path) for path in manifest_paths]}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
