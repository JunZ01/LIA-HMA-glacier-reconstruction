"""Acquire CPC annual fields for years frozen as eligible by the coverage audit."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

import acquire_round80_cpc_unified_allowed_subset_v1 as base


PREFLIGHT = (
    base.PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_80_cpc_unified_daily_coverage_preflight_v1"
)
OUTPUT = (
    base.PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_80_cpc_unified_allowed_subset_acquisition_v3"
)
DRIVER = Path(__file__).resolve()
MIN_VALID_DAYS = 360


def main() -> int:
    coverage_path = PREFLIGHT / "daily_coverage_inventory.csv"
    coverage = pd.read_csv(coverage_path)
    if not np.array_equal(coverage.year.to_numpy(dtype=int), base.YEARS.astype(int)):
        raise RuntimeError("coverage-preflight-year-axis-mismatch")
    eligible = coverage.loc[coverage.finite_day_count >= MIN_VALID_DAYS].copy()
    excluded = coverage.loc[coverage.finite_day_count < MIN_VALID_DAYS].year.astype(int).tolist()
    if excluded != [1985] or len(eligible) != 21:
        raise RuntimeError(f"unexpected-frozen-eligibility:{excluded}:{len(eligible)}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 80 v3 Contract",
                "",
                "## Objective",
                "Acquire full-grid NOAA CPC Unified annual validation fields for the years frozen as eligible by the completed temporal-coverage preflight.",
                "",
                "## Frozen Eligibility",
                f"- Require at least `{MIN_VALID_DAYS}` finite daily totals at the fixed preflight point.",
                "- Eligible years: `1979-1984, 1986-2000` (21 years).",
                "- Excluded year: `1985` (355 finite days with missing dates distributed across seasons).",
                "",
                "## Full-Grid Rules",
                "- NOAA PSL CPC Unified annual files through NCSS; exact project 43x78 public grid; no interpolation.",
                "- Require the complete Gregorian date axis and exact coordinates.",
                "- Every eligible year must reproduce the preflight finite-day count at every HMA-mask cell.",
                "- Each day must be either finite at every HMA-mask cell or missing at every HMA-mask cell; spatially selective missingness fails.",
                "- Annual values are arithmetic means over finite daily totals in mm/day; no gap filling.",
                "- Original source and project files remain read-only.",
                "- The locked 1921-1939 final-test period is not requested or accessed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with h5py.File(base.PUBLIC_SOURCE, "r") as handle:
        target_lat = np.asarray(handle["lat"][...], dtype=np.float64)
        target_lon = np.asarray(handle["lon"][...], dtype=np.float64)
    mask = np.load(base.HMA_MASK).astype(bool)
    years = eligible.year.to_numpy(dtype=np.int16)
    annual = np.full((years.size, target_lat.size, target_lon.size), np.nan, dtype=np.float32)
    counts = np.zeros_like(annual, dtype=np.int16)
    rows: list[dict[str, object]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-reconstruction-audit/1.0"})

    for year_index, year_value in enumerate(years):
        year = int(year_value)
        expected_finite = int(eligible.loc[eligible.year == year, "finite_day_count"].iloc[0])
        expected_missing_dates = str(eligible.loc[eligible.year == year, "missing_dates"].iloc[0])
        if expected_missing_dates == "nan":
            expected_missing_dates = ""
        params = {
            "var": "precip",
            **{key: f"{value:.2f}" for key, value in base.BOUNDS.items()},
            "horizStride": "1",
            "time_start": f"{year}-01-01T00:00:00Z",
            "time_end": f"{year}-12-31T00:00:00Z",
            "timeStride": "1",
            "accept": "netcdf4",
        }
        response = session.get(base.BASE_URL.format(year=year), params=params, timeout=600)
        response.raise_for_status()
        payload = response.content
        with h5py.File(io.BytesIO(payload), "r") as handle:
            lat = np.asarray(handle["lat"][...], dtype=np.float64)
            lon = np.asarray(handle["lon"][...], dtype=np.float64)
            time = np.asarray(handle["time"][...], dtype=np.float64)
            field = np.asarray(handle["precip"][...], dtype=np.float64)
            units = base.attribute_text(handle["time"], "units")
            calendar = base.attribute_text(handle["time"], "calendar")
            precip_units = base.attribute_text(handle["precip"], "units")

        days = base.expected_days(year)
        dates = base.decode_time(time, units)
        expected_dates = [datetime(year, 1, 1) + timedelta(days=index) for index in range(days)]
        if dates != expected_dates or field.shape != (days, target_lat.size, target_lon.size):
            raise RuntimeError(f"daily-axis-or-shape-mismatch:{year}:{field.shape}")
        if not np.array_equal(lat, target_lat) or not np.array_equal(lon, target_lon):
            raise RuntimeError(f"coordinate-mismatch:{year}")
        if calendar.lower() not in {"gregorian", "standard"} or precip_units != "mm":
            raise RuntimeError(f"metadata-mismatch:{year}:{calendar}:{precip_units}")

        finite = np.isfinite(field)
        hma_finite = finite[:, mask]
        daily_counts = hma_finite.sum(axis=1)
        partial = np.flatnonzero((daily_counts > 0) & (daily_counts < int(mask.sum())))
        missing = np.flatnonzero(daily_counts == 0)
        cell_counts = hma_finite.sum(axis=0)
        missing_dates = ";".join(dates[index].strftime("%Y-%m-%d") for index in missing)
        if partial.size or cell_counts.min() != expected_finite or cell_counts.max() != expected_finite:
            raise RuntimeError(
                f"full-grid-coverage-mismatch:{year}:partial={partial.size}:"
                f"min={cell_counts.min()}:max={cell_counts.max()}:expected={expected_finite}"
            )
        if missing_dates != expected_missing_dates:
            raise RuntimeError(f"missing-date-replay-mismatch:{year}:{missing_dates}:{expected_missing_dates}")

        count = finite.sum(axis=0)
        total = np.where(finite, field, 0.0).sum(axis=0)
        annual[year_index] = np.divide(
            total, count, out=np.full_like(total, np.nan), where=count > 0
        ).astype(np.float32)
        counts[year_index] = count.astype(np.int16)
        rows.append(
            {
                "year": year,
                "finite_day_count": expected_finite,
                "missing_day_count": int(missing.size),
                "missing_dates": missing_dates,
                "partial_hma_missing_day_count": int(partial.size),
                "request_url": response.url,
                "http_status": response.status_code,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        print(f"{year}: finite={expected_finite}, bytes={len(payload):,}, full-grid replay=True", flush=True)

    manifest_frame = pd.DataFrame(rows)
    manifest_frame.to_csv(OUTPUT / "request_manifest.csv", index=False)
    target = OUTPUT / "cpc_unified_precip_eligible_1979_2000_public_grid_annual_mm_day.nc"
    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "NOAA CPC Unified eligible annual mean daily precipitation, HMA public grid"
        handle.attrs["source_dataset_page"] = base.DATASET_PAGE
        handle.attrs["aggregation"] = "calendar-year arithmetic mean over finite daily totals"
        handle.attrs["minimum_finite_days"] = MIN_VALID_DAYS
        handle.attrs["excluded_years"] = "1985"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.create_dataset("year", data=years)
        handle.create_dataset("lat", data=target_lat.astype(np.float32))
        handle.create_dataset("lon", data=target_lon.astype(np.float32))
        variable = handle.create_dataset(
            "precip_annual_mean_mm_day", data=annual, compression="gzip", compression_opts=4, shuffle=True
        )
        variable.attrs["units"] = "mm day-1"
        handle.create_dataset("finite_day_count", data=counts, compression="gzip", compression_opts=4, shuffle=True)

    summary = {
        "status": "complete",
        "decision": "advance_cpc_to_fixed_candidate_adjudication",
        "locked_1921_1939_accessed": False,
        "eligible_years": years.astype(int).tolist(),
        "excluded_years": excluded,
        "eligible_year_count": int(years.size),
        "minimum_finite_days": int(manifest_frame.finite_day_count.min()),
        "partial_hma_missing_days_total": int(manifest_frame.partial_hma_missing_day_count.sum()),
        "grid_shape": [int(target_lat.size), int(target_lon.size)],
        "hma_cell_count": int(mask.sum()),
        "west_hma_cell_count": int((mask & (target_lon[None, :] < 80.0)).sum()),
        "request_count": int(len(manifest_frame)),
        "total_transport_bytes": int(manifest_frame.byte_count.sum()),
        "output_file": str(target),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 80 v3 Result",
                "",
                f"- Eligible CPC years acquired: `{years.size}` (`1979-1984, 1986-2000`).",
                "- Excluded by the frozen coverage preflight: `1985`.",
                f"- Minimum retained coverage: `{summary['minimum_finite_days']}` days.",
                f"- Full-grid spatially selective missing days: `{summary['partial_hma_missing_days_total']}`.",
                f"- Exact public grid: `{target_lat.size} x {target_lon.size}`; HMA cells: `{int(mask.sum())}`.",
                "- No interpolation, gap filling, candidate fitting, or locked-period access occurred.",
                "- Decision: `advance_cpc_to_fixed_candidate_adjudication`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = [DRIVER, base.DRIVER, contract, coverage_path, base.HMA_MASK, base.PUBLIC_SOURCE, target, OUTPUT / "request_manifest.csv", OUTPUT / "summary.json", OUTPUT / "FINAL_RESULT.md"]
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "official_dataset_page": base.DATASET_PAGE,
                "files": [base.file_record(path) for path in paths],
                "transport_responses": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
