"""Acquire NOAA CPC daily precipitation with frozen whole-domain gap handling."""

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


OUTPUT = (
    base.PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_80_cpc_unified_allowed_subset_acquisition_v2"
)
DRIVER = Path(__file__).resolve()
MIN_VALID_DAYS = 360


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 80 v2 Contract",
                "",
                "## Reason For Version 2",
                "Round80-v1 stopped at 1981 because CPC contains two whole-HMA missing days (January 1-2), although the complete 365-date axis is present. V2 freezes a missing-day rule before inspecting any later year.",
                "",
                "## Objective",
                "Acquire NOAA CPC Global Unified Gauge-Based Daily Precipitation for 1979-2000 on the exact 0.5-degree public reconstruction grid over HMA and aggregate each calendar year to mean daily precipitation.",
                "",
                "## Frozen Inputs And Rules",
                "- Official source: NOAA PSL CPC Global Unified daily precipitation annual files through NCSS.",
                "- Allowed years only: 1979-2000. The locked 1921-1939 final-test period is not requested or accessed.",
                "- Grid centers must exactly equal the existing public reconstruction 43x78 grid; no interpolation is allowed.",
                "- Every annual response must contain the complete Gregorian date axis for that year.",
                f"- Every HMA-mask cell must have at least {MIN_VALID_DAYS} finite days per year and exactly the same finite-day count.",
                "- A missing day is allowed only when every HMA-mask cell is missing on that day; spatially selective missingness fails the round.",
                "- Annual values are arithmetic means of finite daily totals in mm/day. Missing days are preserved and never filled.",
                "- Preserve missing dates and per-cell finite-day counts plus request URL, status, bytes, and SHA256.",
                "- Original source and project climate files are read-only.",
                "",
                "## Exit Rule",
                "Pass only if all 22 years decode, coordinates match exactly, daily date axes are complete, and the frozen whole-domain gap rule passes every year.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with h5py.File(base.PUBLIC_SOURCE, "r") as handle:
        target_lat = np.asarray(handle["lat"][...], dtype=np.float64)
        target_lon = np.asarray(handle["lon"][...], dtype=np.float64)
    mask = np.load(base.HMA_MASK).astype(bool)
    if mask.shape != (target_lat.size, target_lon.size):
        raise RuntimeError("hma-mask-grid-shape-mismatch")

    annual = np.full((base.YEARS.size, target_lat.size, target_lon.size), np.nan, dtype=np.float32)
    counts = np.zeros_like(annual, dtype=np.int16)
    request_rows: list[dict[str, object]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-reconstruction-audit/1.0"})

    for year_index, year_value in enumerate(base.YEARS):
        year = int(year_value)
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
            required = {"lat", "lon", "time", "precip"}
            if not required.issubset(handle.keys()):
                raise RuntimeError(f"missing-netcdf-variable:{year}:{sorted(required - set(handle.keys()))}")
            lat = np.asarray(handle["lat"][...], dtype=np.float64)
            lon = np.asarray(handle["lon"][...], dtype=np.float64)
            time = np.asarray(handle["time"][...], dtype=np.float64)
            field = np.asarray(handle["precip"][...], dtype=np.float64)
            units = base.attribute_text(handle["time"], "units")
            calendar = base.attribute_text(handle["time"], "calendar")
            precip_units = base.attribute_text(handle["precip"], "units")

        days = base.expected_days(year)
        if not np.array_equal(lat, target_lat) or not np.array_equal(lon, target_lon):
            raise RuntimeError(f"coordinate-mismatch:{year}")
        if field.shape != (days, target_lat.size, target_lon.size):
            raise RuntimeError(f"field-shape-mismatch:{year}:{field.shape}")
        if calendar.lower() not in {"gregorian", "standard"}:
            raise RuntimeError(f"calendar-mismatch:{year}:{calendar}")
        if precip_units != "mm":
            raise RuntimeError(f"precipitation-unit-mismatch:{year}:{precip_units}")
        dates = base.decode_time(time, units)
        expected = [datetime(year, 1, 1) + timedelta(days=index) for index in range(days)]
        if dates != expected:
            raise RuntimeError(f"daily-axis-mismatch:{year}")

        finite = np.isfinite(field)
        hma_finite = finite[:, mask]
        cell_counts = hma_finite.sum(axis=0)
        daily_counts = hma_finite.sum(axis=1)
        partial_missing_days = np.flatnonzero((daily_counts > 0) & (daily_counts < int(mask.sum())))
        missing_day_indices = np.flatnonzero(daily_counts == 0)
        coverage_passes = bool(
            cell_counts.min() >= MIN_VALID_DAYS
            and cell_counts.min() == cell_counts.max()
            and partial_missing_days.size == 0
        )
        if not coverage_passes:
            raise RuntimeError(
                f"whole-domain-gap-rule-failed:{year}:"
                f"min={cell_counts.min()}:max={cell_counts.max()}:partial={partial_missing_days.size}"
            )

        count = finite.sum(axis=0)
        total = np.where(finite, field, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        annual[year_index] = mean.astype(np.float32)
        counts[year_index] = count.astype(np.int16)
        request_rows.append(
            {
                "year": year,
                "request_url": response.url,
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "expected_day_count": days,
                "decoded_day_count": len(dates),
                "hma_finite_day_count": int(cell_counts.min()),
                "whole_hma_missing_day_count": int(missing_day_indices.size),
                "whole_hma_missing_dates": ";".join(dates[index].strftime("%Y-%m-%d") for index in missing_day_indices),
                "partial_hma_missing_day_count": int(partial_missing_days.size),
                "coverage_rule_passes": coverage_passes,
            }
        )
        print(
            f"{year}: {len(payload):,} bytes, finite days={int(cell_counts.min())}, "
            f"whole-HMA missing={missing_day_indices.size}",
            flush=True,
        )

    requests_frame = pd.DataFrame(request_rows)
    requests_frame.to_csv(OUTPUT / "request_manifest.csv", index=False)
    target = OUTPUT / "cpc_unified_precip_1979_2000_public_grid_annual_mm_day.nc"
    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "NOAA CPC Unified annual mean daily precipitation, HMA public grid"
        handle.attrs["source_dataset_page"] = base.DATASET_PAGE
        handle.attrs["aggregation"] = "calendar-year arithmetic mean of finite daily totals"
        handle.attrs["missing_day_rule"] = "minimum 360; whole-HMA days only; no interpolation"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.create_dataset("year", data=base.YEARS)
        handle.create_dataset("lat", data=target_lat.astype(np.float32))
        handle.create_dataset("lon", data=target_lon.astype(np.float32))
        precipitation = handle.create_dataset(
            "precip_annual_mean_mm_day", data=annual, compression="gzip", compression_opts=4, shuffle=True
        )
        precipitation.attrs["units"] = "mm day-1"
        finite_days = handle.create_dataset(
            "finite_day_count", data=counts, compression="gzip", compression_opts=4, shuffle=True
        )
        finite_days.attrs["units"] = "days"

    summary = {
        "status": "complete",
        "decision": "advance_cpc_to_fixed_candidate_adjudication",
        "locked_1921_1939_accessed": False,
        "year_start": int(base.YEARS[0]),
        "year_end": int(base.YEARS[-1]),
        "year_count": int(base.YEARS.size),
        "grid_shape": [int(target_lat.size), int(target_lon.size)],
        "hma_cell_count": int(mask.sum()),
        "west_hma_cell_count": int((mask & (target_lon[None, :] < 80.0)).sum()),
        "minimum_hma_finite_days_any_year": int(requests_frame["hma_finite_day_count"].min()),
        "years_with_whole_hma_missing_days": int((requests_frame["whole_hma_missing_day_count"] > 0).sum()),
        "maximum_whole_hma_missing_days": int(requests_frame["whole_hma_missing_day_count"].max()),
        "partial_hma_missing_days_total": int(requests_frame["partial_hma_missing_day_count"].sum()),
        "all_years_pass_coverage_rule": bool(requests_frame["coverage_rule_passes"].all()),
        "total_transport_bytes": int(requests_frame["byte_count"].sum()),
        "request_count": int(len(requests_frame)),
        "output_file": str(target),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 80 v2 Result",
                "",
                "- Status: `complete`.",
                "- NOAA CPC Unified daily precipitation acquired for `1979-2000` on the exact project public grid.",
                f"- Grid: `{target_lat.size} x {target_lon.size}`; HMA cells: `{int(mask.sum())}`.",
                f"- Minimum annual HMA coverage: `{summary['minimum_hma_finite_days_any_year']}` days.",
                f"- Years with whole-HMA missing days: `{summary['years_with_whole_hma_missing_days']}`; maximum: `{summary['maximum_whole_hma_missing_days']}` days.",
                f"- Spatially selective missing HMA days: `{summary['partial_hma_missing_days_total']}`.",
                "- No interpolation, gap filling, candidate fitting, or locked-period access occurred.",
                "- Decision: `advance_cpc_to_fixed_candidate_adjudication`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_paths = [
        DRIVER,
        base.DRIVER,
        contract,
        base.HMA_MASK,
        base.PUBLIC_SOURCE,
        target,
        OUTPUT / "request_manifest.csv",
        OUTPUT / "summary.json",
        OUTPUT / "FINAL_RESULT.md",
    ]
    manifest = {
        "official_dataset_page": base.DATASET_PAGE,
        "ncss_pattern": base.BASE_URL,
        "files": [base.file_record(path) for path in manifest_paths],
        "transport_responses": request_rows,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
