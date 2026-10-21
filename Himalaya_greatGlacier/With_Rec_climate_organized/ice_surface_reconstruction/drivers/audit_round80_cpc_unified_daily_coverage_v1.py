"""Inventory CPC daily temporal coverage before freezing an annual-use rule."""

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
    / "round_80_cpc_unified_daily_coverage_preflight_v1"
)
DRIVER = Path(__file__).resolve()
POINT = {"north": 35.25, "south": 35.25, "west": 85.25, "east": 85.25}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 80 CPC Coverage Preflight Contract",
                "",
                "## Objective",
                "Inventory the exact missing-day chronology in NOAA CPC Unified precipitation for 1979-2000 before freezing an annual validation eligibility rule.",
                "",
                "## Frozen Rules",
                "- Query one fixed interior HMA grid center at 35.25 N, 85.25 E for every year.",
                "- Require the complete Gregorian date axis; record finite and missing dates without interpolation.",
                "- This preflight does not create annual climate fields, fit candidates, or define the final completeness threshold.",
                "- The locked 1921-1939 period is not requested or accessed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-reconstruction-audit/1.0"})
    for year_value in base.YEARS:
        year = int(year_value)
        params = {
            "var": "precip",
            **{key: f"{value:.2f}" for key, value in POINT.items()},
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
            time = np.asarray(handle["time"][...], dtype=np.float64)
            field = np.asarray(handle["precip"][...], dtype=np.float64).reshape(-1)
            lat = np.asarray(handle["lat"][...], dtype=np.float64)
            lon = np.asarray(handle["lon"][...], dtype=np.float64)
            units = base.attribute_text(handle["time"], "units")
        days = base.expected_days(year)
        dates = base.decode_time(time, units)
        expected = [datetime(year, 1, 1) + timedelta(days=index) for index in range(days)]
        if dates != expected or field.size != days:
            raise RuntimeError(f"daily-axis-mismatch:{year}")
        if not np.array_equal(lat, np.asarray([POINT["north"]])) or not np.array_equal(
            lon, np.asarray([POINT["west"]])
        ):
            raise RuntimeError(f"point-coordinate-mismatch:{year}:{lat}:{lon}")
        missing = np.flatnonzero(~np.isfinite(field))
        rows.append(
            {
                "year": year,
                "expected_day_count": days,
                "finite_day_count": int(np.isfinite(field).sum()),
                "missing_day_count": int(missing.size),
                "missing_dates": ";".join(dates[index].strftime("%Y-%m-%d") for index in missing),
                "first_finite_date": dates[int(np.flatnonzero(np.isfinite(field))[0])].strftime("%Y-%m-%d"),
                "last_finite_date": dates[int(np.flatnonzero(np.isfinite(field))[-1])].strftime("%Y-%m-%d"),
                "request_url": response.url,
                "http_status": response.status_code,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        print(f"{year}: finite={int(np.isfinite(field).sum())}, missing={missing.size}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "daily_coverage_inventory.csv", index=False)
    summary = {
        "status": "complete",
        "decision": "freeze_annual_eligibility_after_full_coverage_inventory",
        "locked_1921_1939_accessed": False,
        "year_count": int(len(frame)),
        "minimum_finite_day_count": int(frame["finite_day_count"].min()),
        "maximum_missing_day_count": int(frame["missing_day_count"].max()),
        "years_with_missing_days": frame.loc[frame.missing_day_count > 0, "year"].astype(int).tolist(),
        "years_with_at_least_360_days": frame.loc[frame.finite_day_count >= 360, "year"].astype(int).tolist(),
        "years_with_at_least_350_days": frame.loc[frame.finite_day_count >= 350, "year"].astype(int).tolist(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 80 CPC Coverage Preflight Result",
                "",
                f"- Years audited: `{len(frame)}` (`1979-2000`).",
                f"- Minimum finite days: `{summary['minimum_finite_day_count']}`.",
                f"- Maximum missing days: `{summary['maximum_missing_day_count']}`.",
                f"- Years with missing days: `{summary['years_with_missing_days']}`.",
                "- No interpolation, annual aggregation, candidate fitting, or locked-period access occurred.",
                "- Decision: `freeze_annual_eligibility_after_full_coverage_inventory`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_paths = [DRIVER, base.DRIVER, contract, OUTPUT / "daily_coverage_inventory.csv", OUTPUT / "summary.json", OUTPUT / "FINAL_RESULT.md"]
    manifest = {
        "official_dataset_page": base.DATASET_PAGE,
        "fixed_point": POINT,
        "files": [base.file_record(path) for path in manifest_paths],
        "transport_responses": rows,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
