"""Acquire the allowed-period NOAA CPC Unified daily precipitation subset.

The source responses are treated as immutable transport objects.  This round
stores annual means and finite-day counts on the project's exact public grid,
plus enough request metadata and hashes to audit every annual response.
"""

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


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
PUBLIC_SOURCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
HMA_MASK = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_input_spatial_audit_v3"
    / "hma_mask_on_netcdf_grid.npy"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_80_cpc_unified_allowed_subset_acquisition_v1"
)
DRIVER = Path(__file__).resolve()
YEARS = np.arange(1979, 2001, dtype=np.int16)
BASE_URL = (
    "https://psl.noaa.gov/thredds/ncss/grid/Datasets/"
    "cpc_global_precip/precip.{year}.nc"
)
DATASET_PAGE = "https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html"
BOUNDS = {"north": 45.75, "south": 24.75, "west": 66.25, "east": 104.75}


def file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def expected_days(year: int) -> int:
    return (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days


def decode_time(values: np.ndarray, units: str) -> list[datetime]:
    prefix = "hours since "
    if not units.startswith(prefix):
        raise RuntimeError(f"unsupported-time-units:{units}")
    origin_text = units[len(prefix) :].strip()
    origin = datetime.fromisoformat(origin_text)
    return [origin + timedelta(hours=float(value)) for value in values]


def attribute_text(dataset: h5py.Dataset, key: str) -> str:
    value = dataset.attrs[key]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 80 Contract",
                "",
                "## Objective",
                "Acquire NOAA CPC Global Unified Gauge-Based Daily Precipitation for 1979-2000 on the exact 0.5-degree public reconstruction grid over HMA, then aggregate each calendar year to mean daily precipitation.",
                "",
                "## Frozen Inputs And Rules",
                "- Official source: NOAA PSL CPC Global Unified daily precipitation annual files through NCSS.",
                "- Allowed years only: 1979-2000. The locked 1921-1939 final-test period is not requested or accessed.",
                "- Grid centers must exactly equal the existing public reconstruction 43x78 grid; no interpolation is allowed.",
                "- Every annual response must have the complete Gregorian daily axis for that year.",
                "- Annual values are arithmetic means of finite daily totals in mm/day. Missing values are not interpolated.",
                "- Preserve per-cell finite-day counts and request URL, status, byte count, and SHA256 for every response.",
                "- Original source and project climate files are read-only.",
                "",
                "## Exit Rule",
                "Pass only if all 22 years decode, all date axes are complete, coordinates match exactly, and every HMA-mask cell has all expected daily values in every year.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with h5py.File(PUBLIC_SOURCE, "r") as handle:
        target_lat = np.asarray(handle["lat"][...], dtype=np.float64)
        target_lon = np.asarray(handle["lon"][...], dtype=np.float64)
    mask = np.load(HMA_MASK).astype(bool)
    if mask.shape != (target_lat.size, target_lon.size):
        raise RuntimeError("hma-mask-grid-shape-mismatch")

    annual = np.full((YEARS.size, target_lat.size, target_lon.size), np.nan, dtype=np.float32)
    counts = np.zeros_like(annual, dtype=np.int16)
    request_rows: list[dict[str, object]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-reconstruction-audit/1.0"})

    for year_index, year_value in enumerate(YEARS):
        year = int(year_value)
        start = f"{year}-01-01T00:00:00Z"
        end = f"{year}-12-31T00:00:00Z"
        params = {
            "var": "precip",
            **{key: f"{value:.2f}" for key, value in BOUNDS.items()},
            "horizStride": "1",
            "time_start": start,
            "time_end": end,
            "timeStride": "1",
            "accept": "netcdf4",
        }
        response = session.get(BASE_URL.format(year=year), params=params, timeout=600)
        response.raise_for_status()
        payload = response.content
        payload_hash = hashlib.sha256(payload).hexdigest()

        with h5py.File(io.BytesIO(payload), "r") as handle:
            required = {"lat", "lon", "time", "precip"}
            if not required.issubset(handle.keys()):
                raise RuntimeError(f"missing-netcdf-variable:{year}:{sorted(required - set(handle.keys()))}")
            lat = np.asarray(handle["lat"][...], dtype=np.float64)
            lon = np.asarray(handle["lon"][...], dtype=np.float64)
            time = np.asarray(handle["time"][...], dtype=np.float64)
            field = np.asarray(handle["precip"][...], dtype=np.float64)
            units = attribute_text(handle["time"], "units")
            calendar = attribute_text(handle["time"], "calendar")
            precip_units = attribute_text(handle["precip"], "units")

        if not np.array_equal(lat, target_lat) or not np.array_equal(lon, target_lon):
            raise RuntimeError(f"coordinate-mismatch:{year}")
        if field.shape != (expected_days(year), target_lat.size, target_lon.size):
            raise RuntimeError(f"field-shape-mismatch:{year}:{field.shape}")
        if calendar.lower() not in {"gregorian", "standard"}:
            raise RuntimeError(f"calendar-mismatch:{year}:{calendar}")
        if precip_units != "mm":
            raise RuntimeError(f"precipitation-unit-mismatch:{year}:{precip_units}")
        dates = decode_time(time, units)
        expected = [datetime(year, 1, 1) + timedelta(days=index) for index in range(expected_days(year))]
        if dates != expected:
            raise RuntimeError(f"daily-axis-mismatch:{year}")

        finite = np.isfinite(field)
        count = finite.sum(axis=0)
        total = np.where(finite, field, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        annual[year_index] = mean.astype(np.float32)
        counts[year_index] = count.astype(np.int16)
        hma_complete = bool(np.all(count[mask] == expected_days(year)))
        request_rows.append(
            {
                "year": year,
                "request_url": response.url,
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "byte_count": len(payload),
                "sha256": payload_hash,
                "expected_day_count": expected_days(year),
                "decoded_day_count": len(dates),
                "hma_complete": hma_complete,
                "hma_min_finite_day_count": int(count[mask].min()),
                "hma_max_finite_day_count": int(count[mask].max()),
            }
        )
        print(f"{year}: {len(payload):,} bytes, HMA complete={hma_complete}", flush=True)

    requests_frame = pd.DataFrame(request_rows)
    requests_frame.to_csv(OUTPUT / "request_manifest.csv", index=False)
    all_hma_complete = bool(requests_frame["hma_complete"].all())
    if not all_hma_complete:
        raise RuntimeError("incomplete-hma-daily-coverage")

    target = OUTPUT / "cpc_unified_precip_1979_2000_public_grid_annual_mm_day.nc"
    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "NOAA CPC Unified annual mean daily precipitation, HMA public grid"
        handle.attrs["source_dataset_page"] = DATASET_PAGE
        handle.attrs["aggregation"] = "calendar-year arithmetic mean of finite daily totals"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.create_dataset("year", data=YEARS)
        handle.create_dataset("lat", data=target_lat.astype(np.float32))
        handle.create_dataset("lon", data=target_lon.astype(np.float32))
        precipitation = handle.create_dataset(
            "precip_annual_mean_mm_day",
            data=annual,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
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
        "year_start": int(YEARS[0]),
        "year_end": int(YEARS[-1]),
        "year_count": int(YEARS.size),
        "grid_shape": [int(target_lat.size), int(target_lon.size)],
        "hma_cell_count": int(mask.sum()),
        "west_hma_cell_count": int((mask & (target_lon[None, :] < 80.0)).sum()),
        "all_hma_cells_complete_daily_coverage": all_hma_complete,
        "total_transport_bytes": int(requests_frame["byte_count"].sum()),
        "request_count": int(len(requests_frame)),
        "output_file": str(target),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 80 Result",
                "",
                "- Status: `complete`.",
                "- NOAA CPC Unified daily precipitation acquired for `1979-2000` on the exact project public grid.",
                f"- Grid: `{target_lat.size} x {target_lon.size}`; HMA mask cells: `{int(mask.sum())}`.",
                f"- Requests: `{len(requests_frame)}`; transport bytes: `{int(requests_frame['byte_count'].sum()):,}`.",
                f"- Complete daily coverage at every HMA cell in every year: `{str(all_hma_complete).lower()}`.",
                "- No interpolation, gap filling, candidate fitting, or locked-period access occurred.",
                "- Decision: `advance_cpc_to_fixed_candidate_adjudication`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_paths = [DRIVER, contract, HMA_MASK, PUBLIC_SOURCE, target, OUTPUT / "request_manifest.csv", OUTPUT / "summary.json", OUTPUT / "FINAL_RESULT.md"]
    manifest = {
        "official_dataset_page": DATASET_PAGE,
        "ncss_pattern": BASE_URL,
        "files": [file_record(path) for path in manifest_paths],
        "transport_responses": request_rows,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
