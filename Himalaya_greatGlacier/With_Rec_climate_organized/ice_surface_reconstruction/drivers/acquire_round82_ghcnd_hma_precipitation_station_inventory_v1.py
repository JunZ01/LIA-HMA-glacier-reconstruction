"""Inventory GHCN-D precipitation stations on the exact HMA mask."""

from __future__ import annotations

import hashlib
import json
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
    / "round_82_ghcnd_hma_precipitation_station_inventory_v1"
)
DRIVER = Path(__file__).resolve()
URLS = {
    "stations": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt",
    "inventory": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt",
    "countries": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-countries.txt",
}
START_YEAR = 1979
END_YEAR = 2000
MIN_OVERLAP_SPAN = 15
MIN_WEST_STATIONS_TO_ADVANCE = 10


def file_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def parse_stations(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if len(line) < 71:
            continue
        rows.append(
            {
                "station_id": line[0:11],
                "latitude": float(line[12:20]),
                "longitude": float(line[21:30]),
                "elevation_m": float(line[31:37]),
                "name": line[41:71].strip(),
            }
        )
    return pd.DataFrame(rows)


def parse_inventory(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if len(line) < 45 or line[31:35] != "PRCP":
            continue
        rows.append(
            {
                "station_id": line[0:11],
                "inventory_latitude": float(line[12:20]),
                "inventory_longitude": float(line[21:30]),
                "element": line[31:35],
                "first_year": int(line[36:40]),
                "last_year": int(line[41:45]),
            }
        )
    return pd.DataFrame(rows)


def parse_countries(text: str) -> dict[str, str]:
    return {line[:2]: line[3:].strip() for line in text.splitlines() if len(line) >= 4}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 82 Contract",
                "",
                "## Objective",
                "Determine whether NOAA GHCN-D contains enough direct precipitation-station records to diagnose the 1979-2000 CPC-versus-gridded-product conflict over HMA, especially west of 80 E.",
                "",
                "## Frozen Rules",
                "- Official NCEI station, inventory, and country metadata only; no station precipitation values are read in this round.",
                "- Map stations to the exact 0.5-degree project grid and HMA mask by containing grid cell; no bounding-box-only count is accepted.",
                "- A long-overlap inventory record spans at least 15 years within 1979-2000 based on catalog endpoints. This does not imply daily completeness.",
                "- Advance to value-level completeness audit only if at least 10 western-HMA stations meet the long-overlap rule.",
                "- This station source may overlap gauge inputs to gridded products and is intended for station-location diagnosis, not an independence claim.",
                "- The locked 1921-1939 period is not requested or accessed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    responses: dict[str, dict[str, object]] = {}
    for label, url in URLS.items():
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        payload = response.content
        path = OUTPUT / f"ghcnd-{label}.txt"
        path.write_bytes(payload)
        responses[label] = {
            "url": response.url,
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        print(f"{label}: {len(payload):,} bytes", flush=True)

    stations = parse_stations((OUTPUT / "ghcnd-stations.txt").read_text(encoding="ascii", errors="replace"))
    inventory = parse_inventory((OUTPUT / "ghcnd-inventory.txt").read_text(encoding="ascii", errors="replace"))
    countries = parse_countries((OUTPUT / "ghcnd-countries.txt").read_text(encoding="ascii", errors="replace"))
    merged = inventory.merge(stations, on="station_id", how="inner", validate="one_to_one")
    if not np.allclose(merged.latitude, merged.inventory_latitude) or not np.allclose(
        merged.longitude, merged.inventory_longitude
    ):
        raise RuntimeError("station-inventory-coordinate-mismatch")

    with h5py.File(PUBLIC_SOURCE, "r") as handle:
        lat = np.asarray(handle["lat"][...], dtype=float)
        lon = np.asarray(handle["lon"][...], dtype=float)
    mask = np.load(HMA_MASK).astype(bool)
    ascending_lat = lat[::-1]
    lat_index_ascending = np.rint((merged.latitude.to_numpy() - ascending_lat[0]) / 0.5).astype(int)
    lon_index = np.rint((merged.longitude.to_numpy() - lon[0]) / 0.5).astype(int)
    inside_grid = (
        (lat_index_ascending >= 0)
        & (lat_index_ascending < lat.size)
        & (lon_index >= 0)
        & (lon_index < lon.size)
    )
    lat_index = lat.size - 1 - lat_index_ascending
    center_distance_ok = np.zeros(len(merged), dtype=bool)
    center_distance_ok[inside_grid] = (
        np.abs(merged.latitude.to_numpy()[inside_grid] - lat[lat_index[inside_grid]]) <= 0.2500001
    ) & (
        np.abs(merged.longitude.to_numpy()[inside_grid] - lon[lon_index[inside_grid]]) <= 0.2500001
    )
    in_hma = np.zeros(len(merged), dtype=bool)
    valid = inside_grid & center_distance_ok
    in_hma[valid] = mask[lat_index[valid], lon_index[valid]]
    selected = merged.loc[in_hma].copy()
    selected["grid_latitude"] = lat[lat_index[in_hma]]
    selected["grid_longitude"] = lon[lon_index[in_hma]]
    selected["country_code"] = selected.station_id.str[:2]
    selected["country"] = selected.country_code.map(countries).fillna("UNKNOWN")
    selected["overlap_start"] = np.maximum(selected.first_year, START_YEAR)
    selected["overlap_end"] = np.minimum(selected.last_year, END_YEAR)
    selected["overlap_span_years"] = np.maximum(0, selected.overlap_end - selected.overlap_start + 1)
    selected["long_overlap"] = selected.overlap_span_years >= MIN_OVERLAP_SPAN
    selected["covers_both_endpoints"] = (selected.first_year <= START_YEAR) & (selected.last_year >= END_YEAR)
    selected["region"] = np.where(selected.longitude < 80.0, "west", "nonwest")
    selected.sort_values(["region", "country", "station_id"]).to_csv(
        OUTPUT / "ghcnd_hma_prcp_station_inventory.csv", index=False
    )

    long_records = selected[selected.long_overlap]
    west_long = long_records[long_records.region == "west"]
    country_counts = (
        long_records.groupby(["region", "country"], as_index=False)
        .size()
        .rename(columns={"size": "station_count"})
    )
    country_counts.to_csv(OUTPUT / "ghcnd_hma_long_overlap_country_counts.csv", index=False)
    advance = int(len(west_long)) >= MIN_WEST_STATIONS_TO_ADVANCE
    summary = {
        "status": "complete",
        "decision": "advance_ghcnd_station_values_to_completeness_audit" if advance else "ghcnd_station_support_insufficient_for_western_adjudication",
        "locked_1921_1939_accessed": False,
        "all_global_prcp_inventory_records": int(len(merged)),
        "hma_prcp_station_count": int(len(selected)),
        "hma_long_overlap_station_count": int(len(long_records)),
        "western_hma_prcp_station_count": int((selected.region == "west").sum()),
        "western_hma_long_overlap_station_count": int(len(west_long)),
        "western_hma_endpoint_covering_station_count": int(
            ((selected.region == "west") & selected.covers_both_endpoints).sum()
        ),
        "minimum_long_overlap_years": MIN_OVERLAP_SPAN,
        "minimum_west_stations_to_advance": MIN_WEST_STATIONS_TO_ADVANCE,
        "countries_with_long_overlap": sorted(long_records.country.unique().tolist()),
        "station_values_read": False,
        "station_independence_claimed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 82 GHCN-D HMA Precipitation Inventory",
                "",
                f"- HMA PRCP stations: `{summary['hma_prcp_station_count']}`; long-overlap: `{summary['hma_long_overlap_station_count']}`.",
                f"- Western HMA PRCP stations: `{summary['western_hma_prcp_station_count']}`; long-overlap: `{summary['western_hma_long_overlap_station_count']}`.",
                f"- Western stations covering both 1979 and 2000 endpoints: `{summary['western_hma_endpoint_covering_station_count']}`.",
                f"- Countries with long-overlap stations: `{summary['countries_with_long_overlap']}`.",
                "- Station values were not read; catalog endpoints do not establish daily completeness.",
                f"- Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_paths = [DRIVER, contract, PUBLIC_SOURCE, HMA_MASK, *[OUTPUT / f"ghcnd-{label}.txt" for label in URLS], OUTPUT / "ghcnd_hma_prcp_station_inventory.csv", OUTPUT / "ghcnd_hma_long_overlap_country_counts.csv", OUTPUT / "summary.json", OUTPUT / "FINAL_RESULT.md"]
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"official_responses": responses, "files": [file_record(path) for path in manifest_paths]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
