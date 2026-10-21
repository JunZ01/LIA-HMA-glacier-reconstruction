"""Metadata-only eligibility audit for 20CRv3 and GHCN-D temperature."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import requests


HOLDOUT_START = 2001
HOLDOUT_END = 2015
MIN_OVERLAP_YEARS = 12
MIN_TOTAL_STATIONS = 30
MIN_REGION_STATIONS = 5
MIN_COUNTRIES = 5
REGIONS = {
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
URLS = {
    "official_description": "https://www.psl.noaa.gov/data/gridded/data.20thC_ReanV3.html",
    "monthly_2m_catalog": "https://psl.noaa.gov/thredds/catalog/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/catalog.xml",
    "monthly_2m_dds": "https://psl.noaa.gov/thredds/dodsC/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/air.2m.mon.mean.nc.dds",
    "monthly_2m_das": "https://psl.noaa.gov/thredds/dodsC/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/air.2m.mon.mean.nc.das",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


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
        if len(line) < 45:
            continue
        element = line[31:35]
        if element not in ("TMAX", "TMIN"):
            continue
        rows.append(
            {
                "station_id": line[0:11],
                "inventory_latitude": float(line[12:20]),
                "inventory_longitude": float(line[21:30]),
                "element": element,
                "first_year": int(line[36:40]),
                "last_year": int(line[41:45]),
            }
        )
    return pd.DataFrame(rows)


def parse_countries(text: str) -> dict[str, str]:
    return {line[:2]: line[3:].strip() for line in text.splitlines() if len(line) >= 4}


def map_to_hma(
    stations: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    ascending_lat = lat[::-1]
    lat_ascending = np.rint((stations.latitude.to_numpy() - ascending_lat[0]) / 0.5).astype(int)
    lon_index = np.rint((stations.longitude.to_numpy() - lon[0]) / 0.5).astype(int)
    inside = (
        (lat_ascending >= 0)
        & (lat_ascending < lat.size)
        & (lon_index >= 0)
        & (lon_index < lon.size)
    )
    lat_index = lat.size - 1 - lat_ascending
    centered = np.zeros(len(stations), dtype=bool)
    centered[inside] = (
        np.abs(stations.latitude.to_numpy()[inside] - lat[lat_index[inside]]) <= 0.2500001
    ) & (
        np.abs(stations.longitude.to_numpy()[inside] - lon[lon_index[inside]]) <= 0.2500001
    )
    in_hma = np.zeros(len(stations), dtype=bool)
    valid = inside & centered
    in_hma[valid] = mask[lat_index[valid], lon_index[valid]]
    result = stations.loc[in_hma].copy()
    result["grid_latitude"] = lat[lat_index[in_hma]]
    result["grid_longitude"] = lon[lon_index[in_hma]]
    return result


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    round82 = loop / "round_82_ghcnd_hma_precipitation_station_inventory_v1"
    output = loop / "round_89_20crv3_ghcnd_temperature_metadata_eligibility_v1"
    public = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    paths = {
        "ghcnd_stations": round82 / "ghcnd-stations.txt",
        "ghcnd_inventory": round82 / "ghcnd-inventory.txt",
        "ghcnd_countries": round82 / "ghcnd-countries.txt",
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "public_grid": public,
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    remote_metadata: dict[str, Any] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-temperature-metadata-audit/1.0"})
    try:
        for name, url in URLS.items():
            response = session.get(url, timeout=180)
            response.raise_for_status()
            payload = response.content
            (output / f"20crv3_{name}.txt").write_bytes(payload)
            remote_metadata[name] = {
                "url": response.url,
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    finally:
        session.close()

    description = (output / "20crv3_official_description.txt").read_text(encoding="utf-8", errors="replace")
    catalog = (output / "20crv3_monthly_2m_catalog.txt").read_text(encoding="utf-8", errors="replace")
    dds = (output / "20crv3_monthly_2m_dds.txt").read_text(encoding="utf-8", errors="replace")
    das = (output / "20crv3_monthly_2m_das.txt").read_text(encoding="utf-8", errors="replace")
    time_match = re.search(r"time\s*=\s*(\d+)", dds)
    lat_match = re.search(r"lat\s*=\s*(\d+)", dds)
    lon_match = re.search(r"lon\s*=\s*(\d+)", dds)
    if not time_match or not lat_match or not lon_match:
        raise RuntimeError("20crv3-dds-dimensions-not-found")
    dimensions = {
        "monthly_time_count": int(time_match.group(1)),
        "latitude_count": int(lat_match.group(1)),
        "longitude_count": int(lon_match.group(1)),
    }
    source_gates = {
        "official_page_identifies_20crv3": "20th Century Reanalysis V3" in description,
        "official_page_states_surface_pressure_observations": "surface pressure" in description.lower(),
        "official_page_lists_monthly_2m_air_temperature": "2 m" in description and "Monthly" in description and "Air Temperature" in description,
        "catalog_contains_monthly_2m_air_file": "air.2m.mon.mean.nc" in catalog,
        "dds_contains_air_time_lat_lon_grid": all(token in dds for token in ("air[time", "[lat", "[lon")),
        "dds_grid_is_1degree_global": dimensions["latitude_count"] == 181 and dimensions["longitude_count"] == 360,
        "dds_month_count_matches_1806_2015": dimensions["monthly_time_count"] == 210 * 12,
        "das_units_are_kelvin": "units \"degK\"" in das or "units \"K\"" in das,
    }

    stations = parse_stations(paths["ghcnd_stations"].read_text(encoding="ascii", errors="replace"))
    inventory = parse_inventory(paths["ghcnd_inventory"].read_text(encoding="ascii", errors="replace"))
    countries = parse_countries(paths["ghcnd_countries"].read_text(encoding="ascii", errors="replace"))
    paired = inventory.pivot(index="station_id", columns="element", values=["first_year", "last_year"])
    paired.columns = [f"{left}_{right.lower()}" for left, right in paired.columns]
    paired = paired.reset_index().dropna()
    paired = paired.merge(stations, on="station_id", how="inner", validate="one_to_one")
    paired["paired_first_year"] = paired[["first_year_tmax", "first_year_tmin"]].max(axis=1).astype(int)
    paired["paired_last_year"] = paired[["last_year_tmax", "last_year_tmin"]].min(axis=1).astype(int)
    paired["overlap_start"] = np.maximum(paired.paired_first_year, HOLDOUT_START)
    paired["overlap_end"] = np.minimum(paired.paired_last_year, HOLDOUT_END)
    paired["catalog_overlap_years"] = np.maximum(0, paired.overlap_end - paired.overlap_start + 1)
    paired["catalog_eligible"] = paired.catalog_overlap_years >= MIN_OVERLAP_YEARS

    with h5py.File(paths["public_grid"], "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
    mask = np.load(paths["hma_mask"]).astype(bool)
    selected = map_to_hma(paired, lat, lon, mask)
    selected["country_code"] = selected.station_id.str[:2]
    selected["country"] = selected.country_code.map(countries).fillna("UNKNOWN")
    selected["region"] = ""
    for region, (lower, upper) in REGIONS.items():
        choose = (selected.longitude >= lower) & (selected.longitude < upper)
        selected.loc[choose, "region"] = region
    selected = selected.sort_values(["region", "country", "station_id"])
    selected.to_csv(output / "ghcnd_hma_paired_temperature_station_inventory.csv", index=False)
    eligible = selected[selected.catalog_eligible].copy()
    region_counts = eligible.groupby("region").size().reindex(REGIONS, fill_value=0)
    country_counts = eligible.groupby(["region", "country"], as_index=False).size()
    country_counts.to_csv(output / "eligible_station_country_counts.csv", index=False)
    station_gates = {
        "at_least_30_hma_catalog_eligible_stations": len(eligible) >= MIN_TOTAL_STATIONS,
        "at_least_5_stations_in_each_region": bool((region_counts >= MIN_REGION_STATIONS).all()),
        "at_least_5_countries": eligible.country.nunique() >= MIN_COUNTRIES,
        "no_station_temperature_values_read": True,
    }
    gates = {**source_gates, **station_gates}
    accepted = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "temperature_values_read": False,
        "holdout_years": [HOLDOUT_START, HOLDOUT_END],
        "source_candidate": "NOAA_CIRES_DOE_20CRv3_monthly_2m_air_ensemble_mean",
        "source_assimilation_scope": "surface_pressure_only_land_air_temperature_not_directly_assimilated",
        "source_dimensions": dimensions,
        "ghcnd_hma_paired_tmax_tmin_station_count": int(len(selected)),
        "ghcnd_catalog_eligible_station_count": int(len(eligible)),
        "eligible_region_counts": {name: int(region_counts[name]) for name in REGIONS},
        "eligible_country_count": int(eligible.country.nunique()),
        "eligible_countries": sorted(eligible.country.unique().tolist()),
        "gates": gates,
        "passes_metadata_eligibility": accepted,
        "decision": (
            "freeze_20crv3_vs_ghcnd_2001_2015_value_validation_protocol"
            if accepted
            else "stop_new_temperature_route_on_metadata_ineligibility"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# 20CRv3 and GHCN-D Temperature Metadata Eligibility",
                "",
                f"- Temperature values read: `false`.",
                f"- 20CRv3 dimensions: `{dimensions}`.",
                f"- HMA paired TMAX/TMIN stations: `{len(selected)}`; catalog-eligible for 2001-2015: `{len(eligible)}`.",
                f"- Eligible region counts: `{summary['eligible_region_counts']}`.",
                f"- Eligible country count: `{summary['eligible_country_count']}`.",
                f"- All metadata gates pass: `{str(accepted).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "local_sources_before": before,
                "local_sources_after": after,
                "local_source_states_unchanged": unchanged,
                "official_metadata_responses": remote_metadata,
                "temperature_values_read": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("round89-local-source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
