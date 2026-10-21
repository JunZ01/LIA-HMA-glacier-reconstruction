"""Audit GHCN-M 1901-1920 HMA coverage without retaining temperatures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_climate_reconstruction_loop_v2_round93_20crv3_ghcnm_v4_temperature_value_validation_v4 as round93


YEARS = range(1901, 1921)
MONTHS = range(1, 13)
MIN_COMPLETE_YEARS = 15
REGIONS = {
    "west_lt80": (55.0, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, 116.000001),
}
EXPECTED_FACTS_SHA256 = "04067a626ca11d5fb2fb2e510ad7f7db51e6e2fbc83e1a686baba529cd21618a"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest(path),
    }


def parse_availability(path: Path, station_ids: set[str]) -> pd.DataFrame:
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 115:
                continue
            station_id = line[0:11]
            if station_id not in station_ids or line[15:19] != "TAVG":
                continue
            year = int(line[11:15])
            if year not in YEARS:
                continue
            for month in MONTHS:
                start = 19 + (month - 1) * 8
                present = line[start : start + 5] != "-9999"
                qc_blank = not line[start + 6 : start + 7].strip()
                rows.append(
                    {
                        "station_id": station_id,
                        "year": year,
                        "month": month,
                        "valid_month": bool(present and qc_blank),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    root = round93.project_root()
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0"
        / "climate_reconstruction_improvement_loop_v2"
        / "round_96_ghcnm_1901_1920_temperature_coverage_readiness_v1"
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": output / "ROUND_CONTRACT.md",
        "ghcnm_data": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_temperature.dat",
        "ghcnm_inventory": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_inventory.inv",
        "ghcnm_countries": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm-countries.txt",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "public_grid": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(paths["facts"]) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(paths['facts'])}")
    before = {name: state(path) for name, path in paths.items()}

    inventory = round93.parse_inventory(paths["ghcnm_inventory"])
    countries = round93.parse_countries(paths["ghcnm_countries"])
    with h5py.File(paths["public_grid"], "r") as handle:
        fine_lat = np.asarray(handle["lat"], dtype=float)
        fine_lon = np.asarray(handle["lon"], dtype=float)
    mask = np.load(paths["hma_mask"]).astype(bool)
    hma = round93.map_to_hma(inventory, fine_lat, fine_lon, mask)
    hma["country_code"] = hma.station_id.str[:2]
    hma["country"] = hma.country_code.map(countries).fillna("UNKNOWN")
    hma["region"] = ""
    for region, (lower, upper) in REGIONS.items():
        selected = (hma.longitude >= lower) & (hma.longitude < upper)
        hma.loc[selected, "region"] = region
    hma.to_csv(output / "ghcnm_hma_station_inventory.csv", index=False)

    availability = parse_availability(paths["ghcnm_data"], set(hma.station_id))
    if availability.empty:
        raise RuntimeError("no-ghcnm-hma-1901-1920-records")
    availability.to_csv(output / "monthly_availability_1901_1920.csv", index=False)
    station_year = (
        availability.groupby(["station_id", "year"], as_index=False).valid_month.sum()
        .assign(complete_year=lambda table: table.valid_month.eq(12))
    )
    complete_counts = station_year.groupby("station_id").complete_year.sum()
    final_ids = complete_counts[complete_counts >= MIN_COMPLETE_YEARS].index
    final = hma[hma.station_id.isin(final_ids)].copy()
    final["complete_year_count"] = final.station_id.map(complete_counts).astype(int)
    final.to_csv(output / "final_station_support.csv", index=False)

    complete = station_year[station_year.station_id.isin(final_ids) & station_year.complete_year].merge(
        final[["station_id", "country", "region"]], on="station_id", how="left", validate="many_to_one"
    )
    annual_hma = complete.groupby("year", as_index=False).agg(
        station_count=("station_id", "nunique"), country_count=("country", "nunique")
    )
    annual_region = complete.groupby(["region", "year"], as_index=False).agg(
        station_count=("station_id", "nunique"), country_count=("country", "nunique")
    )
    full_year_index = pd.MultiIndex.from_product([REGIONS, YEARS], names=["region", "year"])
    annual_region = (
        annual_region.set_index(["region", "year"])
        .reindex(full_year_index, fill_value=0)
        .reset_index()
    )
    annual_hma = (
        annual_hma.set_index("year")
        .reindex(list(YEARS), fill_value=0)
        .rename_axis("year")
        .reset_index()
    )
    annual_hma.to_csv(output / "annual_hma_support.csv", index=False)
    annual_region.to_csv(output / "annual_region_support.csv", index=False)

    region_station_counts = final.groupby("region").size().to_dict()
    region_country_counts = final.groupby("region").country.nunique().to_dict()
    full_gates = {
        "at_least_15_final_stations": len(final) >= 15,
        "at_least_3_final_stations_each_region": all(region_station_counts.get(region, 0) >= 3 for region in REGIONS),
        "at_least_4_final_countries": final.country.nunique() >= 4,
        "at_least_10_stations_every_year": int(annual_hma.station_count.min()) >= 10,
        "at_least_2_stations_each_region_every_year": all(
            int(annual_region[annual_region.region == region].station_count.min()) >= 2 for region in REGIONS
        ),
    }
    subregional_gates = {}
    eligible_regions = []
    for region in REGIONS:
        region_gates = {
            "at_least_5_final_stations": region_station_counts.get(region, 0) >= 5,
            "at_least_2_countries": region_country_counts.get(region, 0) >= 2,
            "at_least_3_stations_every_year": int(
                annual_region[annual_region.region == region].station_count.min()
            ) >= 3,
        }
        subregional_gates[region] = region_gates
        if all(region_gates.values()):
            eligible_regions.append(region)

    full_pass = all(full_gates.values())
    if full_pass:
        decision = "freeze_full_hma_direct_station_phase_adjudication"
    elif eligible_regions:
        decision = "freeze_subregional_direct_station_phase_adjudication"
    else:
        decision = "close_ghcnm_early_direct_station_route_on_support"
    summary = {
        "status": "complete",
        "years": [1901, 1920],
        "temperature_magnitudes_retained_or_used": False,
        "locked_1921_1939_accessed": False,
        "hma_inventory_count": int(len(hma)),
        "final_station_count": int(len(final)),
        "final_region_counts": {region: int(region_station_counts.get(region, 0)) for region in REGIONS},
        "final_country_count": int(final.country.nunique()),
        "minimum_hma_station_count_per_year": int(annual_hma.station_count.min()),
        "minimum_region_station_count_per_year": {
            region: int(annual_region[annual_region.region == region].station_count.min()) for region in REGIONS
        },
        "full_hma_gates": full_gates,
        "passes_full_hma_support": full_pass,
        "subregional_gates": subregional_gates,
        "eligible_subregions": eligible_regions,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# GHCN-M 1901-1920 Coverage Readiness",
                "",
                f"- Full-HMA support: `{str(full_pass).lower()}`.",
                f"- Final stations / regions / countries: `{len(final)}` / `{summary['final_region_counts']}` / `{summary['final_country_count']}`.",
                f"- Minimum annual HMA / regional station support: `{summary['minimum_hma_station_count_per_year']}` / `{summary['minimum_region_station_count_per_year']}`.",
                f"- Eligible subregions: `{eligible_regions}`.",
                "- Temperature magnitudes were not retained or used.",
                "",
                f"Decision: `{decision}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {
        "local_sources_before": before,
        "local_sources_after": after,
        "local_source_states_unchanged": before == after,
        "parsed_fields": ["station_id", "year", "month", "missing_code", "qc_flag"],
        "temperature_magnitudes_retained_or_used": False,
        "locked_1921_1939_accessed": False,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
