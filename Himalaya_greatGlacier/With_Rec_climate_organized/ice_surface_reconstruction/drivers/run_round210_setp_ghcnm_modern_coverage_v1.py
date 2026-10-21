from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_210_setp_ghcnm_modern_coverage_v1"
STATIONS = (
    LOOP
    / "round_209_setp_ghcnm_modern_station_preflight_v1"
    / "ghcnm_stations_within_300km_metadata.csv"
)
DATA = (
    LOOP
    / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4"
    / "ghcnm_qcu_temperature.dat"
)
CONTRACT = OUT / "ROUND_CONTRACT.md"
YEARS = set(range(1940, 2001))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (STATIONS, DATA, CONTRACT):
        if not path.exists():
            raise FileNotFoundError(path)
    before = {"stations": sha256(STATIONS), "data": sha256(DATA)}
    stations = pd.read_csv(STATIONS)
    primary = stations[stations.distance_to_box_km <= 100].copy()
    if len(primary) != 4:
        raise RuntimeError(f"Expected four <=100 km candidates, found {len(primary)}")
    station_ids = set(primary.station_id)

    rows = []
    with DATA.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 115:
                continue
            station_id = line[0:11]
            if station_id not in station_ids or line[15:19] != "TAVG":
                continue
            year = int(line[11:15])
            if year not in YEARS:
                continue
            for month in range(1, 13):
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
    availability = pd.DataFrame(rows)
    if availability.empty:
        raise RuntimeError("No candidate GHCN-M availability rows")

    station_year = (
        availability.groupby(["station_id", "year"], as_index=False)
        .valid_month.sum()
        .rename(columns={"valid_month": "valid_month_count"})
    )
    station_year["complete_year"] = station_year.valid_month_count.eq(12)
    complete_counts = station_year.groupby("station_id").complete_year.sum()
    primary["complete_year_count_1940_2000"] = (
        primary.station_id.map(complete_counts).fillna(0).astype(int)
    )
    primary["eligible_ge30_complete_years"] = primary.complete_year_count_1940_2000 >= 30
    primary.to_csv(OUT / "station_coverage_summary.csv", index=False)
    station_year.to_csv(OUT / "station_year_availability.csv", index=False)

    eligible = set(primary.loc[primary.eligible_ge30_complete_years, "station_id"])
    annual = (
        station_year[
            station_year.station_id.isin(eligible) & station_year.complete_year
        ]
        .groupby("year", as_index=False)
        .station_id.nunique()
        .rename(columns={"station_id": "complete_eligible_station_count"})
        .set_index("year")
        .reindex(range(1940, 2001), fill_value=0)
        .rename_axis("year")
        .reset_index()
    )
    annual.to_csv(OUT / "annual_station_support.csv", index=False)
    supported_fraction = float(
        (annual.complete_eligible_station_count >= 2).mean()
    )
    gates = {
        "at_least_three_eligible_stations": len(eligible) >= 3,
        "two_station_support_at_least_75pct_years": supported_fraction >= 0.75,
    }
    decision = (
        "advance_phase_only_value_adjudication"
        if all(gates.values())
        else "close_direct_station_value_route_on_coverage"
    )
    summary = {
        "round": 210,
        "decision": decision,
        "candidate_station_count": len(primary),
        "eligible_station_count": len(eligible),
        "eligible_station_ids": sorted(eligible),
        "years_with_at_least_two_stations_fraction": supported_fraction,
        "gates": gates,
        "temperature_magnitudes_converted_or_retained": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    after = {"stations": sha256(STATIONS), "data": sha256(DATA)}
    if before != after:
        raise RuntimeError("GHCN-M source changed")
    (OUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
