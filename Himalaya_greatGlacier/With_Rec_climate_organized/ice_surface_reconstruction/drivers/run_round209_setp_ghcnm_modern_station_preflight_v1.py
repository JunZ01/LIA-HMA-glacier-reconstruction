from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_209_setp_ghcnm_modern_station_preflight_v1"
INVENTORY = (
    LOOP
    / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4"
    / "ghcnm_qcu_inventory.inv"
)
COUNTRIES = (
    LOOP
    / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4"
    / "ghcnm-countries.txt"
)
CONTRACT = OUT / "ROUND_CONTRACT.md"
BOX = {"lon_min": 94.5, "lon_max": 97.5, "lat_min": 29.5, "lat_max": 31.5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def distance_to_box_km(latitude: float, longitude: float) -> float:
    nearest_lat = min(max(latitude, BOX["lat_min"]), BOX["lat_max"])
    nearest_lon = min(max(longitude, BOX["lon_min"]), BOX["lon_max"])
    return haversine_km(latitude, longitude, nearest_lat, nearest_lon)


def parse_countries(path: Path) -> dict[str, str]:
    output = {}
    for raw in path.read_bytes().splitlines():
        line = raw.decode("ascii", errors="replace")
        if len(line) >= 5:
            output[line[:2]] = line[3:].strip()
    return output


def main() -> None:
    for path in (INVENTORY, COUNTRIES, CONTRACT):
        if not path.exists():
            raise FileNotFoundError(path)
    before = {"inventory": sha256(INVENTORY), "countries": sha256(COUNTRIES)}
    countries = parse_countries(COUNTRIES)
    rows = []
    for raw in INVENTORY.read_bytes().splitlines():
        line = raw.decode("ascii", errors="replace")
        if len(line) < 68:
            continue
        latitude = float(line[12:20])
        longitude = float(line[21:30])
        distance = distance_to_box_km(latitude, longitude)
        if distance <= 300:
            station_id = line[0:11]
            rows.append(
                {
                    "station_id": station_id,
                    "country": countries.get(station_id[:2], ""),
                    "latitude": latitude,
                    "longitude": longitude,
                    "elevation_m": float(line[31:37]),
                    "name": line[38:68].strip(),
                    "distance_to_box_km": round(distance, 3),
                    "inside_box": distance == 0,
                }
            )
    rows.sort(key=lambda row: (float(row["distance_to_box_km"]), str(row["station_id"])))
    columns = list(rows[0]) if rows else [
        "station_id",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "name",
        "distance_to_box_km",
        "inside_box",
    ]
    with (OUT / "ghcnm_stations_within_300km_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        "inside_box": sum(bool(row["inside_box"]) for row in rows),
        "within_100km": sum(float(row["distance_to_box_km"]) <= 100 for row in rows),
        "within_200km": sum(float(row["distance_to_box_km"]) <= 200 for row in rows),
        "within_300km": len(rows),
    }
    decision = (
        "advance_value_coverage_audit"
        if counts["inside_box"] >= 3 or counts["within_100km"] >= 5
        else "direct_station_geometry_insufficient_for_box_adjudication"
    )
    summary = {
        "round": 209,
        "decision": decision,
        "box": BOX,
        "station_counts": counts,
        "temperature_value_file_opened": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    after = {"inventory": sha256(INVENTORY), "countries": sha256(COUNTRIES)}
    if before != after:
        raise RuntimeError("GHCN-M metadata source changed")
    (OUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
