"""Quantify candidate temporal support using age columns only."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES = {
    "guliya2015_core1_annual": {
        "file": "guliya2015_core1_annual.txt",
        "age_column": "Age_CE",
        "age_scale": "CE",
        "latitude": 35.233,
        "longitude": 81.468258,
        "archive": "ice_core",
        "variable_role": "raw_d18O_temperature_snowfall_proxy",
    },
    "guliya2015_core2_annual": {
        "file": "guliya2015_core2_annual.txt",
        "age_column": "Age_CE",
        "age_scale": "CE",
        "latitude": 35.233,
        "longitude": 81.468258,
        "archive": "ice_core",
        "variable_role": "raw_d18O_temperature_snowfall_proxy",
    },
    "aweng_co_temperature": {
        "file": "aweng_co_temperature.txt",
        "age_column": "age",
        "age_scale": "BP1950",
        "latitude": 32.75,
        "longitude": 81.67,
        "archive": "lake_sediment",
        "variable_role": "reconstructed_surface_air_temperature",
    },
    "heihai_lake_temperature": {
        "file": "heihai_lake_temperature.txt",
        "age_column": "age",
        "age_scale": "BP1950",
        "latitude": 27.36,
        "longitude": 100.07,
        "archive": "lake_sediment",
        "variable_role": "reconstructed_summer_temperature",
    },
    "ngamring_co_maat": {
        "file": "ngamring_co_d2h_maat.txt",
        "age_column": "age_BP",
        "age_scale": "BP1950",
        "latitude": 29.3113,
        "longitude": 87.1878,
        "archive": "lake_sediment",
        "variable_role": "brGDGT_reconstructed_mean_annual_temperature",
    },
    "shenqi_cave_precipitation_index": {
        "file": "shenqi_cave_precipitation_index.txt",
        "age_column": "age_CE",
        "age_scale": "CE",
        "latitude": 28.93,
        "longitude": 103.1,
        "archive": "speleothem",
        "variable_role": "reconstructed_precipitation_index",
    },
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_age_only(path: Path, age_column: str, age_scale: str) -> tuple[np.ndarray, str]:
    header = None
    age_index = None
    ages = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            if header is None:
                if age_column not in parts:
                    continue
                header = stripped
                age_index = parts.index(age_column)
                continue
            if age_index is None or len(parts) <= age_index:
                continue
            token = parts[age_index].strip()
            try:
                age = float(token)
            except ValueError:
                continue
            ages.append(1950.0 - age if age_scale == "BP1950" else age)
    if header is None or not ages:
        raise RuntimeError(f"age-column-read-failed:{path.name}:{age_column}")
    return np.asarray(ages, dtype=float), header


def haversine_km(lat: float, lon: float, other_lat: np.ndarray, other_lon: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(other_lat)
    lon2 = np.radians(other_lon)
    value = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(value))


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_proxy_spatial_gap_discovery_20260818_v1"
    )
    raw = output / "raw_unopened"
    current_path = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_proxy_identity_closure_20260818_v1/"
        "combined_proxy_metadata_only.csv"
    )
    current = pd.read_csv(current_path)
    current_lat = current.latitude.to_numpy(float)
    current_lon = current.longitude.to_numpy(float)
    rows = []
    headers = {}
    source_states = {}
    for name, config in CANDIDATES.items():
        path = raw / config["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        ages, header = read_age_only(path, config["age_column"], config["age_scale"])
        headers[name] = header
        source_states[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        finite = np.unique(ages[np.isfinite(ages)])
        selected = finite[(finite >= 1500) & (finite <= 2000)]
        allowed_early = selected[selected <= 1920]
        allowed_modern = selected[selected >= 1940]
        spacing = np.diff(np.sort(selected))
        distances = haversine_km(
            config["latitude"], config["longitude"], current_lat, current_lon
        )
        nearest = int(np.nanargmin(distances))
        annual_ready = bool(
            len(allowed_modern) >= 50
            and (len(spacing) == 0 or float(np.nanmedian(spacing)) <= 1.1)
            and config["variable_role"].startswith("raw_")
        )
        rows.append(
            {
                "candidate": name,
                "archive": config["archive"],
                "variable_role": config["variable_role"],
                "latitude": config["latitude"],
                "longitude": config["longitude"],
                "first_age_ce": float(np.nanmin(finite)),
                "last_age_ce": float(np.nanmax(finite)),
                "sample_count_1500_2000": int(len(selected)),
                "sample_count_1500_1920": int(len(allowed_early)),
                "sample_count_1940_2000": int(len(allowed_modern)),
                "median_spacing_years_1500_2000": (
                    float(np.nanmedian(spacing)) if len(spacing) else np.nan
                ),
                "nearest_current_record_1based": int(current.iloc[nearest].record_1based),
                "nearest_current_distance_km": float(distances[nearest]),
                "annual_oof_candidate": annual_ready,
                "recommended_role": (
                    "annual_oof_identity_candidate"
                    if annual_ready
                    else "low_frequency_external_benchmark"
                ),
            }
        )
    with (output / "candidate_age_support.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "age_columns_only_decoded": True,
        "proxy_or_reconstruction_values_decoded": False,
        "locked_1921_1939_proxy_or_reconstruction_values_decoded": False,
        "headers": headers,
        "candidates": rows,
        "source_states": source_states,
        "original_sources_modified": False,
    }
    (output / "candidate_age_support_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
