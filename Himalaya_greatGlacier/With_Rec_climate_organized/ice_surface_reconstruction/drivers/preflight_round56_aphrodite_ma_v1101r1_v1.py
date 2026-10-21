"""Metadata-only preflight for registered APHRO_MA V1101R1 NetCDF files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


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


def find_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    lookup = {name.lower(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise RuntimeError(f"required-variable-not-found:{candidates}:{list(dataset.variables)}")


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_56_aphrodite_external_input_readiness_v1"
    input_dir = project / "validation_data/external_inputs/aphrodite_ma_v1101r1_05deg"
    files = sorted([*input_dir.glob("*.nc"), *input_dir.glob("*.nc4")])
    if not files:
        print(json.dumps({"status": "waiting_for_external_input", "input_directory": str(input_dir)}, indent=2))
        return 2

    file_rows: list[dict[str, object]] = []
    all_years: set[int] = set()
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    precipitation_variables: set[str] = set()
    for path in files:
        with xr.open_dataset(path, decode_times=True, chunks=None) as dataset:
            time_name = find_name(dataset, ("time",))
            lat_name = find_name(dataset, ("lat", "latitude", "y"))
            lon_name = find_name(dataset, ("lon", "longitude", "x"))
            precipitation_name = find_name(
                dataset, ("precip", "precipitation", "pr", "daily_precip")
            )
            years = np.asarray(dataset[time_name].dt.year.values, dtype=int).ravel()
            lat = np.asarray(dataset[lat_name].values, dtype=float).ravel()
            lon = np.asarray(dataset[lon_name].values, dtype=float).ravel()
            if reference_lat is None:
                reference_lat = lat
                reference_lon = lon
            elif not np.array_equal(reference_lat, lat) or not np.array_equal(reference_lon, lon):
                raise RuntimeError(f"grid-varies-between-files:{path.name}")
            all_years.update(years.tolist())
            precipitation_variables.add(precipitation_name)
            file_rows.append(
                {
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256(path),
                    "first_year": int(years.min()),
                    "last_year": int(years.max()),
                    "time_count": int(len(years)),
                    "precipitation_variable": precipitation_name,
                    "precipitation_units": str(dataset[precipitation_name].attrs.get("units", "")),
                }
            )

    assert reference_lat is not None and reference_lon is not None
    lat_spacing = float(np.median(np.abs(np.diff(np.sort(reference_lat)))))
    lon_spacing = float(np.median(np.abs(np.diff(np.sort(reference_lon)))))
    required_years = set(range(1951, 2001))
    missing_years = sorted(required_years - all_years)
    gates = {
        "coverage_includes_1951_2000": not missing_years,
        "latitude_spacing_is_0_5_degree": abs(lat_spacing - 0.5) <= 1e-6,
        "longitude_spacing_is_0_5_degree": abs(lon_spacing - 0.5) <= 1e-6,
        "domain_covers_hma_latitudes": float(reference_lat.min()) <= 20.0
        and float(reference_lat.max()) >= 50.0,
        "domain_covers_hma_longitudes": float(reference_lon.min()) <= 55.5
        and float(reference_lon.max()) >= 115.0,
        "single_precipitation_variable_name": len(precipitation_variables) == 1,
    }
    summary = {
        "status": "complete",
        "metadata_only": True,
        "precipitation_values_read": False,
        "locked_1921_1939_accessed": False,
        "file_count": len(files),
        "files": file_rows,
        "available_year_start": min(all_years),
        "available_year_end": max(all_years),
        "missing_required_years": missing_years,
        "latitude_range": [float(reference_lat.min()), float(reference_lat.max())],
        "longitude_range": [float(reference_lon.min()), float(reference_lon.max())],
        "grid_spacing": [lat_spacing, lon_spacing],
        "gates": gates,
        "passes_input_preflight": bool(all(gates.values())),
    }
    target = output / "aphrodite_input_preflight.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passes_input_preflight"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
