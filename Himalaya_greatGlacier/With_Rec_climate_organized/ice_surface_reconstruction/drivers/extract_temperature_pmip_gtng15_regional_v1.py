"""Extract PMIP temperature means on the frozen 0.5-degree GTN-G HMA regions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
PARENT = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2"
CATALOG = PARENT / "remote_source_catalog.json"
REFERENCE = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "round_21_temperature_reference_field_v1/"
    "temperature_reference_family_balanced_allowed_v1.nc"
)
ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "hma_grid_gtng15_assignment.csv"
)
SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "gtng15_region_support_inventory.csv"
)
MODE_SIM = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "round_74_mode_sim_allowed_temperature_acquisition_v3/"
    "mode_sim_temp2_allowed_hma_bbox.nc"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_gtng15_regional_extraction_v1"
LOCKED_START = 1901
LOCKED_END = 1939


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_subset(dataset: Dataset, target_lat: np.ndarray, target_lon: np.ndarray):
    raw_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
    raw_lon = np.mod(np.asarray(dataset.variables["lon"][:], dtype=float), 360.0)
    lat_step = float(np.median(np.abs(np.diff(np.sort(raw_lat)))))
    lon_step = float(np.median(np.abs(np.diff(np.sort(raw_lon)))))
    lat_indices = np.flatnonzero(
        (raw_lat >= target_lat.min() - 1.5 * lat_step)
        & (raw_lat <= target_lat.max() + 1.5 * lat_step)
    )
    lon_indices = np.flatnonzero(
        (raw_lon >= target_lon.min() - 1.5 * lon_step)
        & (raw_lon <= target_lon.max() + 1.5 * lon_step)
    )
    if len(lat_indices) < 2 or len(lon_indices) < 2:
        raise RuntimeError("Insufficient model-grid overlap")
    lat_slice = slice(int(lat_indices.min()), int(lat_indices.max()) + 1)
    lon_slice = slice(int(lon_indices.min()), int(lon_indices.max()) + 1)
    lat = np.asarray(dataset.variables["lat"][lat_slice], dtype=float)
    lon = np.mod(np.asarray(dataset.variables["lon"][lon_slice], dtype=float), 360.0)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    return lat_slice, lon_slice, lat[lat_order], lon[lon_order], lat_order, lon_order


def interpolate_domain_batch(
    dataset: Dataset,
    time_indices: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
    domain_point_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if len(time_indices) == 0 or np.any(np.diff(time_indices) != 1):
        raise RuntimeError("Each remote request must be contiguous")
    lat_slice, lon_slice, source_lat, source_lon, lat_order, lon_order = source_subset(
        dataset, target_lat, target_lon
    )
    tas = dataset.variables["tas"]
    indexers: list[Any] = []
    kept: list[str] = []
    for dimension in tas.dimensions:
        if dimension == "time":
            indexers.append(slice(int(time_indices[0]), int(time_indices[-1]) + 1))
            kept.append("time")
        elif dimension == "lat":
            indexers.append(lat_slice)
            kept.append("lat")
        elif dimension == "lon":
            indexers.append(lon_slice)
            kept.append("lon")
        else:
            indexers.append(0)
    fields = np.asarray(tas[tuple(indexers)], dtype=float).squeeze()
    if kept != ["time", "lat", "lon"] or fields.ndim != 3:
        raise RuntimeError(f"Unexpected tas dimensions: {tas.dimensions} -> {fields.shape}")
    fields = fields[:, lat_order][:, :, lon_order]

    lat_2d, lon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    points = np.column_stack([lat_2d[target_mask], lon_2d[target_mask]])
    point_weights = np.cos(np.deg2rad(points[:, 0]))
    output = {
        domain: np.full(len(time_indices), np.nan) for domain in domain_point_masks
    }
    for time_index, field in enumerate(fields):
        interpolator = RegularGridInterpolator(
            (source_lat, source_lon), field, bounds_error=False, fill_value=np.nan
        )
        values = interpolator(points)
        if int(np.isfinite(values).sum()) < int(0.98 * len(values)):
            raise RuntimeError("Insufficient finite HMA interpolation")
        for domain, domain_mask in domain_point_masks.items():
            valid = domain_mask & np.isfinite(values)
            if int(valid.sum()) == 0:
                raise RuntimeError(f"No finite model cells in region {domain}")
            output[domain][time_index] = float(
                np.average(values[valid], weights=point_weights[valid])
            )
    return output


def read_remote_model(
    model: str,
    files: list[dict[str, Any]],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
    domain_point_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for item in files:
        allowed_ranges = (
            [(1500, 1849)]
            if item["experiment_role"] == "past"
            else [(1850, 1900), (1940, 2000)]
        )
        with Dataset(item["url"]) as dataset:
            time = dataset.variables["time"]
            dates = num2date(
                time[:],
                units=time.units,
                calendar=getattr(time, "calendar", "standard"),
                only_use_cftime_datetimes=True,
            )
            years = np.asarray([int(date.year) + int(item["offset"]) for date in dates])
            allowed = np.zeros(len(years), dtype=bool)
            for start, end in allowed_ranges:
                allowed |= (years >= start) & (years <= end)
            indices = np.flatnonzero(allowed)
            if len(indices) == 0:
                continue
            if np.any((years[indices] >= LOCKED_START) & (years[indices] <= LOCKED_END)):
                raise RuntimeError("Locked remote climate values requested")
            segments = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            for segment in segments:
                domain_values = interpolate_domain_batch(
                    dataset,
                    segment,
                    target_lat,
                    target_lon,
                    target_mask,
                    domain_point_masks,
                )
                for domain, values in domain_values.items():
                    for index, value in zip(segment, values, strict=True):
                        rows.append(
                            {
                                "model": model,
                                "domain": domain,
                                "year": int(years[index]),
                                "month": int(dates[index].month),
                                "tas_K": float(value),
                            }
                        )
            audit.append(
                {
                    "model": model,
                    "title": item["title"],
                    "url": item["url"],
                    "version": item["version"],
                    "checksum": item["checksum"],
                    "requested_month_count": int(len(indices)),
                    "requested_segment_count": int(len(segments)),
                    "locked_climate_values_requested": 0,
                }
            )
    monthly = pd.DataFrame(rows).drop_duplicates(["model", "domain", "year", "month"])
    counts = monthly.groupby(["domain", "year"])["month"].nunique()
    if not bool((counts == 12).all()):
        raise RuntimeError(f"Incomplete monthly coverage for {model}")
    annual = monthly.groupby(["model", "domain", "year"], as_index=False)["tas_K"].mean()
    return annual, audit


def read_mode_sim(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
    domain_point_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lat_2d, lon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    points = np.column_stack([lat_2d[target_mask], lon_2d[target_mask]])
    point_weights = np.cos(np.deg2rad(points[:, 0]))
    with Dataset(MODE_SIM) as dataset:
        years = np.asarray(dataset.variables["time"][:], dtype=int)
        selected = ((years >= 1500) & (years <= 1900)) | (
            (years >= 1940) & (years <= 2000)
        )
        indices = np.flatnonzero(selected)
        if np.any((years[indices] >= LOCKED_START) & (years[indices] <= LOCKED_END)):
            raise RuntimeError("Locked ModE-Sim values requested")
        source_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        source_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        variable = dataset.variables["temp2_anomaly_1961_1990"]
        for index in indices:
            field = np.asarray(variable[int(index)], dtype=float)
            interpolator = RegularGridInterpolator(
                (source_lat, source_lon), field, bounds_error=False, fill_value=np.nan
            )
            values = interpolator(points)
            for domain, domain_mask in domain_point_masks.items():
                valid = domain_mask & np.isfinite(values)
                rows.append(
                    {
                        "model": "ModE-Sim",
                        "domain": domain,
                        "year": int(years[index]),
                        "anomaly_degC": float(
                            np.average(values[valid], weights=point_weights[valid])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# PMIP GTN-G Regional Extraction Contract\n\n"
        "Reproject the already selected MPI, MIROC, and ModE-Sim temperature series to the frozen "
        "0.5-degree HMA reference grid and its 15 GTN-G subregions. Request only 1500-1900 and "
        "1940-2000 climate values, baseline each remote model to 1961-1990, and save regional annual "
        "means only. No correction is fitted and no 1901-1939 climate value is requested.\n",
        encoding="utf-8",
    )
    paths = {
        "remote_catalog": CATALOG,
        "reference_grid": REFERENCE,
        "gtng_assignment": ASSIGNMENT,
        "gtng_support": SUPPORT,
        "mode_sim": MODE_SIM,
        "facts": FACTS,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}

    with Dataset(REFERENCE) as dataset:
        target_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        target_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        target_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    if not bool(assignment.assignment_status.eq("unique").all()):
        raise RuntimeError("GTN-G assignment is not unique")
    domain_masks: dict[str, np.ndarray] = {"hma": target_mask.copy()}
    for region in support.itertuples(index=False):
        mask = np.zeros_like(target_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        if int(mask.sum()) != int(region.hma_grid_cell_count):
            raise RuntimeError(f"Region mask closure failed: {region.o2region}")
        domain_masks[region.o2region] = mask
    domain_point_masks = {
        domain: mask[target_mask] for domain, mask in domain_masks.items()
    }

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    remote_frames = []
    remote_audit = []
    for model in ["MPI-ESM1-2-LR", "MIROC-ES2L"]:
        files = [item for item in catalog if item["model"] == model]
        annual, audit = read_remote_model(
            model,
            files,
            target_lat,
            target_lon,
            target_mask,
            domain_point_masks,
        )
        baseline = annual[annual.year.between(1961, 1990)].groupby("domain")["tas_K"].mean()
        if len(baseline) != 16:
            raise RuntimeError(f"Incomplete regional baseline: {model}")
        annual["anomaly_degC"] = annual["tas_K"] - annual.domain.map(baseline)
        remote_frames.append(annual[["model", "domain", "year", "anomaly_degC"]])
        remote_audit.extend(audit)
    mode = read_mode_sim(
        target_lat, target_lon, target_mask, domain_point_masks
    )
    annual = pd.concat([*remote_frames, mode], ignore_index=True)
    lia_counts = annual[annual.year.between(1500, 1899)].groupby(["model", "domain"])["year"].nunique()
    if len(lia_counts) != 48 or not bool((lia_counts == 400).all()):
        raise RuntimeError("Regional LIA annual coverage failed")
    annual.to_csv(OUTPUT / "pmip_gtng15_annual_temperature_allowed.csv", index=False)
    (OUTPUT / "remote_request_audit.json").write_text(
        json.dumps(remote_audit, indent=2), encoding="utf-8"
    )

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    summary = {
        "status": "complete",
        "model_count": 3,
        "domain_count": 16,
        "gtng_region_count": 15,
        "hma_grid_cell_count": int(target_mask.sum()),
        "locked_1901_1939_climate_values_requested": 0,
        "source_hashes_unchanged": unchanged,
        "output": str(OUTPUT / "pmip_gtng15_annual_temperature_allowed.csv"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# PMIP GTN-G Regional Extraction\n\n"
        "- Models: `3`.\n"
        "- Domains: `HMA + 15 GTN-G regions`.\n"
        f"- HMA grid cells: `{int(target_mask.sum())}`.\n"
        "- Locked 1901-1939 climate values requested: `0`.\n"
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
