"""Verify the public-reference ensemble mean without running OGGM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np


MEMBER_VARIABLES = (
    "tas_rec_ref_HGAO_c05d",
    "tas_rec_ref_LMR_c05d",
    "tas_rec_ref_Nk_AM_c05d",
    "tas_rec_ref_Nk_CCA_c05d",
    "tas_rec_ref_Nk_CPS_c05d",
    "tas_rec_ref_Nk_DA_c05d",
    "tas_rec_ref_Nk_GraphEM_c05d",
    "tas_rec_ref_Nk_PCR_c05d",
    "tas_rec_ref_O_c05d",
    "tas_rec_ref_PHYDA_c05d",
)
TARGET_VARIABLE = "tas_ensemble_mean_rec_ref"
TARGET_LAT_INDEX = 24
TARGET_LON_INDEX = 19
CHUNK_MONTHS = 120

MEMBER_PROVENANCE = (
    ("HGAO", "HGAOEnKF", "HGAOEnKF_clipped_0.5deg.tif"),
    ("LMR", "Last Millennium Reanalysis v2.1", "LMR_clipped_0.5deg.tif"),
    ("Nk_AM", "Neukom analog method", "Neukom_AM_clipped_0.5deg.tif"),
    ("Nk_CCA", "Neukom canonical correlation analysis", "Neukom_CCA_clipped_0.5deg.tif"),
    ("Nk_CPS", "Neukom composite plus scaling", "Neukom_CPS_clipped_0.5deg.tif"),
    ("Nk_DA", "Neukom data assimilation", "Neukom_DA_clipped_0.5deg.tif"),
    ("Nk_GraphEM", "Neukom GraphEM", "Neukom_GraphEM_clipped_0.5deg.tif"),
    ("Nk_PCR", "Neukom principal component regression", "Neukom_PCR_clipped_0.5deg.tif"),
    ("O", "OEnKF", "OEnKF_clipped_0.5deg.tif"),
    ("PHYDA", "PHYDA Common Era hydroclimate reconstruction", "PHYDA_clipped_0.5deg.tif"),
)


def _plain(array: np.ndarray) -> np.ndarray:
    return np.asarray(np.ma.filled(array, np.nan), dtype=np.float64)


def _time_label(dataset: nc.Dataset, index: int) -> str:
    variable = dataset.variables["time"]
    value = nc.num2date(
        variable[index],
        units=variable.units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=True,
    )
    return f"{int(value.year):04d}-{int(value.month):02d}"


def _write_member_table(path: Path, finite_counts: dict[str, int], total: int) -> None:
    rows = []
    for (member_id, product, source_tif), variable in zip(
        MEMBER_PROVENANCE, MEMBER_VARIABLES, strict=True
    ):
        rows.append(
            {
                "member_id": member_id,
                "public_product": product,
                "netcdf_variable": variable,
                "source_rec_ref_tif": source_tif,
                "finite_value_count": finite_counts[variable],
                "finite_fraction": finite_counts[variable] / total,
                "included_in_tas_ensemble_mean_rec_ref": True,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def audit(source: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(source) as dataset:
        missing = [
            name
            for name in (*MEMBER_VARIABLES, TARGET_VARIABLE, "time", "lat", "lon")
            if name not in dataset.variables
        ]
        if missing:
            raise KeyError(f"Missing variables: {missing}")

        target = dataset.variables[TARGET_VARIABLE]
        shape = tuple(int(value) for value in target.shape)
        if target.dimensions != ("time", "lat", "lon"):
            raise ValueError(f"Unexpected target dimensions: {target.dimensions}")
        for name in MEMBER_VARIABLES:
            if dataset.variables[name].shape != target.shape:
                raise ValueError(f"Shape mismatch for {name}")

        member_count_histogram = np.zeros(len(MEMBER_VARIABLES) + 1, dtype=np.int64)
        target_pixel_count_histogram = np.zeros(
            len(MEMBER_VARIABLES) + 1, dtype=np.int64
        )
        member_finite_counts = {name: 0 for name in MEMBER_VARIABLES}
        finite_pair_count = 0
        finite_mismatch_count = 0
        absolute_difference_sum = 0.0
        squared_difference_sum = 0.0
        max_absolute_difference = -1.0
        max_difference_index = (0, 0, 0)
        target_pixel_pair_count = 0
        target_pixel_absolute_difference_sum = 0.0
        target_pixel_max_absolute_difference = -1.0
        target_pixel_max_time_index = 0

        for start in range(0, shape[0], CHUNK_MONTHS):
            stop = min(start + CHUNK_MONTHS, shape[0])
            members = np.stack(
                [_plain(dataset.variables[name][start:stop]) for name in MEMBER_VARIABLES],
                axis=0,
            )
            finite = np.isfinite(members)
            counts = finite.sum(axis=0)
            sums = np.nansum(members, axis=0)
            recomputed = np.full(counts.shape, np.nan, dtype=np.float64)
            valid_recomputed = counts > 0
            recomputed[valid_recomputed] = sums[valid_recomputed] / counts[valid_recomputed]
            stored = _plain(target[start:stop])

            member_count_histogram += np.bincount(
                counts.ravel(), minlength=len(member_count_histogram)
            )
            for member_index, name in enumerate(MEMBER_VARIABLES):
                member_finite_counts[name] += int(finite[member_index].sum())

            stored_finite = np.isfinite(stored)
            finite_mismatch_count += int(
                np.count_nonzero(stored_finite != valid_recomputed)
            )
            comparable = stored_finite & valid_recomputed
            difference = np.abs(stored[comparable] - recomputed[comparable])
            finite_pair_count += int(difference.size)
            absolute_difference_sum += float(difference.sum())
            squared_difference_sum += float(np.square(difference).sum())

            if difference.size:
                local_max = float(difference.max())
                if local_max > max_absolute_difference:
                    dense_difference = np.full(stored.shape, -1.0, dtype=np.float64)
                    dense_difference[comparable] = np.abs(
                        stored[comparable] - recomputed[comparable]
                    )
                    local_index = np.unravel_index(
                        int(np.argmax(dense_difference)), dense_difference.shape
                    )
                    max_absolute_difference = local_max
                    max_difference_index = (
                        start + int(local_index[0]),
                        int(local_index[1]),
                        int(local_index[2]),
                    )

            pixel_counts = counts[:, TARGET_LAT_INDEX, TARGET_LON_INDEX]
            target_pixel_count_histogram += np.bincount(
                pixel_counts, minlength=len(target_pixel_count_histogram)
            )
            pixel_stored = stored[:, TARGET_LAT_INDEX, TARGET_LON_INDEX]
            pixel_recomputed = recomputed[:, TARGET_LAT_INDEX, TARGET_LON_INDEX]
            pixel_comparable = np.isfinite(pixel_stored) & np.isfinite(pixel_recomputed)
            pixel_difference = np.abs(
                pixel_stored[pixel_comparable] - pixel_recomputed[pixel_comparable]
            )
            target_pixel_pair_count += int(pixel_difference.size)
            target_pixel_absolute_difference_sum += float(pixel_difference.sum())
            if pixel_difference.size:
                local_pixel_max = float(pixel_difference.max())
                if local_pixel_max > target_pixel_max_absolute_difference:
                    pixel_dense = np.full(pixel_stored.shape, -1.0, dtype=np.float64)
                    pixel_dense[pixel_comparable] = np.abs(
                        pixel_stored[pixel_comparable] - pixel_recomputed[pixel_comparable]
                    )
                    target_pixel_max_absolute_difference = local_pixel_max
                    target_pixel_max_time_index = start + int(np.argmax(pixel_dense))

        total_grid_months = int(np.prod(shape))
        latitudes = _plain(dataset.variables["lat"][:])
        longitudes = _plain(dataset.variables["lon"][:])
        max_time, max_lat, max_lon = max_difference_index
        summary = {
            "audit_type": "read-only rec_ref ensemble arithmetic check; no OGGM run",
            "source_netcdf": str(source),
            "target_variable": TARGET_VARIABLE,
            "member_variables": list(MEMBER_VARIABLES),
            "member_count": len(MEMBER_VARIABLES),
            "shape_time_lat_lon": list(shape),
            "time_start": _time_label(dataset, 0),
            "time_end": _time_label(dataset, shape[0] - 1),
            "ensemble_rule": "finite-member arithmetic mean at each time/lat/lon cell",
            "global_check": {
                "total_grid_months": total_grid_months,
                "finite_pair_count": finite_pair_count,
                "stored_vs_recomputed_finite_mismatch_count": finite_mismatch_count,
                "mean_absolute_difference_c": absolute_difference_sum
                / finite_pair_count,
                "root_mean_square_difference_c": (
                    squared_difference_sum / finite_pair_count
                )
                ** 0.5,
                "max_absolute_difference_c": max_absolute_difference,
                "max_difference_time": _time_label(dataset, max_time),
                "max_difference_lat_index": max_lat,
                "max_difference_lon_index": max_lon,
                "max_difference_latitude": float(latitudes[max_lat]),
                "max_difference_longitude": float(longitudes[max_lon]),
                "finite_member_count_histogram": {
                    str(index): int(value)
                    for index, value in enumerate(member_count_histogram)
                    if value
                },
            },
            "target_pixel_3584": {
                "lat_index": TARGET_LAT_INDEX,
                "lon_index": TARGET_LON_INDEX,
                "latitude": float(latitudes[TARGET_LAT_INDEX]),
                "longitude": float(longitudes[TARGET_LON_INDEX]),
                "finite_pair_count": target_pixel_pair_count,
                "mean_absolute_difference_c": target_pixel_absolute_difference_sum
                / target_pixel_pair_count,
                "max_absolute_difference_c": target_pixel_max_absolute_difference,
                "max_difference_time": _time_label(
                    dataset, target_pixel_max_time_index
                ),
                "finite_member_count_histogram": {
                    str(index): int(value)
                    for index, value in enumerate(target_pixel_count_histogram)
                    if value
                },
            },
        }

    _write_member_table(
        output_dir / "rec_ref_member_provenance_audit_v1.csv",
        member_finite_counts,
        total_grid_months,
    )
    summary_path = output_dir / "rec_ref_ensemble_arithmetic_audit_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]
    default_output = (
        project_root
        / "validation_data/results/paleo_surface_validation_v0/"
        "layered_coordinate_search_v1/B_3584_forcing_geometry_response_v2/"
        "climate_geometry_read_only_audit_v1"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "E:/3spyderProject/HMA_500/Down_scale_new/out_data/"
            "climate_monthly_abs_05deg_1000-2000.nc"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()
    result = audit(args.source, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
