"""Build the immutable 191-system, 1337-cell GPU spin-up fixture."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE_TRANSACTION = RESULTS / "lia_191_all191_oggm_flowline_1d_v10_203_transaction_v1"
SOURCE_MATRIX = SOURCE_TRANSACTION / "level1_matrix.csv"
SOURCE_MATRIX_SHA256 = "B8A32F08EDD2533DAAFD45AD75BAAF28ECBC54EC1BF1CE8845180761D9693CE2"
INPUT_HASHES = RESULTS / "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json"
FIXTURE_ROOT = RESULTS / "lia_191_gpu_spinup_v10_239_transaction_v1/static_fixture_v1"
SOURCE_GDIR_ROOT = PROJECT_ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"
ELEVATION_ROOT = RESULTS / "lia_elevation_band_core191_length_corrected_v1"
PIXEL_TABLE = RESULTS / "lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"
RECONSTRUCTION = Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/climate_oggm_ensemble_rec_ref.nc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:8] / rgi_id[:11] / rgi_id


def monthly_climatology(dataset: xr.Dataset, name: str) -> np.ndarray:
    values = np.asarray(dataset[name].groupby("time.month").mean("time").values, dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid monthly climatology: {name}")
    return values


def read_matrix() -> list[dict]:
    if sha256(SOURCE_MATRIX) != SOURCE_MATRIX_SHA256:
        raise RuntimeError("frozen v10.203 matrix hash mismatch")
    with SOURCE_MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    systems = sorted({int(row["lia_unique_id"]) for row in rows})
    counts = {system: 0 for system in systems}
    for row in rows:
        counts[int(row["lia_unique_id"])] += 1
    if len(rows) != 1337 or len(systems) != 191 or set(counts.values()) != {7}:
        raise RuntimeError("matrix is not exactly 191 systems x 7 cells")
    expected_delta = {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5}
    for system in systems:
        actual = {
            float(row["delta_t_eq_degC"])
            for row in rows
            if int(row["lia_unique_id"]) == system
        }
        if actual != expected_delta:
            raise RuntimeError(f"delta_t_eq grid mismatch for system {system}")
    return rows


def main() -> int:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=False)
    rows = read_matrix()
    input_hashes = json.loads(INPUT_HASHES.read_text(encoding="utf-8"))
    pixels = pd.read_csv(PIXEL_TABLE)
    systems = []
    seen = set()
    for row in rows:
        system = int(row["lia_unique_id"])
        if system not in seen:
            systems.append(system)
            seen.add(system)
    system_index = {system: index for index, system in enumerate(systems)}

    flowlines = []
    max_nx = 0
    for system in systems:
        row = next(item for item in rows if int(item["lia_unique_id"]) == system)
        path = ELEVATION_ROOT / str(system) / "elevation_band_flowline_lia.pkl.gz"
        expected = input_hashes["member_file_hashes"][str(system)]["corrected_flowline"]
        if sha256(path) != expected or expected != row["geometry_hash"]:
            raise RuntimeError(f"corrected flowline hash mismatch for {system}")
        with gzip.open(path, "rb") as handle:
            loaded = pickle.load(handle)
        if len(loaded) != 1 or loaded[0].__class__.__name__ != "RectangularBedFlowline":
            raise RuntimeError(f"unsupported flowline topology for {system}")
        flowline = loaded[0]
        flowlines.append((path, flowline))
        max_nx = max(max_nx, int(flowline.nx))

    shape = (len(systems), max_nx)
    bed_h = np.zeros(shape, dtype=np.float64)
    widths_m = np.ones(shape, dtype=np.float64)
    thick = np.zeros(shape, dtype=np.float64)
    valid_mask = np.zeros(shape, dtype=np.bool_)
    nx = np.zeros(len(systems), dtype=np.int64)
    dx_meter = np.zeros(len(systems), dtype=np.float64)
    climate_temp = np.zeros((len(systems), 30, 12), dtype=np.float64)
    climate_prcp = np.zeros_like(climate_temp)
    climate_gradient = np.full_like(climate_temp, -0.0065)
    climate_ref_hgt = np.zeros(len(systems), dtype=np.float64)
    system_records = []

    reconstruction_expected = input_hashes["global_inputs"]["reconstruction_climate"]["sha256"]
    if sha256(RECONSTRUCTION) != reconstruction_expected:
        raise RuntimeError("reconstruction climate hash mismatch")
    with xr.open_dataset(RECONSTRUCTION, use_cftime=True) as reconstruction:
        rec_baseline = reconstruction.sel(time=slice("1961-01-01", "1990-12-31"))
        for index, system in enumerate(systems):
            row = next(item for item in rows if int(item["lia_unique_id"]) == system)
            rgi_id = row["rgi_id"]
            path, flowline = flowlines[index]
            n = int(flowline.nx)
            bed_h[index, :n] = np.asarray(flowline.bed_h, dtype=np.float64)
            widths_m[index, :n] = np.asarray(flowline.widths_m, dtype=np.float64)
            thick[index, :n] = np.asarray(flowline.thick, dtype=np.float64)
            valid_mask[index, :n] = True
            nx[index] = n
            dx_meter[index] = float(flowline.dx_meter)

            pixel_rows = pixels[
                pixels["lia_unique_id"].astype(int).eq(system)
                & pixels["rgi_id"].astype(str).eq(rgi_id)
                & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
            ]
            if len(pixel_rows) != 1:
                raise RuntimeError(f"climate pixel row count for {system}: {len(pixel_rows)}")
            pixel = pixel_rows.iloc[0]
            lat_index = int(pixel["climate_lat_index"])
            lon_index = int(pixel["climate_lon_index"])
            rec_temp = reconstruction["temp"].isel(lat=lat_index, lon=lon_index)
            baseline_temp = rec_baseline["temp"].isel(lat=lat_index, lon=lon_index)
            target_temp = rec_temp.sel(time=slice("1441-01-01", "1470-12-31"))
            if target_temp.sizes.get("time") != 360:
                raise RuntimeError(f"incomplete reconstruction window for {system}")
            anomaly = (
                target_temp.groupby("time.month")
                - baseline_temp.groupby("time.month").mean("time")
            ).values.astype(np.float64).reshape(30, 12)

            historical_path = gdir_path(SOURCE_GDIR_ROOT, rgi_id) / "climate_historical.nc"
            expected_historical = input_hashes["member_file_hashes"][str(system)]["source_climate_historical"]
            if sha256(historical_path) != expected_historical:
                raise RuntimeError(f"historical climate hash mismatch for {system}")
            with xr.open_dataset(historical_path, use_cftime=True) as historical:
                baseline = historical.sel(time=slice("1961-01-01", "1990-12-31"))
                if baseline.sizes.get("time") != 360:
                    raise RuntimeError(f"incomplete W5E5 baseline for {system}")
                temp_month = monthly_climatology(baseline, "temp")
                prcp_month = monthly_climatology(baseline, "prcp")
                climate_ref_hgt[index] = float(historical.attrs["ref_hgt"])
            climate_temp[index] = np.tile(temp_month, 30).reshape(30, 12) + anomaly
            climate_prcp[index] = np.tile(prcp_month, 30).reshape(30, 12)
            if not np.isfinite(climate_temp[index]).all() or not np.isfinite(climate_prcp[index]).all():
                raise RuntimeError(f"nonfinite climate for {system}")
            system_records.append({
                "system_index": index,
                "lia_unique_id": system,
                "rgi_id": rgi_id,
                "nx": n,
                "dx_meter": float(flowline.dx_meter),
                "flowline_path": str(path),
                "flowline_sha256": sha256(path),
                "historical_climate_path": str(historical_path),
                "historical_climate_sha256": sha256(historical_path),
                "temperature_min_c": float(np.min(climate_temp[index])),
                "temperature_max_c": float(np.max(climate_temp[index])),
                "precipitation_min_kg_m2_month": float(np.min(climate_prcp[index])),
                "precipitation_max_kg_m2_month": float(np.max(climate_prcp[index])),
            })

    cell_system_index = np.asarray(
        [system_index[int(row["lia_unique_id"])] for row in rows], dtype=np.int64
    )
    arrays = {
        "system_bed_h": bed_h,
        "system_widths_m": widths_m,
        "system_thick": thick,
        "system_valid_mask": valid_mask,
        "system_nx": nx,
        "system_dx_meter": dx_meter,
        "system_climate_temp": climate_temp,
        "system_climate_prcp": climate_prcp,
        "system_climate_gradient": climate_gradient,
        "system_climate_ref_hgt": climate_ref_hgt,
        "cell_system_index": cell_system_index,
        "cell_ordinal": np.asarray([int(row["cell_ordinal"]) for row in rows], dtype=np.int64),
        "cell_glen_a": np.asarray([float(row["glen_a_pa_3_s_1"]) for row in rows], dtype=np.float64),
        "cell_melt_f": np.asarray([float(row["melt_f"]) for row in rows], dtype=np.float64),
        "cell_prcp_fac": np.asarray([float(row["prcp_fac"]) for row in rows], dtype=np.float64),
        "cell_temp_bias": np.asarray([float(row["effective_temp_bias_degC"]) for row in rows], dtype=np.float64),
        "cell_temp_melt": np.asarray([float(row["temp_melt_degC"]) for row in rows], dtype=np.float64),
        "climate_years": np.arange(1441, 1471, dtype=np.int64),
    }
    npz_path = FIXTURE_ROOT / "all191_spinup_inputs_float64_v1.npz"
    temporary = npz_path.with_suffix(".npz.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, npz_path)

    cell_records = []
    for index, row in enumerate(rows):
        cell_records.append({
            "cell_index": index,
            "member_key": f"c{index:04d}",
            "cell_ordinal": int(row["cell_ordinal"]),
            "lia_unique_id": int(row["lia_unique_id"]),
            "rgi_id": row["rgi_id"],
            "source_fingerprint": row["fingerprint"],
            "parameters": {
                "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
                "base_temp_bias_degC": float(row["base_temp_bias_degC"]),
                "effective_temp_bias_degC": float(row["effective_temp_bias_degC"]),
                "melt_f": float(row["melt_f"]),
                "prcp_fac": float(row["prcp_fac"]),
                "glen_a_pa_3_s_1": float(row["glen_a_pa_3_s_1"]),
                "fs": float(row["fs"]),
                "temp_melt_degC": float(row["temp_melt_degC"]),
            },
        })
    manifest = {
        "schema": "lia-191-gpu-spinup-static-fixture-v1",
        "dynamics_performed": False,
        "precision": "float64",
        "population_systems": 191,
        "cells_per_system": 7,
        "cell_count": 1337,
        "max_nx": max_nx,
        "spinup_climate": "ensemble_rec_ref-1441-1470-repeat",
        "precipitation_anomalies": False,
        "source_matrix_path": str(SOURCE_MATRIX.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_matrix_sha256": sha256(SOURCE_MATRIX),
        "input_hashes_path": str(INPUT_HASHES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "input_hashes_sha256": sha256(INPUT_HASHES),
        "reconstruction_climate_path": str(RECONSTRUCTION),
        "reconstruction_climate_sha256": sha256(RECONSTRUCTION),
        "npz_path": str(npz_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "npz_sha256": sha256(npz_path),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "systems": system_records,
        "cells": cell_records,
    }
    atomic_json(FIXTURE_ROOT / "manifest.json", manifest)
    print(json.dumps({
        "status": "fixture-complete",
        "systems": 191,
        "cells": 1337,
        "max_nx": max_nx,
        "npz_sha256": manifest["npz_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
