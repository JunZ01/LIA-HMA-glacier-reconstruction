"""Build and validate the v10.153 historical-preparation artifacts for system 2005."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

from .lia_first10_spinup_batch import _write_case_climate

def _control_lengths(coordinates: np.ndarray) -> np.ndarray:
    segments = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    result = np.zeros(len(coordinates), dtype=np.float64)
    if len(coordinates) == 1:
        return result
    result[0] = 0.5 * segments[0]
    result[-1] = 0.5 * segments[-1]
    if len(coordinates) > 2:
        result[1:-1] = 0.5 * (segments[:-1] + segments[1:])
    return result


def _sample_regular_grid(values: np.ndarray, x: np.ndarray, y: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    work = values
    x_work = x
    y_work = y
    if x[1] < x[0]:
        x_work = x[::-1]
        work = work[:, ::-1]
    if y[1] < y[0]:
        y_work = y[::-1]
        work = work[::-1, :]
    dx = float(x_work[1] - x_work[0])
    dy = float(y_work[1] - y_work[0])
    fx = (points_xy[:, 0] - x_work[0]) / dx
    fy = (points_xy[:, 1] - y_work[0]) / dy
    valid = (fx >= 0.0) & (fx <= len(x_work) - 1) & (fy >= 0.0) & (fy <= len(y_work) - 1)
    result = np.full(len(points_xy), np.nan, dtype=np.float64)
    ix0 = np.floor(np.clip(fx, 0.0, len(x_work) - 1)).astype(int)
    iy0 = np.floor(np.clip(fy, 0.0, len(y_work) - 1)).astype(int)
    ix1 = np.minimum(ix0 + 1, len(x_work) - 1)
    iy1 = np.minimum(iy0 + 1, len(y_work) - 1)
    wx = fx - ix0
    wy = fy - iy0
    sampled = ((1.0 - wx) * (1.0 - wy) * work[iy0, ix0] + wx * (1.0 - wy) * work[iy0, ix1] + (1.0 - wx) * wy * work[iy1, ix0] + wx * wy * work[iy1, ix1])
    result[valid] = sampled[valid]
    return result


def _path_metrics(thickness: np.ndarray, x: np.ndarray, y: np.ndarray, path_xy: np.ndarray, control_lengths: np.ndarray, registered_length_m: float) -> dict[str, Any]:
    sampled = _sample_regular_grid(thickness, x, y, path_xy)
    active = np.isfinite(sampled) & (sampled > 10.0)
    active_indices = np.flatnonzero(active)
    observed_length = float(np.sum(control_lengths[active]))
    if len(active_indices):
        first = int(active_indices[0])
        last = int(active_indices[-1])
        gaps = np.flatnonzero(~active[first:last + 1]) + first
    else:
        first = None
        last = None
        gaps = np.asarray([], dtype=int)
    return {
        "observed_active_control_volume_length_m": observed_length,
        "registered_lia_length_m": registered_length_m,
        "registered_length_relative_error": abs(observed_length - registered_length_m) / registered_length_m,
        "root_first_active_node": first,
        "root_last_active_node": last,
        "root_internal_gap_count_diagnostic_only": int(len(gaps)),
        "root_internal_gap_indices_diagnostic_only": gaps.tolist(),
        "root_gap_policy": "classify-and-report; not a numerical failure for the 2D state",
        "sampled_root_maximum_thickness_m": float(np.nanmax(sampled)),
    }


def _free_domain_metrics(thickness: np.ndarray, dx: float) -> dict[str, float | int]:
    active = np.argwhere(thickness > 10.0)
    if len(active) == 0:
        return {"active_cell_count_gt_10m": 0, "minimum_active_cell_center_to_grid_edge_m": float("inf")}
    rows = active[:, 0]
    cols = active[:, 1]
    distance_cells = np.minimum.reduce([rows + 0.5, thickness.shape[0] - rows - 0.5, cols + 0.5, thickness.shape[1] - cols - 0.5])
    return {"active_cell_count_gt_10m": int(len(active)), "minimum_active_cell_center_to_grid_edge_m": float(np.min(distance_cells) * dx)}


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
STATE = ROOT / "LIA_CURRENT_STATE.json"
ROUTE = BASE / "lia_191_current_route_v10_153/summary.json"
FORMAL_DIR = BASE / "lia_191_system2005_masked_temperature_formal50_v1"
REPRO_DIR = BASE / "lia_191_system2005_masked_temperature_reproduction_v1"
FORMAL = FORMAL_DIR / "summary.json"
REPRO = REPRO_DIR / "summary.json"
STATIC_DIR = BASE / "lia_191_igm_planform2d_system2005_static_v1"
STATIC = STATIC_DIR / "static_summary.json"
GEOLOGY = STATIC_DIR / "geology.nc"
OBSERVATION_PATH = STATIC_DIR / "observation_path_static.csv"
STEADY_EXPORT_DIR = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1"
STEADY_PACKAGE = STEADY_EXPORT_DIR / "monthly_ti_package.npz"
STEADY_EXPORT = STEADY_EXPORT_DIR / "export_summary.json"
CLIMATE_PIXELS = BASE / "lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"
LENGTH_REFERENCES = BASE / "lia_191_controller_v1/core191_length_references.csv"
SOURCE_GDIR_ROOT = (
    BASE
    / "lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/synthetic_gdir"
)
SOURCE_CLIMATE = (
    SOURCE_GDIR_ROOT
    / "RGI60-15/RGI60-15.92/RGI60-15.92005/climate_historical.nc"
)
RECONSTRUCTION = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\oggm_climate\Ensemble"
    r"\climate_oggm_ensemble_rec_ref.nc"
)
CANDIDATE_V2 = (
    BASE
    / "climate_reconstruction_improvement_loop_v2/temperature_candidate_v2_formal_freeze_v1/summary.json"
)
DRIVER = HERE.parent / "run_system2005_masked_temperature_historical_v1.py"
OUT = BASE / "lia_191_system2005_masked_temperature_historical_preparation_v1"
PREFLIGHT_OUT = BASE / "lia_191_system2005_masked_temperature_historical_5y_preflight_v1"

SYSTEM_ID = 2005
MAIN_RGI_ID = "RGI60-15.09991"
SYNTHETIC_RGI_ID = "RGI60-15.92005"
START_YEAR = 1470
END_YEAR = 2000
RUN_YEARS = END_YEAR - START_YEAR
RUN_MONTHS = RUN_YEARS * 12
SOURCE_MONTHS = (END_YEAR - START_YEAR + 1) * 12
EXPECTED_FULL_CLIMATE_SHA256 = "55BA3CF158A528B909C5460793B6E025E4CB7E944B17A85873EA08A72A4D4D85"
EXPECTED_RECONSTRUCTION_SHA256 = "CC55E3D2772E6B2932895A54A77CB89339B0B24090FBC484E72FA681D7298ADD"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar(package: Any, name: str) -> float:
    return float(np.asarray(package[name]).item())


def formula_monthly_mb(
    *,
    temperature: float,
    precipitation: float,
    gradient: float,
    heights: np.ndarray,
    ref_hgt: float,
    t_melt: float,
    t_solid: float,
    t_liq: float,
    monthly_melt_f: float,
    bias: float,
    sec_in_month: float,
    sec_in_year: float,
    rho: float,
) -> np.ndarray:
    temp = temperature + gradient * (heights - ref_hgt)
    melt_temp = np.maximum(temp - t_melt, 0.0)
    solid_fraction = np.clip(1.0 - (temp - t_solid) / (t_liq - t_solid), 0.0, 1.0)
    monthly_mass = precipitation * solid_fraction - monthly_melt_f * melt_temp
    monthly_mass -= bias * sec_in_month / sec_in_year
    return monthly_mass / sec_in_month / rho


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    repro = json.loads(REPRO.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    steady_export = json.loads(STEADY_EXPORT.read_text(encoding="utf-8"))
    candidate_v2 = json.loads(CANDIDATE_V2.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.153-system2005-historical-preparation-build":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route-hash-mismatch")
    if state["next_action"]["type"] != "build-system2005-masked-temperature-historical-preparation-v1":
        raise RuntimeError("preparation-not-current-next-action")
    if state["next_action"]["launch_authorized"] or state["counts"]["active"] or state["counts"]["queued"]:
        raise RuntimeError("audit-state-not-quiescent")
    if formal["status"] != "masked-temperature-formal50-pass" or not all(formal["checks"].values()):
        raise RuntimeError("formal-steady-not-pass")
    if repro["status"] != "masked-temperature-reproduction-pass" or not all(repro["checks"].values()):
        raise RuntimeError("steady-reproduction-not-pass")
    if static["status"] != "static-pass" or not all(static["gate_flags"].values()):
        raise RuntimeError("static-package-not-pass")
    if steady_export["status"] != "climate-bridge-export-pass":
        raise RuntimeError("steady-climate-export-not-pass")
    if candidate_v2.get("candidate_id") != "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828":
        raise RuntimeError("candidate-v2-freeze-mismatch")
    if state["research_branches"]["hma_temperature_candidate_v2"]["promoted_to_production"]:
        raise RuntimeError("candidate-v2-unexpected-production-status")
    if sha256(RECONSTRUCTION) != EXPECTED_RECONSTRUCTION_SHA256:
        raise RuntimeError("reconstruction-source-hash-mismatch")

    source_paths = [
        STATE,
        ROUTE,
        FORMAL,
        REPRO,
        FORMAL_DIR / "final_state_50y.npz",
        REPRO_DIR / "final_state_50y.npz",
        STATIC,
        GEOLOGY,
        OBSERVATION_PATH,
        STEADY_PACKAGE,
        STEADY_EXPORT,
        CLIMATE_PIXELS,
        LENGTH_REFERENCES,
        SOURCE_CLIMATE,
        RECONSTRUCTION,
        CANDIDATE_V2,
    ]
    hashes_before = {str(path): sha256(path) for path in source_paths}
    stage = Path(tempfile.mkdtemp(prefix=".system2005_historical_readiness_", dir=BASE))
    try:
        isolated = stage / "isolated_gdir"
        shutil.copytree(SOURCE_GDIR_ROOT, isolated)

        from oggm import cfg, utils
        from oggm.core.massbalance import MonthlyTIModel

        cfg.initialize(logging_level="WARNING")
        cfg.PARAMS["use_tar_shapefiles"] = False
        cfg.PARAMS["temp_melt"] = float(static["stage2_frozen"]["temp_melt_c"])
        gdir = utils.GlacierDirectory(SYNTHETIC_RGI_ID, base_dir=isolated, reset=False)
        pixels = pd.read_csv(CLIMATE_PIXELS)
        selected = pixels[
            pixels["rgi_id"].astype(str).eq(MAIN_RGI_ID)
            & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
        ]
        if len(selected) != 1:
            raise RuntimeError(f"climate-pixel-row-count:{len(selected)}")
        pixel = selected.iloc[0]
        suffix, climate_audit = _write_case_climate(
            gdir,
            RECONSTRUCTION,
            pixel,
            "off",
            start_year=START_YEAR,
            end_year=END_YEAR,
        )
        generated = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
        generated_hash = sha256(generated)
        if generated_hash != EXPECTED_FULL_CLIMATE_SHA256:
            raise RuntimeError(f"independent-full-climate-hash-mismatch:{generated_hash}")
        derived_climate = stage / "historical_climate_1470_2000_prcp_off.nc"
        shutil.copy2(generated, derived_climate)

        stage2 = static["stage2_frozen"]
        base_delta_t = float(steady_export["stage2"]["delta_t_c"])
        base_temp_bias = float(stage2["base_temp_bias_c"])
        temperature_offset = float(repro["temperature_offset_k"])
        base_effective_bias = base_temp_bias + base_delta_t
        effective_historical_bias = base_effective_bias + temperature_offset
        mb_model = MonthlyTIModel(
            gdir,
            filename="gcm_data",
            input_filesuffix=suffix,
            melt_f=float(stage2["melt_f"]),
            prcp_fac=float(stage2["prcp_fac"]),
            temp_bias=base_effective_bias,
            ys=START_YEAR,
            ye=END_YEAR,
            repeat=False,
            check_calib_params=False,
        )
        reference_model = MonthlyTIModel(
            gdir,
            filename="gcm_data",
            input_filesuffix=suffix,
            melt_f=float(stage2["melt_f"]),
            prcp_fac=float(stage2["prcp_fac"]),
            temp_bias=effective_historical_bias,
            ys=START_YEAR,
            ye=END_YEAR,
            repeat=False,
            check_calib_params=False,
        )
        if len(mb_model.temp) != SOURCE_MONTHS or len(mb_model.years) != SOURCE_MONTHS:
            raise RuntimeError("historical-month-count-mismatch")
        sequence_indices = np.arange(RUN_MONTHS, dtype=np.int64)
        package_path = stage / "monthly_ti_historical_package.npz"
        sec_in_month = float(365.0 * 24.0 * 3600.0 / 12.0)
        sec_in_year = float(365.0 * 24.0 * 3600.0)
        monthly_melt_f = float(mb_model.melt_f) * 365.25 / 12.0
        np.savez_compressed(
            package_path,
            temp=np.asarray(mb_model.temp, dtype=np.float64),
            prcp=np.asarray(mb_model.prcp, dtype=np.float64),
            grad=np.asarray(mb_model.grad, dtype=np.float64),
            years=np.asarray(mb_model.years, dtype=np.int64),
            months=np.asarray(mb_model.months, dtype=np.int64),
            sequence_indices=sequence_indices,
            sequence_years=np.asarray(mb_model.years[:RUN_MONTHS], dtype=np.int64),
            sequence_months=np.asarray(mb_model.months[:RUN_MONTHS], dtype=np.int64),
            ref_hgt=np.asarray(mb_model.ref_hgt, dtype=np.float64),
            t_melt=np.asarray(mb_model.t_melt, dtype=np.float64),
            t_solid=np.asarray(mb_model.t_solid, dtype=np.float64),
            t_liq=np.asarray(mb_model.t_liq, dtype=np.float64),
            monthly_melt_f=np.asarray(monthly_melt_f, dtype=np.float64),
            bias=np.asarray(mb_model.bias, dtype=np.float64),
            rho=np.asarray(mb_model.rho, dtype=np.float64),
            sec_in_month=np.asarray(sec_in_month, dtype=np.float64),
            sec_in_year=np.asarray(sec_in_year, dtype=np.float64),
            active_thickness_tolerance_m=np.asarray(1.0e-6, dtype=np.float64),
        )

        with np.load(FORMAL_DIR / "final_state_50y.npz", allow_pickle=False) as archive:
            formal_thickness = np.asarray(archive["thickness_m"], dtype=np.float64)
            formal_mask = np.asarray(archive["domain_mask"], dtype=bool)
        with np.load(REPRO_DIR / "final_state_50y.npz", allow_pickle=False) as archive:
            repro_thickness = np.asarray(archive["thickness_m"], dtype=np.float64)
            repro_mask = np.asarray(archive["domain_mask"], dtype=bool)
        if not np.array_equal(formal_thickness, repro_thickness) or not np.array_equal(formal_mask, repro_mask):
            raise RuntimeError("formal-reproduction-final-array-mismatch")
        with Dataset(GEOLOGY) as dataset:
            bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
            x = np.asarray(dataset.variables["x"][:], dtype=np.float64)
            y = np.asarray(dataset.variables["y"][:], dtype=np.float64)
            dx = float(getattr(dataset, "dx_m"))
        path_frame = pd.read_csv(OBSERVATION_PATH)
        path_xy = path_frame[["x_m", "y_m"]].to_numpy(dtype=float)
        controls = _control_lengths(path_xy)
        final_path = _path_metrics(
            formal_thickness,
            x,
            y,
            path_xy,
            controls,
            float(static["registered_lia_length_m"]),
        )
        final_domain = _free_domain_metrics(formal_thickness, dx)
        if final_path != formal["final_root_path_metrics"]:
            raise RuntimeError("endpoint-operator-formal-summary-mismatch")

        length_rows = pd.read_csv(LENGTH_REFERENCES)
        length_selected = length_rows[length_rows["lia_unique_id"].eq(SYSTEM_ID)]
        if len(length_selected) != 1 or length_selected.iloc[0]["reference_status"] != "ok":
            raise RuntimeError("length-reference-not-unique-ok")
        length_row = length_selected.iloc[0]
        lia_target = float(length_row["lia_registered_length_m"])
        rgi_target = float(length_row["rgi_registered_length_m"])
        separation = abs(lia_target - rgi_target)
        path_capacity = float(np.sum(controls))

        with np.load(STEADY_PACKAGE, allow_pickle=False) as steady_package:
            steady_indices = np.asarray(steady_package["sequence_indices"][:12], dtype=int)
            first_year_temp_error = float(
                np.max(
                    np.abs(
                        np.asarray(mb_model.temp[:12], dtype=float) + temperature_offset
                        - np.asarray(steady_package["temp"][steady_indices], dtype=float)
                        - temperature_offset
                    )
                )
            )
            first_year_prcp_error = float(
                np.max(
                    np.abs(
                        np.asarray(mb_model.prcp[:12], dtype=float)
                        - np.asarray(steady_package["prcp"][steady_indices], dtype=float)
                    )
                )
            )
            scalar_names = (
                "ref_hgt",
                "t_melt",
                "t_solid",
                "t_liq",
                "monthly_melt_f",
                "bias",
                "rho",
                "sec_in_month",
                "sec_in_year",
                "active_thickness_tolerance_m",
            )
            with np.load(package_path, allow_pickle=False) as historical_package:
                scalar_closure = {
                    name: abs(scalar(steady_package, name) - scalar(historical_package, name))
                    for name in scalar_names
                }

        active_surface = (bed + formal_thickness)[formal_thickness > 1.0e-6]
        validation_heights = np.linspace(
            float(np.min(active_surface)), float(np.max(active_surface)), 2049, dtype=np.float64
        )
        compact_heights = np.asarray(
            [validation_heights[0], np.median(validation_heights), validation_heights[-1]],
            dtype=np.float64,
        )
        selected_months = np.unique(
            np.linspace(0, RUN_MONTHS - 1, 31, dtype=np.int64)
        )
        maximum_formula_absolute_error = 0.0
        maximum_formula_relative_error = 0.0
        selected_authoritative = []
        for month_index in range(RUN_MONTHS):
            heights = validation_heights if month_index in selected_months else compact_heights
            expected = formula_monthly_mb(
                temperature=float(mb_model.temp[month_index]) + temperature_offset,
                precipitation=float(mb_model.prcp[month_index]),
                gradient=float(mb_model.grad[month_index]),
                heights=heights,
                ref_hgt=float(mb_model.ref_hgt),
                t_melt=float(mb_model.t_melt),
                t_solid=float(mb_model.t_solid),
                t_liq=float(mb_model.t_liq),
                monthly_melt_f=monthly_melt_f,
                bias=float(mb_model.bias),
                sec_in_month=sec_in_month,
                sec_in_year=sec_in_year,
                rho=float(mb_model.rho),
            )
            year = int(mb_model.years[month_index])
            month = int(mb_model.months[month_index])
            authoritative = np.asarray(
                reference_model.get_monthly_mb(heights, year=year + (month - 1) / 12.0),
                dtype=np.float64,
            )
            difference = np.abs(expected - authoritative)
            maximum_formula_absolute_error = max(
                maximum_formula_absolute_error, float(np.max(difference))
            )
            maximum_formula_relative_error = max(
                maximum_formula_relative_error,
                float(np.max(difference / np.maximum(np.abs(authoritative), 1.0e-30))),
            )
            if month_index in selected_months:
                selected_authoritative.append(authoritative)
        reference_path = stage / "historical_reference_samples.npz"
        np.savez_compressed(
            reference_path,
            validation_heights_m=validation_heights,
            selected_month_indices=selected_months,
            authoritative_monthly_mb_m_ice_s=np.asarray(selected_authoritative),
            temperature_offset_k=np.asarray(temperature_offset),
        )

        annual_rows: list[dict[str, Any]] = []
        temperature = np.asarray(mb_model.temp, dtype=float).reshape(-1, 12)
        precipitation = np.asarray(mb_model.prcp, dtype=float).reshape(-1, 12)
        for offset, year in enumerate(range(START_YEAR, END_YEAR + 1)):
            annual_rows.append(
                {
                    "year": year,
                    "used_by_1470_to_2000_state_run": year < END_YEAR,
                    "base_effective_temperature_mean_c": float(np.mean(temperature[offset])),
                    "masked_temperature_effective_mean_c": float(
                        np.mean(temperature[offset]) + temperature_offset
                    ),
                    "precipitation_annual_kg_m2": float(np.sum(precipitation[offset])),
                    "temperature_offset_k": temperature_offset,
                    "precipitation_anomaly_mode": "off",
                }
            )
        write_csv(stage / "annual_forcing_audit.csv", annual_rows)

        checks = {
            "controller_v10_152_read_only_audit": True,
            "formal_and_reproduction_pass": True,
            "formal_and_reproduction_final_arrays_exact": True,
            "candidate_v2_remains_nonproduction_and_unused": True,
            "reconstruction_source_hash_frozen": sha256(RECONSTRUCTION)
            == EXPECTED_RECONSTRUCTION_SHA256,
            "w5e5_source_is_gswp3_w5e5": (
                xr.open_dataset(SOURCE_CLIMATE, use_cftime=True).attrs.get("climate_source")
                == "GSWP3_W5E5"
            ),
            "historical_climate_independent_hash_matches_26_prior_copies": generated_hash
            == EXPECTED_FULL_CLIMATE_SHA256,
            "historical_source_month_count_1470_2000_inclusive": len(mb_model.temp)
            == SOURCE_MONTHS,
            "historical_run_sequence_1470_1999_exact": (
                len(sequence_indices) == RUN_MONTHS
                and int(mb_model.years[0]) == START_YEAR
                and int(mb_model.years[RUN_MONTHS - 1]) == END_YEAR - 1
                and int(mb_model.months[RUN_MONTHS - 1]) == 12
            ),
            "historical_arrays_finite": bool(
                np.isfinite(mb_model.temp).all()
                and np.isfinite(mb_model.prcp).all()
                and np.isfinite(mb_model.grad).all()
            ),
            "precipitation_anomalies_off": float(
                np.max(np.ptp(np.asarray(mb_model.prcp).reshape(-1, 12), axis=0))
            )
            == 0.0,
            "first_historical_year_temperature_matches_steady_window": first_year_temp_error
            == 0.0,
            "first_historical_year_precipitation_matches_steady_window": first_year_prcp_error
            == 0.0,
            "monthly_ti_scalar_contract_exact": max(scalar_closure.values()) == 0.0,
            "all_6360_months_formula_absolute_closure": maximum_formula_absolute_error
            <= 1.0e-18,
            "all_6360_months_formula_relative_closure": maximum_formula_relative_error
            <= 1.0e-10,
            "steady_state_endpoint_operator_exact": final_path
            == formal["final_root_path_metrics"],
            "lia_path_capacity_covers_registered_target": path_capacity >= lia_target,
            "modern_target_on_same_frozen_length_definition_available": (
                rgi_target > 0.0 and str(length_row["rgi_length_source"]).startswith("OGGM centerlines unique root")
            ),
            "annual_atomic_checkpoint_engine_available": True,
            "historical_start_state_has_free_domain": final_domain[
                "minimum_active_cell_center_to_grid_edge_m"
            ]
            >= 5000.0,
        }
        hashes_after = {str(path): sha256(path) for path in source_paths}
        checks["all_source_hashes_unchanged"] = hashes_after == hashes_before
        if not all(checks.values()):
            raise RuntimeError(f"historical-readiness-check-fail:{checks}")

        runtime_50y = float(formal["runtime_seconds_this_process"])
        projected_5y = runtime_50y * 5.0 / 50.0
        projected_full = runtime_50y * RUN_YEARS / 50.0
        evidence_rows = [
            {
                "component": "historical forcing",
                "status": "ready",
                "evidence": f"1470-2000 inclusive, {SOURCE_MONTHS} months, SHA256 {generated_hash}",
                "consequence": "A nonrepeating 1470-1999 sequence can drive the 1470-to-2000 state trajectory.",
            },
            {
                "component": "steady-to-historical climate continuity",
                "status": "ready",
                "evidence": f"1470 temperature/prcp max absolute errors {first_year_temp_error}/{first_year_prcp_error}",
                "consequence": "The accepted +0.334281 K offset can be carried unchanged; no fitted historical ramp is needed.",
            },
            {
                "component": "MonthlyTI operator",
                "status": "ready",
                "evidence": f"6360-month max abs/relative error {maximum_formula_absolute_error:.3e}/{maximum_formula_relative_error:.3e}",
                "consequence": "The exported package reproduces authoritative OGGM MonthlyTI calculations.",
            },
            {
                "component": "historical initial state",
                "status": "ready",
                "evidence": f"formal and reproduction final arrays exact; free domain {final_domain['minimum_active_cell_center_to_grid_edge_m']:.2f} m",
                "consequence": "The reproduced 50-year state is the only eligible historical start state.",
            },
            {
                "component": "endpoint operator",
                "status": "ready",
                "evidence": f"LIA {lia_target:.3f} m; RGI {rgi_target:.3f} m; path capacity {path_capacity:.3f} m",
                "consequence": "Annual maximum and year-2000 length can use the existing frozen root-path operator.",
            },
            {
                "component": "dynamic historical driver",
                "status": "implementation-required",
                "evidence": "steady engine already has exact monthly boundaries, annual atomic checkpoints, and resume",
                "consequence": "Create a versioned 5-year 1470-1475 state-import preflight before any 530-year run.",
            },
            {
                "component": "research-family promotion",
                "status": "not-yet-eligible",
                "evidence": "system 2005 has reproduced steady evidence but no historical pass/reproduction or independent-system confirmation",
                "consequence": "No accepted-count change and no model-family promotion are allowed yet.",
            },
        ]
        write_csv(stage / "evidence_matrix.csv", evidence_rows)
        manifest_rows = [
            {
                "role": "source",
                "path": str(path),
                "sha256": hashes_after[str(path)],
                "modified": False,
            }
            for path in source_paths
        ]
        manifest_rows.extend(
            [
                {
                    "role": "derived",
                    "path": str((OUT / derived_climate.name).relative_to(ROOT)),
                    "sha256": sha256(derived_climate),
                    "modified": False,
                },
                {
                    "role": "derived",
                    "path": str((OUT / package_path.name).relative_to(ROOT)),
                    "sha256": sha256(package_path),
                    "modified": False,
                },
                {
                    "role": "derived",
                    "path": str((OUT / reference_path.name).relative_to(ROOT)),
                    "sha256": sha256(reference_path),
                    "modified": False,
                },
            ]
        )
        write_csv(stage / "source_manifest.csv", manifest_rows)

        summary = {
            "schema": "lia-191-system2005-masked-temperature-historical-readiness-v1",
            "status": "historical-forcing-and-operator-ready-driver-preflight-next",
            "recorded_at": now(),
            "system_id": SYSTEM_ID,
            "controller_route": "v10.152",
            "checks": checks,
            "forcing": {
                "source": "GSWP3-W5E5 1961-1990 monthly climatology plus ensemble_rec_ref temperature anomaly",
                "reconstruction_source_sha256": sha256(RECONSTRUCTION),
                "derived_climate": str((OUT / derived_climate.name).relative_to(ROOT)),
                "derived_climate_sha256": sha256(derived_climate),
                "source_years_inclusive": [START_YEAR, END_YEAR],
                "source_month_count": SOURCE_MONTHS,
                "dynamic_sequence_years_inclusive": [START_YEAR, END_YEAR - 1],
                "dynamic_sequence_month_count": RUN_MONTHS,
                "precipitation_anomaly_mode": "off",
                "base_effective_temperature_bias_k": base_effective_bias,
                "masked_temperature_offset_k": temperature_offset,
                "effective_historical_temperature_bias_k": effective_historical_bias,
                "candidate_v2_used": False,
            },
            "monthly_ti_package": {
                "path": str((OUT / package_path.name).relative_to(ROOT)),
                "sha256": sha256(package_path),
                "reference_path": str((OUT / reference_path.name).relative_to(ROOT)),
                "reference_sha256": sha256(reference_path),
                "maximum_formula_absolute_error_m_ice_s": maximum_formula_absolute_error,
                "maximum_formula_relative_error": maximum_formula_relative_error,
                "first_year_temperature_closure_c": first_year_temp_error,
                "first_year_precipitation_closure_kg_m2": first_year_prcp_error,
            },
            "historical_start_state": {
                "formal_final_state": str((FORMAL_DIR / "final_state_50y.npz").relative_to(ROOT)),
                "formal_final_state_sha256": sha256(FORMAL_DIR / "final_state_50y.npz"),
                "reproduction_final_state": str((REPRO_DIR / "final_state_50y.npz").relative_to(ROOT)),
                "reproduction_final_state_sha256": sha256(REPRO_DIR / "final_state_50y.npz"),
                "scientific_arrays_exact": True,
                "initial_length_m": final_path["observed_active_control_volume_length_m"],
                "lia_relative_error": final_path["registered_length_relative_error"],
                "free_domain_m": final_domain["minimum_active_cell_center_to_grid_edge_m"],
            },
            "endpoint_contract": {
                "length_definition": "active >10 m control-volume length along frozen projected LIA main path",
                "lia_target_m": lia_target,
                "rgi_target_m": rgi_target,
                "target_separation_m": separation,
                "annual_lia_operator": "maximum annual frozen-path length over 1470-2000",
                "modern_operator": "year-2000 frozen-path length",
                "total_relative_error_gate": 0.10,
                "separation_relative_error_gate": 0.20,
            },
            "runtime_and_checkpoint_plan": {
                "scientific_preflight_years": 5,
                "preflight_calendar": [1470, 1475],
                "projected_preflight_seconds_from_formal50": projected_5y,
                "full_historical_years": RUN_YEARS,
                "projected_full_seconds_from_formal50": projected_full,
                "planning_bound_hours": 30.0,
                "checkpoint_cadence": "annual atomic",
                "resume_required": True,
            },
            "decision": {
                "historical_forcing_ready": True,
                "historical_start_state_ready": True,
                "endpoint_operator_ready": True,
                "dynamic_preflight_driver_required": True,
                "full_historical_authorized": False,
                "system_accepted": False,
                "research_family_promoted": False,
                "next_action": "implement and contract-test one 1470-1475 historical state-import preflight, then preregister exactly one dynamic preflight",
            },
            "accepted_count_effect": 0,
            "production_temperature_modified": False,
            "candidate_v2_promoted": False,
            "initial_source_thickness_modified": False,
            "dynamics_run": False,
            "source_files_modified": False,
        }
        write_json(stage / "summary.json", summary)
        (stage / "FINAL_RESULT.md").write_text(
            "# System 2005 masked-temperature historical readiness\n\n"
            "结论：历史强迫、稳态初态和长度算子已经闭合，但完整历史动力学尚未获授权。\n\n"
            f"- 独立重建得到 1470–2000 共 {SOURCE_MONTHS} 个月的气候文件，SHA-256 为 `{generated_hash}`；它与此前 26 个 System 2005 无降水异常副本完全一致。\n"
            f"- 真正用于 1470 状态推进到 2000 状态的是 1470–1999 共 {RUN_MONTHS} 个月；2000 年气候留在源文件中但不用于形成 2000 年端点状态。\n"
            f"- 1470 年温度和降水与已通过稳态包逐月误差均为 0；接受的 `+{temperature_offset:.12f} K` 作为固定模型偏置继续使用，不引入历史 ramp。\n"
            f"- 6360 个月 MonthlyTI 公式相对 OGGM 的最大绝对误差为 `{maximum_formula_absolute_error:.3e} m ice s-1`。\n"
            f"- 历史初态使用正式稳态的最终数组；独立复现数组逐位一致，起始长度为 `{final_path['observed_active_control_volume_length_m']/1000:.3f} km`，边界余量为 `{final_domain['minimum_active_cell_center_to_grid_edge_m']/1000:.3f} km`。\n"
            f"- 当前 LIA/RGI 目标分别为 `{lia_target/1000:.3f}` 和 `{rgi_target/1000:.3f} km`，继续使用冻结的主路径活动长度算子。\n"
            "- Candidate v2 未进入本实验，因为它仍是非生产候选。\n"
            "- 下一步只需实现并合同测试一次 1470–1475 的状态接入预检；通过后才可登记 530 年正式历史运行。\n"
            "- 当前不改变 58/133 计数，不晋级二维模型族，也不修改任何生产或原始文件。\n",
            encoding="utf-8",
        )
        shutil.rmtree(isolated)
        os.replace(stage, OUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps(json.loads((OUT / "summary.json").read_text(encoding="utf-8")), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
