"""Read-only population audit of parent-climate and frozen-state equilibrium.

The same deterministic operator is applied to all 93 eligible systems.  It
uses only frozen Stage-2 parameters, the registered parent climate, and the
frozen initial grid.  Endpoint targets are carried only as cohort metadata and
are never used by the equilibrium calculation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
COHORTS = BASE / "lia_191_population_extension_cohorts_v1/population_extension_cohort_assignment.csv"
STAGE2 = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06/stage2_freeze_stage3_routing_v0"
    / "stage2_member_ensemble.csv"
)
MASTER = (
    BASE
    / "lia_191_controller_v1/diagnostics/unresolved_master_router_v1"
    / "unresolved_master_queue.csv"
)
PIXELS = BASE / "lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"
GRIDS = BASE / "core191_anchor_reroute_f19_v3"
GDIRS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0"
    / "oggm_workdir/per_glacier"
)
RECONSTRUCTION = Path(
    "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/"
    "climate_oggm_ensemble_rec_ref.nc"
)
OUT = BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v1"

START_YEAR = 1441
END_YEAR = 1470
YEARS = END_YEAR - START_YEAR + 1
RHO = 900.0
GRAD = -0.0065
T_MELT = -1.0
T_SOLID = 0.0
T_LIQ = 2.0
ROOT_LOW_C = -10.0
ROOT_HIGH_C = 10.0
ROOT_ITERATIONS = 64
POSITIVE_ICE_M = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def scalar(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"nonfinite-scalar:{value}")
    return result


@dataclass(frozen=True)
class SurfaceDistribution:
    heights: np.ndarray
    prefix: np.ndarray

    @classmethod
    def from_values(cls, values: np.ndarray) -> "SurfaceDistribution":
        heights = np.sort(np.asarray(values, dtype=np.float64))
        if heights.ndim != 1 or heights.size == 0 or not np.isfinite(heights).all():
            raise RuntimeError("invalid-active-surface-distribution")
        return cls(heights=heights, prefix=np.concatenate(([0.0], np.cumsum(heights))))

    @property
    def count(self) -> int:
        return int(self.heights.size)

    def _sum_between(self, start: int, end: int) -> float:
        return float(self.prefix[end] - self.prefix[start])

    def monthly_mean_mass(
        self,
        source_temp_c: float,
        source_prcp_kg_m2: float,
        ref_hgt_m: float,
        monthly_melt_f: float,
        delta_t_c: float,
    ) -> float:
        intercept = source_temp_c + delta_t_c - GRAD * ref_hgt_m

        melt_threshold = (T_MELT - intercept) / GRAD
        melt_end = int(np.searchsorted(self.heights, melt_threshold, side="left"))
        melt_sum = (
            melt_end * (intercept - T_MELT)
            + GRAD * self._sum_between(0, melt_end)
        )
        melt_mean = max(melt_sum, 0.0) / self.count

        liquid_threshold = (T_LIQ - intercept) / GRAD
        solid_threshold = (T_SOLID - intercept) / GRAD
        middle_start = int(np.searchsorted(self.heights, liquid_threshold, side="right"))
        middle_end = int(np.searchsorted(self.heights, solid_threshold, side="left"))
        middle_end = max(middle_end, middle_start)
        middle_count = middle_end - middle_start
        middle_temp_sum = (
            middle_count * intercept
            + GRAD * self._sum_between(middle_start, middle_end)
        )
        middle_solid_sum = middle_count - (
            middle_temp_sum - middle_count * T_SOLID
        ) / (T_LIQ - T_SOLID)
        fully_solid_count = self.count - middle_end
        solid_mean = np.clip(
            (fully_solid_count + middle_solid_sum) / self.count, 0.0, 1.0
        )
        return float(source_prcp_kg_m2 * solid_mean - monthly_melt_f * melt_mean)


def annual_mean_mb_m_ice(
    surface: SurfaceDistribution,
    temp_c: np.ndarray,
    prcp_kg_m2: np.ndarray,
    ref_hgt_m: float,
    monthly_melt_f: float,
    delta_t_c: float,
) -> float:
    monthly = [
        surface.monthly_mean_mass(t, p, ref_hgt_m, monthly_melt_f, delta_t_c)
        for t, p in zip(temp_c, prcp_kg_m2, strict=True)
    ]
    return float(np.sum(monthly) / YEARS / RHO)


def solve_equilibrium_delta(
    surface: SurfaceDistribution,
    temp_c: np.ndarray,
    prcp_kg_m2: np.ndarray,
    ref_hgt_m: float,
    monthly_melt_f: float,
) -> tuple[float, float, float]:
    def objective(delta: float) -> float:
        return annual_mean_mb_m_ice(
            surface, temp_c, prcp_kg_m2, ref_hgt_m, monthly_melt_f, delta
        )

    low = ROOT_LOW_C
    high = ROOT_HIGH_C
    low_mb = objective(low)
    high_mb = objective(high)
    if not (low_mb > 0.0 and high_mb < 0.0):
        raise RuntimeError(f"equilibrium-root-not-bracketed:{low_mb}:{high_mb}")
    for _ in range(ROOT_ITERATIONS):
        middle = (low + high) / 2.0
        middle_mb = objective(middle)
        if middle_mb > 0.0:
            low = middle
        else:
            high = middle
    root = (low + high) / 2.0
    return root, objective(root), (high - low)


def annual_mb_at_heights(
    heights_m: np.ndarray,
    temp_c: np.ndarray,
    prcp_kg_m2: np.ndarray,
    ref_hgt_m: float,
    monthly_melt_f: float,
    delta_t_c: float,
) -> np.ndarray:
    heights = np.asarray(heights_m, dtype=np.float64)
    temperature = temp_c[:, None] + GRAD * (heights[None, :] - ref_hgt_m) + delta_t_c
    solid = np.clip(1.0 - (temperature - T_SOLID) / (T_LIQ - T_SOLID), 0.0, 1.0)
    melt = np.maximum(temperature - T_MELT, 0.0)
    monthly_mass = prcp_kg_m2[:, None] * solid - monthly_melt_f * melt
    return np.sum(monthly_mass, axis=0) / YEARS / RHO


def equilibrium_line_altitude(
    temp_c: np.ndarray,
    prcp_kg_m2: np.ndarray,
    ref_hgt_m: float,
    monthly_melt_f: float,
    delta_t_c: float,
    minimum_surface_m: float,
    maximum_surface_m: float,
) -> float:
    low = minimum_surface_m - 3000.0
    high = maximum_surface_m + 3000.0
    values = annual_mb_at_heights(
        np.asarray([low, high]), temp_c, prcp_kg_m2, ref_hgt_m, monthly_melt_f, delta_t_c
    )
    if values[0] >= 0.0:
        return low
    if values[1] <= 0.0:
        return high
    for _ in range(64):
        middle = (low + high) / 2.0
        value = annual_mb_at_heights(
            np.asarray([middle]), temp_c, prcp_kg_m2, ref_hgt_m,
            monthly_melt_f, delta_t_c,
        )[0]
        if value < 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def gdir_climate_path(rgi_id: str) -> Path:
    return GDIRS / rgi_id.split(".")[0] / rgi_id[:-3] / rgi_id / "climate_historical.nc"


def system_grid_path(system_id: int, rgi_id: str) -> Path:
    return GRIDS / f"lia_{system_id}_{rgi_id}_system_grid_v1.nc"


def climate_arrays(
    baseline_path: Path,
    reconstruction: xr.Dataset,
    pixel: pd.Series,
    stage2: pd.Series,
    delta_t_c: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    with xr.open_dataset(baseline_path, use_cftime=True) as baseline:
        period = baseline.sel(time=slice("1961-01-01", "1990-12-31"))
        if period.sizes.get("time") != 360:
            raise RuntimeError(f"baseline-month-count:{baseline_path}")
        monthly_temp = np.asarray(period["temp"].groupby("time.month").mean("time"), dtype=float)
        monthly_prcp = np.asarray(period["prcp"].groupby("time.month").mean("time"), dtype=float)
        ref_hgt_m = scalar(baseline.attrs["ref_hgt"])

    lat = int(pixel["climate_lat_index"])
    lon = int(pixel["climate_lon_index"])
    rec_baseline = reconstruction.sel(time=slice("1961-01-01", "1990-12-31"))
    rec_monthly = rec_baseline["temp"].isel(lat=lat, lon=lon).groupby("time.month").mean("time")
    target = reconstruction["temp"].isel(lat=lat, lon=lon).sel(
        time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    )
    if target.sizes.get("time") != YEARS * 12:
        raise RuntimeError(f"target-month-count:{lat}:{lon}")
    anomaly = np.asarray(target.groupby("time.month") - rec_monthly, dtype=float)
    temp = (
        np.tile(monthly_temp, YEARS)
        + anomaly
        + scalar(stage2["temp_bias"])
        + float(delta_t_c)
    )
    prcp = np.tile(monthly_prcp, YEARS) * scalar(stage2["prcp_fac"])
    if not np.isfinite(temp).all() or not np.isfinite(prcp).all() or np.any(prcp < 0.0):
        raise RuntimeError("invalid-climate-arrays")
    return temp, prcp, ref_hgt_m


def load_grid(path: Path) -> tuple[SurfaceDistribution, float, float, dict[str, float]]:
    with xr.open_dataset(path) as dataset:
        thickness = np.asarray(dataset["thickness_m"].isel(member=0), dtype=float)
        surface = np.asarray(dataset["surface_elevation_m"].isel(member=0), dtype=float)
        valid = np.asarray(dataset["valid_mask"].isel(member=0), dtype=bool)
        lia = np.asarray(dataset["lia_mask"], dtype=bool)
        x = np.asarray(dataset["x"], dtype=float)
        y = np.asarray(dataset["y"], dtype=float)
    active = valid & lia & np.isfinite(surface) & np.isfinite(thickness) & (thickness > POSITIVE_ICE_M)
    if not active.any():
        raise RuntimeError(f"no-active-ice:{path}")
    dx = scalar(np.median(np.diff(x)))
    dy = abs(scalar(np.median(np.diff(y))))
    if dx <= 0.0 or dy <= 0.0 or not np.isclose(dx, dy):
        raise RuntimeError(f"invalid-grid-spacing:{path}:{dx}:{dy}")
    cell_area = dx * dy
    heights = surface[active]
    metrics = {
        "active_cell_count": int(active.sum()),
        "active_area_m2": float(active.sum() * cell_area),
        "initial_volume_m3": float(np.sum(thickness[active]) * cell_area),
        "surface_min_m": float(np.min(heights)),
        "surface_p10_m": float(np.quantile(heights, 0.10)),
        "surface_median_m": float(np.median(heights)),
        "surface_p90_m": float(np.quantile(heights, 0.90)),
        "surface_max_m": float(np.max(heights)),
    }
    return SurfaceDistribution.from_values(heights), cell_area, float(np.sum(thickness[active])), metrics


def evaluate_system(
    cohort: pd.Series,
    stage2: pd.Series,
    master: pd.Series,
    pixel: pd.Series,
    reconstruction: xr.Dataset,
) -> tuple[dict[str, object], dict[str, object]]:
    system_id = int(cohort["system_id"])
    rgi_id = str(cohort["rgi_id"])
    grid = system_grid_path(system_id, rgi_id)
    baseline = gdir_climate_path(rgi_id)
    inventory = {
        "system_id": system_id,
        "rgi_id": rgi_id,
        "cohort": str(cohort["cohort"]),
        "grid_exists": grid.is_file(),
        "baseline_climate_exists": baseline.is_file(),
        "stage2_main_midturnover_row_count": 1,
        "master_route_row_count": 1,
        "climate_pixel_row_count": 1,
    }
    if not grid.is_file() or not baseline.is_file():
        raise FileNotFoundError(f"missing-system-input:{system_id}:{inventory}")

    surface, _, _, grid_metrics = load_grid(grid)
    registered_delta = scalar(master["spinup_delta_t_c"])
    base_temp, base_prcp, ref_hgt = climate_arrays(
        baseline, reconstruction, pixel, stage2, delta_t_c=0.0
    )
    monthly_melt_f = scalar(stage2["melt_f"]) * 365.0 / 12.0
    mb_base = annual_mean_mb_m_ice(
        surface, base_temp, base_prcp, ref_hgt, monthly_melt_f, 0.0
    )
    mb_registered = annual_mean_mb_m_ice(
        surface, base_temp, base_prcp, ref_hgt, monthly_melt_f, registered_delta
    )
    equilibrium_delta, equilibrium_residual, root_width = solve_equilibrium_delta(
        surface, base_temp, base_prcp, ref_hgt, monthly_melt_f
    )
    mb_registered_warm = annual_mean_mb_m_ice(
        surface, base_temp, base_prcp, ref_hgt, monthly_melt_f, registered_delta + 0.1
    )
    ela_registered = equilibrium_line_altitude(
        base_temp, base_prcp, ref_hgt, monthly_melt_f, registered_delta,
        grid_metrics["surface_min_m"], grid_metrics["surface_max_m"],
    )
    ela_equilibrium = equilibrium_line_altitude(
        base_temp, base_prcp, ref_hgt, monthly_melt_f, equilibrium_delta,
        grid_metrics["surface_min_m"], grid_metrics["surface_max_m"],
    )
    result = {
        "system_id": system_id,
        "rgi_id": rgi_id,
        "rgi_region": int(cohort["rgi_region"]),
        "cohort": str(cohort["cohort"]),
        "cohort_order": int(cohort["cohort_order"]),
        "size_class": str(cohort["size_class"]),
        "retreat_class": str(cohort["retreat_class"]),
        "stage2_melt_f": scalar(stage2["melt_f"]),
        "stage2_prcp_fac": scalar(stage2["prcp_fac"]),
        "stage2_temp_bias_c": scalar(stage2["temp_bias"]),
        "registered_spinup_delta_t_c": registered_delta,
        "static_parent_mb_stage2_base_m_ice_yr": mb_base,
        "static_parent_mb_registered_m_ice_yr": mb_registered,
        "static_parent_source_registered_m3_yr": mb_registered * grid_metrics["active_area_m2"],
        "static_mb_sensitivity_m_ice_yr_per_c_near_registered": (
            mb_registered_warm - mb_registered
        ) / 0.1,
        "equilibrium_delta_t_c_from_stage2_base": equilibrium_delta,
        "equilibrium_correction_from_registered_c": equilibrium_delta - registered_delta,
        "equilibrium_residual_m_ice_yr": equilibrium_residual,
        "equilibrium_root_final_width_c": root_width,
        "registered_parent_ela_m": ela_registered,
        "equilibrium_parent_ela_m": ela_equilibrium,
        "registered_accumulation_area_fraction": float(
            np.mean(surface.heights >= ela_registered)
        ),
        "equilibrium_accumulation_area_fraction": float(
            np.mean(surface.heights >= ela_equilibrium)
        ),
        **grid_metrics,
    }
    return inventory, result


def validate_reference_packages(
    rows: pd.DataFrame,
    stage2_table: pd.DataFrame,
    pixels: pd.DataFrame,
    reconstruction: xr.Dataset,
) -> list[dict[str, object]]:
    validations = []
    for system_id, output_name in [
        (7, "w7generic2dsteady50_parent1441_1470_v1"),
        (79, "w79generic2dsteady50_parent1441_1470_v1"),
    ]:
        run_summary = json.loads((ROOT / output_name / "summary.json").read_text(encoding="utf-8"))
        frozen = run_summary["stage2_frozen"]
        rgi_id = str(frozen.get("main_rgi_id") or rows.loc[system_id, "rgi_id"])
        stage2_matches = stage2_table[
            stage2_table["lia_unique_id"].eq(system_id)
            & stage2_table["ensemble_role"].eq("mid-turnover")
            & stage2_table["rgi_id"].astype(str).eq(rgi_id)
        ]
        if len(stage2_matches) != 1:
            raise RuntimeError(f"reference-stage2-row-count:{system_id}:{len(stage2_matches)}")
        stage2 = stage2_matches.iloc[0].copy()
        for key, source_key in [
            ("melt_f", "melt_f"), ("prcp_fac", "prcp_fac"), ("temp_bias", "base_temp_bias_c")
        ]:
            stage2[key] = frozen[source_key]
        delta = scalar(frozen["delta_t_c"])
        pixel_matches = pixels[
            pixels["rgi_id"].astype(str).eq(rgi_id)
            & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
        ]
        if len(pixel_matches) != 1:
            raise RuntimeError(f"reference-pixel-row-count:{system_id}:{len(pixel_matches)}")
        temp, prcp, ref_hgt = climate_arrays(
            gdir_climate_path(rgi_id), reconstruction, pixel_matches.iloc[0], stage2, delta
        )
        package_root = BASE / f"lia_191_generic_planform2d_system{system_id}_parent_1441_1470_monthly_ti_v1"
        package_path = package_root / "monthly_ti_historical_1441_1470.npz"
        with np.load(package_path) as package:
            temp_error = float(np.max(np.abs(temp - package["temp"])))
            prcp_error = float(np.max(np.abs(prcp - package["prcp"])))
            ref_error = abs(ref_hgt - scalar(package["ref_hgt"]))
            grad_error = float(np.max(np.abs(np.asarray(package["grad"]) - GRAD)))
        grid = system_grid_path(system_id, rgi_id)
        surface, _, _, metrics = load_grid(grid)
        predicted_source = annual_mean_mb_m_ice(
            surface, temp, prcp, ref_hgt, scalar(frozen["melt_f"]) * 365.0 / 12.0, 0.0
        ) * metrics["active_area_m2"]
        year1 = run_summary["checkpoint_metrics"]["1"]
        actual_year1_source = scalar(year1["cumulative_source_m3"])
        validations.append({
            "system_id": system_id,
            "run_delta_t_c": delta,
            "maximum_temperature_array_error_c": temp_error,
            "maximum_precipitation_array_error_kg_m2_month": prcp_error,
            "reference_height_error_m": ref_error,
            "gradient_array_error_c_m": grad_error,
            "fixed_initial_static_source_m3_yr": predicted_source,
            "dynamic_year1_cumulative_source_m3": actual_year1_source,
            "initial_tendency_sign_match": bool(np.sign(predicted_source) == np.sign(actual_year1_source)),
            "dynamic_50y_cumulative_source_m3": scalar(run_summary["cumulative_climate_source_m3"]),
            "dynamic_50y_volume_change_fraction": scalar(run_summary["final_volume_relative_change"]),
        })
    return validations


def describe_group(group: pd.DataFrame) -> dict[str, object]:
    correction = group["equilibrium_correction_from_registered_c"]
    mb = group["static_parent_mb_registered_m_ice_yr"]
    return {
        "count": int(len(group)),
        "registered_mb_positive_count": int((mb > 0.01).sum()),
        "registered_mb_near_zero_count_abs_le_0p01": int((mb.abs() <= 0.01).sum()),
        "registered_mb_negative_count": int((mb < -0.01).sum()),
        "median_registered_mb_m_ice_yr": float(mb.median()),
        "median_equilibrium_correction_c": float(correction.median()),
        "equilibrium_correction_q25_c": float(correction.quantile(0.25)),
        "equilibrium_correction_q75_c": float(correction.quantile(0.75)),
        "absolute_correction_gt_0p5_count": int((correction.abs() > 0.5).sum()),
        "absolute_correction_gt_1p0_count": int((correction.abs() > 1.0).sum()),
        "median_equilibrium_accumulation_area_fraction": float(
            group["equilibrium_accumulation_area_fraction"].median()
        ),
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    OUT.mkdir(parents=True)

    cohorts = pd.read_csv(COHORTS)
    stage2_table = pd.read_csv(STAGE2)
    stage2_table["lia_unique_id"] = stage2_table["lia_unique_id"].astype(int)
    master_table = pd.read_csv(MASTER)
    master_table["lia_unique_id"] = master_table["lia_unique_id"].astype(int)
    pixels = pd.read_csv(PIXELS)
    if len(cohorts) != 93 or cohorts["system_id"].nunique() != 93:
        raise RuntimeError(f"cohort-count-mismatch:{len(cohorts)}")

    inventories: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    with xr.open_dataset(RECONSTRUCTION, use_cftime=True) as reconstruction:
        for cohort in cohorts.itertuples(index=False):
            cohort_series = pd.Series(cohort._asdict())
            system_id = int(cohort_series["system_id"])
            rgi_id = str(cohort_series["rgi_id"])
            stage2_rows = stage2_table[
                stage2_table["lia_unique_id"].eq(system_id)
                & stage2_table["ensemble_role"].eq("mid-turnover")
                & stage2_table["rgi_id"].astype(str).eq(rgi_id)
            ]
            master_rows = master_table[master_table["lia_unique_id"].eq(system_id)]
            pixel_rows = pixels[
                pixels["rgi_id"].astype(str).eq(rgi_id)
                & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
            ]
            if len(stage2_rows) != 1 or len(master_rows) != 1 or len(pixel_rows) != 1:
                raise RuntimeError(
                    f"registered-row-count:{system_id}:stage2={len(stage2_rows)}:"
                    f"master={len(master_rows)}:pixel={len(pixel_rows)}"
                )
            inventory, result = evaluate_system(
                cohort_series, stage2_rows.iloc[0], master_rows.iloc[0],
                pixel_rows.iloc[0], reconstruction,
            )
            inventories.append(inventory)
            results.append(result)

        result_table = pd.DataFrame(results).sort_values(["cohort", "cohort_order", "system_id"])
        reference_validation = validate_reference_packages(
            result_table.set_index("system_id"), stage2_table, pixels, reconstruction
        )

    inventory_table = pd.DataFrame(inventories).sort_values("system_id")
    validation_table = pd.DataFrame(reference_validation).sort_values("system_id")
    inventory_table.to_csv(OUT / "input_inventory_93.csv", index=False)
    result_table.to_csv(OUT / "static_equilibrium_attribution_93.csv", index=False)
    validation_table.to_csv(OUT / "system7_79_reference_validation.csv", index=False)

    package_formula_pass = bool(
        (validation_table["maximum_temperature_array_error_c"] <= 1.0e-5).all()
        and (validation_table["maximum_precipitation_array_error_kg_m2_month"] <= 1.0e-5).all()
        and (validation_table["reference_height_error_m"] <= 1.0e-9).all()
        and (validation_table["gradient_array_error_c_m"] <= 1.0e-12).all()
    )
    tendency_sign_pass = bool(validation_table["initial_tendency_sign_match"].all())
    all_roots_pass = bool(
        (result_table["equilibrium_residual_m_ice_yr"].abs() <= 1.0e-10).all()
        and (result_table["equilibrium_root_final_width_c"] <= 1.0e-12).all()
    )
    group_summary = {
        "population93": describe_group(result_table),
        **{
            cohort: describe_group(group)
            for cohort, group in result_table.groupby("cohort", sort=False)
        },
    }
    checks = {
        "population_count_93": len(result_table) == 93,
        "input_inventory_complete": bool(
            inventory_table[["grid_exists", "baseline_climate_exists"]].all(axis=None)
        ),
        "one_registered_row_per_system": bool(
            inventory_table[
                ["stage2_main_midturnover_row_count", "master_route_row_count", "climate_pixel_row_count"]
            ].eq(1).all(axis=None)
        ),
        "reference_package_formula": package_formula_pass,
        "reference_initial_tendency_sign": tendency_sign_pass,
        "equilibrium_roots": all_roots_pass,
        "no_dynamics_run": True,
        "endpoint_targets_not_used_by_operator": True,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "old_output_modified": False,
        "launch_authorized": False,
    }
    summary = {
        "schema": "lia-191-population93-static-climate-state-equilibrium-audit-v1",
        "status": "read-only-static-equilibrium-audit-pass" if all(checks.values()) else "read-only-static-equilibrium-audit-fail",
        "population": 93,
        "operator": {
            "name": "fixed-parent-climate-area-integrated-zero-net-mb-bisection-v1",
            "inputs_allowed_to_vary_by_system": [
                "frozen Stage-2 mid-turnover melt_f, prcp_fac, temp_bias",
                "registered main-glacier W5E5 baseline climate",
                "registered reconstruction climate pixel",
                "frozen V3/F19 initial surface distribution",
            ],
            "fixed_for_all_systems": {
                "parent_climate_years": [START_YEAR, END_YEAR],
                "precipitation_anomaly_mode": "off",
                "temp_melt_c": T_MELT,
                "temp_solid_c": T_SOLID,
                "temp_liquid_c": T_LIQ,
                "temperature_lapse_rate_c_m": GRAD,
                "ice_density_kg_m3": RHO,
                "root_bracket_c": [ROOT_LOW_C, ROOT_HIGH_C],
                "root_iterations": ROOT_ITERATIONS,
            },
            "objective": "area-weighted 30-year mean surface mass balance on the frozen initial ice mask equals zero",
            "endpoint_length_targets_used": False,
            "glacier_specific_manual_tuning": False,
        },
        "checks": checks,
        "group_summary": group_summary,
        "reference_validation": reference_validation,
        "interpretation": [
            "The completed system7 and system79 runs conserve the climate source in the volume budget; their scientific failure is not numerical mass leakage.",
            "This audit measures the initial climate-state mismatch and its common-rule equilibrium correction. It does not claim that zero initial net mass balance alone guarantees a 50-year dynamic steady state.",
            "The discovery cohort must test the frozen operator as one class. No named glacier may receive a custom correction or search range.",
        ],
        "next_recommended_control_action": (
            "register the deterministic equilibrium-centering operator and a fixed discovery16 execution contract; keep heldout16 sealed"
            if all(checks.values())
            else "do not authorize discovery; reconcile the failed static-audit checks"
        ),
        "dynamics_run": False,
        "model_launch_performed": False,
        "launch_authorized": False,
        "accepted_count_effect": 0,
        "controller_state_modified": False,
    }
    write_json(OUT / "summary.json", summary)

    report = [
        "# Population-93 Static Climate-State Equilibrium Audit",
        "",
        f"Status: {summary['status']}",
        "",
        "- One deterministic, endpoint-independent operator was applied to all 93 systems.",
        "- The operator uses the 1441-1470 parent climate and frozen initial geometry.",
        "- No dynamics, thickness edit, old-output edit, or model launch occurred.",
        "- The equilibrium correction is a physics-derived diagnostic, not a per-glacier manual fit.",
        "- A zero initial net balance is necessary evidence, not proof of 50-year dynamic steadiness.",
        "",
        "The next control transaction may preregister discovery16 with this fixed operator. Heldout16 remains sealed.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
