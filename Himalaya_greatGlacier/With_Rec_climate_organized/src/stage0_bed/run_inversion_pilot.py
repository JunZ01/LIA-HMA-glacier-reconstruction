"""Run provisional W5E5/MB calibration and Farinotti-constrained inversion."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy import optimize

from oggm import cfg, tasks, workflow
from oggm.core.massbalance import (
    MB_GLOBAL_PARAMS,
    MonthlyTIModel,
    decide_winter_precip_factor,
)
from oggm.shop import w5e5

from build_modern_gdirs_pilot import (
    configure_oggm,
    load_selection,
    resolve_path,
    resolve_system_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("climate-mb", "inversion", "all"), default="all"
    )
    parser.add_argument("--reset-climate", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=False)


def load_context(config_path: Path):
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    with open(config_path, encoding="utf-8") as source:
        config = json.load(source)

    manifest_path = resolve_path(repo_root, config["manifest_path"])
    geometry_path = resolve_path(repo_root, config["modern_geometry_path"])
    dem_path = resolve_path(repo_root, config["surface_dem_path"])
    output_dir = resolve_path(repo_root, config["output_dir"])
    system_ids = resolve_system_ids(manifest_path, config["lia_system_ids"])
    selected, systems = load_selection(
        manifest_path, geometry_path, system_ids
    )
    configure_oggm(config, output_dir, dem_path)
    cfg.PARAMS["baseline_climate"] = "GSWP3_W5E5"
    cfg.PARAMS["use_winter_prcp_fac"] = True
    # Reopen existing directories by RGI ID. Passing full entities would make
    # OGGM rewrite the outline shapefile on every process restart; the gdirs
    # produced by Stage 0 already contain tarred outlines and are read-only at
    # this boundary.
    gdirs = workflow.init_glacier_directories(
        selected["RGIId"].astype(str).tolist(), reset=False
    )
    center_by_id = {
        row.RGIId: row.geometry.representative_point().coords[0]
        for row in selected.itertuples(index=False)
    }
    for gdir in gdirs:
        center_lon, center_lat = center_by_id[gdir.rgi_id]
        gdir.cenlon = float(center_lon)
        gdir.cenlat = float(center_lat)
    gdir_by_id = {gdir.rgi_id: gdir for gdir in gdirs}
    return config, repo_root, output_dir, selected, systems, gdirs, gdir_by_id


def configure_local_w5e5(repo_root: Path, config: dict) -> dict[str, Path]:
    paths = {
        key: resolve_path(repo_root, value)
        for key, value in config["w5e5_files"].items()
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing local W5E5 files: {missing}")

    def local_file(dataset="GSWP3_W5E5", var=None):
        if dataset != "GSWP3_W5E5" or var not in paths:
            raise ValueError(f"Unsupported local W5E5 request: {dataset}/{var}")
        return str(paths[var])

    w5e5.get_gswp3_w5e5_file = local_file
    return paths


def load_hugonnet(config: dict, period: str) -> pd.DataFrame:
    frames = [pd.read_csv(Path(path)) for path in config["hugonnet_paths"]]
    data = pd.concat(frames, ignore_index=True)
    return data[data["period"] == period].copy().set_index("rgiid")


def system_members(selected: pd.DataFrame, system_id: int) -> list[str]:
    return selected.loc[selected["lia_unique_id"] == system_id, "RGIId"].tolist()


def system_modeled_mb(
    gdir_by_id,
    member_ids: list[str],
    hugonnet: pd.DataFrame,
    p0_by_id: dict[str, float],
    melt_f: float,
    precip_multiplier: float,
    temp_bias: float,
    years: np.ndarray,
) -> tuple[float, dict[str, float]]:
    weighted_sum = 0.0
    weight_sum = 0.0
    member_values = {}
    for rgi_id in member_ids:
        if rgi_id not in hugonnet.index or pd.isna(hugonnet.loc[rgi_id, "dmdtda"]):
            continue
        gdir = gdir_by_id[rgi_id]
        prcp_fac = float(
            np.clip(
                p0_by_id[rgi_id] * precip_multiplier,
                cfg.PARAMS["prcp_fac_min"],
                cfg.PARAMS["prcp_fac_max"],
            )
        )
        model = MonthlyTIModel(
            gdir,
            melt_f=melt_f,
            prcp_fac=prcp_fac,
            temp_bias=temp_bias,
            check_calib_params=False,
        )
        fls = gdir.read_pickle("inversion_flowlines")
        modeled = float(np.mean(model.get_specific_mb(fls=fls, year=years)))
        weight = float(hugonnet.loc[rgi_id, "area"])
        weighted_sum += weight * modeled
        weight_sum += weight
        member_values[rgi_id] = modeled
    if weight_sum <= 0:
        raise ValueError("No valid Hugonnet-weighted members in system")
    return weighted_sum / weight_sum, member_values


def write_mb_calibration(
    gdir,
    system_id: int,
    melt_f: float,
    prcp_fac: float,
    temp_bias: float,
    reference_mb: float,
    reference_mb_err: float,
    reference_period: str,
) -> None:
    record = {
        "rgi_id": gdir.rgi_id,
        "bias": 0,
        "melt_f": melt_f,
        "prcp_fac": prcp_fac,
        "temp_bias": temp_bias,
        "reference_mb": reference_mb,
        "reference_mb_err": reference_mb_err,
        "reference_period": reference_period,
        "mb_global_params": {key: cfg.PARAMS[key] for key in MB_GLOBAL_PARAMS},
        "baseline_climate_source": gdir.get_climate_info()[
            "baseline_climate_source"
        ],
    }
    gdir.write_json(record, "mb_calib")


def run_climate_mb(
    config: dict,
    repo_root: Path,
    output_dir: Path,
    selected: pd.DataFrame,
    systems: pd.DataFrame,
    gdirs,
    gdir_by_id,
    reset_climate: bool,
) -> dict:
    started = time.time()
    w5e5_paths = configure_local_w5e5(repo_root, config)
    max_ref_distance_m = float(
        config.get("max_climate_ref_distance_m", 100_000)
    )
    system_id_by_rgi = selected.set_index("RGIId")["lia_unique_id"].to_dict()
    preprocessing_failures: dict[int, list[str]] = {}
    for gdir in gdirs:
        system_id = int(system_id_by_rgi[gdir.rgi_id])
        try:
            if reset_climate and gdir.has_file("climate_historical"):
                Path(gdir.get_filepath("climate_historical")).unlink()
            if not gdir.has_file("climate_historical"):
                w5e5.process_gswp3_w5e5_data(
                    gdir,
                    y0=1901,
                    y1=2019,
                    reset=True,
                    continue_on_error=False,
                )
            with xr.open_dataset(gdir.get_filepath("climate_historical")) as ds:
                ref_distance_m = float(ds.attrs.get("ref_pix_dis", np.nan))
                ref_lon = float(ds.attrs.get("ref_pix_lon", np.nan))
                ref_lat = float(ds.attrs.get("ref_pix_lat", np.nan))
            if not np.isfinite(ref_distance_m) or ref_distance_m > max_ref_distance_m:
                raise RuntimeError(
                    "Climate reference pixel is invalid: "
                    f"distance={ref_distance_m:.1f} m, "
                    f"reference=({ref_lon:.3f}, {ref_lat:.3f}), "
                    f"glacier=({gdir.cenlon:.3f}, {gdir.cenlat:.3f})"
                )
        except Exception as exc:
            preprocessing_failures.setdefault(system_id, []).append(
                f"{gdir.rgi_id}: {type(exc).__name__}: {exc}"
            )

    period = config["mb_period"]
    years = np.arange(int(period[:4]), int(period[11:15]))
    hugonnet = load_hugonnet(config, period)
    p0_by_id = {}
    for gdir in gdirs:
        system_id = int(system_id_by_rgi[gdir.rgi_id])
        if system_id in preprocessing_failures:
            continue
        try:
            p0_by_id[gdir.rgi_id] = decide_winter_precip_factor(gdir)
        except Exception as exc:
            preprocessing_failures.setdefault(system_id, []).append(
                f"{gdir.rgi_id}: {type(exc).__name__}: {exc}"
            )
    precip_center = float(
        config["mb_candidate"]["precip_prior_multiplier"]
    )
    default_temp_bias = float(config["mb_candidate"]["temp_bias_k"])
    temp_fallback_min, temp_fallback_max = map(
        float, config["mb_candidate"]["temp_bias_fallback_bounds_k"]
    )
    default_melt_f = float(config["mb_candidate"]["default_melt_f"])
    melt_min, melt_max = map(
        float, config["mb_candidate"]["melt_f_bounds_kg_m2_day_k"]
    )

    system_rows = []
    member_rows = []
    failure_rows = []
    systems_by_id = systems.set_index("lia_unique_id")
    for system_id in sorted(systems_by_id.index.astype(int)):
        member_ids = system_members(selected, system_id)
        if system_id in preprocessing_failures:
            failure_rows.append(
                {
                    "lia_unique_id": system_id,
                    "stage": "climate-preprocessing",
                    "status": "failed-system-retained-for-review",
                    "error_type": "MemberPreprocessingFailure",
                    "error": " | ".join(preprocessing_failures[system_id]),
                    "member_count": len(member_ids),
                }
            )
            continue
        try:
            temp_bias = default_temp_bias
            target = float(systems_by_id.loc[system_id, "hugonnet_mb_kg_m2_yr"])
            target_err = (
                float(
                    systems_by_id.loc[
                        system_id, "hugonnet_err_fully_correlated_mwea"
                    ]
                )
                * 1000
            )
            precip_min = float(
                systems_by_id.loc[system_id, "precip_prior_multiplier_p025"]
            )
            precip_max = float(
                systems_by_id.loc[system_id, "precip_prior_multiplier_p975"]
            )

            def residual_melt(melt_f):
                modeled, _ = system_modeled_mb(
                    gdir_by_id,
                    member_ids,
                    hugonnet,
                    p0_by_id,
                    melt_f,
                    precip_center,
                    temp_bias,
                    years,
                )
                return modeled - target

            residual_min = residual_melt(melt_min)
            residual_max = residual_melt(melt_max)
            if residual_min * residual_max <= 0:
                melt_f = float(optimize.brentq(residual_melt, melt_min, melt_max))
                precip_multiplier = precip_center
                calibration_mode = "melt_f-at-precip-prior-center"
            else:

                def residual_precip(multiplier):
                    modeled, _ = system_modeled_mb(
                        gdir_by_id,
                        member_ids,
                        hugonnet,
                        p0_by_id,
                        default_melt_f,
                        multiplier,
                        temp_bias,
                        years,
                    )
                    return modeled - target

                precip_residual_min = residual_precip(precip_min)
                precip_residual_max = residual_precip(precip_max)
                if precip_residual_min * precip_residual_max <= 0:
                    precip_multiplier = float(
                        optimize.brentq(residual_precip, precip_min, precip_max)
                    )
                    melt_f = default_melt_f
                    calibration_mode = "precip-multiplier-at-default-melt_f"
                else:
                    precip_multiplier = precip_max
                    melt_f = default_melt_f

                    def residual_temp(candidate_temp_bias):
                        modeled, _ = system_modeled_mb(
                            gdir_by_id,
                            member_ids,
                            hugonnet,
                            p0_by_id,
                            melt_f,
                            precip_multiplier,
                            candidate_temp_bias,
                            years,
                        )
                        return modeled - target

                    temp_residual_min = residual_temp(temp_fallback_min)
                    temp_residual_max = residual_temp(temp_fallback_max)
                    if temp_residual_min * temp_residual_max > 0:
                        raise RuntimeError(
                            "MB target is outside all configured parameter "
                            "brackets; temp residuals are "
                            f"{temp_residual_min:.2f}, {temp_residual_max:.2f}"
                        )
                    temp_bias = float(
                        optimize.brentq(
                            residual_temp, temp_fallback_min, temp_fallback_max
                        )
                    )
                    calibration_mode = (
                        "temp-bias-at-upper-precip-and-default-melt_f"
                    )
            precip_at_prior_bound = bool(
                np.isclose(precip_multiplier, precip_min)
                or np.isclose(precip_multiplier, precip_max)
            )
            melt_f_at_bound = bool(
                np.isclose(melt_f, melt_min) or np.isclose(melt_f, melt_max)
            )
            identifiability_status = (
                "weak-identifiability"
                if precip_at_prior_bound or melt_f_at_bound
                else "provisional-candidate"
            )
            modeled_system, modeled_members = system_modeled_mb(
                gdir_by_id,
                member_ids,
                hugonnet,
                p0_by_id,
                melt_f,
                precip_multiplier,
                temp_bias,
                years,
            )

            for rgi_id in member_ids:
                prcp_fac = float(
                    np.clip(
                        p0_by_id[rgi_id] * precip_multiplier,
                        cfg.PARAMS["prcp_fac_min"],
                        cfg.PARAMS["prcp_fac_max"],
                    )
                )
                write_mb_calibration(
                    gdir_by_id[rgi_id],
                    system_id,
                    melt_f,
                    prcp_fac,
                    temp_bias,
                    target,
                    target_err,
                    period,
                )
                member_rows.append(
                    {
                        "lia_unique_id": system_id,
                        "rgi_id": rgi_id,
                        "hugonnet_valid": rgi_id in modeled_members,
                        "winter_prcp_fac_center": p0_by_id[rgi_id],
                        "precip_prior_multiplier": precip_multiplier,
                        "prcp_fac": prcp_fac,
                        "melt_f": melt_f,
                        "temp_bias": temp_bias,
                        "calibration_mode": calibration_mode,
                        "identifiability_status": identifiability_status,
                        "modeled_mb_kg_m2_yr": modeled_members.get(rgi_id, np.nan),
                        "calibration_role": "provisional-pilot-candidate",
                    }
                )
            system_rows.append(
                {
                    "lia_unique_id": system_id,
                    "member_count": len(member_ids),
                    "hugonnet_valid_member_count": len(modeled_members),
                    "target_mb_kg_m2_yr": target,
                    "modeled_mb_kg_m2_yr": modeled_system,
                    "residual_kg_m2_yr": modeled_system - target,
                    "target_error_upper_kg_m2_yr": target_err,
                    "melt_f": melt_f,
                    "precip_prior_multiplier": precip_multiplier,
                    "temp_bias": temp_bias,
                    "calibration_mode": calibration_mode,
                    "precip_at_prior_bound": precip_at_prior_bound,
                    "melt_f_at_bound": melt_f_at_bound,
                    "identifiability_status": identifiability_status,
                    "status": "provisional-pilot-candidate",
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "lia_unique_id": system_id,
                    "stage": "climate-mb",
                    "status": "failed-system-retained-for-review",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "member_count": len(member_ids),
                }
            )

    member_table = pd.DataFrame(member_rows)
    system_table = pd.DataFrame(system_rows)
    failure_table = pd.DataFrame(
        failure_rows,
        columns=[
            "lia_unique_id",
            "stage",
            "status",
            "error_type",
            "error",
            "member_count",
        ],
    )
    member_table.to_csv(output_dir / "mb_member_parameters.csv", index=False)
    system_table.to_csv(output_dir / "mb_system_candidates.csv", index=False)
    failure_table.to_csv(output_dir / "climate_mb_failures.csv", index=False)
    failed_system_ids = sorted(
        set(int(value) for value in failure_table["lia_unique_id"].tolist())
    )
    failed_system_fraction = len(failed_system_ids) / len(systems_by_id)
    summary = {
        "pilot_id": config["pilot_id"],
        "phase": "climate-mb",
        "status": "provisional-pilot-candidate-not-final-posterior",
        "system_count": len(system_table),
        "member_count": len(member_table),
        "failed_system_count": len(failed_system_ids),
        "failed_system_ids": failed_system_ids,
        "failed_system_fraction": failed_system_fraction,
        "climate_ready_count": sum(
            int(gdir.has_file("climate_historical")) for gdir in gdirs
        ),
        "w5e5_period": "1901-2019",
        "max_climate_ref_distance_m": max_ref_distance_m,
        "w5e5_local_files": {key: str(path) for key, path in w5e5_paths.items()},
        "mb_period": period,
        "initial_temp_bias_k": default_temp_bias,
        "parameter_selection_rule": (
            "fit system melt_f at precipitation-prior center; if infeasible, "
            "fit precipitation multiplier inside its registered prior while "
            "holding melt_f at the OGGM default; if still infeasible, hold "
            "precipitation at its registered upper bound and fit temp_bias "
            "inside the configured fallback interval"
        ),
        "calibration_mode_counts": system_table["calibration_mode"].value_counts().to_dict()
        if not system_table.empty
        else {},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "climate_mb_summary.json", summary)
    max_failed_fraction = float(config.get("max_failed_system_fraction", 1.0))
    if failed_system_fraction > max_failed_fraction:
        raise RuntimeError(
            "Climate-MB failed-system fraction "
            f"{failed_system_fraction:.3f} exceeds {max_failed_fraction:.3f}"
        )
    return summary


def filtered_system_volume(gdirs, glen_a: float, fs: float) -> tuple[float, dict]:
    member_volumes = {}
    for gdir in gdirs:
        tasks.mass_conservation_inversion(
            gdir, glen_a=glen_a, fs=fs, write=True
        )
        volume = float(tasks.filter_inversion_output(gdir))
        member_volumes[gdir.rgi_id] = volume
    return sum(member_volumes.values()), member_volumes


def write_member_bed(gdir, variable: str, scenario_id: str) -> dict:
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as dataset:
        raw_thickness = dataset[variable].values.astype("float64")
        mask = dataset["glacier_mask"].values.astype(bool)
    with rasterio.open(gdir.get_filepath("dem")) as source:
        surface = source.read(1).astype("float64")
        profile = source.profile.copy()
        pixel_area = abs(source.transform.a * source.transform.e)
        nodata = source.nodata

    valid_surface = np.isfinite(surface)
    if nodata is not None:
        valid_surface &= surface != nodata
    negative_thickness_pixels = int(
        np.sum((raw_thickness < -1e-5) & mask & np.isfinite(raw_thickness))
    )
    nonfinite_thickness_pixels = int(np.sum(~np.isfinite(raw_thickness) & mask))
    thickness = np.where(
        mask & np.isfinite(raw_thickness), np.maximum(raw_thickness, 0), 0
    )
    bed = surface.copy()
    bed[mask & valid_surface] = surface[mask & valid_surface] - thickness[
        mask & valid_surface
    ]
    bed[~valid_surface] = -9999.0

    output_path = Path(gdir.dir) / f"bed_{scenario_id}.tif"
    profile.update(dtype="float32", nodata=-9999.0, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as target:
        target.write(bed.astype("float32"), 1)

    return {
        "distributed_volume_m3": float(np.sum(thickness[mask]) * pixel_area),
        "negative_thickness_pixels": negative_thickness_pixels,
        "nonfinite_thickness_pixels": nonfinite_thickness_pixels,
        "bed_above_surface_pixels": int(
            np.sum((bed > surface + 1e-5) & mask & valid_surface)
        ),
        "max_thickness_m": float(np.max(thickness[mask])),
        "bed_path": str(output_path),
    }


def run_inversion(
    config: dict,
    repo_root: Path,
    output_dir: Path,
    selected: pd.DataFrame,
    gdirs,
    gdir_by_id,
) -> dict:
    started = time.time()
    mb_source_output_dir = resolve_path(
        repo_root, config.get("mb_source_output_dir", str(output_dir))
    )
    required = mb_source_output_dir / "mb_system_candidates.csv"
    if not required.exists():
        raise FileNotFoundError("Run the climate-mb phase before inversion")
    candidate_table = pd.read_csv(required)
    candidate_system_ids = set(
        int(value) for value in candidate_table["lia_unique_id"].tolist()
    )
    climate_failure_path = mb_source_output_dir / "climate_mb_failures.csv"
    climate_failed_system_ids = set()
    if climate_failure_path.exists():
        climate_failures = pd.read_csv(climate_failure_path)
        if not climate_failures.empty:
            climate_failed_system_ids = set(
                int(value) for value in climate_failures["lia_unique_id"].tolist()
            )

    thickness_stats = pd.read_csv(
        resolve_path(repo_root, config["thickness_target_stats_path"])
    )
    product = config["thickness_target_product"]
    volume_column = {
        "farinotti2019": "far_volume_km3",
        "millan2022": "millan_volume_km3",
    }[product]
    target_by_id = thickness_stats.set_index("rgi_id")[volume_column].to_dict()
    scenario = config["inversion_scenario"]
    scenario_id = scenario["scenario_id"]
    filesuffix = scenario["filesuffix"]
    fs = float(scenario["fs"])
    default_glen_a = float(cfg.PARAMS["glen_a"])
    multipliers = [float(value) for value in scenario["glen_a_multiplier_scan"]]

    scan_rows = []
    system_rows = []
    member_rows = []
    failure_rows = []
    system_id_by_rgi = selected.set_index("RGIId")["lia_unique_id"].to_dict()
    prep_failed_system_ids = set()

    # Prepare only systems with a successful MB candidate. A failure in one
    # member is recorded against its system and must not stop the batch.
    for gdir in gdirs:
        system_id = int(system_id_by_rgi[gdir.rgi_id])
        if system_id not in candidate_system_ids:
            continue
        if system_id in climate_failed_system_ids:
            continue
        if not gdir.has_file("mb_calib"):
            prep_failed_system_ids.add(system_id)
            failure_rows.append(
                {
                    "lia_unique_id": system_id,
                    "stage": "inversion-preparation",
                    "status": "failed-system-retained-for-review",
                    "error_type": "MissingCalibration",
                    "error": f"Missing mb_calib for {gdir.rgi_id}",
                    "member_count": len(system_members(selected, system_id)),
                }
            )
            continue
        try:
            cfg.PARAMS["auto_skip_task"] = False
            tasks.apparent_mb_from_any_mb(gdir, mb_years=(2000, 2020))
            tasks.prepare_for_inversion(gdir)
        except Exception as exc:
            prep_failed_system_ids.add(system_id)
            failure_rows.append(
                {
                    "lia_unique_id": system_id,
                    "stage": "inversion-preparation",
                    "status": "failed-system-retained-for-review",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "member_count": len(system_members(selected, system_id)),
                }
            )

    for system_id in sorted(selected["lia_unique_id"].unique()):
        system_id = int(system_id)
        member_ids = system_members(selected, system_id)
        if system_id in climate_failed_system_ids:
            failure_rows.append(
                {
                    "lia_unique_id": system_id,
                    "stage": "inversion",
                    "status": "skipped-after-climate-mb-failure",
                    "error_type": "ClimateMBFailure",
                    "error": "No valid MB candidate was produced for this system",
                    "member_count": len(member_ids),
                }
            )
            continue
        if system_id not in candidate_system_ids or system_id in prep_failed_system_ids:
            continue

        try:
            system_gdirs = [gdir_by_id[rgi_id] for rgi_id in member_ids]
            target_volume_m3 = sum(target_by_id[rgi_id] for rgi_id in member_ids) * 1e9
            evaluated = {}

            def residual_log10(log10_multiplier):
                multiplier = 10 ** float(log10_multiplier)
                if multiplier not in evaluated:
                    volume, _ = filtered_system_volume(
                        system_gdirs, default_glen_a * multiplier, fs
                    )
                    evaluated[multiplier] = volume
                    scan_rows.append(
                        {
                            "lia_unique_id": system_id,
                            "glen_a_multiplier": multiplier,
                            "glen_a_pa_3_s_1": default_glen_a * multiplier,
                            "filtered_volume_km3": volume / 1e9,
                            "target_volume_km3": target_volume_m3 / 1e9,
                            "residual_fraction": volume / target_volume_m3 - 1,
                        }
                    )
                return evaluated[multiplier] - target_volume_m3

            scan_residuals = [
                (multiplier, residual_log10(np.log10(multiplier)))
                for multiplier in multipliers
            ]
            bracket = None
            for left, right in zip(scan_residuals[:-1], scan_residuals[1:]):
                if left[1] == 0 or left[1] * right[1] < 0:
                    bracket = (left[0], right[0])
                    break
            if bracket is None:
                raise RuntimeError(
                    "Thickness target is outside the configured glen_a scan"
                )

            log_root = optimize.brentq(
                residual_log10,
                np.log10(bracket[0]),
                np.log10(bracket[1]),
                xtol=1e-7,
            )
            multiplier = 10 ** log_root
            glen_a = default_glen_a * multiplier
            final_volume_m3, member_line_volumes = filtered_system_volume(
                system_gdirs, glen_a, fs
            )

            for gdir in system_gdirs:
                inversion_output = gdir.read_pickle("inversion_output")
                gdir.write_pickle(
                    inversion_output, "inversion_output", filesuffix=filesuffix
                )
                base_inversion_output = Path(gdir.get_filepath("inversion_output"))
                suffixed_inversion_output = base_inversion_output.with_name(
                    f"{base_inversion_output.stem}{filesuffix}"
                    f"{base_inversion_output.suffix}"
                )
                if not suffixed_inversion_output.exists():
                    shutil.copy2(
                        base_inversion_output,
                        suffixed_inversion_output,
                    )
                tasks.distribute_thickness_per_altitude(
                    gdir, varname_suffix=filesuffix
                )
                variable = "distributed_thickness" + filesuffix
                bed_qc = write_member_bed(gdir, variable, scenario_id)
                member_target_m3 = target_by_id[gdir.rgi_id] * 1e9
                member_rows.append(
                    {
                        "lia_unique_id": system_id,
                        "rgi_id": gdir.rgi_id,
                        "scenario_id": scenario_id,
                        "thickness_target_product": product,
                        "glen_a_multiplier": multiplier,
                        "glen_a_pa_3_s_1": glen_a,
                        "fs": fs,
                        "target_volume_km3": member_target_m3 / 1e9,
                        "line_inversion_volume_km3": member_line_volumes[gdir.rgi_id]
                        / 1e9,
                        "distributed_volume_km3": bed_qc["distributed_volume_m3"]
                        / 1e9,
                        "distributed_minus_line_fraction": bed_qc[
                            "distributed_volume_m3"
                        ]
                        / member_line_volumes[gdir.rgi_id]
                        - 1,
                        "negative_thickness_pixels": bed_qc[
                            "negative_thickness_pixels"
                        ],
                        "nonfinite_thickness_pixels": bed_qc[
                            "nonfinite_thickness_pixels"
                        ],
                        "bed_above_surface_pixels": bed_qc[
                            "bed_above_surface_pixels"
                        ],
                        "max_thickness_m": bed_qc["max_thickness_m"],
                        "bed_path": bed_qc["bed_path"],
                        "status": "provisional-modern-member-bed-ready",
                    }
                )

            system_rows.append(
                {
                    "lia_unique_id": system_id,
                    "scenario_id": scenario_id,
                    "thickness_target_product": product,
                    "member_count": len(member_ids),
                    "glen_a_multiplier": multiplier,
                    "glen_a_pa_3_s_1": glen_a,
                    "fs": fs,
                    "target_volume_km3": target_volume_m3 / 1e9,
                    "filtered_inversion_volume_km3": final_volume_m3 / 1e9,
                    "residual_fraction": final_volume_m3 / target_volume_m3 - 1,
                    "within_tolerance": abs(final_volume_m3 / target_volume_m3 - 1)
                    <= float(scenario["system_volume_tolerance_fraction"]),
                    "status": "provisional-pilot-candidate",
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "lia_unique_id": system_id,
                    "stage": "inversion",
                    "status": "failed-system-retained-for-review",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "member_count": len(member_ids),
                }
            )

    pd.DataFrame(scan_rows).to_csv(output_dir / "glen_a_scan.csv", index=False)
    member_table = pd.DataFrame(member_rows)
    system_table = pd.DataFrame(system_rows)
    failure_table = pd.DataFrame(
        failure_rows,
        columns=[
            "lia_unique_id",
            "stage",
            "status",
            "error_type",
            "error",
            "member_count",
        ],
    )
    member_table.to_csv(output_dir / "inversion_member_status.csv", index=False)
    system_table.to_csv(output_dir / "inversion_system_summary.csv", index=False)
    failure_table.to_csv(output_dir / "inversion_failures.csv", index=False)
    failed_system_ids = sorted(
        set(int(value) for value in failure_table["lia_unique_id"].tolist())
    )
    total_system_count = int(selected["lia_unique_id"].nunique())
    failed_system_fraction = len(failed_system_ids) / total_system_count
    summary = {
        "pilot_id": config["pilot_id"],
        "phase": "inversion",
        "status": "provisional-modern-member-bed-not-final-lia-system-bed",
        "scenario_id": scenario_id,
        "thickness_target_product": product,
        "mb_source_output_dir": str(mb_source_output_dir),
        "system_count": len(system_table),
        "member_count": len(member_table),
        "candidate_system_count": len(candidate_system_ids),
        "failed_system_count": len(failed_system_ids),
        "failed_system_ids": failed_system_ids,
        "failed_system_fraction": failed_system_fraction,
        "system_volume_tolerance_fraction": float(
            scenario["system_volume_tolerance_fraction"]
        ),
        "systems_within_tolerance": int(system_table["within_tolerance"].sum())
        if not system_table.empty
        else 0,
        "members_with_negative_thickness": int(
            (member_table["negative_thickness_pixels"] > 0).sum()
        )
        if not member_table.empty
        else 0,
        "members_with_nonfinite_thickness": int(
            (member_table["nonfinite_thickness_pixels"] > 0).sum()
        )
        if not member_table.empty
        else 0,
        "members_with_bed_above_surface": int(
            (member_table["bed_above_surface_pixels"] > 0).sum()
        )
        if not member_table.empty
        else 0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "inversion_summary.json", summary)
    max_failed_fraction = float(config.get("max_failed_system_fraction", 1.0))
    if failed_system_fraction > max_failed_fraction:
        raise RuntimeError(
            "Inversion failed-system fraction "
            f"{failed_system_fraction:.3f} exceeds {max_failed_fraction:.3f}"
        )
    return summary


def main() -> None:
    args = parse_args()
    (
        config,
        repo_root,
        output_dir,
        selected,
        systems,
        gdirs,
        gdir_by_id,
    ) = load_context(args.config)

    results = {}
    if args.phase in ("climate-mb", "all"):
        results["climate_mb"] = run_climate_mb(
            config,
            repo_root,
            output_dir,
            selected,
            systems,
            gdirs,
            gdir_by_id,
            args.reset_climate,
        )
    if args.phase in ("inversion", "all"):
        results["inversion"] = run_inversion(
            config, repo_root, output_dir, selected, gdirs, gdir_by_id
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
