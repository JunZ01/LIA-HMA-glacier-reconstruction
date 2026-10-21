"""Attribute ModE-RA HMA temperature to ModE-Sim and assimilation increment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOWS = tuple((start, start + 49) for start in range(1450, 1900, 50))
ALLOWED_YEARS = np.concatenate((np.arange(1421, 1921), np.arange(1940, 2001)))


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_series(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=1)
    return np.nansum(values * weights[None], axis=1) / np.where(
        denominator == 0, np.nan, denominator
    )


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.sqrt(np.sum(weights[valid] * values[valid] ** 2) / np.sum(weights[valid])))


def weighted_covariance(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.sum(weights[valid] * left[valid] * right[valid]) / np.sum(weights[valid]))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 4 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half : index + half + 1]
        if np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def cell_edges(centers: np.ndarray) -> np.ndarray:
    if not np.all(np.diff(centers) > 0):
        raise RuntimeError("cell-centers-not-increasing")
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def extract_coarse(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    support: pd.DataFrame,
) -> np.ndarray:
    cells = []
    for row in support.itertuples(index=False):
        i = np.flatnonzero(np.isclose(lat, row.latitude, atol=1e-8))
        j = np.flatnonzero(np.isclose(lon, row.longitude, atol=1e-8))
        if len(i) != 1 or len(j) != 1:
            raise RuntimeError(f"support-grid-match-failed:{row.latitude}:{row.longitude}")
        cells.append(field[:, int(i[0]), int(j[0])])
    return np.stack(cells, axis=1)


def aggregate_reference(
    values: np.ndarray,
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    hma_mask: np.ndarray,
    support: pd.DataFrame,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> np.ndarray:
    lat_edges = cell_edges(coarse_lat)
    lon_edges = cell_edges(coarse_lon)
    result = np.full((values.shape[0], len(support)), np.nan)
    for cell_index, row in enumerate(support.itertuples(index=False)):
        i = int(np.flatnonzero(np.isclose(coarse_lat, row.latitude, atol=1e-8))[0])
        j = int(np.flatnonzero(np.isclose(coarse_lon, row.longitude, atol=1e-8))[0])
        fine_i = np.flatnonzero((fine_lat >= lat_edges[i]) & (fine_lat < lat_edges[i + 1]))
        fine_j = np.flatnonzero((fine_lon >= lon_edges[j]) & (fine_lon < lon_edges[j + 1]))
        local_mask = hma_mask[np.ix_(fine_i, fine_j)]
        ii, jj = np.where(local_mask)
        selected_i = fine_i[ii]
        selected_j = fine_j[jj]
        if len(selected_i) != int(row.fine_hma_cell_count):
            raise RuntimeError(
                f"coarse-support-count-conflict:{cell_index}:{len(selected_i)}:"
                f"{row.fine_hma_cell_count}"
            )
        subset = values[:, selected_i, selected_j]
        local_weights = np.cos(np.deg2rad(fine_lat[selected_i]))
        valid = np.isfinite(subset)
        denominator = np.sum(valid * local_weights[None], axis=1)
        result[:, cell_index] = np.nansum(
            subset * local_weights[None], axis=1
        ) / np.where(denominator == 0, np.nan, denominator)
    return result


def component_classification(
    forcing_share: float,
    forcing_sign_count: int,
    increment_share: float,
    increment_sign_count: int,
) -> str:
    if forcing_share >= 0.67 and forcing_sign_count >= 8:
        return "mode_sim_forcing_dominant"
    if increment_share >= 0.67 and increment_sign_count >= 8:
        return "assimilation_increment_dominant"
    return "mixed_or_opposing"


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_75_mode_ra_prior_assimilation_attribution_v1"
    round67 = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    round68 = loop / "round_68_mode_ra_modern_independent_validation_v1"
    round69 = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v3"
    round70 = loop / "round_70_mode_ra_lia_regional_adjudication_v2"
    round71 = loop / "round_71_mode_ra_lia_spatial_consensus_v1"
    round73 = loop / "round_73_mode_ra_hma_source_overlap_v1"
    round74 = loop / "round_74_mode_sim_allowed_temperature_acquisition_v3"
    round21 = loop / "round_21_temperature_reference_field_v1"
    paths = {
        "mode_ra_paleo": round69 / "mode_ra_temp2_1421_1920_hma_bbox.nc",
        "mode_ra_modern": round67 / "mode_ra_ensmean_1940_2000_hma_bbox.nc",
        "mode_sim_allowed": round74 / "mode_sim_temp2_allowed_hma_bbox.nc",
        "round74_summary": round74 / "summary.json",
        "round73_summary": round73 / "summary.json",
        "coarse_support": round68 / "mode_ra_hma_coarse_support.csv",
        "modern_curves": round68 / "mode_ra_hma_curves.csv",
        "regional_windows_control": round70 / "mode_ra_lia_50year_adjudication.csv",
        "regional_annual_control": round70 / "mode_ra_project_reference_annual_curves.csv",
        "spatial_windows_control": round71 / "mode_ra_reference_lia_spatial_window_scores.csv",
        "reference_field": round21 / "temperature_reference_family_balanced_allowed_v1.nc",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    if not json.loads(paths["round74_summary"].read_text(encoding="utf-8"))[
        "passes_acquisition_preflight"
    ]:
        raise RuntimeError("round74-acquisition-did-not-pass")

    with h5py.File(paths["mode_ra_paleo"], "r") as handle:
        paleo_years = np.asarray(handle["time"], dtype=int)
        ra_lat = np.asarray(handle["lat"], dtype=float)
        ra_lon = np.asarray(handle["lon"], dtype=float)
        ra_paleo = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)
    with h5py.File(paths["mode_ra_modern"], "r") as handle:
        modern_years = np.asarray(handle["time"], dtype=int)
        modern_lat = np.asarray(handle["lat"], dtype=float)
        modern_lon = np.asarray(handle["lon"], dtype=float)
        ra_modern_raw = np.asarray(handle["temp2_anomaly"], dtype=float)
    with h5py.File(paths["mode_sim_allowed"], "r") as handle:
        sim_years = np.asarray(handle["time"], dtype=int)
        sim_lat = np.asarray(handle["lat"], dtype=float)
        sim_lon = np.asarray(handle["lon"], dtype=float)
        sim = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)
    if not np.array_equal(paleo_years, np.arange(1421, 1921)):
        raise RuntimeError("mode-ra-paleo-year-axis-failed")
    if not np.array_equal(modern_years, np.arange(1940, 2001)):
        raise RuntimeError("mode-ra-modern-year-axis-failed")
    if not np.array_equal(sim_years, ALLOWED_YEARS):
        raise RuntimeError("mode-sim-allowed-year-axis-failed")
    if np.any((sim_years >= 1921) & (sim_years <= 1939)):
        raise RuntimeError("locked-period-present-in-mode-sim")
    if not (
        np.array_equal(ra_lat, modern_lat)
        and np.array_equal(ra_lon, modern_lon)
        and np.array_equal(ra_lat, sim_lat)
        and np.array_equal(ra_lon, sim_lon)
    ):
        raise RuntimeError("mode-ra-mode-sim-grid-conflict")
    baseline = (modern_years >= 1961) & (modern_years <= 1990)
    ra_modern = ra_modern_raw - np.mean(ra_modern_raw[baseline], axis=0)[None]
    ra = np.concatenate((ra_paleo, ra_modern), axis=0)
    if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(sim)):
        raise RuntimeError("nonfinite-mode-ra-or-mode-sim")
    increment = ra - sim
    field_closure = float(np.max(np.abs(ra - (sim + increment))))

    support = pd.read_csv(paths["coarse_support"])
    if len(support) != 140:
        raise RuntimeError(f"unexpected-coarse-support-count:{len(support)}")
    weights = (
        np.cos(np.deg2rad(support.latitude.to_numpy(float)))
        * support.fine_hma_cell_count.to_numpy(float)
    )
    ra_coarse = extract_coarse(ra, ra_lat, ra_lon, support)
    sim_coarse = extract_coarse(sim, sim_lat, sim_lon, support)
    increment_coarse = ra_coarse - sim_coarse
    coarse_closure = float(np.max(np.abs(ra_coarse - (sim_coarse + increment_coarse))))
    ra_curve = weighted_series(ra_coarse, weights)
    sim_curve = weighted_series(sim_coarse, weights)
    increment_curve = weighted_series(increment_coarse, weights)
    regional_closure = float(np.max(np.abs(ra_curve - (sim_curve + increment_curve))))

    annual = pd.DataFrame(
        {
            "year": ALLOWED_YEARS,
            "mode_ra_hma_degC": ra_curve,
            "mode_sim_hma_degC": sim_curve,
            "assimilation_increment_hma_degC": increment_curve,
        }
    )
    annual.to_csv(output / "mode_ra_mode_sim_increment_annual_hma.csv", index=False)

    paleo_selection = (ALLOWED_YEARS >= 1450) & (ALLOWED_YEARS <= 1899)
    if not np.array_equal(ALLOWED_YEARS[paleo_selection], np.arange(1450, 1900)):
        raise RuntimeError("paleo-comparison-axis-failed")
    with netcdf_file(paths["reference_field"], "r", mmap=False) as handle:
        reference_years = np.asarray(handle.variables["year"][:], dtype=int)
        selected_indices = np.flatnonzero(
            (reference_years >= 1450) & (reference_years <= 1899)
        )
        if not np.array_equal(reference_years[selected_indices], np.arange(1450, 1900)):
            raise RuntimeError("reference-comparison-axis-failed")
        selection = slice(int(selected_indices[0]), int(selected_indices[-1]) + 1)
        fine_lat = np.asarray(handle.variables["lat"][:], dtype=float)
        fine_lon = np.asarray(handle.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(handle.variables["hma_mask"][:], dtype=int).astype(bool)
        reference = np.asarray(
            handle.variables["tas_reference_family_balanced"][selection], dtype=float
        )
        fill = float(handle.variables["tas_reference_family_balanced"]._FillValue)
    reference[np.isclose(reference, fill) | (reference < -1e20)] = np.nan
    reference_coarse = aggregate_reference(
        reference, fine_lat, fine_lon, hma_mask, support, sim_lat, sim_lon
    )
    if not np.all(np.isfinite(reference_coarse)):
        raise RuntimeError("nonfinite-reference-coarse-field")

    paleo_years_common = np.arange(1450, 1900)
    ra_paleo_coarse = ra_coarse[paleo_selection]
    sim_paleo_coarse = sim_coarse[paleo_selection]
    increment_paleo_coarse = increment_coarse[paleo_selection]
    regional_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for start, end in WINDOWS:
        selected = (paleo_years_common >= start) & (paleo_years_common <= end)
        ra_window = np.mean(ra_paleo_coarse[selected], axis=0)
        sim_window = np.mean(sim_paleo_coarse[selected], axis=0)
        increment_window = np.mean(increment_paleo_coarse[selected], axis=0)
        reference_window = np.mean(reference_coarse[selected], axis=0)
        ra_mean = weighted_mean(ra_window, weights)
        sim_mean = weighted_mean(sim_window, weights)
        increment_mean = weighted_mean(increment_window, weights)
        reference_mean = weighted_mean(reference_window, weights)
        regional_denominator = abs(sim_mean) + abs(increment_mean)
        regional_rows.append(
            {
                "start_year": start,
                "end_year": end,
                "mode_ra_mean_degC": ra_mean,
                "mode_sim_mean_degC": sim_mean,
                "assimilation_increment_mean_degC": increment_mean,
                "four_family_reference_mean_degC": reference_mean,
                "closure_error_degC": ra_mean - sim_mean - increment_mean,
                "mode_sim_absolute_share": abs(sim_mean) / regional_denominator,
                "assimilation_increment_absolute_share": abs(increment_mean)
                / regional_denominator,
                "mode_sim_same_sign_as_mode_ra": bool(np.sign(sim_mean) == np.sign(ra_mean)),
                "increment_same_sign_as_mode_ra": bool(
                    np.sign(increment_mean) == np.sign(ra_mean)
                ),
                "mode_ra_abs_error_vs_reference_degC": abs(ra_mean - reference_mean),
                "mode_sim_abs_error_vs_reference_degC": abs(sim_mean - reference_mean),
            }
        )

        ra_residual = ra_window - ra_mean
        sim_residual = sim_window - sim_mean
        increment_residual = increment_window - increment_mean
        reference_residual = reference_window - reference_mean
        residual_closure = float(
            np.max(np.abs(ra_residual - sim_residual - increment_residual))
        )
        ra_rms = weighted_rms(ra_residual, weights)
        sim_rms = weighted_rms(sim_residual, weights)
        increment_rms = weighted_rms(increment_residual, weights)
        reference_rms = weighted_rms(reference_residual, weights)
        covariance = weighted_covariance(sim_residual, increment_residual, weights)
        variance_closure = ra_rms**2 - sim_rms**2 - increment_rms**2 - 2 * covariance
        rms_denominator = sim_rms + increment_rms
        ra_reference_rmse = weighted_rms(ra_residual - reference_residual, weights)
        sim_reference_rmse = weighted_rms(sim_residual - reference_residual, weights)
        increment_reference_rmse = weighted_rms(
            increment_residual - reference_residual, weights
        )
        spatial_rows.append(
            {
                "start_year": start,
                "end_year": end,
                "mode_ra_residual_rms_degC": ra_rms,
                "mode_sim_residual_rms_degC": sim_rms,
                "assimilation_increment_residual_rms_degC": increment_rms,
                "reference_residual_rms_degC": reference_rms,
                "mode_sim_rms_share": sim_rms / rms_denominator,
                "assimilation_increment_rms_share": increment_rms / rms_denominator,
                "mode_sim_increment_weighted_covariance_degC2": covariance,
                "variance_closure_error_degC2": variance_closure,
                "residual_field_closure_max_abs_degC": residual_closure,
                "mode_sim_to_mode_ra_residual_correlation": correlation(
                    sim_residual, ra_residual
                ),
                "increment_to_mode_ra_residual_correlation": correlation(
                    increment_residual, ra_residual
                ),
                "mode_ra_to_reference_residual_correlation": correlation(
                    ra_residual, reference_residual
                ),
                "mode_sim_to_reference_residual_correlation": correlation(
                    sim_residual, reference_residual
                ),
                "increment_to_reference_residual_correlation": correlation(
                    increment_residual, reference_residual
                ),
                "mode_ra_reference_residual_skill": 1.0
                - ra_reference_rmse / reference_rms,
                "mode_sim_reference_residual_skill": 1.0
                - sim_reference_rmse / reference_rms,
                "increment_reference_residual_skill": 1.0
                - increment_reference_rmse / reference_rms,
            }
        )
    regional = pd.DataFrame(regional_rows)
    spatial = pd.DataFrame(spatial_rows)
    regional.to_csv(output / "regional_50year_attribution.csv", index=False)
    spatial.to_csv(output / "spatial_50year_attribution.csv", index=False)

    regional_control = pd.read_csv(paths["regional_windows_control"])
    spatial_control = pd.read_csv(paths["spatial_windows_control"])
    regional_control_closure = float(
        np.max(
            np.abs(
                regional.mode_ra_mean_degC.to_numpy()
                - regional_control.mode_ra_mean_degC.to_numpy()
            )
        )
    )
    spatial_control_corr_closure = float(
        np.max(
            np.abs(
                spatial.mode_ra_to_reference_residual_correlation.to_numpy()
                - spatial_control.spatial_residual_correlation.to_numpy()
            )
        )
    )
    spatial_control_skill_closure = float(
        np.max(
            np.abs(
                spatial.mode_ra_reference_residual_skill.to_numpy()
                - spatial_control.spatial_residual_skill.to_numpy()
            )
        )
    )

    modern_curves = pd.read_csv(paths["modern_curves"])
    modern_curves = modern_curves[modern_curves.variable == "temperature"].copy()
    modern_sim = dict(zip(ALLOWED_YEARS.tolist(), sim_curve.tolist()))
    modern_ra = dict(zip(ALLOWED_YEARS.tolist(), ra_curve.tolist()))
    modern_rows: list[dict[str, Any]] = []
    modern_ra_curve_closure = 0.0
    for observation, group in modern_curves.groupby("observation"):
        group = group.sort_values("year")
        years = group.year.to_numpy(int)
        observed = group.observed.to_numpy(float)
        ra_values = np.asarray([modern_ra[int(year)] for year in years])
        sim_values = np.asarray([modern_sim[int(year)] for year in years])
        modern_ra_curve_closure = max(
            modern_ra_curve_closure,
            float(np.max(np.abs(ra_values - group.mode_ra.to_numpy(float)))),
        )
        ra_rmse = float(np.sqrt(np.mean((ra_values - observed) ** 2)))
        sim_rmse = float(np.sqrt(np.mean((sim_values - observed) ** 2)))
        modern_rows.append(
            {
                "observation": observation,
                "start_year": int(years.min()),
                "end_year": int(years.max()),
                "year_count": len(years),
                "mode_ra_rmse_degC": ra_rmse,
                "mode_sim_rmse_degC": sim_rmse,
                "assimilation_rmse_reduction_fraction": 1.0 - ra_rmse / sim_rmse,
                "mode_ra_annual_correlation": correlation(ra_values, observed),
                "mode_sim_annual_correlation": correlation(sim_values, observed),
                "mode_ra_15year_correlation": correlation(
                    smooth(ra_values), smooth(observed)
                ),
                "mode_sim_15year_correlation": correlation(
                    smooth(sim_values), smooth(observed)
                ),
            }
        )
    modern = pd.DataFrame(modern_rows)
    modern.to_csv(output / "modern_hma_regional_assimilation_gain.csv", index=False)

    median_regional_sim_share = float(regional.mode_sim_absolute_share.median())
    median_regional_increment_share = float(
        regional.assimilation_increment_absolute_share.median()
    )
    regional_sim_sign_count = int(regional.mode_sim_same_sign_as_mode_ra.sum())
    regional_increment_sign_count = int(regional.increment_same_sign_as_mode_ra.sum())
    regional_class = component_classification(
        median_regional_sim_share,
        regional_sim_sign_count,
        median_regional_increment_share,
        regional_increment_sign_count,
    )
    median_spatial_sim_share = float(spatial.mode_sim_rms_share.median())
    median_spatial_increment_share = float(spatial.assimilation_increment_rms_share.median())
    median_sim_ra_spatial_corr = float(
        spatial.mode_sim_to_mode_ra_residual_correlation.median()
    )
    median_increment_ra_spatial_corr = float(
        spatial.increment_to_mode_ra_residual_correlation.median()
    )
    if median_spatial_sim_share >= 0.67 and median_sim_ra_spatial_corr >= 0.80:
        spatial_class = "mode_sim_forcing_dominant"
    elif median_spatial_increment_share >= 0.67 and median_increment_ra_spatial_corr >= 0.80:
        spatial_class = "assimilation_increment_dominant"
    else:
        spatial_class = "mixed_or_opposing"
    modern_improved_count = int((modern.assimilation_rmse_reduction_fraction > 0).sum())

    gates = {
        "field_and_regional_closure_below_1e_12": max(
            field_closure, coarse_closure, regional_closure
        )
        <= 1e-12,
        "residual_and_variance_closure_below_1e_12": max(
            float(spatial.residual_field_closure_max_abs_degC.max()),
            float(np.abs(spatial.variance_closure_error_degC2).max()),
        )
        <= 1e-12,
        "round70_regional_control_closure_below_1e_6": regional_control_closure <= 1e-6,
        "round71_spatial_control_closure_below_1e_6": max(
            spatial_control_corr_closure, spatial_control_skill_closure
        )
        <= 1e-6,
        "modern_round68_ra_curve_closure_below_1e_6": modern_ra_curve_closure <= 1e-6,
        "locked_1921_1939_absent": True,
    }
    summary = {
        "status": "complete",
        "allowed_periods": [[1421, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "coarse_hma_cell_count": len(support),
        "lia_window_count": len(regional),
        "identity": "mode_ra_equals_mode_sim_plus_assimilation_increment",
        "closure": {
            "field_max_abs_degC": field_closure,
            "coarse_max_abs_degC": coarse_closure,
            "regional_max_abs_degC": regional_closure,
            "residual_field_max_abs_degC": float(
                spatial.residual_field_closure_max_abs_degC.max()
            ),
            "variance_max_abs_degC2": float(
                np.abs(spatial.variance_closure_error_degC2).max()
            ),
            "round70_regional_max_abs_degC": regional_control_closure,
            "round71_spatial_correlation_max_abs": spatial_control_corr_closure,
            "round71_spatial_skill_max_abs": spatial_control_skill_closure,
            "round68_modern_ra_curve_max_abs_degC": modern_ra_curve_closure,
        },
        "regional_attribution": {
            "classification": regional_class,
            "median_mode_sim_absolute_share": median_regional_sim_share,
            "median_assimilation_increment_absolute_share": median_regional_increment_share,
            "mode_sim_same_sign_window_count": regional_sim_sign_count,
            "increment_same_sign_window_count": regional_increment_sign_count,
            "mode_sim_closer_to_four_family_than_mode_ra_window_count": int(
                (
                    regional.mode_sim_abs_error_vs_reference_degC
                    < regional.mode_ra_abs_error_vs_reference_degC
                ).sum()
            ),
        },
        "spatial_attribution": {
            "classification": spatial_class,
            "median_mode_sim_rms_share": median_spatial_sim_share,
            "median_assimilation_increment_rms_share": median_spatial_increment_share,
            "median_mode_sim_to_mode_ra_residual_correlation": median_sim_ra_spatial_corr,
            "median_increment_to_mode_ra_residual_correlation": median_increment_ra_spatial_corr,
            "median_mode_sim_to_reference_residual_correlation": float(
                spatial.mode_sim_to_reference_residual_correlation.median()
            ),
            "median_increment_to_reference_residual_correlation": float(
                spatial.increment_to_reference_residual_correlation.median()
            ),
            "median_mode_ra_reference_residual_skill": float(
                spatial.mode_ra_reference_residual_skill.median()
            ),
            "median_mode_sim_reference_residual_skill": float(
                spatial.mode_sim_reference_residual_skill.median()
            ),
        },
        "modern_assimilation_diagnostic": {
            "observation_product_count": len(modern),
            "mode_ra_lower_rmse_product_count": modern_improved_count,
            "median_rmse_reduction_fraction": float(
                modern.assimilation_rmse_reduction_fraction.median()
            ),
            "median_mode_ra_annual_correlation": float(
                modern.mode_ra_annual_correlation.median()
            ),
            "median_mode_sim_annual_correlation": float(
                modern.mode_sim_annual_correlation.median()
            ),
        },
        "source_independence_caveat": (
            "mode_sim_is_a_forced_model_prior; mode_ra_assimilation_uses_proxy_evidence_"
            "that_overlaps_existing_hma_collections"
        ),
        "gates": gates,
        "passes_mechanical_attribution": bool(all(gates.values())),
        "decision": (
            f"regional_{regional_class};spatial_{spatial_class};"
            "retain_mode_ra_as_modern_validated_but_not_proxy_evidence_independent"
        ),
        "production_release_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.7), constrained_layout=True)
    paleo_annual = annual[(annual.year >= 1421) & (annual.year <= 1920)]
    axes[0, 0].plot(
        paleo_annual.year,
        smooth(paleo_annual.mode_ra_hma_degC.to_numpy(float)),
        color="#111111",
        linewidth=2.1,
        label="ModE-RA",
    )
    axes[0, 0].plot(
        paleo_annual.year,
        smooth(paleo_annual.mode_sim_hma_degC.to_numpy(float)),
        color="#0072B2",
        linewidth=1.8,
        label="ModE-Sim",
    )
    axes[0, 0].plot(
        paleo_annual.year,
        smooth(paleo_annual.assimilation_increment_hma_degC.to_numpy(float)),
        color="#D55E00",
        linewidth=1.6,
        label="Assimilation increment",
    )
    axes[0, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 0].set(title="HMA annual attribution, 15-year means", ylabel="Anomaly (degC)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)

    x = np.arange(len(regional))
    labels = [f"{row.start_year}-{row.end_year}" for row in regional.itertuples()]
    axes[0, 1].bar(x - 0.25, regional.mode_sim_mean_degC, width=0.25, color="#0072B2", label="ModE-Sim")
    axes[0, 1].bar(x, regional.assimilation_increment_mean_degC, width=0.25, color="#D55E00", label="Increment")
    axes[0, 1].bar(x + 0.25, regional.mode_ra_mean_degC, width=0.25, color="#333333", label="ModE-RA")
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    axes[0, 1].set(title="LIA 50-year regional means", ylabel="Anomaly (degC)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(x, regional.mode_sim_absolute_share * 100, marker="o", color="#0072B2", label="Regional |Sim| share")
    axes[1, 0].plot(x, spatial.mode_sim_rms_share * 100, marker="s", color="#009E73", label="Spatial Sim RMS share")
    axes[1, 0].axhline(67, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 0].set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    axes[1, 0].set(title="Prior contribution shares", ylabel="Share (%)", ylim=(0, 105))
    axes[1, 0].legend(frameon=False, fontsize=8)

    modern_x = np.arange(len(modern))
    axes[1, 1].bar(modern_x - 0.18, modern.mode_sim_rmse_degC, width=0.36, color="#0072B2", label="ModE-Sim")
    axes[1, 1].bar(modern_x + 0.18, modern.mode_ra_rmse_degC, width=0.36, color="#CC79A7", label="ModE-RA")
    axes[1, 1].set_xticks(modern_x, modern.observation, rotation=25, ha="right")
    axes[1, 1].set(title="Modern HMA regional RMSE", ylabel="RMSE (degC)")
    axes[1, 1].legend(frameon=False)
    figure.savefig(output / "mode_ra_prior_assimilation_attribution.png", dpi=180)
    plt.close(figure)

    lines = [
        "# ModE-RA Prior and Assimilation Attribution",
        "",
        f"- Regional classification: `{regional_class}`; median ModE-Sim absolute share `{median_regional_sim_share:.1%}`.",
        f"- Spatial classification: `{spatial_class}`; median ModE-Sim residual-RMS share `{median_spatial_sim_share:.1%}`.",
        f"- Median Sim/RA residual correlation: `{median_sim_ra_spatial_corr:.3f}`.",
        f"- Assimilation lowers modern HMA regional RMSE for `{modern_improved_count}/{len(modern)}` products; median reduction `{modern.assimilation_rmse_reduction_fraction.median():.1%}`.",
        f"- Mechanical attribution gates pass: `{str(summary['passes_mechanical_attribution']).lower()}`.",
        "",
        "Interpretation: ModE-Sim is a forced-model prior, while the assimilation increment uses proxy evidence that overlaps existing HMA collections. Neither component is an independent observation.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
