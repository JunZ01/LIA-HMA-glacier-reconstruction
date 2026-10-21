"""Audit 3584 climate chronology and Core-to-spin-up geometry without running OGGM."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


SYSTEMS = {
    1044: {
        "rgi_id": "RGI60-15.06826",
        "candidate": (
            "validation_data/results/paleo_surface_validation_v0/"
            "layered_coordinate_search_v1/"
            "layer_B_melt_delta_boundary_refine_1044_round2_v1/"
            "system_1044/mf_4p666_dt_p1p192"
        ),
    },
    3584: {
        "rgi_id": "RGI60-14.18847",
        "candidate": (
            "validation_data/results/paleo_surface_validation_v0/"
            "layered_coordinate_search_v1/"
            "B_3584_glen_shape_refine_v1/glen_x0p75_dt_0p7"
        ),
    },
}
BASELINE_YEARS = (1441, 1470)
EARLY_YEARS = (1801, 1900)
MODERN_YEARS = (1971, 2000)
ROLLING_WINDOW_YEARS = 30
NEIGHBORHOOD_RADIUS = 2


def _time_parts(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    years = np.asarray([int(value.year) for value in dataset.time.values], dtype=int)
    months = np.asarray([int(value.month) for value in dataset.time.values], dtype=int)
    return years, months


def _annual_temperature(
    dataset: xr.Dataset, lat_index: int, lon_index: int
) -> pd.Series:
    years, _ = _time_parts(dataset)
    values = np.asarray(
        dataset["temp"].isel(lat=lat_index, lon=lon_index).values, dtype=float
    )
    table = pd.DataFrame({"year": years, "temperature_c": values})
    counts = table.groupby("year").size()
    if not counts.eq(12).all():
        raise ValueError("source-climate-does-not-have-12-months-per-year")
    annual = table.groupby("year")["temperature_c"].mean()
    annual.index = annual.index.astype(int)
    return annual


def _annual_anomaly_b_window(annual_temperature: pd.Series) -> pd.Series:
    baseline = annual_temperature.loc[BASELINE_YEARS[0] : BASELINE_YEARS[1]]
    if len(baseline) != 30:
        raise ValueError("missing-1441-1470-baseline")
    return annual_temperature - float(baseline.mean())


def _first_crossing(
    rolling: pd.Series, threshold: float, warming_direction: float
) -> float:
    available = rolling.loc[rolling.index >= EARLY_YEARS[0]].dropna()
    if warming_direction >= 0:
        crossed = available[available >= threshold]
    else:
        crossed = available[available <= threshold]
    return float(crossed.index[0]) if len(crossed) else math.nan


def _sustained_crossing(
    rolling: pd.Series, threshold: float, warming_direction: float
) -> float:
    available = rolling.loc[rolling.index >= EARLY_YEARS[0]].dropna()
    values = available.to_numpy(dtype=float)
    if warming_direction >= 0:
        warm_side = values >= threshold
    else:
        warm_side = values <= threshold
    for index in range(len(values)):
        if bool(np.all(warm_side[index:])):
            return float(available.index[index])
    return math.nan


def _series_metrics(anomaly: pd.Series) -> dict[str, float]:
    rolling = anomaly.rolling(ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS).mean()
    early = float(anomaly.loc[EARLY_YEARS[0] : EARLY_YEARS[1]].mean())
    modern = float(anomaly.loc[MODERN_YEARS[0] : MODERN_YEARS[1]].mean())
    warming = modern - early
    metrics: dict[str, float] = {
        "annual_anomaly_min_c": float(anomaly.min()),
        "annual_anomaly_max_c": float(anomaly.max()),
        "annual_anomaly_range_c": float(anomaly.max() - anomaly.min()),
        "mean_1801_1900_c": early,
        "mean_1901_1930_c": float(anomaly.loc[1901:1930].mean()),
        "mean_1921_1950_c": float(anomaly.loc[1921:1950].mean()),
        "mean_1951_1980_c": float(anomaly.loc[1951:1980].mean()),
        "mean_1971_2000_c": modern,
        "early_to_modern_change_c": warming,
    }
    for fraction in (0.25, 0.5, 0.75):
        label = int(round(100 * fraction))
        threshold = early + fraction * warming
        metrics[f"warming_{label}_threshold_c"] = float(threshold)
        metrics[f"warming_{label}_first_rolling30_end_year"] = _first_crossing(
            rolling, threshold, warming
        )
        metrics[f"warming_{label}_sustained_rolling30_end_year"] = _sustained_crossing(
            rolling, threshold, warming
        )
    valid_rolling = rolling.loc[rolling.index >= 1470].dropna()
    metrics["coldest_rolling30_end_year"] = float(valid_rolling.idxmin())
    metrics["coldest_rolling30_mean_c"] = float(valid_rolling.min())
    metrics["warmest_rolling30_end_year"] = float(valid_rolling.idxmax())
    metrics["warmest_rolling30_mean_c"] = float(valid_rolling.max())
    return metrics


def _midrank_percentile(values: np.ndarray, target: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(target) or not len(finite):
        return math.nan
    less = float(np.sum(finite < target))
    equal = float(np.sum(np.isclose(finite, target, rtol=0.0, atol=1e-10)))
    return 100.0 * (less + 0.5 * equal) / float(len(finite))


def _climate_product_audit(
    registry: pd.DataFrame,
    pixel_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    target_rows: list[dict] = []
    neighborhood_rows: list[dict] = []
    neighborhood_summary_rows: list[dict] = []
    target_series: dict[str, pd.Series] = {}
    lat_index = int(pixel_row["climate_lat_index"])
    lon_index = int(pixel_row["climate_lon_index"])

    for registry_row in registry.itertuples(index=False):
        climate_id = str(registry_row.climate_id)
        source_path = Path(str(registry_row.path))
        with xr.open_dataset(source_path, use_cftime=True) as dataset:
            annual = _annual_anomaly_b_window(
                _annual_temperature(dataset, lat_index, lon_index)
            )
            target_series[climate_id] = annual
            target_metric = _series_metrics(annual)
            target_rows.append(
                {
                    "climate_id": climate_id,
                    "source_path": str(source_path),
                    "source_temp": str(dataset.attrs.get("source_temp", "")),
                    "source_prcp": str(dataset.attrs.get("source_prcp", "")),
                    "lat_index": lat_index,
                    "lon_index": lon_index,
                    "longitude": float(dataset.lon.values[lon_index]),
                    "latitude": float(dataset.lat.values[lat_index]),
                    **target_metric,
                }
            )

            lat_indices = range(lat_index - NEIGHBORHOOD_RADIUS, lat_index + NEIGHBORHOOD_RADIUS + 1)
            lon_indices = range(lon_index - NEIGHBORHOOD_RADIUS, lon_index + NEIGHBORHOOD_RADIUS + 1)
            product_neighborhood: list[dict] = []
            for neighbor_lat in lat_indices:
                for neighbor_lon in lon_indices:
                    neighbor_annual = _annual_anomaly_b_window(
                        _annual_temperature(dataset, neighbor_lat, neighbor_lon)
                    )
                    metrics = _series_metrics(neighbor_annual)
                    row = {
                        "climate_id": climate_id,
                        "lat_index": neighbor_lat,
                        "lon_index": neighbor_lon,
                        "latitude": float(dataset.lat.values[neighbor_lat]),
                        "longitude": float(dataset.lon.values[neighbor_lon]),
                        "is_target_pixel": neighbor_lat == lat_index and neighbor_lon == lon_index,
                        **metrics,
                    }
                    neighborhood_rows.append(row)
                    product_neighborhood.append(row)

            neighborhood_frame = pd.DataFrame(product_neighborhood)
            for metric_name in (
                "mean_1921_1950_c",
                "mean_1971_2000_c",
                "early_to_modern_change_c",
                "warming_50_first_rolling30_end_year",
                "warming_50_sustained_rolling30_end_year",
            ):
                values = neighborhood_frame[metric_name].to_numpy(dtype=float)
                target_value = float(target_metric[metric_name])
                neighborhood_summary_rows.append(
                    {
                        "climate_id": climate_id,
                        "metric": metric_name,
                        "target_value": target_value,
                        "neighborhood_min": float(np.nanmin(values)),
                        "neighborhood_median": float(np.nanmedian(values)),
                        "neighborhood_max": float(np.nanmax(values)),
                        "neighborhood_std": float(np.nanstd(values)),
                        "target_midrank_percentile": _midrank_percentile(values, target_value),
                    }
                )

    return (
        pd.DataFrame(target_rows),
        pd.DataFrame(neighborhood_rows),
        pd.DataFrame(neighborhood_summary_rows),
        target_series,
    )


def _monthly_anomaly_from_1961_1990(
    dataset: xr.Dataset, values: np.ndarray
) -> np.ndarray:
    years, months = _time_parts(dataset)
    reference = (years >= 1961) & (years <= 1990)
    if int(reference.sum()) != 360:
        raise ValueError("missing-1961-1990-monthly-reference")
    climatology = np.full(13, np.nan, dtype=float)
    for month in range(1, 13):
        month_values = values[reference & (months == month)]
        if len(month_values) != 30:
            raise ValueError(f"invalid-reference-month:{month}")
        climatology[month] = float(np.mean(month_values))
    return values - climatology[months]


def _chronology_audit(
    raw_path: Path,
    generated_path: Path,
    lat_index: int,
    lon_index: int,
) -> tuple[pd.DataFrame, dict]:
    with (
        xr.open_dataset(raw_path, use_cftime=True) as raw,
        xr.open_dataset(generated_path, use_cftime=True) as generated,
    ):
        raw_years, raw_months = _time_parts(raw)
        generated_years, generated_months = _time_parts(generated)
        raw_values = np.asarray(
            raw["temp"].isel(lat=lat_index, lon=lon_index).values, dtype=float
        )
        generated_values = np.asarray(generated["temp"].values, dtype=float)
        raw_anomaly = _monthly_anomaly_from_1961_1990(raw, raw_values)
        generated_anomaly = _monthly_anomaly_from_1961_1990(generated, generated_values)
        subset = (raw_years >= int(generated_years.min())) & (
            raw_years <= int(generated_years.max())
        )
        raw_subset_years = raw_years[subset]
        raw_subset_months = raw_months[subset]
        raw_subset_anomaly = raw_anomaly[subset]
        timestamps_match = bool(
            np.array_equal(raw_subset_years, generated_years)
            and np.array_equal(raw_subset_months, generated_months)
        )
        if not timestamps_match:
            raise ValueError("raw-generated-timestamps-do-not-match")
        difference = generated_anomaly - raw_subset_anomaly
        monthly = pd.DataFrame(
            {
                "year": generated_years,
                "month": generated_months,
                "raw_monthly_anomaly_c": raw_subset_anomaly,
                "generated_monthly_anomaly_c": generated_anomaly,
                "difference_c": difference,
            }
        )
        annual = monthly.groupby("year", as_index=False).agg(
            raw_annual_anomaly_c=("raw_monthly_anomaly_c", "mean"),
            generated_annual_anomaly_c=("generated_monthly_anomaly_c", "mean"),
            annual_max_monthly_range_raw_c=("raw_monthly_anomaly_c", lambda x: float(np.max(x) - np.min(x))),
            annual_max_monthly_range_generated_c=("generated_monthly_anomaly_c", lambda x: float(np.max(x) - np.min(x))),
        )
        annual["annual_difference_c"] = (
            annual["generated_annual_anomaly_c"] - annual["raw_annual_anomaly_c"]
        )
        offset_rows = []
        raw_series = annual.set_index("year")["raw_annual_anomaly_c"]
        generated_series = annual.set_index("year")["generated_annual_anomaly_c"]
        for offset in range(-3, 4):
            shifted = raw_series.copy()
            shifted.index = shifted.index - offset
            aligned = pd.concat(
                [generated_series.rename("generated"), shifted.rename("raw")], axis=1
            ).dropna()
            rmse = float(np.sqrt(np.mean((aligned["generated"] - aligned["raw"]) ** 2)))
            offset_rows.append({"offset_years": offset, "rmse_c": rmse})
        offset_table = pd.DataFrame(offset_rows)
        best_offset = int(offset_table.loc[offset_table["rmse_c"].idxmin(), "offset_years"])
        summary = {
            "raw_path": str(raw_path),
            "generated_path": str(generated_path),
            "timestamps_match_exactly": timestamps_match,
            "month_count": int(len(monthly)),
            "year_start": int(generated_years.min()),
            "year_end": int(generated_years.max()),
            "max_abs_monthly_anomaly_difference_c": float(np.max(np.abs(difference))),
            "max_abs_annual_anomaly_difference_c": float(
                np.max(np.abs(annual["annual_difference_c"]))
            ),
            "max_raw_within_year_monthly_anomaly_range_c": float(
                annual["annual_max_monthly_range_raw_c"].max()
            ),
            "max_generated_within_year_monthly_anomaly_range_c": float(
                annual["annual_max_monthly_range_generated_c"].max()
            ),
            "best_year_offset": best_offset,
            "offset_rmse_c": offset_table.to_dict(orient="records"),
        }
    return annual, summary


def _load_single_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        value = pickle.load(stream)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected-single-flowline:{path}:{len(value)}")
        return value[0]
    return value


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> dict[str, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    selected_values = values[valid]
    selected_weights = weights[valid]
    order = np.argsort(selected_values)
    selected_values = selected_values[order]
    selected_weights = selected_weights[order]
    cumulative = np.cumsum(selected_weights) - 0.5 * selected_weights
    cumulative = cumulative / np.sum(selected_weights)
    return {
        f"p{int(round(100 * quantile)):02d}": float(
            np.interp(quantile, cumulative, selected_values)
        )
        for quantile in quantiles
    }


def _geometry_audit(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    profile_rows: list[pd.DataFrame] = []
    segment_rows: list[dict] = []
    for system_id, details in SYSTEMS.items():
        candidate_dir = project_root / str(details["candidate"])
        summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
        core_path = (
            project_root
            / "validation_data/results/paleo_surface_validation_v0/"
            "lia_elevation_band_core191_length_corrected_v1"
            / str(system_id)
            / "elevation_band_flowline_lia.pkl.gz"
        )
        spinup_path = candidate_dir / "spinup_state.pkl.gz"
        core = _load_single_flowline(core_path)
        spinup = _load_single_flowline(spinup_path)
        core_thickness = np.asarray(core.thick, dtype=float)
        spinup_thickness = np.asarray(spinup.thick, dtype=float)
        core_bed = np.asarray(core.bed_h, dtype=float)
        spinup_bed = np.asarray(spinup.bed_h, dtype=float)
        core_surface = np.asarray(core.surface_h, dtype=float)
        spinup_surface = np.asarray(spinup.surface_h, dtype=float)
        widths = np.asarray(core.widths_m, dtype=float)
        if not (
            core_thickness.shape
            == spinup_thickness.shape
            == core_bed.shape
            == spinup_bed.shape
            == widths.shape
        ):
            raise ValueError(f"geometry-shape-mismatch:{system_id}")
        if not np.allclose(widths, np.asarray(spinup.widths_m, dtype=float), rtol=0.0, atol=1e-8):
            raise ValueError(f"geometry-width-mismatch:{system_id}")
        bed_max_abs_difference = float(np.max(np.abs(spinup_bed - core_bed)))
        dx_meter = float(core.dx_meter)
        if not math.isclose(dx_meter, float(spinup.dx_meter), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"geometry-dx-mismatch:{system_id}")
        weights = widths * dx_meter
        core_ice = core_thickness > 0
        spinup_ice = spinup_thickness > 0
        common_ice = core_ice & spinup_ice
        thickness_difference = spinup_thickness - core_thickness
        surface_difference = spinup_surface - core_surface
        core_area = float(core.area_m2)
        spinup_area = float(spinup.area_m2)
        core_volume = float(core.volume_m3)
        spinup_volume = float(spinup.volume_m3)
        core_domain_weights = weights * core_ice
        common_weights = weights * common_ice
        core_quantiles = _weighted_quantile(
            thickness_difference[core_ice],
            weights[core_ice],
            (0.05, 0.25, 0.5, 0.75, 0.95),
        )
        common_quantiles = _weighted_quantile(
            thickness_difference[common_ice],
            weights[common_ice],
            (0.05, 0.25, 0.5, 0.75, 0.95),
        )
        summary_rows.append(
            {
                "lia_unique_id": system_id,
                "candidate_dir": str(candidate_dir),
                "core_flowline_path": str(core_path),
                "spinup_state_path": str(spinup_path),
                "dx_meter": dx_meter,
                "bed_max_abs_difference_m": bed_max_abs_difference,
                "core_length_m": float(core.length_m),
                "spinup_length_m": float(spinup.length_m),
                "spinup_minus_core_length_m": float(spinup.length_m - core.length_m),
                "core_area_m2": core_area,
                "spinup_area_m2": spinup_area,
                "area_ratio_spinup_core": spinup_area / core_area,
                "area_change_percent": 100.0 * (spinup_area / core_area - 1.0),
                "core_volume_m3": core_volume,
                "spinup_volume_m3": spinup_volume,
                "volume_ratio_spinup_core": spinup_volume / core_volume,
                "volume_change_percent": 100.0 * (spinup_volume / core_volume - 1.0),
                "core_mean_thickness_m": core_volume / core_area,
                "spinup_mean_thickness_m": spinup_volume / spinup_area,
                "mean_thickness_change_m": spinup_volume / spinup_area - core_volume / core_area,
                "mean_thickness_change_percent": 100.0
                * ((spinup_volume / spinup_area) / (core_volume / core_area) - 1.0),
                "core_domain_area_weighted_surface_difference_m": _weighted_mean(
                    surface_difference[core_ice], weights[core_ice]
                ),
                "core_domain_median_surface_difference_m": float(
                    np.median(surface_difference[core_ice])
                ),
                "core_domain_thickness_rmse_m": float(
                    np.sqrt(
                        np.average(
                            thickness_difference[core_ice] ** 2,
                            weights=weights[core_ice],
                        )
                    )
                ),
                "common_ice_area_weighted_surface_difference_m": _weighted_mean(
                    surface_difference[common_ice], weights[common_ice]
                ),
                "common_ice_median_surface_difference_m": float(
                    np.median(surface_difference[common_ice])
                ),
                "common_ice_thickness_rmse_m": float(
                    np.sqrt(
                        np.average(
                            thickness_difference[common_ice] ** 2,
                            weights=weights[common_ice],
                        )
                    )
                ),
                "core_domain_fraction_area_thicker_in_spinup": float(
                    np.sum(core_domain_weights[thickness_difference > 0])
                    / np.sum(core_domain_weights)
                ),
                "common_ice_fraction_area_thicker_in_spinup": float(
                    np.sum(common_weights[thickness_difference > 0])
                    / np.sum(common_weights)
                ),
                **{f"core_domain_thickness_difference_{key}_m": value for key, value in core_quantiles.items()},
                **{f"common_ice_thickness_difference_{key}_m": value for key, value in common_quantiles.items()},
            }
        )

        target_separation = float(
            summary["targets"]["lia_length_m"] - summary["targets"]["rgi_length_m"]
        )
        core_indices = np.flatnonzero(core_ice)
        core_last_index = int(core_indices[-1])
        distance_to_core_terminus = (core_last_index - np.arange(len(core_thickness))) * dx_meter
        segment = core_ice & (distance_to_core_terminus >= 0) & (
            distance_to_core_terminus <= target_separation
        )
        core_segment_volume = float(np.sum(np.asarray(core.section)[segment]) * dx_meter)
        spinup_segment_volume = float(np.sum(np.asarray(spinup.section)[segment]) * dx_meter)
        segment_rows.append(
            {
                "lia_unique_id": system_id,
                "target_lia_rgi_separation_m": target_separation,
                "fixed_core_terminus_segment_node_count": int(np.sum(segment)),
                "fixed_core_terminus_segment_length_m": float(np.sum(segment) * dx_meter),
                "core_segment_volume_m3": core_segment_volume,
                "spinup_segment_volume_m3": spinup_segment_volume,
                "segment_volume_ratio_spinup_core": spinup_segment_volume / core_segment_volume,
                "core_segment_area_weighted_mean_thickness_m": _weighted_mean(
                    core_thickness[segment], weights[segment]
                ),
                "spinup_segment_area_weighted_mean_thickness_m": _weighted_mean(
                    spinup_thickness[segment], weights[segment]
                ),
                "spinup_segment_ice_covered_length_m": float(
                    np.sum(segment & spinup_ice) * dx_meter
                ),
                "spinup_segment_ice_covered_fraction": float(
                    np.sum(segment & spinup_ice) / np.sum(segment)
                ),
                "segment_area_weighted_thickness_difference_m": _weighted_mean(
                    thickness_difference[segment], weights[segment]
                ),
                "segment_median_thickness_difference_m": float(
                    np.median(thickness_difference[segment])
                ),
            }
        )
        profile_rows.append(
            pd.DataFrame(
                {
                    "lia_unique_id": system_id,
                    "node_index": np.arange(len(core_thickness)),
                    "distance_from_head_m": (np.arange(len(core_thickness)) + 1) * dx_meter,
                    "distance_to_core_terminus_m": distance_to_core_terminus,
                    "in_fixed_core_terminus_target_segment": segment,
                    "width_m": widths,
                    "bed_h_m": core_bed,
                    "core_thickness_m": core_thickness,
                    "spinup_thickness_m": spinup_thickness,
                    "thickness_difference_m": thickness_difference,
                    "core_surface_h_m": core_surface,
                    "spinup_surface_h_m": spinup_surface,
                    "surface_difference_m": surface_difference,
                    "core_ice": core_ice,
                    "spinup_ice": spinup_ice,
                }
            )
        )
    return pd.DataFrame(summary_rows), pd.concat(profile_rows, ignore_index=True), pd.DataFrame(segment_rows)


def _plot_climate(
    target_series: dict[str, pd.Series],
    neighborhood: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for index, (climate_id, annual) in enumerate(sorted(target_series.items())):
        rolling = annual.rolling(ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS).mean()
        axes[0].plot(
            rolling.index,
            rolling.values,
            linewidth=2.0,
            color=colors(index),
            label=climate_id.replace("ensemble_", ""),
        )
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_ylabel("30-year temperature anomaly (degC)\nrelative to 1441-1470")
    axes[0].set_title("System 3584 climate products at the registered pixel")
    axes[0].legend(ncol=3, frameon=False)
    axes[0].grid(alpha=0.25)

    rec_ref_rows = neighborhood[neighborhood["climate_id"].eq("ensemble_rec_ref")]
    annual_neighbors = []
    target = target_series["ensemble_rec_ref"]
    raw_path = Path(
        target_series["ensemble_rec_ref"].attrs.get("source_path", "")
    )
    if raw_path.exists():
        with xr.open_dataset(raw_path, use_cftime=True) as dataset:
            for row in rec_ref_rows.itertuples(index=False):
                annual_neighbors.append(
                    _annual_anomaly_b_window(
                        _annual_temperature(dataset, int(row.lat_index), int(row.lon_index))
                    )
                    .rolling(ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS)
                    .mean()
                )
    if annual_neighbors:
        matrix = pd.concat(annual_neighbors, axis=1)
        axes[1].fill_between(
            matrix.index,
            matrix.quantile(0.1, axis=1).to_numpy(),
            matrix.quantile(0.9, axis=1).to_numpy(),
            color="#9ecae1",
            alpha=0.45,
            label="5x5 neighborhood 10-90%",
        )
        axes[1].plot(
            matrix.index,
            matrix.median(axis=1),
            color="#2171b5",
            linewidth=1.6,
            label="5x5 neighborhood median",
        )
    target_rolling = target.rolling(ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS).mean()
    axes[1].plot(
        target_rolling.index,
        target_rolling.values,
        color="#cb181d",
        linewidth=2.2,
        label="3584 target pixel",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("30-year temperature anomaly (degC)\nrelative to 1441-1470")
    axes[1].set_xlabel("Rolling-window end year")
    axes[1].set_title("ensemble_rec_ref spatial neighborhood")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    axes[1].set_xlim(1470, 2000)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_geometry(profile: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for axis, system_id in zip(axes, (1044, 3584)):
        system = profile[profile["lia_unique_id"].eq(system_id)]
        visible = system[system["core_ice"] | system["spinup_ice"]]
        x = visible["distance_from_head_m"].to_numpy(dtype=float) / 1000.0
        axis.plot(x, visible["core_thickness_m"], linewidth=2.0, label="Core-191 initial")
        axis.plot(x, visible["spinup_thickness_m"], linewidth=2.0, label="stable spin-up")
        segment = visible["in_fixed_core_terminus_target_segment"].to_numpy(dtype=bool)
        if np.any(segment):
            axis.axvspan(float(x[segment].min()), float(x[segment].max()), color="#fdae6b", alpha=0.22)
        axis.set_ylabel("Ice thickness (m)")
        axis.set_title(f"System {system_id}: Core-191 versus saved stable spin-up")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("Distance along elevation-band flowline (km)")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "validation_data/results/paleo_surface_validation_v0/"
            "layered_coordinate_search_v1/B_3584_forcing_geometry_response_v2/"
            "climate_geometry_read_only_audit_v1"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = (
        project_root
        / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/"
        "019f730c-ee32-74d0-8b77-fdb7832f6a06/stage3_initial_state_pilot_v0/"
        "climate_registry_6.csv"
    )
    pixel_path = (
        project_root
        / "validation_data/results/paleo_surface_validation_v0/"
        "selected_five_lia_systems_v1/selected_five_climate_pixels.csv"
    )
    registry = pd.read_csv(registry_path)
    pixels = pd.read_csv(pixel_path)
    pixel_row = pixels[pixels["rgi_id"].eq(SYSTEMS[3584]["rgi_id"])]
    if len(pixel_row) != 1:
        raise ValueError(f"3584-pixel-row-count:{len(pixel_row)}")
    pixel_row = pixel_row.iloc[0]

    target_table, neighborhood, neighborhood_summary, target_series = _climate_product_audit(
        registry, pixel_row
    )
    for climate_id, annual in target_series.items():
        source_path = Path(
            registry.loc[registry["climate_id"].eq(climate_id), "path"].iloc[0]
        )
        annual.attrs["source_path"] = str(source_path)
    target_table.to_csv(output_dir / "climate_product_target_pixel_audit.csv", index=False)
    neighborhood.to_csv(output_dir / "climate_5x5_neighborhood_metrics.csv", index=False)
    neighborhood_summary.to_csv(
        output_dir / "climate_5x5_target_percentiles.csv", index=False
    )
    annual_table = pd.DataFrame({"year": target_series["ensemble_rec_ref"].index})
    for climate_id, annual in sorted(target_series.items()):
        annual_table[climate_id] = annual.to_numpy(dtype=float)
        annual_table[f"{climate_id}_rolling30"] = annual.rolling(
            ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
        ).mean().to_numpy(dtype=float)
    annual_table.to_csv(output_dir / "climate_product_annual_curves.csv", index=False)

    representative_summary_path = (
        project_root / str(SYSTEMS[3584]["candidate"]) / "summary.json"
    )
    representative_summary = json.loads(
        representative_summary_path.read_text(encoding="utf-8")
    )
    rec_ref_path = Path(
        registry.loc[registry["climate_id"].eq("ensemble_rec_ref"), "path"].iloc[0]
    )
    generated_path = Path(
        representative_summary["climate"]["forward"]["climate_file"]
    )
    chronology_annual, chronology_summary = _chronology_audit(
        rec_ref_path,
        generated_path,
        int(pixel_row["climate_lat_index"]),
        int(pixel_row["climate_lon_index"]),
    )
    chronology_annual.to_csv(output_dir / "rec_ref_generated_chronology_annual.csv", index=False)

    geometry_summary, geometry_profile, segment_summary = _geometry_audit(project_root)
    geometry_summary.to_csv(output_dir / "core_spinup_geometry_summary.csv", index=False)
    geometry_profile.to_csv(output_dir / "core_spinup_geometry_profiles.csv", index=False)
    segment_summary.to_csv(output_dir / "core_terminus_target_segment_summary.csv", index=False)

    _plot_climate(
        target_series,
        neighborhood,
        output_dir / "climate_product_and_neighborhood_curves.png",
    )
    _plot_geometry(
        geometry_profile,
        output_dir / "core_spinup_thickness_profiles.png",
    )

    transition_metric = "warming_50_sustained_rolling30_end_year"
    warming_products = target_table[target_table["early_to_modern_change_c"] > 0].copy()
    nonwarming_products = target_table[target_table["early_to_modern_change_c"] <= 0].copy()
    transition_values = warming_products[transition_metric].to_numpy(dtype=float)
    rec_ref_percentile = neighborhood_summary[
        neighborhood_summary["climate_id"].eq("ensemble_rec_ref")
        & neighborhood_summary["metric"].eq(transition_metric)
    ].iloc[0]
    summary = {
        "audit_type": "read-only; no OGGM model constructed or advanced",
        "climate_target": {
            "lia_unique_id": 3584,
            "rgi_id": SYSTEMS[3584]["rgi_id"],
            "lat_index": int(pixel_row["climate_lat_index"]),
            "lon_index": int(pixel_row["climate_lon_index"]),
            "longitude": float(pixel_row["climate_pixel_longitude"]),
            "latitude": float(pixel_row["climate_pixel_latitude"]),
            "product_count": int(len(target_table)),
            "products_with_positive_1801_1900_to_1971_2000_change": warming_products[
                "climate_id"
            ].tolist(),
            "products_without_positive_1801_1900_to_1971_2000_change": nonwarming_products[
                "climate_id"
            ].tolist(),
            "positive_change_product_count": int(len(warming_products)),
            "positive_change_product_sustained_half_warming_rolling30_end_year_min": float(
                np.nanmin(transition_values)
            ),
            "positive_change_product_sustained_half_warming_rolling30_end_year_median": float(
                np.nanmedian(transition_values)
            ),
            "positive_change_product_sustained_half_warming_rolling30_end_year_max": float(
                np.nanmax(transition_values)
            ),
            "rec_ref_target_neighborhood_percentile": float(
                rec_ref_percentile["target_midrank_percentile"]
            ),
            "rec_ref_neighborhood_min": float(rec_ref_percentile["neighborhood_min"]),
            "rec_ref_neighborhood_median": float(
                rec_ref_percentile["neighborhood_median"]
            ),
            "rec_ref_neighborhood_max": float(rec_ref_percentile["neighborhood_max"]),
            "per_product_metrics": target_table.to_dict(orient="records"),
        },
        "chronology": chronology_summary,
        "geometry": geometry_summary.to_dict(orient="records"),
        "core_terminus_target_segments": segment_summary.to_dict(orient="records"),
        "files": {
            "target_product_metrics": "climate_product_target_pixel_audit.csv",
            "annual_climate_curves": "climate_product_annual_curves.csv",
            "neighborhood_metrics": "climate_5x5_neighborhood_metrics.csv",
            "neighborhood_percentiles": "climate_5x5_target_percentiles.csv",
            "chronology_annual": "rec_ref_generated_chronology_annual.csv",
            "geometry_summary": "core_spinup_geometry_summary.csv",
            "geometry_profiles": "core_spinup_geometry_profiles.csv",
            "terminus_segment_summary": "core_terminus_target_segment_summary.csv",
            "climate_figure": "climate_product_and_neighborhood_curves.png",
            "geometry_figure": "core_spinup_thickness_profiles.png",
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
