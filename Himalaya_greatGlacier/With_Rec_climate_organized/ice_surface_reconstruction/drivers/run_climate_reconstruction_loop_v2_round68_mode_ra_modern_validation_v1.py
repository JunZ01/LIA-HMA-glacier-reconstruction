"""Validate the allowed-period ModE-RA HMA fields against modern products."""

from __future__ import annotations

import calendar
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
TEMPERATURE_OBSERVATIONS = {
    "Berkeley": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}
PRECIPITATION_OBSERVATIONS = {
    "CRU": "pr_obs_cru_pre_05deg",
    "GPCC": "pr_obs_gpcp_05deg",
}
PRECIP_CHUNKS = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))


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


def text_attr(value: Any) -> str:
    item = np.asarray(value).reshape(-1)[0]
    return item.decode() if isinstance(item, bytes) else str(item)


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).reshape(-1)[0])
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def recenter(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990)
    if int(baseline.sum()) != 30:
        raise RuntimeError(f"baseline-year-count-failed:{int(baseline.sum())}")
    center = np.nanmean(values[baseline], axis=0)
    return values - center[None]


def decode_precl_time(values: np.ndarray, units: str) -> list[datetime]:
    if units != "hours since 1800-1-1 00:00:0.0":
        raise RuntimeError(f"unexpected-precl-time-units:{units}")
    origin = datetime(1800, 1, 1)
    return [origin + timedelta(hours=float(value)) for value in values]


def annualize_precl(paths: list[Path], target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dates: list[datetime] = []
    arrays: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            lat = np.asarray(handle["lat"], dtype=float)
            lon = np.asarray(handle["lon"], dtype=float)
            local_dates = decode_precl_time(
                np.asarray(handle["time"], dtype=float),
                text_attr(handle["time"].attrs["units"]),
            )
            values = np.asarray(handle["precip"], dtype=float)
            fill = float(np.asarray(handle["precip"].attrs.get("missing_value", -9.96921e36)).reshape(-1)[0])
            values[np.isclose(values, fill) | (values < -1e20)] = np.nan
        lat_index = np.asarray([np.flatnonzero(np.isclose(lat, value, atol=1e-7))[0] for value in target_lat])
        lon_index = np.asarray([np.flatnonzero(np.isclose(lon, value, atol=1e-7))[0] for value in target_lon])
        dates.extend(local_dates)
        arrays.append(values[:, lat_index, :][:, :, lon_index])
    years = np.arange(1948, 2001)
    expected = [(int(year), month) for year in years for month in range(1, 13)]
    if [(date.year, date.month) for date in dates] != expected:
        raise RuntimeError("precl-month-axis-failed")
    monthly = np.concatenate(arrays)
    annual = np.full((len(years), len(target_lat), len(target_lon)), np.nan)
    for index, year in enumerate(years):
        values = monthly[index * 12 : (index + 1) * 12]
        weights = np.asarray([calendar.monthrange(int(year), month)[1] for month in range(1, 13)])
        complete = np.all(np.isfinite(values), axis=0)
        annual[index] = np.where(
            complete,
            np.sum(values * weights[:, None, None], axis=0) / weights.sum(),
            np.nan,
        )
    return years, annual


def annualize_udel(path: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        units = text_attr(handle["time"].attrs["units"])
        if units != "days since 1900-01-01 00:00:00":
            raise RuntimeError(f"unexpected-udel-time-units:{units}")
        dates = [datetime(1900, 1, 1) + timedelta(days=float(value)) for value in handle["time"]]
        selected = [index for index, date in enumerate(dates) if 1940 <= date.year <= 2000]
        expected = [(year, month) for year in range(1940, 2001) for month in range(1, 13)]
        if [(dates[index].year, dates[index].month) for index in selected] != expected:
            raise RuntimeError("udel-month-axis-failed")
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        lat_index = np.asarray([np.flatnonzero(np.isclose(lat, value, atol=1e-7))[0] for value in target_lat])
        lon_index = np.asarray([np.flatnonzero(np.isclose(lon, value, atol=1e-7))[0] for value in target_lon])
        monthly = np.asarray(handle["precip"][selected][:, lat_index, :][:, :, lon_index], dtype=float)
    years = np.arange(1940, 2001)
    annual = np.full((len(years), len(target_lat), len(target_lon)), np.nan)
    for index, year in enumerate(years):
        values = monthly[index * 12 : (index + 1) * 12]
        complete = np.all(np.isfinite(values), axis=0)
        annual[index] = np.where(
            complete,
            np.sum(values, axis=0) * 10.0 / (366 if calendar.isleap(int(year)) else 365),
            np.nan,
        )
    return years, annual


def cell_edges(centers: np.ndarray) -> np.ndarray:
    if not np.all(np.diff(centers) > 0):
        raise RuntimeError("cell-centers-not-increasing")
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def build_coarse_support(
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    hma_mask: np.ndarray,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> list[dict[str, Any]]:
    lat_edges = cell_edges(coarse_lat)
    lon_edges = cell_edges(coarse_lon)
    cells: list[dict[str, Any]] = []
    for i, latitude in enumerate(coarse_lat):
        fine_i = np.flatnonzero((fine_lat >= lat_edges[i]) & (fine_lat < lat_edges[i + 1]))
        for j, longitude in enumerate(coarse_lon):
            fine_j = np.flatnonzero((fine_lon >= lon_edges[j]) & (fine_lon < lon_edges[j + 1]))
            if not len(fine_i) or not len(fine_j):
                continue
            local_mask = hma_mask[np.ix_(fine_i, fine_j)]
            ii, jj = np.where(local_mask)
            if len(ii) < 3:
                continue
            cells.append(
                {
                    "coarse_i": i,
                    "coarse_j": j,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "fine_i": fine_i[ii],
                    "fine_j": fine_j[jj],
                    "fine_cell_count": int(len(ii)),
                }
            )
    if len(cells) < 40:
        raise RuntimeError(f"insufficient-mode-ra-hma-cells:{len(cells)}")
    return cells


def aggregate_fine(values: np.ndarray, fine_lat: np.ndarray, cells: list[dict[str, Any]]) -> np.ndarray:
    result = np.full((values.shape[0], len(cells)), np.nan)
    for cell_index, cell in enumerate(cells):
        lat_index = np.asarray(cell["fine_i"], dtype=int)
        lon_index = np.asarray(cell["fine_j"], dtype=int)
        subset = values[:, lat_index, lon_index]
        weights = np.cos(np.deg2rad(fine_lat[lat_index]))
        valid = np.isfinite(subset)
        denominator = np.sum(valid * weights[None], axis=1)
        result[:, cell_index] = np.nansum(subset * weights[None], axis=1) / np.where(
            denominator == 0, np.nan, denominator
        )
    return result


def extract_coarse(values: np.ndarray, cells: list[dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [values[:, int(cell["coarse_i"]), int(cell["coarse_j"])] for cell in cells],
        axis=1,
    )


def weighted_series(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=1)
    return np.nansum(values * weights[None], axis=1) / np.where(denominator == 0, np.nan, denominator)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
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


def pair_metrics(candidate: np.ndarray, observation: np.ndarray, years: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    common = np.all(np.isfinite(candidate), axis=0) & np.all(np.isfinite(observation), axis=0)
    if int(common.sum()) < 4:
        raise RuntimeError(f"insufficient-common-coarse-cells:{int(common.sum())}")
    candidate = candidate[:, common]
    observation = observation[:, common]
    weights = weights[common]
    candidate_series = weighted_series(candidate, weights)
    observation_series = weighted_series(observation, weights)
    regional_rmse = float(np.sqrt(np.mean((candidate_series - observation_series) ** 2)))
    zero_regional_rmse = float(np.sqrt(np.mean(observation_series**2)))
    candidate_residual = candidate - candidate_series[:, None]
    observation_residual = observation - observation_series[:, None]
    weight_matrix = np.broadcast_to(weights[None], observation_residual.shape)
    spatial_rmse = float(
        np.sqrt(np.sum(weight_matrix * (candidate_residual - observation_residual) ** 2) / np.sum(weight_matrix))
    )
    zero_spatial_rmse = float(
        np.sqrt(np.sum(weight_matrix * observation_residual**2) / np.sum(weight_matrix))
    )
    pattern_correlations = [
        correlation(candidate_residual[index], observation_residual[index])
        for index in range(len(years))
    ]
    return {
        "common_cell_count": int(common.sum()),
        "regional_rmse": regional_rmse,
        "zero_regional_rmse": zero_regional_rmse,
        "regional_skill": float(1.0 - regional_rmse / zero_regional_rmse),
        "annual_correlation": correlation(candidate_series, observation_series),
        "correlation_15year": correlation(smooth(candidate_series), smooth(observation_series)),
        "spatial_residual_rmse": spatial_rmse,
        "zero_spatial_residual_rmse": zero_spatial_rmse,
        "spatial_residual_skill": float(1.0 - spatial_rmse / zero_spatial_rmse),
        "median_annual_spatial_pattern_correlation": float(np.nanmedian(pattern_correlations)),
        "candidate_series": candidate_series,
        "observation_series": observation_series,
    }


def summarize_variable(region_scores: pd.DataFrame, block_scores: pd.DataFrame) -> dict[str, Any]:
    hma = region_scores[region_scores.region == "hma"]
    regional_gates = {
        "median_region_product_skill_at_least_5pct": float(region_scores.regional_skill.median()) >= 0.05,
        "worst_region_product_regression_at_most_5pct": float(region_scores.regional_skill.min()) >= -0.05,
        "at_least_75pct_region_product_pairs_improve": float(np.mean(region_scores.regional_skill > 0)) >= 0.75,
        "median_annual_correlation_at_least_0_30": float(region_scores.annual_correlation.median()) >= 0.30,
        "median_15year_correlation_at_least_0_30": float(region_scores.correlation_15year.median()) >= 0.30,
        "at_least_70pct_available_blocks_improve": float(np.mean(block_scores.regional_skill > 0)) >= 0.70,
    }
    spatial_gates = {
        "median_hma_spatial_residual_skill_positive": float(hma.spatial_residual_skill.median()) > 0,
        "worst_hma_spatial_residual_regression_at_most_5pct": float(hma.spatial_residual_skill.min()) >= -0.05,
        "at_least_75pct_hma_products_have_positive_spatial_skill": float(np.mean(hma.spatial_residual_skill > 0)) >= 0.75,
        "median_hma_spatial_pattern_correlation_at_least_0_20": float(
            hma.median_annual_spatial_pattern_correlation.median()
        ) >= 0.20,
    }
    return {
        "product_count": int(hma.observation.nunique()),
        "region_product_pair_count": int(len(region_scores)),
        "median_region_product_skill": float(region_scores.regional_skill.median()),
        "worst_region_product_skill": float(region_scores.regional_skill.min()),
        "positive_region_product_fraction": float(np.mean(region_scores.regional_skill > 0)),
        "median_annual_correlation": float(region_scores.annual_correlation.median()),
        "median_15year_correlation": float(region_scores.correlation_15year.median()),
        "positive_block_fraction": float(np.mean(block_scores.regional_skill > 0)),
        "median_hma_spatial_residual_skill": float(hma.spatial_residual_skill.median()),
        "worst_hma_spatial_residual_skill": float(hma.spatial_residual_skill.min()),
        "positive_hma_spatial_product_fraction": float(np.mean(hma.spatial_residual_skill > 0)),
        "median_hma_spatial_pattern_correlation": float(hma.median_annual_spatial_pattern_correlation.median()),
        "regional_gates": regional_gates,
        "spatial_gates": spatial_gates,
        "passes_regional_curve": bool(all(regional_gates.values())),
        "passes_spatial_field": bool(all(regional_gates.values()) and all(spatial_gates.values())),
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_68_mode_ra_modern_independent_validation_v1"
    round67 = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    public = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    mask_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    udel = loop / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc"
    precl_dir = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    precl_paths = [precl_dir / f"precl_{start}_{end}.nc" for start, end in PRECIP_CHUNKS]
    paths = {
        "mode_ra": round67 / "mode_ra_ensmean_1940_2000_hma_bbox.nc",
        "round67_summary": round67 / "summary.json",
        "public_observations": public,
        "hma_mask": mask_path,
        "udel": udel,
        **{f"precl_{start}_{end}": path for (start, end), path in zip(PRECIP_CHUNKS, precl_paths)},
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    round67_summary = json.loads(paths["round67_summary"].read_text(encoding="utf-8"))
    if not round67_summary["passes_acquisition_preflight"]:
        raise RuntimeError("round67-acquisition-did-not-pass")

    with h5py.File(paths["mode_ra"], "r") as handle:
        mode_years = np.asarray(handle["time"], dtype=int)
        mode_lat = np.asarray(handle["lat"], dtype=float)
        mode_lon = np.asarray(handle["lon"], dtype=float)
        mode_temp = np.asarray(handle["temp2_anomaly"], dtype=float)
        mode_precip = np.asarray(handle["totprec_anomaly"], dtype=float)
    if not np.array_equal(mode_years, YEARS):
        raise RuntimeError("mode-ra-year-axis-failed")
    if np.any((mode_years >= 1921) & (mode_years <= 1939)):
        raise RuntimeError("locked-period-in-mode-ra-subset")

    hma_mask = np.load(mask_path).astype(bool)
    with h5py.File(public, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((source_years >= 1940) & (source_years <= 2000))
        if not np.array_equal(source_years[indices], YEARS):
            raise RuntimeError("public-observation-year-axis-failed")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        fine_lat = np.asarray(handle["lat"], dtype=float).ravel()
        fine_lon = np.asarray(handle["lon"], dtype=float).ravel()
        if hma_mask.shape != (len(fine_lat), len(fine_lon)):
            raise RuntimeError("hma-mask-shape-failed")
        temperature = {
            name: (YEARS.copy(), clean(handle[variable], selection))
            for name, variable in TEMPERATURE_OBSERVATIONS.items()
        }
        precipitation = {
            name: (YEARS.copy(), clean(handle[variable], selection))
            for name, variable in PRECIPITATION_OBSERVATIONS.items()
        }
    precipitation["UDel"] = annualize_udel(udel, fine_lat, fine_lon)
    precipitation["PREC/L"] = annualize_precl(precl_paths, fine_lat, fine_lon)

    cells = build_coarse_support(fine_lat, fine_lon, hma_mask, mode_lat, mode_lon)
    cell_lat = np.asarray([cell["latitude"] for cell in cells])
    cell_lon = np.asarray([cell["longitude"] for cell in cells])
    support_count = np.asarray([cell["fine_cell_count"] for cell in cells], dtype=float)
    base_weights = np.cos(np.deg2rad(cell_lat)) * support_count
    candidates = {
        "temperature": recenter(extract_coarse(mode_temp, cells), YEARS),
        "precipitation": recenter(extract_coarse(mode_precip, cells), YEARS),
    }
    observations = {"temperature": temperature, "precipitation": precipitation}

    region_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for variable, products in observations.items():
        candidate_full = candidates[variable]
        for product, (product_years, fine_values) in products.items():
            coarse_observation = recenter(aggregate_fine(fine_values, fine_lat, cells), product_years)
            mode_index = np.asarray([np.flatnonzero(YEARS == year)[0] for year in product_years])
            candidate = candidate_full[mode_index]
            for region, (lower, upper) in REGIONS.items():
                selected_cells = (cell_lon >= lower) & (cell_lon < upper)
                metrics = pair_metrics(
                    candidate[:, selected_cells],
                    coarse_observation[:, selected_cells],
                    product_years,
                    base_weights[selected_cells],
                )
                region_rows.append(
                    {
                        "variable": variable,
                        "observation": product,
                        "region": region,
                        **{key: value for key, value in metrics.items() if not key.endswith("_series")},
                    }
                )
                if region == "hma":
                    for index, year in enumerate(product_years):
                        curve_rows.append(
                            {
                                "variable": variable,
                                "observation": product,
                                "year": int(year),
                                "mode_ra": float(metrics["candidate_series"][index]),
                                "observed": float(metrics["observation_series"][index]),
                            }
                        )
                for block_index, (start, end) in enumerate(BLOCKS, start=1):
                    selected_years = (product_years >= start) & (product_years <= end)
                    if int(selected_years.sum()) < 8:
                        continue
                    candidate_series = metrics["candidate_series"][selected_years]
                    observation_series = metrics["observation_series"][selected_years]
                    rmse = float(np.sqrt(np.mean((candidate_series - observation_series) ** 2)))
                    zero_rmse = float(np.sqrt(np.mean(observation_series**2)))
                    block_rows.append(
                        {
                            "variable": variable,
                            "observation": product,
                            "region": region,
                            "block": block_index,
                            "start_year": start,
                            "end_year": end,
                            "year_count": int(selected_years.sum()),
                            "regional_skill": float(1.0 - rmse / zero_rmse),
                        }
                    )

    region_scores = pd.DataFrame(region_rows)
    block_scores = pd.DataFrame(block_rows)
    curves = pd.DataFrame(curve_rows)
    region_scores.to_csv(output / "mode_ra_region_product_scores.csv", index=False)
    block_scores.to_csv(output / "mode_ra_block_scores.csv", index=False)
    curves.to_csv(output / "mode_ra_hma_curves.csv", index=False)
    pd.DataFrame(
        [
            {
                "coarse_cell": index,
                "latitude": cell["latitude"],
                "longitude": cell["longitude"],
                "fine_hma_cell_count": cell["fine_cell_count"],
            }
            for index, cell in enumerate(cells)
        ]
    ).to_csv(output / "mode_ra_hma_coarse_support.csv", index=False)

    variables = {
        variable: summarize_variable(
            region_scores[region_scores.variable == variable],
            block_scores[block_scores.variable == variable],
        )
        for variable in ("temperature", "precipitation")
    }
    decisions = {}
    for variable, result in variables.items():
        if result["passes_spatial_field"]:
            decisions[variable] = "accept_mode_ra_as_external_regional_and_spatial_evidence"
        elif result["passes_regional_curve"]:
            decisions[variable] = "retain_mode_ra_as_external_regional_curve_only"
        else:
            decisions[variable] = "reject_mode_ra_as_validated_external_candidate"
    summary = {
        "status": "complete",
        "years": [1940, 2000],
        "locked_1921_1939_accessed": False,
        "mode_ra_hma_coarse_cell_count": len(cells),
        "independent_recentering_baseline": [1961, 1990],
        "temperature_products": list(temperature),
        "precipitation_products": list(precipitation),
        "variables": variables,
        "decision": decisions,
        "production_release_authorized": False,
        "preinstrumental_mode_ra_validation_authorized": bool(
            any(result["passes_regional_curve"] for result in variables.values())
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for row, variable in enumerate(("temperature", "precipitation")):
        variable_curves = curves[curves.variable == variable]
        mode_curves = []
        for color, product in zip(colors, variable_curves.observation.unique()):
            values = variable_curves[variable_curves.observation == product]
            axes[row, 0].plot(values.year, values.observed, color=color, linewidth=1.0, alpha=0.75, label=product)
            mode_curves.append(values.set_index("year").mode_ra)
        common_mode = pd.concat(mode_curves, axis=1).mean(axis=1)
        axes[row, 0].plot(common_mode.index, common_mode, color="#111111", linewidth=2.2, label="ModE-RA")
        axes[row, 0].axhline(0, color="#888888", linewidth=0.7)
        axes[row, 0].set(
            title=f"HMA {variable} curves",
            ylabel="Anomaly (degC)" if variable == "temperature" else "Anomaly (mm/day)",
        )
        axes[row, 0].legend(frameon=False, ncol=3, fontsize=8)
        hma_scores = region_scores[(region_scores.variable == variable) & (region_scores.region == "hma")]
        x = np.arange(len(hma_scores))
        axes[row, 1].bar(x - 0.18, hma_scores.regional_skill * 100, width=0.36, color="#0072B2", label="Regional")
        axes[row, 1].bar(x + 0.18, hma_scores.spatial_residual_skill * 100, width=0.36, color="#D55E00", label="Spatial residual")
        axes[row, 1].axhline(0, color="#555555", linewidth=0.8)
        axes[row, 1].set_xticks(x, hma_scores.observation, rotation=25, ha="right")
        axes[row, 1].set(title=f"HMA {variable} skill over zero", ylabel="RMSE skill (%)")
        axes[row, 1].legend(frameon=False)
    axes[1, 0].set_xlabel("Year")
    figure.savefig(output / "mode_ra_modern_validation_curves_and_skill.png", dpi=180)
    plt.close(figure)

    lines = ["# ModE-RA Modern Independent Validation", ""]
    for variable, result in variables.items():
        lines.extend(
            [
                f"## {variable.title()}",
                f"- Median/worst regional skill: `{result['median_region_product_skill']:.2%}` / `{result['worst_region_product_skill']:.2%}`.",
                f"- Positive region-product pairs: `{result['positive_region_product_fraction']:.1%}`.",
                f"- Median annual/15-year correlations: `{result['median_annual_correlation']:.3f}` / `{result['median_15year_correlation']:.3f}`.",
                f"- Median/worst HMA spatial-residual skill: `{result['median_hma_spatial_residual_skill']:.2%}` / `{result['worst_hma_spatial_residual_skill']:.2%}`.",
                f"- Regional/spatial pass: `{str(result['passes_regional_curve']).lower()}` / `{str(result['passes_spatial_field']).lower()}`.",
                "",
            ]
        )
    lines.append(f"Decision: `{decisions}`.")
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged},
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
