"""Run a leakage-aware instrumental precipitation baseline diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_NETCDF = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)

OBSERVATIONS = {
    "CRU TS": "pr_obs_cru_pre_05deg",
    "GPCC-derived": "pr_obs_gpcp_05deg",
}

PRODUCTION = {
    "production ensemble mean": "pr_ensemble_mean_rec_result",
    "EnKF proxy6": "pr_rec_result_O_6_m_c05d",
    "EnKF proxy9": "pr_rec_result_O_9_m_c05d",
    "AOEnKF proxy6": "pr_rec_result_AO_6_m_c05d",
    "AOEnKF proxy9": "pr_rec_result_AO_9_m_c05d",
    "HGAOEnKF proxy6": "pr_rec_result_HGAO_6_m_c05d",
    "HGAOEnKF proxy9": "pr_rec_result_HGAO_9_m_c05d",
}

COMPARISON_ONLY = {
    "rec_ref ensemble mean": "pr_ensemble_mean_rec_ref",
}

PERIODS = {
    "gpcc_early_context": (1891, 1900),
    "development_out_of_sample": (1901, 1920),
    "locked_final_out_of_sample": (1921, 1939),
    "calibration_consistency": (1940, 2000),
}

FILL_VALUE_LIMIT = -9000.0


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


def decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return decode_attr(value.reshape(-1)[0])
    return str(value)


def read_slice(
    handle: h5py.File,
    variable: str,
    time_indices: np.ndarray,
) -> np.ndarray:
    values = np.asarray(handle[variable][time_indices, :, :], dtype=float)
    values[values <= FILL_VALUE_LIMIT] = np.nan
    return values


def weighted_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights_2d = np.cos(np.deg2rad(latitudes))[:, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights_2d[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights_2d[None, :, :], axis=(1, 2))
    return np.divide(
        numerator,
        denominator,
        out=np.full(field.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    x = left[valid]
    y = right[valid]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def trend_per_century(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    x = years[valid].astype(float)
    y = values[valid]
    return float(100.0 * np.polyfit(x - x.mean(), y, 1)[0])


def cellwise_metrics(
    reconstruction: np.ndarray,
    observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = reconstruction.shape[1:]
    corr = np.full((rows, columns), np.nan, dtype=float)
    rmse = np.full((rows, columns), np.nan, dtype=float)
    bias = np.full((rows, columns), np.nan, dtype=float)
    for row in range(rows):
        for column in range(columns):
            rec = reconstruction[:, row, column]
            obs = observation[:, row, column]
            valid = np.isfinite(rec) & np.isfinite(obs)
            if int(valid.sum()) < 3:
                continue
            difference = rec[valid] - obs[valid]
            rmse[row, column] = float(np.sqrt(np.mean(difference**2)))
            bias[row, column] = float(np.mean(difference))
            corr[row, column] = correlation(rec[valid], obs[valid])
    return corr, rmse, bias


def summarize_pair(
    years: np.ndarray,
    latitudes: np.ndarray,
    reconstruction: np.ndarray,
    observation: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(reconstruction) & np.isfinite(observation)
    difference = np.where(valid, reconstruction - observation, np.nan)
    rec_mean = weighted_mean(np.where(valid, reconstruction, np.nan), latitudes)
    obs_mean = weighted_mean(np.where(valid, observation, np.nan), latitudes)
    mean_difference = rec_mean - obs_mean

    spatial_correlations = []
    spatial_rmses = []
    for index in range(len(years)):
        rec = reconstruction[index].reshape(-1)
        obs = observation[index].reshape(-1)
        pair = np.isfinite(rec) & np.isfinite(obs)
        if int(pair.sum()) < 3:
            continue
        spatial_correlations.append(correlation(rec[pair], obs[pair]))
        spatial_rmses.append(float(np.sqrt(np.mean((rec[pair] - obs[pair]) ** 2))))

    cell_corr, cell_rmse, cell_bias = cellwise_metrics(reconstruction, observation)
    finite_cell_corr = cell_corr[np.isfinite(cell_corr)]
    finite_cell_rmse = cell_rmse[np.isfinite(cell_rmse)]
    finite_cell_bias = cell_bias[np.isfinite(cell_bias)]
    metrics: dict[str, float | int] = {
        "finite_cell_year_count": int(valid.sum()),
        "field_bias_mm_day": float(np.nanmean(difference)),
        "field_mae_mm_day": float(np.nanmean(np.abs(difference))),
        "field_rmse_mm_day": float(np.sqrt(np.nanmean(difference**2))),
        "hma_mean_bias_mm_day": float(np.nanmean(mean_difference)),
        "hma_mean_mae_mm_day": float(np.nanmean(np.abs(mean_difference))),
        "hma_mean_rmse_mm_day": float(np.sqrt(np.nanmean(mean_difference**2))),
        "hma_mean_correlation": correlation(rec_mean, obs_mean),
        "reconstruction_hma_trend_mm_day_per_century": trend_per_century(years, rec_mean),
        "observation_hma_trend_mm_day_per_century": trend_per_century(years, obs_mean),
        "spatial_correlation_median": float(np.nanmedian(spatial_correlations)),
        "spatial_correlation_minimum": float(np.nanmin(spatial_correlations)),
        "spatial_rmse_mean_mm_day": float(np.nanmean(spatial_rmses)),
        "gridcell_temporal_correlation_median": float(np.nanmedian(finite_cell_corr)),
        "gridcell_temporal_correlation_q25": float(np.nanquantile(finite_cell_corr, 0.25)),
        "gridcell_temporal_correlation_q75": float(np.nanquantile(finite_cell_corr, 0.75)),
        "gridcell_positive_temporal_correlation_fraction": float(np.mean(finite_cell_corr > 0)),
        "gridcell_rmse_median_mm_day": float(np.nanmedian(finite_cell_rmse)),
        "gridcell_abs_bias_median_mm_day": float(np.nanmedian(np.abs(finite_cell_bias))),
    }
    return metrics, cell_corr, cell_rmse, cell_bias


def plot_hma_means(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(13.2, 5.6), constrained_layout=True)
    axis.axvspan(1921, 1939, color="#d9d9d9", alpha=0.65, label="Locked final period")
    styles = {
        "CRU TS": ("#1b9e77", 2.0),
        "GPCC-derived": ("#7570b3", 2.0),
        "production ensemble mean": ("#d95f02", 2.4),
        "rec_ref ensemble mean": ("#666666", 1.4),
    }
    for name, (color, width) in styles.items():
        subset = table[table["series"].eq(name)]
        axis.plot(subset["year"], subset["hma_mean_mm_day"], color=color, lw=width, label=name)
    axis.axvline(1940, color="#222222", lw=0.9, ls="--")
    axis.axhline(0.0, color="#555555", lw=0.8)
    axis.set_xlim(1891, 2000)
    axis.set_xlabel("Year")
    axis.set_ylabel("HMA area-weighted precipitation anomaly (mm/day)")
    axis.set_title("Precipitation baseline: disclosed periods only")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3, fontsize=9)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_development_maps(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    maps: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    for axis, (name, values) in zip(axes, maps.items()):
        image = axis.pcolormesh(longitudes, latitudes, values, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        axis.set_title(name)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    figure.colorbar(image, ax=axes, label="1901-1920 temporal correlation", shrink=0.84)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_method_rmse(metrics: pd.DataFrame, output_path: Path) -> None:
    selected = metrics[metrics["reconstruction"].isin(PRODUCTION)].copy()
    pivot = selected.pivot(index="reconstruction", columns="observation", values="hma_mean_rmse_mm_day")
    pivot = pivot.reindex(list(PRODUCTION))
    axis = pivot.plot.bar(figsize=(13.2, 5.8), color=["#1b9e77", "#7570b3"], width=0.78)
    axis.set_ylabel("1901-1920 HMA-mean RMSE (mm/day)")
    axis.set_xlabel("")
    axis.set_title("Development-period error; not final model selection")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_instrumental_baseline_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)

    source_hash_before = sha256(INPUT_NETCDF)
    metric_rows: list[dict[str, float | int | str]] = []
    grid_rows: list[pd.DataFrame] = []
    disclosed_series_rows: list[dict[str, float | int | str]] = []
    development_maps: dict[str, np.ndarray] = {}

    with h5py.File(INPUT_NETCDF, "r") as handle:
        years = np.asarray(handle["time"], dtype=int)
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        required = {*OBSERVATIONS.values(), *PRODUCTION.values(), *COMPARISON_ONLY.values()}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"missing-required-variables:{missing}")

        variable_audit = {
            variable: {
                "shape": list(handle[variable].shape),
                "units": decode_attr(handle[variable].attrs.get("units", "")),
                "reference_period": decode_attr(handle[variable].attrs.get("reference_period", "")),
                "data_type": decode_attr(handle[variable].attrs.get("data_type", "")),
            }
            for variable in sorted(required)
        }
        if any(item["shape"] != [1001, 43, 78] for item in variable_audit.values()):
            raise ValueError("unexpected-variable-shape")
        if any(item["units"] != "mm/day" for item in variable_audit.values()):
            raise ValueError("unexpected-variable-units")

        disclosed_periods = ["gpcc_early_context", "development_out_of_sample", "calibration_consistency"]
        all_series = {**OBSERVATIONS, **PRODUCTION, **COMPARISON_ONLY}
        for period_name in disclosed_periods:
            start, end = PERIODS[period_name]
            indices = np.where((years >= start) & (years <= end))[0]
            if period_name == "gpcc_early_context":
                allowed = {"GPCC-derived": OBSERVATIONS["GPCC-derived"], **PRODUCTION, **COMPARISON_ONLY}
            else:
                allowed = all_series
            fields = {name: read_slice(handle, variable, indices) for name, variable in allowed.items()}
            for name, field in fields.items():
                for year, value in zip(years[indices], weighted_mean(field, latitudes)):
                    disclosed_series_rows.append(
                        {"period": period_name, "year": int(year), "series": name, "hma_mean_mm_day": float(value)}
                    )

            observations_for_period = ["GPCC-derived"] if period_name == "gpcc_early_context" else list(OBSERVATIONS)
            for reconstruction_name in [*PRODUCTION, *COMPARISON_ONLY]:
                reconstruction = fields[reconstruction_name]
                for observation_name in observations_for_period:
                    metrics, cell_corr, cell_rmse, cell_bias = summarize_pair(
                        years[indices], latitudes, reconstruction, fields[observation_name]
                    )
                    metric_rows.append(
                        {
                            "period": period_name,
                            "reconstruction": reconstruction_name,
                            "role": "production" if reconstruction_name in PRODUCTION else "comparison_only",
                            "observation": observation_name,
                            **metrics,
                        }
                    )
                    if period_name == "development_out_of_sample":
                        lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
                        frame = pd.DataFrame(
                            {
                                "latitude": lat_grid.reshape(-1),
                                "longitude": lon_grid.reshape(-1),
                                "reconstruction": reconstruction_name,
                                "role": "production" if reconstruction_name in PRODUCTION else "comparison_only",
                                "observation": observation_name,
                                "temporal_correlation": cell_corr.reshape(-1),
                                "rmse_mm_day": cell_rmse.reshape(-1),
                                "bias_mm_day": cell_bias.reshape(-1),
                            }
                        )
                        grid_rows.append(frame)
                        if reconstruction_name == "production ensemble mean":
                            development_maps[f"Production mean vs {observation_name}"] = cell_corr

        observation_rows = []
        for period_name in ["development_out_of_sample", "calibration_consistency"]:
            start, end = PERIODS[period_name]
            indices = np.where((years >= start) & (years <= end))[0]
            cru = read_slice(handle, OBSERVATIONS["CRU TS"], indices)
            gpcc = read_slice(handle, OBSERVATIONS["GPCC-derived"], indices)
            metrics, _, _, _ = summarize_pair(years[indices], latitudes, cru, gpcc)
            observation_rows.append({"period": period_name, **metrics})

    metrics = pd.DataFrame(metric_rows)
    grid_metrics = pd.concat(grid_rows, ignore_index=True)
    observation_disagreement = pd.DataFrame(observation_rows)
    disclosed_series = pd.DataFrame(disclosed_series_rows)
    metrics.to_csv(output / "reconstruction_metrics_disclosed_periods.csv", index=False)
    grid_metrics.to_csv(output / "gridcell_metrics_development_1901_1920.csv", index=False)
    observation_disagreement.to_csv(output / "cru_gpcc_disagreement.csv", index=False)
    disclosed_series.to_csv(output / "hma_mean_series_disclosed_periods.csv", index=False)

    locked_manifest = {
        "period": "locked_final_out_of_sample",
        "years": list(PERIODS["locked_final_out_of_sample"]),
        "status": "locked_not_evaluated",
        "rule": "No metric, table, ranking, or figure may access 1921-1939 before a reconstruction method is frozen.",
        "source_sha256": source_hash_before,
    }
    (output / "LOCKED_FINAL_PERIOD.json").write_text(json.dumps(locked_manifest, indent=2), encoding="utf-8")

    plot_hma_means(disclosed_series, output / "figure_hma_mean_disclosed_periods.png")
    plot_development_maps(latitudes, longitudes, development_maps, output / "figure_development_temporal_correlation_maps.png")
    development_metrics = metrics[metrics["period"].eq("development_out_of_sample")]
    plot_method_rmse(development_metrics, output / "figure_development_method_rmse.png")

    production_mean = development_metrics[
        development_metrics["reconstruction"].eq("production ensemble mean")
    ]
    median_reconstruction_rmse = float(production_mean["hma_mean_rmse_mm_day"].median())
    development_obs_rmse = float(
        observation_disagreement.loc[
            observation_disagreement["period"].eq("development_out_of_sample"),
            "hma_mean_rmse_mm_day",
        ].iloc[0]
    )
    production_components = disclosed_series[
        disclosed_series["period"].eq("development_out_of_sample")
        & disclosed_series["series"].isin([name for name in PRODUCTION if name != "production ensemble mean"])
    ].pivot(index="year", columns="series", values="hma_mean_mm_day")
    mean_component_spread = float(production_components.std(axis=1, ddof=0).mean())

    if median_reconstruction_rmse > 1.5 * development_obs_rmse:
        observation_diagnosis = "reconstruction_error_exceeds_product_disagreement"
    else:
        observation_diagnosis = "observation_product_uncertainty_is_comparable"
    if mean_component_spread < 0.5 * median_reconstruction_rmse:
        method_diagnosis = "shared_reconstruction_structure_dominates_method_spread"
    else:
        method_diagnosis = "method_or_proxy_network_spread_is_material"

    source_hash_after = sha256(INPUT_NETCDF)
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_instrumental_baseline_v1",
        "date": "2026-08-13",
        "status": "diagnostic_complete",
        "no_oggm": True,
        "no_parameter_change": True,
        "input_netcdf": str(INPUT_NETCDF),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "psm_calibration_period": [1940, 2000],
        "development_out_of_sample_period": [1901, 1920],
        "locked_final_out_of_sample_period": [1921, 1939],
        "locked_final_period_evaluated": False,
        "calibration_consistency_period": [1940, 2000],
        "gpcc_label_correction": "Stored pr_obs_gpcp_05deg is GPCC-derived.",
        "rec_ref_role": "comparison_only",
        "variable_audit": variable_audit,
        "development_diagnostics": {
            "production_mean_median_hma_rmse_across_cru_gpcc_mm_day": median_reconstruction_rmse,
            "cru_gpcc_hma_rmse_mm_day": development_obs_rmse,
            "mean_six_component_hma_spread_mm_day": mean_component_spread,
            "observation_diagnosis": observation_diagnosis,
            "method_diagnosis": method_diagnosis,
        },
        "decision_rule": (
            "Use only the disclosed 1901-1920 diagnostics to choose the next single repair target. "
            "Do not open 1921-1939 and do not select a final method from 1940-2000 consistency."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Precipitation Instrumental Baseline V1",
        "",
        "Status: **diagnostic complete**",
        "",
        f"- Production-mean median HMA RMSE against CRU/GPCC in 1901-1920: `{median_reconstruction_rmse:.6g} mm/day`",
        f"- CRU-GPCC HMA RMSE in 1901-1920: `{development_obs_rmse:.6g} mm/day`",
        f"- Mean six-component HMA spread in 1901-1920: `{mean_component_spread:.6g} mm/day`",
        f"- Observation diagnosis: `{observation_diagnosis}`",
        f"- Method diagnosis: `{method_diagnosis}`",
        f"- Source hash unchanged: `{source_hash_before == source_hash_after}`",
        "",
        "The 1921-1939 final out-of-sample period remains locked and was not evaluated.",
        "The 1940-2000 results are calibration-period consistency diagnostics only.",
        "",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
