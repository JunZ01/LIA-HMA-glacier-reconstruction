"""Compare 3584 temperature reconstructions with instrumental products only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


TARGET_LATITUDE = 33.75
TARGET_LONGITUDE = 75.75
ANALYSIS_YEARS = (1901, 2000)
REFERENCE_YEARS = (1961, 1990)
ROLLING_WINDOW_YEARS = 30

OBSERVATION_VARIABLES = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU TS": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}
RECONSTRUCTION_VARIABLES = {
    "rec_ref": "tas_ensemble_mean_rec_ref",
    "rec_result": "tas_ensemble_mean_rec_result",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coordinate_index(values: np.ndarray, target: float, name: str) -> int:
    index = int(np.argmin(np.abs(np.asarray(values, dtype=float) - target)))
    selected = float(values[index])
    if not math.isclose(selected, target, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"missing-exact-{name}-coordinate:{target}:nearest={selected}")
    return index


def _extract_and_recenter(
    dataset: xr.Dataset,
    variable: str,
    lat_index: int,
    lon_index: int,
) -> tuple[pd.Series, dict[str, float | int | str]]:
    years = np.asarray(dataset.time.values, dtype=int)
    values = np.asarray(
        dataset[variable].isel(lat=lat_index, lon=lon_index).values,
        dtype=float,
    )
    reference = (years >= REFERENCE_YEARS[0]) & (years <= REFERENCE_YEARS[1])
    analysis = (years >= ANALYSIS_YEARS[0]) & (years <= ANALYSIS_YEARS[1])
    if int(reference.sum()) != 30:
        raise ValueError(f"invalid-reference-year-count:{variable}:{int(reference.sum())}")
    if int(analysis.sum()) != 100:
        raise ValueError(f"invalid-analysis-year-count:{variable}:{int(analysis.sum())}")
    if not np.isfinite(values[reference]).all():
        raise ValueError(f"non-finite-reference-values:{variable}")
    if not np.isfinite(values[analysis]).all():
        raise ValueError(f"non-finite-analysis-values:{variable}")
    reference_mean = float(np.mean(values[reference]))
    series = pd.Series(
        values[analysis] - reference_mean,
        index=years[analysis],
        name=variable,
        dtype=float,
    )
    audit = {
        "variable": variable,
        "stored_units": str(dataset[variable].attrs.get("units", "")),
        "stored_reference_period": str(
            dataset[variable].attrs.get("reference_period", "")
        ),
        "stored_1961_1990_mean_c": reference_mean,
        "recentered_1961_1990_mean_c": float(series.loc[1961:1990].mean()),
        "analysis_finite_year_count": int(series.notna().sum()),
        "analysis_start_year": int(series.index.min()),
        "analysis_end_year": int(series.index.max()),
    }
    return series, audit


def _trend_per_century(series: pd.Series) -> float:
    clean = series.dropna()
    x = clean.index.to_numpy(dtype=float)
    y = clean.to_numpy(dtype=float)
    slope = float(np.polyfit(x - float(x.mean()), y, 1)[0])
    return 100.0 * slope


def _comparison_metrics(
    reconstruction: pd.Series,
    observation: pd.Series,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [reconstruction.rename("reconstruction"), observation.rename("observation")],
        axis=1,
    ).dropna()
    difference = aligned["reconstruction"] - aligned["observation"]
    reconstruction_rolling = aligned["reconstruction"].rolling(
        ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
    ).mean()
    observation_rolling = aligned["observation"].rolling(
        ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
    ).mean()
    rolling = pd.concat(
        [
            reconstruction_rolling.rename("reconstruction"),
            observation_rolling.rename("observation"),
        ],
        axis=1,
    ).dropna()
    rolling_difference = rolling["reconstruction"] - rolling["observation"]
    reconstruction_trend = _trend_per_century(aligned["reconstruction"])
    observation_trend = _trend_per_century(aligned["observation"])
    return {
        "annual_year_count": int(len(aligned)),
        "annual_bias_c": float(difference.mean()),
        "annual_mae_c": float(difference.abs().mean()),
        "annual_rmse_c": float(np.sqrt(np.mean(difference**2))),
        "annual_pearson_r": float(
            aligned["reconstruction"].corr(aligned["observation"])
        ),
        "rolling30_window_count": int(len(rolling)),
        "rolling30_bias_c": float(rolling_difference.mean()),
        "rolling30_mae_c": float(rolling_difference.abs().mean()),
        "rolling30_rmse_c": float(np.sqrt(np.mean(rolling_difference**2))),
        "rolling30_pearson_r": float(
            rolling["reconstruction"].corr(rolling["observation"])
        ),
        "reconstruction_trend_c_per_century": reconstruction_trend,
        "observation_trend_c_per_century": observation_trend,
        "trend_difference_c_per_century": reconstruction_trend - observation_trend,
    }


def _period_summary(name: str, series: pd.Series, role: str) -> dict[str, float | str]:
    return {
        "series": name,
        "role": role,
        "mean_1901_1930_c": float(series.loc[1901:1930].mean()),
        "mean_1931_1960_c": float(series.loc[1931:1960].mean()),
        "mean_1961_1990_c": float(series.loc[1961:1990].mean()),
        "mean_1971_2000_c": float(series.loc[1971:2000].mean()),
        "change_1901_1930_to_1971_2000_c": float(
            series.loc[1971:2000].mean() - series.loc[1901:1930].mean()
        ),
        "trend_1901_2000_c_per_century": _trend_per_century(series.loc[1901:2000]),
        "annual_std_c": float(series.loc[1901:2000].std(ddof=0)),
    }


def _plot_curves(
    series: dict[str, pd.Series],
    observation_names: list[str],
    output_path: Path,
) -> None:
    observation_frame = pd.DataFrame(
        {name: series[name] for name in observation_names}
    )
    observation_mean = series["Observation mean"]
    observation_min = observation_frame.min(axis=1)
    observation_max = observation_frame.max(axis=1)
    rolling_observations = observation_frame.rolling(
        ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
    ).mean()
    rolling_mean = observation_mean.rolling(
        ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
    ).mean()

    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].fill_between(
        observation_frame.index,
        observation_min.to_numpy(dtype=float),
        observation_max.to_numpy(dtype=float),
        color="#bdbdbd",
        alpha=0.28,
        label="Instrumental product range",
    )
    for index, name in enumerate(observation_names):
        axes[0].plot(
            series[name].index,
            series[name].values,
            color=colors(index),
            linewidth=0.9,
            alpha=0.55,
            label=name,
        )
    axes[0].plot(
        observation_mean.index,
        observation_mean.values,
        color="black",
        linewidth=2.0,
        label="Instrumental mean",
    )
    axes[0].plot(
        series["rec_ref"].index,
        series["rec_ref"].values,
        color="#2166ac",
        linewidth=2.1,
        label="rec_ref",
    )
    axes[0].plot(
        series["rec_result"].index,
        series["rec_result"].values,
        color="#d95f02",
        linewidth=2.1,
        label="rec_result",
    )
    axes[0].axhline(0.0, color="#4d4d4d", linewidth=0.8)
    axes[0].set_ylabel("Annual temperature anomaly (degC)")
    axes[0].set_title(
        "System 3584 pixel (33.75N, 75.75E): annual instrumental comparison"
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(ncol=3, frameon=False, fontsize=9)

    rolling_min = rolling_observations.min(axis=1)
    rolling_max = rolling_observations.max(axis=1)
    axes[1].fill_between(
        rolling_observations.index,
        rolling_min.to_numpy(dtype=float),
        rolling_max.to_numpy(dtype=float),
        color="#bdbdbd",
        alpha=0.35,
        label="Instrumental rolling range",
    )
    for index, name in enumerate(observation_names):
        rolling = series[name].rolling(
            ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
        ).mean()
        axes[1].plot(
            rolling.index,
            rolling.values,
            color=colors(index),
            linewidth=1.0,
            alpha=0.55,
        )
    axes[1].plot(
        rolling_mean.index,
        rolling_mean.values,
        color="black",
        linewidth=2.2,
        label="Instrumental rolling mean",
    )
    for name, color in (("rec_ref", "#2166ac"), ("rec_result", "#d95f02")):
        rolling = series[name].rolling(
            ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
        ).mean()
        axes[1].plot(
            rolling.index,
            rolling.values,
            color=color,
            linewidth=2.3,
            label=f"{name} rolling mean",
        )
    axes[1].axhline(0.0, color="#4d4d4d", linewidth=0.8)
    axes[1].set_ylabel("Trailing 30-year anomaly (degC)")
    axes[1].set_xlabel("Year / rolling-window end year")
    axes[1].set_title("Low-frequency comparison after a common 1961-1990 recentering")
    axes[1].grid(alpha=0.22)
    axes[1].legend(ncol=2, frameon=False, fontsize=9)
    axes[1].set_xlim(1901, 2000)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-netcdf",
        type=Path,
        default=Path(
            r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "validation_data/results/climate_validation_v0/"
            "3584_rec_ref_vs_rec_result_instrumental_v1"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = args.input_netcdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_variables = {
        **OBSERVATION_VARIABLES,
        **RECONSTRUCTION_VARIABLES,
    }
    all_series: dict[str, pd.Series] = {}
    baseline_audits: list[dict[str, float | int | str]] = []
    with xr.open_dataset(input_path) as dataset:
        lat_index = _coordinate_index(dataset.lat.values, TARGET_LATITUDE, "latitude")
        lon_index = _coordinate_index(dataset.lon.values, TARGET_LONGITUDE, "longitude")
        for name, variable in expected_variables.items():
            if variable not in dataset:
                raise ValueError(f"missing-required-variable:{variable}")
            extracted, audit = _extract_and_recenter(
                dataset,
                variable,
                lat_index,
                lon_index,
            )
            extracted.name = name
            all_series[name] = extracted
            baseline_audits.append({"series": name, **audit})
        source_attributes = {key: str(value) for key, value in dataset.attrs.items()}
        selected_latitude = float(dataset.lat.values[lat_index])
        selected_longitude = float(dataset.lon.values[lon_index])

    observation_names = list(OBSERVATION_VARIABLES)
    observation_frame = pd.DataFrame(
        {name: all_series[name] for name in observation_names}
    )
    all_series["Observation mean"] = observation_frame.mean(axis=1)
    all_series["Observation minimum"] = observation_frame.min(axis=1)
    all_series["Observation maximum"] = observation_frame.max(axis=1)

    annual_table = pd.DataFrame({"year": all_series["rec_ref"].index})
    for name, series in all_series.items():
        column = name.lower().replace(" ", "_")
        annual_table[column] = series.to_numpy(dtype=float)
        annual_table[f"{column}_rolling30"] = series.rolling(
            ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
        ).mean().to_numpy(dtype=float)
    annual_table.to_csv(output_dir / "annual_temperature_curves_3584.csv", index=False)
    pd.DataFrame(baseline_audits).to_csv(
        output_dir / "reference_period_recenter_audit.csv", index=False
    )

    metric_rows: list[dict[str, float | int | str]] = []
    comparison_targets = [*observation_names, "Observation mean"]
    for reconstruction_name in RECONSTRUCTION_VARIABLES:
        for observation_name in comparison_targets:
            metric_rows.append(
                {
                    "reconstruction": reconstruction_name,
                    "reference": observation_name,
                    **_comparison_metrics(
                        all_series[reconstruction_name],
                        all_series[observation_name],
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        output_dir / "reconstruction_vs_instrumental_metrics.csv", index=False
    )

    observation_pair_rows: list[dict[str, float | int | str]] = []
    for left, right in itertools.combinations(observation_names, 2):
        observation_pair_rows.append(
            {
                "product_a": left,
                "product_b": right,
                **_comparison_metrics(all_series[left], all_series[right]),
            }
        )
    observation_pairs = pd.DataFrame(observation_pair_rows)
    observation_pairs.to_csv(
        output_dir / "instrumental_product_pairwise_metrics.csv", index=False
    )

    period_rows = [
        _period_summary(name, all_series[name], "observation")
        for name in observation_names
    ]
    period_rows.append(
        _period_summary(
            "Observation mean", all_series["Observation mean"], "observation_ensemble"
        )
    )
    period_rows.extend(
        _period_summary(name, all_series[name], "reconstruction")
        for name in RECONSTRUCTION_VARIABLES
    )
    period_summary = pd.DataFrame(period_rows)
    period_summary.to_csv(output_dir / "period_and_trend_summary.csv", index=False)

    observation_min = all_series["Observation minimum"]
    observation_max = all_series["Observation maximum"]
    rolling_observation_min = observation_frame.rolling(
        ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
    ).mean().min(axis=1)
    rolling_observation_max = observation_frame.rolling(
        ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
    ).mean().max(axis=1)
    envelope_rows = []
    for reconstruction_name in RECONSTRUCTION_VARIABLES:
        reconstruction = all_series[reconstruction_name]
        annual_inside = (reconstruction >= observation_min) & (
            reconstruction <= observation_max
        )
        rolling = reconstruction.rolling(
            ROLLING_WINDOW_YEARS, min_periods=ROLLING_WINDOW_YEARS
        ).mean()
        valid_rolling = rolling.notna() & rolling_observation_min.notna()
        rolling_inside = (rolling >= rolling_observation_min) & (
            rolling <= rolling_observation_max
        )
        envelope_rows.append(
            {
                "reconstruction": reconstruction_name,
                "annual_within_instrumental_range_fraction": float(annual_inside.mean()),
                "rolling30_within_instrumental_range_fraction": float(
                    rolling_inside[valid_rolling].mean()
                ),
            }
        )
    envelope_table = pd.DataFrame(envelope_rows)
    envelope_table.to_csv(
        output_dir / "instrumental_envelope_coverage.csv", index=False
    )

    _plot_curves(
        all_series,
        observation_names,
        output_dir / "temperature_annual_and_rolling30_3584.png",
    )

    mean_metrics = metrics[metrics["reference"].eq("Observation mean")].set_index(
        "reconstruction"
    )
    rec_ref_wins = int(
        sum(
            float(
                metrics[
                    metrics["reconstruction"].eq("rec_ref")
                    & metrics["reference"].eq(name)
                ]["annual_rmse_c"].iloc[0]
            )
            < float(
                metrics[
                    metrics["reconstruction"].eq("rec_result")
                    & metrics["reference"].eq(name)
                ]["annual_rmse_c"].iloc[0]
            )
            for name in observation_names
        )
    )
    summary = {
        "audit_type": "read-only instrumental-temperature comparison; no OGGM run",
        "input_netcdf": str(input_path),
        "input_sha256": _sha256(input_path),
        "source_attributes": source_attributes,
        "target": {
            "lia_unique_id": 3584,
            "latitude": selected_latitude,
            "longitude": selected_longitude,
        },
        "analysis_years": list(ANALYSIS_YEARS),
        "common_reference_years": list(REFERENCE_YEARS),
        "recenter_rule": (
            "subtract each product's own finite 1961-1990 mean at the target pixel"
        ),
        "observation_products": observation_names,
        "observation_product_count": len(observation_names),
        "reference_period_audit": baseline_audits,
        "metrics_against_observation_mean": mean_metrics.reset_index().to_dict(
            orient="records"
        ),
        "annual_rmse_wins_against_individual_observations": {
            "rec_ref": rec_ref_wins,
            "rec_result": len(observation_names) - rec_ref_wins,
        },
        "instrumental_pairwise_context": {
            "annual_rmse_c_min": float(observation_pairs["annual_rmse_c"].min()),
            "annual_rmse_c_median": float(observation_pairs["annual_rmse_c"].median()),
            "annual_rmse_c_max": float(observation_pairs["annual_rmse_c"].max()),
            "rolling30_rmse_c_min": float(
                observation_pairs["rolling30_rmse_c"].min()
            ),
            "rolling30_rmse_c_median": float(
                observation_pairs["rolling30_rmse_c"].median()
            ),
            "rolling30_rmse_c_max": float(
                observation_pairs["rolling30_rmse_c"].max()
            ),
        },
        "instrumental_envelope_coverage": envelope_table.to_dict(orient="records"),
        "files": {
            "annual_curves": "annual_temperature_curves_3584.csv",
            "reference_period_audit": "reference_period_recenter_audit.csv",
            "reconstruction_metrics": "reconstruction_vs_instrumental_metrics.csv",
            "instrumental_pairwise_metrics": "instrumental_product_pairwise_metrics.csv",
            "period_and_trend_summary": "period_and_trend_summary.csv",
            "instrumental_envelope_coverage": "instrumental_envelope_coverage.csv",
            "curve_figure": "temperature_annual_and_rolling30_3584.png",
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
