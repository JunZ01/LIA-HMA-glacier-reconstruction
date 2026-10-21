from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trend(years: np.ndarray, values: np.ndarray) -> float:
    x = years.astype(float)
    x -= np.mean(x)
    return float(100.0 * np.polyfit(x, values, 1)[0])


def _trend_map(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    x = years.astype(float)
    x -= np.mean(x)
    return 100.0 * np.tensordot(x, values, axes=(0, 0)) / np.sum(x**2)


def _recenter(
    years: np.ndarray, values: np.ndarray, reference_years: tuple[int, int]
) -> np.ndarray:
    selected = (years >= reference_years[0]) & (years <= reference_years[1])
    if int(selected.sum()) != 30 or not np.isfinite(values[selected]).all():
        raise ValueError("invalid-reference-period-series")
    return values - float(np.mean(values[selected]))


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def _metrics(
    years: np.ndarray,
    reconstruction: np.ndarray,
    observation: np.ndarray,
    window: int,
) -> dict[str, float | int]:
    difference = reconstruction - observation
    reconstruction_rolling = _rolling(reconstruction, window)
    observation_rolling = _rolling(observation, window)
    valid = np.isfinite(reconstruction_rolling) & np.isfinite(observation_rolling)
    return {
        "annual_year_count": int(len(years)),
        "annual_bias_c": float(np.mean(difference)),
        "annual_mae_c": float(np.mean(np.abs(difference))),
        "annual_rmse_c": float(np.sqrt(np.mean(difference**2))),
        "annual_pearson_r": float(np.corrcoef(reconstruction, observation)[0, 1]),
        "rolling30_window_count": int(valid.sum()),
        "rolling30_bias_c": float(
            np.mean(reconstruction_rolling[valid] - observation_rolling[valid])
        ),
        "rolling30_mae_c": float(
            np.mean(np.abs(reconstruction_rolling[valid] - observation_rolling[valid]))
        ),
        "rolling30_rmse_c": float(
            np.sqrt(
                np.mean(
                    (reconstruction_rolling[valid] - observation_rolling[valid]) ** 2
                )
            )
        ),
        "rolling30_pearson_r": float(
            np.corrcoef(reconstruction_rolling[valid], observation_rolling[valid])[0, 1]
        ),
        "reconstruction_trend_c_per_century": _trend(years, reconstruction),
        "observation_trend_c_per_century": _trend(years, observation),
        "trend_difference_c_per_century": _trend(years, reconstruction)
        - _trend(years, observation),
    }


def _period_summary(
    name: str,
    role: str,
    scope: str,
    years: np.ndarray,
    values: np.ndarray,
    early_years: tuple[int, int],
    late_years: tuple[int, int],
) -> dict[str, float | str]:
    early = (years >= early_years[0]) & (years <= early_years[1])
    late = (years >= late_years[0]) & (years <= late_years[1])
    return {
        "scope": scope,
        "series": name,
        "role": role,
        "mean_1901_1930_c": float(np.mean(values[early])),
        "mean_1961_1990_c": float(np.mean(values[(years >= 1961) & (years <= 1990)])),
        "mean_1971_2000_c": float(np.mean(values[late])),
        "change_1901_1930_to_1971_2000_c": float(
            np.mean(values[late]) - np.mean(values[early])
        ),
        "trend_1901_2000_c_per_century": _trend(years, values),
        "annual_std_c": float(np.std(values)),
    }


def _regional_series(
    field: xr.DataArray,
    time_indices: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(field.isel(time=time_indices).values, dtype=float)
    selected = values[:, mask]
    if not np.isfinite(selected).all():
        raise ValueError(f"regional-nonfinite:{field.name}")
    normalized = weights[mask] / float(np.sum(weights[mask]))
    return np.sum(selected * normalized[None, :], axis=1)


def _plot_scope(
    years: np.ndarray,
    series: dict[str, np.ndarray],
    observation_names: list[str],
    corrected_member_names: list[str],
    title: str,
    output_path: Path,
    window: int,
) -> None:
    observations = np.stack([series[name] for name in observation_names])
    observation_min = np.min(observations, axis=0)
    observation_max = np.max(observations, axis=0)
    observation_mean = series["Observation mean"]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    axes[0].fill_between(
        years,
        observation_min,
        observation_max,
        color="#bdbdbd",
        alpha=0.32,
        label="five-product range",
    )
    axes[0].plot(years, observation_mean, color="black", lw=2.2, label="five-product mean")
    member_colors = plt.get_cmap("tab10")
    for index, name in enumerate(corrected_member_names):
        axes[0].plot(
            years,
            series[name],
            color=member_colors(index),
            lw=1.0,
            alpha=0.62,
            label=name,
        )
    axes[0].plot(
        years,
        series["corrected_ensemble"],
        color="#15803d",
        lw=2.6,
        label="corrected six-member mean",
    )
    axes[0].plot(years, series["rec_ref"], color="#2166ac", lw=2.1, label="rec_ref")
    axes[0].plot(
        years,
        series["public_rec_result"],
        color="#d95f02",
        lw=2.0,
        ls="--",
        label="current public rec_result",
    )
    axes[0].axhline(0.0, color="0.35", lw=0.8)
    axes[0].set_ylabel("Annual anomaly (degC)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.22)
    axes[0].legend(ncol=4, frameon=False, fontsize=8)

    rolling_observations = np.stack([_rolling(series[name], window) for name in observation_names])
    rolling_valid = np.isfinite(rolling_observations).any(axis=0)
    rolling_min = np.full(len(years), np.nan, dtype=float)
    rolling_max = np.full(len(years), np.nan, dtype=float)
    rolling_min[rolling_valid] = np.nanmin(rolling_observations[:, rolling_valid], axis=0)
    rolling_max[rolling_valid] = np.nanmax(rolling_observations[:, rolling_valid], axis=0)
    axes[1].fill_between(
        years,
        rolling_min,
        rolling_max,
        color="#bdbdbd",
        alpha=0.35,
        label="five-product rolling range",
    )
    axes[1].plot(
        years,
        _rolling(observation_mean, window),
        color="black",
        lw=2.2,
        label="five-product rolling mean",
    )
    for index, name in enumerate(corrected_member_names):
        axes[1].plot(
            years,
            _rolling(series[name], window),
            color=member_colors(index),
            lw=1.0,
            alpha=0.62,
        )
    for name, color, style, width in (
        ("corrected_ensemble", "#15803d", "-", 2.6),
        ("rec_ref", "#2166ac", "-", 2.1),
        ("public_rec_result", "#d95f02", "--", 2.0),
    ):
        axes[1].plot(
            years,
            _rolling(series[name], window),
            color=color,
            ls=style,
            lw=width,
            label=f"{name} rolling mean",
        )
    axes[1].axhline(0.0, color="0.35", lw=0.8)
    axes[1].set_ylabel("Trailing 30-year anomaly (degC)")
    axes[1].set_xlabel("Year / rolling-window end year")
    axes[1].grid(alpha=0.22)
    axes[1].legend(ncol=2, frameon=False, fontsize=8)
    axes[1].set_xlim(int(years[0]), int(years[-1]))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_trend_maps(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    corrected: np.ndarray,
    public: np.ndarray,
    target_lon: float,
    target_lat: float,
    output_path: Path,
) -> None:
    corrected = np.where(mask, corrected, np.nan)
    public = np.where(mask, public, np.nan)
    difference = corrected - public
    finite = np.concatenate([corrected[np.isfinite(corrected)], public[np.isfinite(public)]])
    limit = max(0.5, float(np.nanpercentile(np.abs(finite), 98)))
    difference_limit = max(0.25, float(np.nanpercentile(np.abs(difference), 98)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    for axis, field, title, vlim in (
        (axes[0], corrected, "Corrected six-member trend", limit),
        (axes[1], public, "Current public rec_result trend", limit),
        (axes[2], difference, "Corrected minus public", difference_limit),
    ):
        mesh = axis.pcolormesh(
            lon,
            lat,
            field,
            shading="nearest",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
        )
        axis.scatter(
            [target_lon],
            [target_lat],
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
        )
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis, shrink=0.82, label="degC per century")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("experiment_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_coordinate_corrected_05deg_v1"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    corrected_path = output / "coordinate_corrected_rec_result_05deg_1000_2000.nc"
    reference_path = Path(config["reference_netcdf"]).resolve()
    generation = json.loads((output / "regeneration_manifest.json").read_text(encoding="utf-8"))
    if generation["status"] != "pass":
        raise ValueError("regeneration-hard-gates-did-not-pass")
    hashes_before = {
        "corrected_netcdf": _sha256(corrected_path),
        "reference_netcdf": _sha256(reference_path),
    }

    analysis_years = tuple(config["analysis_years"])
    reference_years = tuple(config["reference_years"])
    early_years = tuple(config["early_years"])
    late_years = tuple(config["late_years"])
    target_lat = float(config["target"]["latitude"])
    target_lon = float(config["target"]["longitude"])
    window = int(config["rolling_window_years"])
    member_names = list(config["members"])
    member_variables = {
        name: spec["netcdf"] for name, spec in config["members"].items()
    }
    observation_variables = config["observation_variables"]
    context_variables = config["context_reconstructions"]

    with xr.open_dataset(corrected_path) as corrected, xr.open_dataset(reference_path) as reference:
        years_all = np.asarray(corrected.time.values, dtype=int)
        if not np.array_equal(years_all, np.asarray(reference.time.values, dtype=int)):
            raise ValueError("corrected-reference-year-conflict")
        analysis = (years_all >= analysis_years[0]) & (years_all <= analysis_years[1])
        analysis_indices = np.where(analysis)[0]
        years = years_all[analysis]
        lat = np.asarray(corrected.lat.values, dtype=float)
        lon = np.asarray(corrected.lon.values, dtype=float)
        if not np.array_equal(lat, np.asarray(reference.lat.values, dtype=float)):
            raise ValueError("corrected-reference-latitude-conflict")
        if not np.array_equal(lon, np.asarray(reference.lon.values, dtype=float)):
            raise ValueError("corrected-reference-longitude-conflict")
        lat_index = int(np.where(np.isclose(lat, target_lat))[0][0])
        lon_index = int(np.where(np.isclose(lon, target_lon))[0][0])
        hma_mask = np.asarray(corrected["hma_mask"].values, dtype=bool)
        common_mask = hma_mask.copy()
        for variable in observation_variables.values():
            common_mask &= np.isfinite(
                np.asarray(reference[variable].isel(time=analysis_indices).values, dtype=float)
            ).all(axis=0)
        area_weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))

        scopes: dict[str, dict[str, np.ndarray]] = {"target_3584": {}, "hma_regional": {}}
        roles: dict[str, str] = {}
        for name, variable in member_variables.items():
            target_values = np.asarray(
                corrected[variable].isel(time=analysis_indices, lat=lat_index, lon=lon_index).values,
                dtype=float,
            )
            regional_values = _regional_series(
                corrected[variable], analysis_indices, common_mask, area_weights
            )
            scopes["target_3584"][name] = _recenter(years, target_values, reference_years)
            scopes["hma_regional"][name] = _recenter(years, regional_values, reference_years)
            roles[name] = "corrected_member"

        ensemble_variable = config["ensemble_variable"]
        for scope, values in (
            (
                "target_3584",
                np.asarray(
                    corrected[ensemble_variable]
                    .isel(time=analysis_indices, lat=lat_index, lon=lon_index)
                    .values,
                    dtype=float,
                ),
            ),
            (
                "hma_regional",
                _regional_series(
                    corrected[ensemble_variable], analysis_indices, common_mask, area_weights
                ),
            ),
        ):
            scopes[scope]["corrected_ensemble"] = _recenter(years, values, reference_years)
        roles["corrected_ensemble"] = "corrected_ensemble"

        for name, variable in context_variables.items():
            target_values = np.asarray(
                reference[variable].isel(time=analysis_indices, lat=lat_index, lon=lon_index).values,
                dtype=float,
            )
            regional_values = _regional_series(
                reference[variable], analysis_indices, common_mask, area_weights
            )
            scopes["target_3584"][name] = _recenter(years, target_values, reference_years)
            scopes["hma_regional"][name] = _recenter(years, regional_values, reference_years)
            roles[name] = "context_reconstruction"

        for name, variable in observation_variables.items():
            target_values = np.asarray(
                reference[variable].isel(time=analysis_indices, lat=lat_index, lon=lon_index).values,
                dtype=float,
            )
            regional_values = _regional_series(
                reference[variable], analysis_indices, common_mask, area_weights
            )
            scopes["target_3584"][name] = _recenter(years, target_values, reference_years)
            scopes["hma_regional"][name] = _recenter(years, regional_values, reference_years)
            roles[name] = "observation"

        for scope in scopes:
            observation_stack = np.stack(
                [scopes[scope][name] for name in observation_variables]
            )
            scopes[scope]["Observation mean"] = np.mean(observation_stack, axis=0)
        roles["Observation mean"] = "observation_ensemble"

        corrected_trend_map = _trend_map(
            years,
            np.asarray(corrected[ensemble_variable].isel(time=analysis_indices).values, dtype=float),
        )
        public_trend_map = _trend_map(
            years,
            np.asarray(
                reference[context_variables["public_rec_result"]]
                .isel(time=analysis_indices)
                .values,
                dtype=float,
            ),
        )

    metric_rows: list[dict[str, float | int | str]] = []
    period_rows: list[dict[str, float | str]] = []
    observation_names = list(observation_variables)
    reconstruction_names = [*member_names, "corrected_ensemble", *context_variables]
    for scope, scope_series in scopes.items():
        annual = pd.DataFrame({"year": years})
        for name, values in scope_series.items():
            column = name.lower().replace(" ", "_").replace("-", "_")
            annual[column] = values
            annual[f"{column}_rolling30"] = _rolling(values, window)
            period_rows.append(
                _period_summary(
                    name,
                    roles[name],
                    scope,
                    years,
                    values,
                    early_years,
                    late_years,
                )
            )
        annual.to_csv(output / f"annual_temperature_curves_{scope}.csv", index=False)
        for reconstruction_name in reconstruction_names:
            for observation_name in [*observation_names, "Observation mean"]:
                metric_rows.append(
                    {
                        "scope": scope,
                        "reconstruction": reconstruction_name,
                        "reconstruction_role": roles[reconstruction_name],
                        "reference": observation_name,
                        **_metrics(
                            years,
                            scope_series[reconstruction_name],
                            scope_series[observation_name],
                            window,
                        ),
                    }
                )
        _plot_scope(
            years,
            scope_series,
            observation_names,
            member_names,
            (
                "System 3584: coordinate-corrected members and five instrumental products"
                if scope == "target_3584"
                else "HMA fixed-mask regional mean: corrected members and five products"
            ),
            output / f"temperature_annual_and_rolling30_{scope}.png",
            window,
        )

    metrics = pd.DataFrame(metric_rows)
    periods = pd.DataFrame(period_rows)
    metrics.to_csv(output / "five_product_instrumental_metrics.csv", index=False)
    periods.to_csv(output / "period_and_trend_summary.csv", index=False)
    _plot_trend_maps(
        lon,
        lat,
        hma_mask,
        corrected_trend_map,
        public_trend_map,
        target_lon,
        target_lat,
        output / "corrected_vs_public_trend_maps.png",
    )

    against_mean = metrics[metrics["reference"].eq("Observation mean")]
    ensemble_metrics = against_mean[
        against_mean["reconstruction"].eq("corrected_ensemble")
    ].set_index("scope")
    context_metrics = against_mean[
        against_mean["reconstruction"].isin(context_variables)
    ]
    trend_lookup = periods.set_index(["scope", "series"])[
        "trend_1901_2000_c_per_century"
    ].to_dict()
    hashes_after = {
        "corrected_netcdf": _sha256(corrected_path),
        "reference_netcdf": _sha256(reference_path),
    }
    sources_unchanged = hashes_before == hashes_after
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "complete" if sources_unchanged else "source-purity-failed",
        "scientific_scope": "climate-only; no OGGM simulation and no glacier tuning",
        "coordinate_regeneration": generation,
        "diagnostic_masks": {
            "hma_mask_cell_count": int(hma_mask.sum()),
            "five_product_common_complete_cell_count": int(common_mask.sum()),
            "regional_weighting": "fixed common mask with cosine-latitude area weights",
        },
        "trends_c_per_century": {
            "target_3584_corrected_ensemble": float(
                trend_lookup[("target_3584", "corrected_ensemble")]
            ),
            "target_3584_public_rec_result": float(
                trend_lookup[("target_3584", "public_rec_result")]
            ),
            "target_3584_rec_ref": float(trend_lookup[("target_3584", "rec_ref")]),
            "target_3584_observation_mean": float(
                trend_lookup[("target_3584", "Observation mean")]
            ),
            "hma_corrected_ensemble": float(
                trend_lookup[("hma_regional", "corrected_ensemble")]
            ),
            "hma_public_rec_result": float(
                trend_lookup[("hma_regional", "public_rec_result")]
            ),
            "hma_rec_ref": float(trend_lookup[("hma_regional", "rec_ref")]),
            "hma_observation_mean": float(
                trend_lookup[("hma_regional", "Observation mean")]
            ),
        },
        "corrected_ensemble_metrics_against_five_product_mean": {
            scope: {
                key: float(ensemble_metrics.loc[scope, key])
                for key in (
                    "annual_rmse_c",
                    "annual_pearson_r",
                    "rolling30_rmse_c",
                    "rolling30_pearson_r",
                    "trend_difference_c_per_century",
                )
            }
            for scope in ensemble_metrics.index
        },
        "context_metrics_against_five_product_mean": context_metrics.to_dict(
            orient="records"
        ),
        "source_purity": {
            "unchanged": sources_unchanged,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
        },
        "decision": (
            "The coordinate-corrected file supersedes the current public rec_result "
            "file for any future local or HMA rec_result diagnostic. Instrumental "
            "performance is reported separately and does not authorize proxy, "
            "operator, or glacier-parameter tuning."
        ),
        "files": {
            "corrected_netcdf": str(corrected_path),
            "metrics": "five_product_instrumental_metrics.csv",
            "period_summary": "period_and_trend_summary.csv",
            "target_curves": "temperature_annual_and_rolling30_target_3584.png",
            "regional_curves": "temperature_annual_and_rolling30_hma_regional.png",
            "trend_maps": "corrected_vs_public_trend_maps.png",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    target_metric = ensemble_metrics.loc["target_3584"]
    regional_metric = ensemble_metrics.loc["hma_regional"]
    final_text = f"""# Coordinate-corrected 0.5-degree reconstruction result

## Hard-gate result

The six members were regenerated directly from the verified 2-degree center
coordinates. All grid, target-node, trend-interpolation, member-mean, mask,
and source-purity gates passed. The new NetCDF is independent of the existing
MAT, TIF, and public NetCDF files.

The HMA mask contains {int(hma_mask.sum())} cells. The fixed five-product
instrumental comparison uses {int(common_mask.sum())} cells with complete
1901-2000 coverage. Three retained cells at 24.75 N use the declared 0.25
degree linear boundary extrapolation.

## System 3584

- Corrected six-member trend: {trend_lookup[("target_3584", "corrected_ensemble")]:+.4f} degC/century.
- Current public rec_result trend: {trend_lookup[("target_3584", "public_rec_result")]:+.4f} degC/century.
- rec_ref trend: {trend_lookup[("target_3584", "rec_ref")]:+.4f} degC/century.
- Five-product mean trend: {trend_lookup[("target_3584", "Observation mean")]:+.4f} degC/century.
- Corrected ensemble versus five-product mean: annual RMSE
  {target_metric['annual_rmse_c']:.3f} degC, annual correlation
  {target_metric['annual_pearson_r']:+.3f}, trailing-30-year RMSE
  {target_metric['rolling30_rmse_c']:.3f} degC, and trailing-30-year correlation
  {target_metric['rolling30_pearson_r']:+.3f}.

## HMA regional mean

- Corrected six-member trend: {trend_lookup[("hma_regional", "corrected_ensemble")]:+.4f} degC/century.
- Current public rec_result trend: {trend_lookup[("hma_regional", "public_rec_result")]:+.4f} degC/century.
- rec_ref trend: {trend_lookup[("hma_regional", "rec_ref")]:+.4f} degC/century.
- Five-product mean trend: {trend_lookup[("hma_regional", "Observation mean")]:+.4f} degC/century.
- Corrected ensemble versus five-product mean: annual RMSE
  {regional_metric['annual_rmse_c']:.3f} degC, annual correlation
  {regional_metric['annual_pearson_r']:+.3f}, trailing-30-year RMSE
  {regional_metric['rolling30_rmse_c']:.3f} degC, and trailing-30-year correlation
  {regional_metric['rolling30_pearson_r']:+.3f}.

## Decision

Use the new coordinate-corrected file, not the current public rec_result file,
for any future rec_result climate or glacier diagnostic. These instrumental
metrics characterize the corrected reconstruction; they do not yet authorize
changing proxies, localization, climate offsets, or glacier parameters.

No original source file was modified. Source-purity check passed:
{sources_unchanged}.
"""
    (output / "FINAL_RESULT.md").write_text(final_text, encoding="utf-8")
    patch_text = f"""*** Begin Patch
*** Update File: LIA_PROJECT_FACTS.md
@@
+- The independent coordinate-corrected 0.5-degree reconstruction experiment
+  regenerated all six rec_result members directly from the verified 2-degree
+  MAT cell centers and passed source-node, target-node, trend, mean, mask,
+  coordinate-bound, and source-purity gates. At system 3584 the corrected
+  six-member 1901-2000 trend is
+  {trend_lookup[("target_3584", "corrected_ensemble")]:+.4f} degC/century versus
+  {trend_lookup[("target_3584", "public_rec_result")]:+.4f} in the current
+  public misregistered product. The fixed {int(common_mask.sum())}-cell HMA
+  five-product diagnostic gives a corrected regional trend of
+  {trend_lookup[("hma_regional", "corrected_ensemble")]:+.4f} degC/century.
+  Use the independent corrected NetCDF for future rec_result diagnostics;
+  retain instrumental validation as separate evidence and do not tune glacier
+  curves or physical parameters against it.
*** End Patch
"""
    (output / "lia_facts_update_coordinate_corrected_05deg_v1.patch").write_text(
        patch_text, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if sources_unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
