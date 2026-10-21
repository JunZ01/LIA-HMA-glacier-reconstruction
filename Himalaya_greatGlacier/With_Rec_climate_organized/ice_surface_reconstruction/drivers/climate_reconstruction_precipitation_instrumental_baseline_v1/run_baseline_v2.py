"""Run the corrected leakage-aware precipitation baseline diagnosis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_base_module():
    path = Path(__file__).resolve().with_name("run_baseline.py")
    spec = importlib.util.spec_from_file_location("precipitation_baseline_v1_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base-module-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def paired_hma_means(
    reconstruction: np.ndarray,
    observation: np.ndarray,
    latitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(reconstruction) & np.isfinite(observation)
    return (
        base.weighted_mean(np.where(valid, reconstruction, np.nan), latitudes),
        base.weighted_mean(np.where(valid, observation, np.nan), latitudes),
    )


def plot_paired_hma_means(table: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.3), constrained_layout=True)
    colors = {"CRU TS": "#1b9e77", "GPCC-derived": "#7570b3"}
    for axis, period, title in (
        (axes[0], "development_out_of_sample", "Development out-of-sample: 1901-1920"),
        (axes[1], "calibration_consistency", "Calibration consistency only: 1940-2000"),
    ):
        subset = table[table["period"].eq(period)]
        for observation, color in colors.items():
            paired = subset[subset["observation"].eq(observation)]
            axis.plot(
                paired["year"],
                paired["observation_hma_mean_mm_day"],
                color=color,
                lw=1.8,
                label=observation,
            )
            axis.plot(
                paired["year"],
                paired["reconstruction_hma_mean_mm_day"],
                color=color,
                lw=2.2,
                ls="--",
                label=f"production mean on {observation} mask",
            )
        axis.axhline(0.0, color="#555555", lw=0.8)
        axis.set_ylabel("Precipitation anomaly (mm/day)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=2, fontsize=9)
    axes[1].set_xlabel("Year")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> int:
    root = base.project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_instrumental_baseline_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)

    source_hash_before = base.sha256(base.INPUT_NETCDF)
    metric_rows: list[dict[str, float | int | str]] = []
    grid_rows: list[pd.DataFrame] = []
    paired_series_rows: list[dict[str, float | int | str]] = []
    development_maps: dict[str, np.ndarray] = {}

    with h5py.File(base.INPUT_NETCDF, "r") as handle:
        years = np.asarray(handle["time"], dtype=int)
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        required = {
            *base.OBSERVATIONS.values(),
            *base.PRODUCTION.values(),
            *base.COMPARISON_ONLY.values(),
        }
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"missing-required-variables:{missing}")

        variable_audit = {
            variable: {
                "shape": list(handle[variable].shape),
                "units": base.decode_attr(handle[variable].attrs.get("units", "")),
                "reference_period": base.decode_attr(
                    handle[variable].attrs.get("reference_period", "")
                ),
                "data_type": base.decode_attr(
                    handle[variable].attrs.get("data_type", "")
                ),
            }
            for variable in sorted(required)
        }
        if any(item["shape"] != [1001, 43, 78] for item in variable_audit.values()):
            raise ValueError("unexpected-variable-shape")
        if any(item["units"] != "mm/day" for item in variable_audit.values()):
            raise ValueError("unexpected-variable-units")

        disclosed_periods = [
            "gpcc_early_context",
            "development_out_of_sample",
            "calibration_consistency",
        ]
        all_series = {
            **base.OBSERVATIONS,
            **base.PRODUCTION,
            **base.COMPARISON_ONLY,
        }
        for period_name in disclosed_periods:
            start, end = base.PERIODS[period_name]
            indices = np.where((years >= start) & (years <= end))[0]
            if period_name == "gpcc_early_context":
                allowed = {
                    "GPCC-derived": base.OBSERVATIONS["GPCC-derived"],
                    **base.PRODUCTION,
                    **base.COMPARISON_ONLY,
                }
                observations_for_period = ["GPCC-derived"]
            else:
                allowed = all_series
                observations_for_period = list(base.OBSERVATIONS)
            fields = {
                name: base.read_slice(handle, variable, indices)
                for name, variable in allowed.items()
            }

            for reconstruction_name in [*base.PRODUCTION, *base.COMPARISON_ONLY]:
                reconstruction = fields[reconstruction_name]
                for observation_name in observations_for_period:
                    observation = fields[observation_name]
                    metrics, cell_corr, cell_rmse, cell_bias = base.summarize_pair(
                        years[indices], latitudes, reconstruction, observation
                    )
                    metric_rows.append(
                        {
                            "period": period_name,
                            "reconstruction": reconstruction_name,
                            "role": (
                                "production"
                                if reconstruction_name in base.PRODUCTION
                                else "comparison_only"
                            ),
                            "observation": observation_name,
                            **metrics,
                        }
                    )
                    if reconstruction_name == "production ensemble mean":
                        rec_mean, obs_mean = paired_hma_means(
                            reconstruction, observation, latitudes
                        )
                        for year, rec_value, obs_value in zip(
                            years[indices], rec_mean, obs_mean
                        ):
                            paired_series_rows.append(
                                {
                                    "period": period_name,
                                    "year": int(year),
                                    "observation": observation_name,
                                    "reconstruction_hma_mean_mm_day": float(rec_value),
                                    "observation_hma_mean_mm_day": float(obs_value),
                                }
                            )
                    if period_name == "development_out_of_sample":
                        lat_grid, lon_grid = np.meshgrid(
                            latitudes, longitudes, indexing="ij"
                        )
                        grid_rows.append(
                            pd.DataFrame(
                                {
                                    "latitude": lat_grid.reshape(-1),
                                    "longitude": lon_grid.reshape(-1),
                                    "reconstruction": reconstruction_name,
                                    "role": (
                                        "production"
                                        if reconstruction_name in base.PRODUCTION
                                        else "comparison_only"
                                    ),
                                    "observation": observation_name,
                                    "temporal_correlation": cell_corr.reshape(-1),
                                    "rmse_mm_day": cell_rmse.reshape(-1),
                                    "bias_mm_day": cell_bias.reshape(-1),
                                }
                            )
                        )
                        if reconstruction_name == "production ensemble mean":
                            development_maps[
                                f"Production mean vs {observation_name}"
                            ] = cell_corr

        observation_rows = []
        for period_name in [
            "development_out_of_sample",
            "calibration_consistency",
        ]:
            start, end = base.PERIODS[period_name]
            indices = np.where((years >= start) & (years <= end))[0]
            cru = base.read_slice(handle, base.OBSERVATIONS["CRU TS"], indices)
            gpcc = base.read_slice(
                handle, base.OBSERVATIONS["GPCC-derived"], indices
            )
            metrics, _, _, _ = base.summarize_pair(
                years[indices], latitudes, cru, gpcc
            )
            observation_rows.append({"period": period_name, **metrics})

    metrics = pd.DataFrame(metric_rows)
    grid_metrics = pd.concat(grid_rows, ignore_index=True)
    observation_disagreement = pd.DataFrame(observation_rows)
    paired_series = pd.DataFrame(paired_series_rows)
    metrics.to_csv(
        output / "reconstruction_metrics_disclosed_periods.csv", index=False
    )
    grid_metrics.to_csv(
        output / "gridcell_metrics_development_1901_1920.csv", index=False
    )
    observation_disagreement.to_csv(
        output / "cru_gpcc_disagreement.csv", index=False
    )
    paired_series.to_csv(
        output / "paired_hma_mean_series_disclosed_periods.csv", index=False
    )

    locked_manifest = {
        "period": "locked_final_out_of_sample",
        "years": list(base.PERIODS["locked_final_out_of_sample"]),
        "status": "locked_not_evaluated",
        "rule": (
            "No metric, table, ranking, or figure may access 1921-1939 before "
            "a reconstruction method is frozen."
        ),
        "source_sha256": source_hash_before,
    }
    (output / "LOCKED_FINAL_PERIOD.json").write_text(
        json.dumps(locked_manifest, indent=2), encoding="utf-8"
    )

    plot_paired_hma_means(
        paired_series, output / "figure_paired_hma_mean_disclosed_periods.png"
    )
    base.plot_development_maps(
        latitudes,
        longitudes,
        development_maps,
        output / "figure_development_temporal_correlation_maps.png",
    )
    development_metrics = metrics[
        metrics["period"].eq("development_out_of_sample")
    ]
    base.plot_method_rmse(
        development_metrics, output / "figure_development_method_rmse.png"
    )

    production_mean = development_metrics[
        development_metrics["reconstruction"].eq("production ensemble mean")
    ]
    median_reconstruction_rmse = float(
        production_mean["hma_mean_rmse_mm_day"].median()
    )
    development_obs_rmse = float(
        observation_disagreement.loc[
            observation_disagreement["period"].eq(
                "development_out_of_sample"
            ),
            "hma_mean_rmse_mm_day",
        ].iloc[0]
    )

    component_rows = development_metrics[
        development_metrics["reconstruction"].isin(
            [name for name in base.PRODUCTION if name != "production ensemble mean"]
        )
    ]
    component_rmse_spread = float(
        component_rows.groupby("observation")["hma_mean_rmse_mm_day"]
        .std(ddof=0)
        .median()
    )
    if median_reconstruction_rmse > 1.5 * development_obs_rmse:
        observation_diagnosis = (
            "reconstruction_error_exceeds_product_disagreement"
        )
    else:
        observation_diagnosis = "observation_product_uncertainty_is_comparable"
    if component_rmse_spread < 0.25 * median_reconstruction_rmse:
        method_diagnosis = (
            "shared_reconstruction_structure_dominates_method_spread"
        )
    else:
        method_diagnosis = "method_or_proxy_network_spread_is_material"

    source_hash_after = base.sha256(base.INPUT_NETCDF)
    summary = {
        "experiment_id": (
            "climate_reconstruction_precipitation_instrumental_baseline_v1"
        ),
        "driver_entry_point": "run_baseline_v2.py",
        "date": "2026-08-13",
        "status": "diagnostic_complete",
        "no_oggm": True,
        "no_parameter_change": True,
        "input_netcdf": str(base.INPUT_NETCDF),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "psm_calibration_period": [1940, 2000],
        "development_out_of_sample_period": [1901, 1920],
        "locked_final_out_of_sample_period": [1921, 1939],
        "locked_final_period_evaluated": False,
        "calibration_consistency_period": [1940, 2000],
        "gpcc_label_correction": (
            "Stored pr_obs_gpcp_05deg is GPCC-derived."
        ),
        "rec_ref_role": "comparison_only",
        "variable_audit": variable_audit,
        "development_diagnostics": {
            "production_mean_median_hma_rmse_across_cru_gpcc_mm_day": (
                median_reconstruction_rmse
            ),
            "cru_gpcc_hma_rmse_mm_day": development_obs_rmse,
            "six_component_rmse_spread_median_mm_day": component_rmse_spread,
            "observation_diagnosis": observation_diagnosis,
            "method_diagnosis": method_diagnosis,
        },
        "decision_rule": (
            "Use only the disclosed 1901-1920 diagnostics to choose the next "
            "single repair target. Do not open 1921-1939 and do not select a "
            "final method from 1940-2000 consistency."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# Precipitation Instrumental Baseline V1",
        "",
        "Status: **diagnostic complete**",
        "",
        (
            "- Production-mean median HMA RMSE against CRU/GPCC in "
            f"1901-1920: `{median_reconstruction_rmse:.6g} mm/day`"
        ),
        (
            "- CRU-GPCC HMA RMSE in 1901-1920: "
            f"`{development_obs_rmse:.6g} mm/day`"
        ),
        (
            "- Median spread of six component RMSE values: "
            f"`{component_rmse_spread:.6g} mm/day`"
        ),
        f"- Observation diagnosis: `{observation_diagnosis}`",
        f"- Method diagnosis: `{method_diagnosis}`",
        f"- Source hash unchanged: `{source_hash_before == source_hash_after}`",
        "",
        "The 1921-1939 final out-of-sample period remains locked and was not evaluated.",
        "The 1940-2000 results are calibration-period consistency diagnostics only.",
        "",
    ]
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
