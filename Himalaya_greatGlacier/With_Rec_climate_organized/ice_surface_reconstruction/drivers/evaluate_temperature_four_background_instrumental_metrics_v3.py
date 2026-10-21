from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_temperature_four_background_instrumental_metrics_v1 as core


core.OUT = (
    Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_four_background_instrumental_metrics_v3"
)


def bootstrap_differences_fast(frame: pd.DataFrame, observation: str) -> pd.DataFrame:
    rng = np.random.default_rng(core.RANDOM_SEED + (0 if observation.startswith("qcf") else 1))
    station_ids = np.array(sorted(frame["station_id"].unique()))
    annual: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    decadal: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for branch, column in core.BRANCHES.items():
        annual[branch] = {}
        decadal[branch] = {}
        for station_id in station_ids:
            group = frame[frame["station_id"].eq(station_id)]
            valid = group[[observation, column]].notna().all(axis=1)
            annual[branch][station_id] = (
                group.loc[valid, observation].to_numpy(float),
                group.loc[valid, column].to_numpy(float),
            )
            blocks = core.decadal_pairs(group, observation, column)
            decadal[branch][station_id] = (
                blocks["observed_degC"].to_numpy(float),
                blocks["predicted_degC"].to_numpy(float),
            )

    distributions = {
        branch: {metric: np.empty(core.BOOTSTRAP_REPLICATES, dtype=float) for metric in (
            "annual_rmse", "annual_r", "decadal_rmse", "decadal_r"
        )}
        for branch in core.BRANCHES
    }
    for replicate_index in range(core.BOOTSTRAP_REPLICATES):
        sampled = rng.choice(station_ids, size=len(station_ids), replace=True)
        calculated: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
        for branch in core.BRANCHES:
            annual_obs = np.concatenate([annual[branch][station_id][0] for station_id in sampled])
            annual_pred = np.concatenate([annual[branch][station_id][1] for station_id in sampled])
            decade_obs = np.concatenate([decadal[branch][station_id][0] for station_id in sampled])
            decade_pred = np.concatenate([decadal[branch][station_id][1] for station_id in sampled])
            calculated[branch] = (
                core.paired_metrics(annual_obs, annual_pred),
                core.paired_metrics(decade_obs, decade_pred),
            )
        current_annual, current_decadal = calculated["current_CESM"]
        for branch, (branch_annual, branch_decadal) in calculated.items():
            distributions[branch]["annual_rmse"][replicate_index] = (
                branch_annual["rmse_degC"] - current_annual["rmse_degC"]
            )
            distributions[branch]["annual_r"][replicate_index] = (
                branch_annual["correlation_r"] - current_annual["correlation_r"]
            )
            distributions[branch]["decadal_rmse"][replicate_index] = (
                branch_decadal["rmse_degC"] - current_decadal["rmse_degC"]
            )
            distributions[branch]["decadal_r"][replicate_index] = (
                branch_decadal["correlation_r"] - current_decadal["correlation_r"]
            )

    rows: list[dict[str, object]] = []
    for branch, metrics in distributions.items():
        for metric, values in metrics.items():
            finite = values[np.isfinite(values)]
            lower, median, upper = np.quantile(finite, [0.025, 0.5, 0.975])
            lower_is_better = metric.endswith("rmse")
            rows.append({
                "observation": observation.removesuffix("_station_degC"),
                "branch": branch,
                "metric_delta_vs_current": metric,
                "replicate_count": int(len(finite)),
                "delta_median": float(median),
                "delta_ci95_lower": float(lower),
                "delta_ci95_upper": float(upper),
                "probability_better_than_current": (
                    float(np.mean(finite < 0)) if lower_is_better else float(np.mean(finite > 0))
                ),
            })
    return pd.DataFrame(rows)


core.bootstrap_differences = bootstrap_differences_fast


if __name__ == "__main__":
    core.main()
