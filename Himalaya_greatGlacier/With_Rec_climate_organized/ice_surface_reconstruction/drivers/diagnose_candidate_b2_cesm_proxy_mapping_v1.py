from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import build_temperature_candidate_b2_allowed_input_v1 as source


RESULTS = source.RESULTS
LOOP = source.LOOP
STATE = RESULTS / (
    "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1/"
    "four_prior_state_pools_1500_1848_v1.npz"
)
RAW = LOOP / (
    "temperature_balanced_pooled_candidate_b2_allowed_input_v15/"
    "candidate_b2_allowed_raw_input_v1.npz"
)
OUT = LOOP / "temperature_candidate_b2_cesm_proxy_mapping_diagnostic_v1"


def metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    difference = candidate - reference
    return (
        float(np.sqrt(np.mean(difference * difference))),
        float(np.max(np.abs(difference))),
        float(np.corrcoef(reference, candidate)[0, 1]),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    state = np.load(STATE, allow_pickle=False)
    raw = np.load(RAW, allow_pickle=False)
    lat = np.asarray(state["native_lat"], dtype=float)
    lon = np.asarray(state["native_lon"], dtype=float)
    state_points_lat, state_points_lon = np.meshgrid(lat, lon, indexing="ij")
    state_points = np.column_stack(
        [state_points_lat.reshape(-1, order="F"), state_points_lon.reshape(-1, order="F")]
    )
    tas_state = np.asarray(state["cesm_tas_state"], dtype=float)
    pr_state = np.asarray(state["cesm_pr_state"], dtype=float)
    rows = []
    for network in (6, 9):
        wanted = np.asarray(raw[f"proxy{network}_record_indices_0based"], dtype=int)
        point_indices = np.asarray(raw[f"proxy{network}_unique_point_indices"], dtype=int) + int(raw["state_count"])
        raw_t = np.asarray(raw["cesm_tas_anomaly_1961_1990_degC"], dtype=float)[:349, point_indices]
        raw_p = np.asarray(raw["cesm_pr_mm_day"], dtype=float)[:349, point_indices]
        with h5py.File(source.PROXY[network], "r") as handle:
            refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
            for column, record_index in enumerate(wanted):
                proxy = handle[refs[record_index]]
                location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
                mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")[499:848]
                mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")[499:848]
                distance = (state_points[:, 0] - location[0]) ** 2 + (
                    (state_points[:, 1] - location[1]) * np.cos(np.deg2rad(location[0]))
                ) ** 2
                nearest = int(np.argmin(distance))
                nearest_t = tas_state[:, nearest]
                nearest_p = pr_state[:, nearest]
                bilinear_t = np.empty(349)
                bilinear_p = np.empty(349)
                query = np.asarray([[location[0], np.mod(location[1], 360.0)]])
                for year in range(349):
                    t_field = tas_state[year].reshape(len(lat), len(lon), order="F")
                    p_field = pr_state[year].reshape(len(lat), len(lon), order="F")
                    bilinear_t[year] = RegularGridInterpolator((lat, np.mod(lon, 360.0)), t_field)(query)[0]
                    bilinear_p[year] = RegularGridInterpolator((lat, np.mod(lon, 360.0)), p_field)(query)[0]
                for variable, reference, nearest_values, bilinear_values, raw_values in (
                    ("M_t", mt, nearest_t, bilinear_t, raw_t[:, column]),
                    ("M_p", mp, nearest_p, bilinear_p, raw_p[:, column]),
                ):
                    nearest_metric = metrics(reference, nearest_values)
                    bilinear_metric = metrics(reference, bilinear_values)
                    raw_metric = metrics(reference, raw_values)
                    rows.append(
                        {
                            "network": network,
                            "record_index_0based": int(record_index),
                            "variable": variable,
                            "nearest_rmse": nearest_metric[0],
                            "nearest_max_abs": nearest_metric[1],
                            "nearest_correlation": nearest_metric[2],
                            "state_bilinear_rmse": bilinear_metric[0],
                            "state_bilinear_max_abs": bilinear_metric[1],
                            "state_bilinear_correlation": bilinear_metric[2],
                            "raw_full_grid_bilinear_rmse": raw_metric[0],
                            "raw_full_grid_bilinear_max_abs": raw_metric[1],
                            "raw_full_grid_bilinear_correlation": raw_metric[2],
                        }
                    )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "record_metrics.csv", index=False)
    summary_rows = (
        table.groupby(["network", "variable"])
        .agg(
            record_count=("record_index_0based", "size"),
            median_nearest_rmse=("nearest_rmse", "median"),
            median_nearest_correlation=("nearest_correlation", "median"),
            median_state_bilinear_rmse=("state_bilinear_rmse", "median"),
            median_state_bilinear_correlation=("state_bilinear_correlation", "median"),
            median_raw_bilinear_rmse=("raw_full_grid_bilinear_rmse", "median"),
            median_raw_bilinear_correlation=("raw_full_grid_bilinear_correlation", "median"),
        )
        .reset_index()
    )
    summary_rows.to_csv(OUT / "summary_metrics.csv", index=False)
    best_t = table[table.variable == "M_t"][["nearest_rmse", "state_bilinear_rmse", "raw_full_grid_bilinear_rmse"]].median().idxmin()
    best_p = table[table.variable == "M_p"][["nearest_rmse", "state_bilinear_rmse", "raw_full_grid_bilinear_rmse"]].median().idxmin()
    result = {
        "experiment": OUT.name,
        "status": "complete",
        "record_metric_count": len(table),
        "best_temperature_mapping_by_median_rmse": best_t,
        "best_precipitation_mapping_by_median_rmse": best_p,
        "sealed_values_used": False,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(summary_rows.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
