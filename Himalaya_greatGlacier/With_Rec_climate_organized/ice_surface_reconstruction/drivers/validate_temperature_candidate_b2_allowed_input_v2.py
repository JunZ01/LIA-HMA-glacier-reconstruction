from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import validate_temperature_candidate_b2_allowed_input_v1 as v1


OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_preflight_v2"


def main(input_path: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    before = {"raw_input": v1.sha256(input_path), "stage_b": v1.sha256(v1.STAGE_B)}
    raw = np.load(input_path, allow_pickle=False)
    stage = np.load(v1.STAGE_B, allow_pickle=False)
    years = np.asarray(raw["years"], dtype=int)
    expected = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)]
    if not np.array_equal(years, expected):
        raise RuntimeError("allowed-year-axis-failed")
    state_count = int(np.asarray(raw["state_count"]))
    pool = (years >= 1500) & (years <= 1848)
    evaluation = (years >= 1515) & (years <= 1833)
    pool_eval = (np.arange(1500, 1849) >= 1515) & (np.arange(1500, 1849) <= 1833)
    state_means = []
    state_closure = []
    operator_means = {network: [] for network in v1.NETWORKS}
    operator_closure = []
    for model_index, model in enumerate(v1.MODELS):
        tas = np.asarray(raw[f"{model}_tas_anomaly_1961_1990_degC"], dtype=float)
        pr = np.asarray(raw[f"{model}_pr_mm_day"], dtype=float)
        state_low = v1.lowpass(tas[:, :state_count], years)
        state_means.append(state_low)
        pool_anomaly = tas[pool, :state_count] - tas[pool, :state_count].mean(axis=0)
        rebuilt = v1.pool_lowpass(pool_anomaly)[pool_eval]
        archived = np.asarray(stage["prior_temperature_forced_mean"][model_index], dtype=float)
        state_closure.append(float(np.max(np.abs(rebuilt - archived))))
        for network in v1.NETWORKS:
            if model == "cesm":
                operator = np.asarray(raw[f"proxy{network}_cesm_archived_operator"], dtype=float)
                operator_source = "archived_HXb"
            else:
                b = np.asarray(raw[f"proxy{network}_psm_b0_b1_b2"], dtype=float)
                point_index = np.asarray(raw[f"proxy{network}_unique_point_indices"], dtype=int) + state_count
                operator = (
                    b[:, 0][None, :]
                    + b[:, 1][None, :] * tas[:, point_index]
                    + b[:, 2][None, :] * pr[:, point_index]
                )
                operator_source = "b0_plus_b1_t_plus_b2_p"
            anomaly = operator - operator[pool].mean(axis=0)
            operator_low = v1.lowpass(anomaly, years)
            operator_means[network].append(operator_low)
            rebuilt_operator = v1.pool_lowpass(anomaly[pool])[pool_eval]
            archived_operator = np.asarray(
                stage[f"proxy{network}_prior_operator_forced_mean"][model_index], dtype=float
            )
            operator_closure.append(
                {
                    "model": model,
                    "proxy_network": network,
                    "operator_source": operator_source,
                    "maximum_abs_closure": float(
                        np.max(np.abs(rebuilt_operator - archived_operator))
                    ),
                }
            )
    state_max = max(state_closure)
    operator_max = max(row["maximum_abs_closure"] for row in operator_closure)
    gates = {
        "exact_allowed_year_axis": True,
        "sealed_year_absent": not np.any((years >= 1921) & (years <= 1939)),
        "all_raw_values_finite": all(
            np.all(np.isfinite(raw[name])) for name in raw.files if raw[name].dtype.kind in "fiu"
        ),
        "state_stage_b_closure_at_most_2e_5": state_max <= 2e-5,
        "operator_stage_b_closure_at_most_2e_5": operator_max <= 2e-5,
    }
    passed = all(gates.values())
    arrays = {
        "years": years,
        "balanced_state_forced_mean": np.mean(np.asarray(state_means), axis=0).astype(np.float32),
        "pooled_state_residual_library": np.concatenate(
            np.asarray(stage["prior_temperature_residual_library"], dtype=float), axis=0
        ).astype(np.float32),
    }
    for network in v1.NETWORKS:
        arrays[f"proxy{network}_balanced_operator_forced_mean"] = np.mean(
            np.asarray(operator_means[network]), axis=0
        ).astype(np.float32)
        arrays[f"proxy{network}_pooled_operator_residual_library"] = np.concatenate(
            np.asarray(stage[f"proxy{network}_prior_operator_residual_library"], dtype=float), axis=0
        ).astype(np.float32)
        arrays[f"proxy{network}_residual_variance"] = stage[f"proxy{network}_residual_variance"]
    if passed:
        np.savez_compressed(OUT / "candidate_b2_stage_c_prior_input_v2.npz", **arrays)
    pd.DataFrame(operator_closure).to_csv(OUT / "operator_stage_b_closure.csv", index=False)
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "input_preflight_pass": passed,
        "gates": gates,
        "maximum_state_stage_b_closure": state_max,
        "maximum_operator_stage_b_closure": operator_max,
        "pooled_member_count": 1396,
        "cesm_operator_branch": "archived HXb, exactly matching Stage B",
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "source_hashes": before,
        "next_action": "request_stage_c_authorization" if passed else "reject_or_repair_raw_input",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    main(args.input)
