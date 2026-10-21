from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STAGE_B = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2/stage_b_pseudoproxy_input_v1.npz"
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_preflight_v1"
MODELS = ("cesm", "mode", "mpi", "miroc")
NETWORKS = (6, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lowpass(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        mask = (years >= start) & (years <= end)
        output[mask] = (
            pd.DataFrame(values[mask])
            .rolling(31, center=True, min_periods=16)
            .mean()
            .to_numpy(float)
        )
    return output


def pool_lowpass(values: np.ndarray) -> np.ndarray:
    return (
        pd.DataFrame(values)
        .rolling(31, center=True, min_periods=16)
        .mean()
        .to_numpy(float)
    )


def main(input_path: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    before = {"raw_input": sha256(input_path), "stage_b": sha256(STAGE_B)}
    raw = np.load(input_path, allow_pickle=False)
    stage = np.load(STAGE_B, allow_pickle=False)
    years = np.asarray(raw["years"], dtype=int)
    expected = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)]
    if not np.array_equal(years, expected):
        raise RuntimeError("allowed-year-axis-failed")
    state_count = int(np.asarray(raw["state_count"]))
    if state_count != 333:
        raise RuntimeError("state-count-failed")
    pool = (years >= 1500) & (years <= 1848)
    evaluation = (years >= 1515) & (years <= 1833)
    pool_eval = (np.arange(1500, 1849) >= 1515) & (np.arange(1500, 1849) <= 1833)

    state_means = []
    state_closure = []
    operator_means = {network: [] for network in NETWORKS}
    operator_closure = []
    for model_index, model in enumerate(MODELS):
        tas = np.asarray(raw[f"{model}_tas_anomaly_1961_1990_degC"], dtype=float)
        pr = np.asarray(raw[f"{model}_pr_mm_day"], dtype=float)
        if tas.shape[0] != len(years) or pr.shape != tas.shape:
            raise RuntimeError(f"shape:{model}:{tas.shape}:{pr.shape}")
        if not np.all(np.isfinite(tas)) or not np.all(np.isfinite(pr)):
            raise RuntimeError(f"nonfinite:{model}")
        state_low = lowpass(tas[:, :state_count], years)
        state_means.append(state_low)
        pool_anomaly = tas[pool, :state_count] - tas[pool, :state_count].mean(axis=0)
        rebuilt = pool_lowpass(pool_anomaly)[pool_eval]
        archived = np.asarray(stage["prior_temperature_forced_mean"][model_index], dtype=float)
        state_closure.append(float(np.max(np.abs(rebuilt - archived))))

        for network in NETWORKS:
            b = np.asarray(raw[f"proxy{network}_psm_b0_b1_b2"], dtype=float)
            point_index = np.asarray(raw[f"proxy{network}_unique_point_indices"], dtype=int) + state_count
            operator = (
                b[:, 0][None, :]
                + b[:, 1][None, :] * tas[:, point_index]
                + b[:, 2][None, :] * pr[:, point_index]
            )
            anomaly = operator - operator[pool].mean(axis=0)
            operator_low = lowpass(anomaly, years)
            operator_means[network].append(operator_low)
            rebuilt_operator = pool_lowpass(anomaly[pool])[pool_eval]
            archived_operator = np.asarray(
                stage[f"proxy{network}_prior_operator_forced_mean"][model_index], dtype=float
            )
            operator_closure.append({
                "model": model, "proxy_network": network,
                "maximum_abs_closure": float(np.max(np.abs(rebuilt_operator - archived_operator))),
            })

    state_closure_max = max(state_closure)
    operator_closure_max = max(row["maximum_abs_closure"] for row in operator_closure)
    gates = {
        "exact_allowed_year_axis": True,
        "sealed_year_absent": not np.any((years >= 1921) & (years <= 1939)),
        "all_raw_values_finite": True,
        "state_stage_b_closure_at_most_2e_5": state_closure_max <= 2e-5,
        "operator_stage_b_closure_at_most_2e_5": operator_closure_max <= 2e-5,
    }
    passed = all(gates.values())
    arrays = {
        "years": years,
        "balanced_state_forced_mean": np.mean(np.asarray(state_means), axis=0).astype(np.float32),
        "pooled_state_residual_library": np.concatenate(
            np.asarray(stage["prior_temperature_residual_library"], dtype=float), axis=0
        ).astype(np.float32),
    }
    for network in NETWORKS:
        arrays[f"proxy{network}_balanced_operator_forced_mean"] = np.mean(
            np.asarray(operator_means[network]), axis=0
        ).astype(np.float32)
        arrays[f"proxy{network}_pooled_operator_residual_library"] = np.concatenate(
            np.asarray(stage[f"proxy{network}_prior_operator_residual_library"], dtype=float), axis=0
        ).astype(np.float32)
        arrays[f"proxy{network}_residual_variance"] = stage[f"proxy{network}_residual_variance"]
    if passed:
        np.savez_compressed(OUT / "candidate_b2_stage_c_prior_input_v1.npz", **arrays)
    pd.DataFrame(operator_closure).to_csv(OUT / "operator_stage_b_closure.csv", index=False)
    summary = {
        "experiment": OUT.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "input_preflight_pass": passed, "gates": gates,
        "maximum_state_stage_b_closure": state_closure_max,
        "maximum_operator_stage_b_closure": operator_closure_max,
        "pooled_member_count": 1396, "real_proxy_observations_read": False,
        "stage_c_enkf_run": False, "sealed_values_used": False,
        "production_modified": False, "oggm_run": False,
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
