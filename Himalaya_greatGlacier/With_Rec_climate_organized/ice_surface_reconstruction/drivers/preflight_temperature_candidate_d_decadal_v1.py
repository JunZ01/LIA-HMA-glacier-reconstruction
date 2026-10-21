from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decompose_temperature_candidate_b2_stage_c_increment_v1 import hma_gain
from preflight_temperature_candidate_c_intercept_synthetic_v1 import estimate_intercepts


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
CONTRACT = LOOP / "temp_candidate_d_decadal_prereg_v1/CANDIDATE_D_CONTRACT.md"
OUTPUT = LOOP / "temp_candidate_d_decadal_synthetic_v1"
EXPECTED_STATIC_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"
BRANCHES = {"independent": 1.0, "conservative": 0.5, "unchanged": 0.0}
REPLICATES = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def block_definitions(years: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    blocks = []
    for first, last in ((1500, 1919), (1940, 1999)):
        for start in range(first, last + 1, 10):
            indices = np.flatnonzero((years >= start) & (years <= start + 9))
            if len(indices) != 10:
                raise RuntimeError(f"incomplete block {start}")
            blocks.append((start, start + 9, indices))
    return blocks


def block_observations(
    observations: np.ndarray,
    operator_mean: np.ndarray,
    blocks: list[tuple[int, int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.zeros((len(blocks), observations.shape[1]), dtype=int)
    block_obs = np.full(count.shape, np.nan, dtype=float)
    block_operator = np.full(count.shape, np.nan, dtype=float)
    for block_index, (_, _, indices) in enumerate(blocks):
        values = observations[indices]
        active = np.isfinite(values)
        count[block_index] = active.sum(axis=0)
        admitted = count[block_index] >= 6
        block_obs[block_index, admitted] = np.nanmean(values[:, admitted], axis=0)
        for proxy_index in np.flatnonzero(admitted):
            rows = active[:, proxy_index]
            block_operator[block_index, proxy_index] = operator_mean[indices[rows], proxy_index].mean()
    return block_obs, block_operator, count


def prepare_block_gains(
    block_obs: np.ndarray,
    count: np.ndarray,
    state_residual: np.ndarray,
    operator_residual: np.ndarray,
    variance: np.ndarray,
    weights: np.ndarray,
    exponent: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    output = []
    for block_index, row in enumerate(block_obs):
        active = np.isfinite(row)
        block_variance = variance[active] / np.power(count[block_index, active], exponent)
        output.append((active, hma_gain(state_residual, operator_residual[:, active], block_variance, weights)))
    return output


def block_updates(
    block_obs: np.ndarray,
    block_operator: np.ndarray,
    gains: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    result = np.zeros(block_obs.shape[0], dtype=float)
    for index, (active, gain) in enumerate(gains):
        result[index] = (block_obs[index, active] - block_operator[index, active]) @ gain
    return result


def slope(centers: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(centers, values, 1)[0])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    if sha256(STATIC) != EXPECTED_STATIC_SHA256:
        raise RuntimeError("static package hash mismatch")
    static = np.load(STATIC)
    years = np.asarray(static["years"], dtype=int)
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("sealed years present")
    blocks = block_definitions(years)
    centers = np.asarray([(start + end) / 2.0 for start, end, _ in blocks])
    state_residual = np.asarray(static["pooled_state_residual_library"], dtype=float)
    operator_mean = np.asarray(static["proxy6_balanced_operator_forced_mean"], dtype=float)
    operator_residual = np.asarray(static["proxy6_pooled_operator_residual_library"], dtype=float)
    variance = np.asarray(static["proxy6_residual_variance"], dtype=float)
    weights = np.asarray(static["hma_weights"], dtype=float)
    weights /= weights.sum()

    member_index = 217
    proxy_pattern = operator_residual[member_index] - operator_residual.mean(axis=0)
    time_signal = (years.astype(float) - 1674.0) / 220.0 + 0.32 * np.sin(2.0 * np.pi * (years - 1500.0) / 47.0)
    baseline = (years >= 1500) & (years <= 1848)
    time_signal -= time_signal[baseline].mean()
    signal = time_signal[:, None] * proxy_pattern[None, :]
    offsets = 1.8 * np.sin(np.arange(operator_mean.shape[1]) * 0.73) + 0.7 * np.cos(np.arange(operator_mean.shape[1]) * 0.19)
    row = np.arange(len(years))[:, None]
    column = np.arange(operator_mean.shape[1])[None, :]
    mask = ((row + 3 * column) % 17) != 0
    mask[:, :5] = False
    mask[np.ix_(np.flatnonzero(baseline)[:20], np.arange(5))] = True

    clean = operator_mean + signal
    clean[~mask] = np.nan
    corrected_clean, _, _ = estimate_intercepts(years, clean, operator_mean)
    clean_block, clean_operator, block_count = block_observations(corrected_clean, operator_mean, blocks)
    branch_gains = {
        name: prepare_block_gains(clean_block, block_count, state_residual, operator_residual, variance, weights, exponent)
        for name, exponent in BRANCHES.items()
    }
    noise_free = {
        name: block_updates(clean_block, clean_operator, gains)
        for name, gains in branch_gains.items()
    }

    rows = []
    for replicate in range(REPLICATES):
        rng = np.random.default_rng(2026082800 + replicate)
        noise = rng.normal(size=operator_mean.shape) * np.sqrt(variance)[None, :]
        offset_observations = operator_mean + signal + offsets[None, :] + noise
        paired_clean = operator_mean + signal + noise
        no_signal = operator_mean + offsets[None, :] + noise
        for array in (offset_observations, paired_clean, no_signal):
            array[~mask] = np.nan
        corrected_offset, _, _ = estimate_intercepts(years, offset_observations, operator_mean)
        corrected_paired, _, _ = estimate_intercepts(years, paired_clean, operator_mean)
        corrected_no_signal, _, _ = estimate_intercepts(years, no_signal, operator_mean)
        offset_block, offset_operator, _ = block_observations(corrected_offset, operator_mean, blocks)
        paired_block, paired_operator, _ = block_observations(corrected_paired, operator_mean, blocks)
        no_signal_block, no_signal_operator, _ = block_observations(corrected_no_signal, operator_mean, blocks)
        for branch, gains in branch_gains.items():
            update = block_updates(offset_block, offset_operator, gains)
            paired_update = block_updates(paired_block, paired_operator, gains)
            no_signal_update = block_updates(no_signal_block, no_signal_operator, gains)
            expected = noise_free[branch]
            error = update - expected
            rows.append(
                {
                    "branch": branch,
                    "replicate": replicate,
                    "offset_invariance_max_abs_degC": float(np.max(np.abs(update - paired_update))),
                    "correlation_with_noise_free": float(np.corrcoef(update, expected)[0, 1]),
                    "rmse_vs_noise_free_degC": float(np.sqrt(np.mean(error**2))),
                    "no_signal_update_rms_degC": float(np.sqrt(np.mean(no_signal_update**2))),
                    "trend_sign_retained": bool(np.sign(slope(centers, update)) == np.sign(slope(centers, expected))),
                }
            )
    replicates = pd.DataFrame(rows)
    replicates.to_csv(OUTPUT / "decadal_synthetic_replicates.csv", index=False)
    aggregate = replicates.groupby("branch").agg(
        offset_invariance_max_abs_degC=("offset_invariance_max_abs_degC", "max"),
        correlation_median=("correlation_with_noise_free", "median"),
        correlation_p05=("correlation_with_noise_free", lambda x: float(np.quantile(x, 0.05))),
        rmse_median_degC=("rmse_vs_noise_free_degC", "median"),
        rmse_p95_degC=("rmse_vs_noise_free_degC", lambda x: float(np.quantile(x, 0.95))),
        no_signal_rms_median_degC=("no_signal_update_rms_degC", "median"),
        trend_sign_retention_fraction=("trend_sign_retained", "mean"),
    ).reset_index()
    aggregate["noise_free_response_std_degC"] = aggregate["branch"].map({name: float(np.std(values)) for name, values in noise_free.items()})
    aggregate.to_csv(OUTPUT / "decadal_synthetic_summary.csv", index=False)

    primary = aggregate[aggregate["branch"].eq("conservative")].iloc[0]
    gates = {
        "offset_invariance_max_le_1e_10": float(primary["offset_invariance_max_abs_degC"]) <= 1e-10,
        "median_correlation_ge_0p8": float(primary["correlation_median"]) >= 0.8,
        "median_rmse_le_noise_free_std": float(primary["rmse_median_degC"]) <= float(primary["noise_free_response_std_degC"]),
        "trend_sign_retention_ge_0p9": float(primary["trend_sign_retention_fraction"]) >= 0.9,
        "median_no_signal_rms_le_noise_free_std": float(primary["no_signal_rms_median_degC"]) <= float(primary["noise_free_response_std_degC"]),
    }
    passed = all(gates.values())

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    order = list(BRANCHES)
    for axis, column_name, ylabel in (
        (axes[0], "correlation_with_noise_free", "Correlation"),
        (axes[1], "rmse_vs_noise_free_degC", "RMSE (degC)"),
        (axes[2], "no_signal_update_rms_degC", "No-signal RMS (degC)"),
    ):
        data = [replicates.loc[replicates["branch"].eq(branch), column_name].to_numpy() for branch in order]
        axis.boxplot(data, labels=order, showfliers=False)
        axis.tick_params(axis="x", rotation=20)
        axis.set(ylabel=ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Candidate D decadal synthetic noise sensitivity")
    fig.savefig(OUTPUT / "candidate_d_decadal_synthetic_sensitivity.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "synthetic_preflight_pass": passed,
        "primary_branch": "conservative_R_over_sqrt_n",
        "block_count": len(blocks),
        "replicates": REPLICATES,
        "gates": gates,
        "primary_metrics": primary.to_dict(),
        "all_branch_metrics": aggregate.to_dict(orient="records"),
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "prepare Candidate D real-proxy static preflight and authorization boundary" if passed else "advance to temporal state model",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate D decadal synthetic preflight\n\n"
        f"Conservative branch pass: `{passed}`. Median correlation `{primary['correlation_median']:.3f}`, "
        f"median RMSE `{primary['rmse_median_degC']:.3f} degC`, trend-sign retention "
        f"`{primary['trend_sign_retention_fraction']:.1%}`.\n",
        encoding="utf-8",
    )
    manifest = {
        "inputs": {str(STATIC): sha256(STATIC), str(CONTRACT): sha256(CONTRACT)},
        "outputs": {str(path): sha256(path) for path in sorted(OUTPUT.iterdir()) if path.name != "source_manifest.json"},
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
