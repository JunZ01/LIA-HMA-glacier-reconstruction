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
CONTRACT = LOOP / "temp_candidate_c_intercept_prereg_v1/NOISE_SENSITIVITY_CONTRACT.md"
OUTPUT = LOOP / "temp_candidate_c_intercept_noise_v1"
EXPECTED_STATIC_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"
MULTIPLIERS = (0.5, 1.0, 2.0)
REPLICATES = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def prepare_gains(
    observations: np.ndarray,
    state_residual: np.ndarray,
    operator_residual: np.ndarray,
    variance: np.ndarray,
    weights: np.ndarray,
) -> dict[bytes, tuple[np.ndarray, np.ndarray]]:
    gains: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for row in observations:
        active = np.isfinite(row)
        key = np.packbits(active, bitorder="little").tobytes()
        if key not in gains and active.any():
            gains[key] = (
                active,
                hma_gain(state_residual, operator_residual[:, active], variance[active], weights),
            )
    return gains


def hma_update(
    observations: np.ndarray,
    operator_mean: np.ndarray,
    gains: dict[bytes, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    result = np.zeros(observations.shape[0], dtype=float)
    for index, row in enumerate(observations):
        active = np.isfinite(row)
        if not active.any():
            continue
        key = np.packbits(active, bitorder="little").tobytes()
        _, gain = gains[key]
        result[index] = (row[active] - operator_mean[index, active]) @ gain
    return result


def slope(years: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(years.astype(float), values, 1)[0])


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
    low_coverage = np.arange(5)
    mask[:, low_coverage] = False
    mask[np.ix_(np.flatnonzero(baseline)[:20], low_coverage)] = True

    clean_observations = operator_mean + signal
    clean_observations[~mask] = np.nan
    corrected_clean, _, _ = estimate_intercepts(years, clean_observations, operator_mean)
    gains = prepare_gains(corrected_clean, state_residual, operator_residual, variance, weights)
    noise_free_update = hma_update(corrected_clean, operator_mean, gains)
    noise_free_std = float(np.std(noise_free_update))
    noise_free_slope = slope(years, noise_free_update)

    rows = []
    for multiplier in MULTIPLIERS:
        for replicate in range(REPLICATES):
            rng = np.random.default_rng(20260828 + int(multiplier * 1000) * 100 + replicate)
            noise = rng.normal(size=operator_mean.shape) * np.sqrt(variance)[None, :] * multiplier
            offset_observations = operator_mean + signal + offsets[None, :] + noise
            paired_clean_observations = operator_mean + signal + noise
            no_signal_observations = operator_mean + offsets[None, :] + noise
            for array in (offset_observations, paired_clean_observations, no_signal_observations):
                array[~mask] = np.nan
            corrected_offset, _, _ = estimate_intercepts(years, offset_observations, operator_mean)
            corrected_paired_clean, _, _ = estimate_intercepts(years, paired_clean_observations, operator_mean)
            corrected_no_signal, _, _ = estimate_intercepts(years, no_signal_observations, operator_mean)
            offset_update = hma_update(corrected_offset, operator_mean, gains)
            paired_clean_update = hma_update(corrected_paired_clean, operator_mean, gains)
            no_signal_update = hma_update(corrected_no_signal, operator_mean, gains)
            error = offset_update - noise_free_update
            correlation = float(np.corrcoef(offset_update, noise_free_update)[0, 1])
            rows.append(
                {
                    "noise_multiplier": multiplier,
                    "replicate": replicate,
                    "offset_invariance_max_abs_degC": float(np.max(np.abs(offset_update - paired_clean_update))),
                    "correlation_with_noise_free": correlation,
                    "rmse_vs_noise_free_degC": float(np.sqrt(np.mean(error**2))),
                    "no_signal_update_rms_degC": float(np.sqrt(np.mean(no_signal_update**2))),
                    "trend_sign_retained": bool(np.sign(slope(years, offset_update)) == np.sign(noise_free_slope)),
                }
            )
    replicate_table = pd.DataFrame(rows)
    replicate_table.to_csv(OUTPUT / "noise_sensitivity_replicates.csv", index=False)
    aggregate = replicate_table.groupby("noise_multiplier").agg(
        offset_invariance_max_abs_degC=("offset_invariance_max_abs_degC", "max"),
        correlation_median=("correlation_with_noise_free", "median"),
        correlation_p05=("correlation_with_noise_free", lambda x: float(np.quantile(x, 0.05))),
        rmse_median_degC=("rmse_vs_noise_free_degC", "median"),
        rmse_p95_degC=("rmse_vs_noise_free_degC", lambda x: float(np.quantile(x, 0.95))),
        no_signal_rms_median_degC=("no_signal_update_rms_degC", "median"),
        trend_sign_retention_fraction=("trend_sign_retained", "mean"),
    ).reset_index()
    aggregate.to_csv(OUTPUT / "noise_sensitivity_summary.csv", index=False)

    primary = aggregate[aggregate["noise_multiplier"].eq(1.0)].iloc[0]
    gates_result = {
        "offset_invariance_max_le_1e_10": float(primary["offset_invariance_max_abs_degC"]) <= 1e-10,
        "median_correlation_ge_0p8": float(primary["correlation_median"]) >= 0.8,
        "median_rmse_le_noise_free_std": float(primary["rmse_median_degC"]) <= noise_free_std,
        "trend_sign_retention_ge_0p9": float(primary["trend_sign_retention_fraction"]) >= 0.9,
        "median_no_signal_rms_le_noise_free_std": float(primary["no_signal_rms_median_degC"]) <= noise_free_std,
    }
    passed = all(gates_result.values())

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, column_name, ylabel in (
        (axes[0], "correlation_with_noise_free", "Correlation"),
        (axes[1], "rmse_vs_noise_free_degC", "RMSE (degC)"),
        (axes[2], "no_signal_update_rms_degC", "No-signal RMS (degC)"),
    ):
        data = [replicate_table.loc[replicate_table["noise_multiplier"].eq(value), column_name].to_numpy() for value in MULTIPLIERS]
        axis.boxplot(data, tick_labels=[str(value) for value in MULTIPLIERS], showfliers=False)
        axis.set(xlabel="Noise multiplier", ylabel=ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Candidate C intercept uncertainty sensitivity")
    fig.savefig(OUTPUT / "candidate_c_intercept_noise_sensitivity.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "noise_sensitivity_pass": passed,
        "primary_noise_multiplier": 1.0,
        "replicates_per_multiplier": REPLICATES,
        "noise_free_hma_response_std_degC": noise_free_std,
        "gates": gates_result,
        "primary_metrics": primary.to_dict(),
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "prepare real-proxy static preflight and authorization boundary" if passed else "close intercept mechanism",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate C intercept noise sensitivity\n\n"
        f"Pass: `{passed}`. Primary median correlation `{primary['correlation_median']:.3f}`, "
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
