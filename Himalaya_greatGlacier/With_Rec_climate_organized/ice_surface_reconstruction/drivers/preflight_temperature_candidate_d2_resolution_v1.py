from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decompose_temperature_candidate_b2_stage_c_increment_v1 import hma_gain
from preflight_temperature_candidate_c_intercept_synthetic_v1 import estimate_intercepts
from preflight_temperature_candidate_d_decadal_v1 import block_observations, prepare_block_gains, block_updates, slope


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
CONTRACT = LOOP / "temp_candidate_d2_resolution_prereg_v1/CANDIDATE_D2_CONTRACT.md"
OUTPUT = LOOP / "temp_candidate_d2_resolution_synthetic_v1"
EXPECTED_STATIC_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"
RESOLUTIONS = (10, 20, 30)
REPLICATES = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def make_blocks(years: np.ndarray, length: int) -> list[tuple[int, int, np.ndarray]]:
    output = []
    for first, last in ((1500, 1920), (1940, 2000)):
        start = first
        while start + length - 1 <= last:
            end = start + length - 1
            indices = np.flatnonzero((years >= start) & (years <= end))
            if len(indices) != length:
                raise RuntimeError(f"invalid block {start}-{end}")
            output.append((start, end, indices))
            start += length
    return output


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
    mask[:, :5] = False
    mask[np.ix_(np.flatnonzero(baseline)[:20], np.arange(5))] = True

    clean = operator_mean + signal
    clean[~mask] = np.nan
    corrected_clean, _, _ = estimate_intercepts(years, clean, operator_mean)

    fixtures = {}
    for resolution in RESOLUTIONS:
        blocks = make_blocks(years, resolution)
        centers = np.asarray([(start + end) / 2.0 for start, end, _ in blocks])
        clean_block, clean_operator, count = block_observations(corrected_clean, operator_mean, blocks)
        gains = prepare_block_gains(clean_block, count, state_residual, operator_residual, variance, weights, 0.0)
        noise_free = block_updates(clean_block, clean_operator, gains)
        fixtures[resolution] = (blocks, centers, gains, noise_free)

    rows = []
    for replicate in range(REPLICATES):
        rng = np.random.default_rng(2026082900 + replicate)
        noise = rng.normal(size=operator_mean.shape) * np.sqrt(variance)[None, :]
        offset_observations = operator_mean + signal + offsets[None, :] + noise
        paired_clean = operator_mean + signal + noise
        no_signal = operator_mean + offsets[None, :] + noise
        for array in (offset_observations, paired_clean, no_signal):
            array[~mask] = np.nan
        corrected_offset, _, _ = estimate_intercepts(years, offset_observations, operator_mean)
        corrected_paired, _, _ = estimate_intercepts(years, paired_clean, operator_mean)
        corrected_no_signal, _, _ = estimate_intercepts(years, no_signal, operator_mean)
        for resolution in RESOLUTIONS:
            blocks, centers, gains, expected = fixtures[resolution]
            offset_block, offset_operator, _ = block_observations(corrected_offset, operator_mean, blocks)
            paired_block, paired_operator, _ = block_observations(corrected_paired, operator_mean, blocks)
            no_signal_block, no_signal_operator, _ = block_observations(corrected_no_signal, operator_mean, blocks)
            update = block_updates(offset_block, offset_operator, gains)
            paired_update = block_updates(paired_block, paired_operator, gains)
            no_signal_update = block_updates(no_signal_block, no_signal_operator, gains)
            error = update - expected
            rows.append(
                {
                    "resolution_years": resolution,
                    "replicate": replicate,
                    "offset_invariance_max_abs_degC": float(np.max(np.abs(update - paired_update))),
                    "correlation_with_noise_free": float(np.corrcoef(update, expected)[0, 1]),
                    "rmse_vs_noise_free_degC": float(np.sqrt(np.mean(error**2))),
                    "no_signal_update_rms_degC": float(np.sqrt(np.mean(no_signal_update**2))),
                    "trend_sign_retained": bool(np.sign(slope(centers, update)) == np.sign(slope(centers, expected))),
                }
            )
    replicates = pd.DataFrame(rows)
    replicates.to_csv(OUTPUT / "resolution_ladder_replicates.csv", index=False)
    aggregate = replicates.groupby("resolution_years").agg(
        offset_invariance_max_abs_degC=("offset_invariance_max_abs_degC", "max"),
        correlation_median=("correlation_with_noise_free", "median"),
        correlation_p05=("correlation_with_noise_free", lambda x: float(np.quantile(x, 0.05))),
        rmse_median_degC=("rmse_vs_noise_free_degC", "median"),
        rmse_p95_degC=("rmse_vs_noise_free_degC", lambda x: float(np.quantile(x, 0.95))),
        no_signal_rms_median_degC=("no_signal_update_rms_degC", "median"),
        trend_sign_retention_fraction=("trend_sign_retained", "mean"),
    ).reset_index()
    aggregate["noise_free_response_std_degC"] = aggregate["resolution_years"].map({value: float(np.std(fixtures[value][3])) for value in RESOLUTIONS})
    aggregate["pass"] = (
        (aggregate["offset_invariance_max_abs_degC"] <= 1e-10)
        & (aggregate["correlation_median"] >= 0.8)
        & (aggregate["rmse_median_degC"] <= aggregate["noise_free_response_std_degC"])
        & (aggregate["trend_sign_retention_fraction"] >= 0.9)
        & (aggregate["no_signal_rms_median_degC"] <= aggregate["noise_free_response_std_degC"])
    )
    aggregate.to_csv(OUTPUT / "resolution_ladder_summary.csv", index=False)
    passing = aggregate.loc[aggregate["pass"], "resolution_years"].astype(int).tolist()
    selected = min(passing) if passing else None

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, column_name, ylabel in (
        (axes[0], "correlation_with_noise_free", "Correlation"),
        (axes[1], "rmse_vs_noise_free_degC", "RMSE (degC)"),
        (axes[2], "no_signal_update_rms_degC", "No-signal RMS (degC)"),
    ):
        data = [replicates.loc[replicates["resolution_years"].eq(value), column_name].to_numpy() for value in RESOLUTIONS]
        axis.boxplot(data, labels=[str(value) for value in RESOLUTIONS], showfliers=False)
        axis.set(xlabel="Block years", ylabel=ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Candidate D2 ordered temporal-resolution ladder")
    fig.savefig(OUTPUT / "candidate_d2_resolution_ladder.png", dpi=220)
    plt.close(fig)

    passed = selected is not None
    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "resolution_ladder_pass": passed,
        "tested_resolutions_years": list(RESOLUTIONS),
        "ordered_selection_rule": "shortest passing block",
        "selected_resolution_years": selected,
        "all_metrics": aggregate.to_dict(orient="records"),
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "validate selected-resolution interpolation and endpoint behavior" if passed else "close block-mean mechanism and advance to temporal state model",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate D2 resolution ladder\n\n"
        f"Passing resolutions: `{passing}`. Selected shortest resolution: `{selected}`.\n",
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
