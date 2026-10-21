from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from temperature_candidate_b2_stage_c_core_v1 import posterior_by_active_mask


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
CONTRACT = LOOP / "temp_candidate_c_intercept_prereg_v1/CANDIDATE_C_CONTRACT.md"
OUTPUT = LOOP / "temp_candidate_c_intercept_synthetic_v1"
EXPECTED_STATIC_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def estimate_intercepts(
    years: np.ndarray,
    observations: np.ndarray,
    operator_mean: np.ndarray,
    minimum_count: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = (years >= 1500) & (years <= 1848)
    innovation = observations[baseline] - operator_mean[baseline]
    counts = np.isfinite(innovation).sum(axis=0)
    centers = np.full(observations.shape[1], np.nan, dtype=float)
    identified = counts >= minimum_count
    centers[identified] = np.nanmean(innovation[:, identified], axis=0)
    corrected = observations - centers[None, :]
    corrected[:, ~identified] = np.nan
    return corrected, centers, counts


def hma(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=float) @ weights


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

    state_mean = np.asarray(static["balanced_state_forced_mean"], dtype=float)
    state_residual = np.asarray(static["pooled_state_residual_library"], dtype=float)
    operator_mean = np.asarray(static["proxy6_balanced_operator_forced_mean"], dtype=float)
    operator_residual = np.asarray(static["proxy6_pooled_operator_residual_library"], dtype=float)
    residual_variance = np.asarray(static["proxy6_residual_variance"], dtype=float)
    weights = np.asarray(static["hma_weights"], dtype=float)
    weights /= weights.sum()

    member_index = 217
    proxy_pattern = operator_residual[member_index] - operator_residual.mean(axis=0)
    time_signal = (years.astype(float) - 1674.0) / 220.0 + 0.32 * np.sin(2.0 * np.pi * (years - 1500.0) / 47.0)
    baseline = (years >= 1500) & (years <= 1848)
    time_signal -= time_signal[baseline].mean()
    offsets = 1.8 * np.sin(np.arange(operator_mean.shape[1]) * 0.73) + 0.7 * np.cos(np.arange(operator_mean.shape[1]) * 0.19)

    mask = np.ones(operator_mean.shape, dtype=bool)
    row = np.arange(len(years))[:, None]
    column = np.arange(operator_mean.shape[1])[None, :]
    mask &= ((row + 3 * column) % 17) != 0
    low_coverage = np.arange(5)
    mask[:, low_coverage] = False
    baseline_rows = np.flatnonzero(baseline)[:20]
    mask[np.ix_(baseline_rows, low_coverage)] = True

    signal = time_signal[:, None] * proxy_pattern[None, :]
    observations_offset = operator_mean + signal + offsets[None, :]
    observations_clean = operator_mean + signal
    observations_no_signal = operator_mean + offsets[None, :]
    for array in (observations_offset, observations_clean, observations_no_signal):
        array[~mask] = np.nan

    corrected_offset, estimated_offsets, counts = estimate_intercepts(years, observations_offset, operator_mean)
    corrected_clean, _, _ = estimate_intercepts(years, observations_clean, operator_mean)
    corrected_no_signal, _, _ = estimate_intercepts(years, observations_no_signal, operator_mean)

    posterior_offset, _, masks_offset = posterior_by_active_mask(
        state_mean, state_residual, operator_mean, operator_residual,
        residual_variance, corrected_offset,
    )
    posterior_clean, _, masks_clean = posterior_by_active_mask(
        state_mean, state_residual, operator_mean, operator_residual,
        residual_variance, corrected_clean,
    )
    posterior_no_signal, _, masks_no_signal = posterior_by_active_mask(
        state_mean, state_residual, operator_mean, operator_residual,
        residual_variance, corrected_no_signal,
    )
    posterior_uncorrected, _, masks_uncorrected = posterior_by_active_mask(
        state_mean, state_residual, operator_mean, operator_residual,
        residual_variance, observations_offset,
    )

    offset_invariance = float(np.max(np.abs(posterior_offset - posterior_clean)))
    no_signal_update = float(np.max(np.abs(posterior_no_signal - state_mean)))
    corrected_hma_increment = hma(posterior_offset - state_mean, weights)
    clean_hma_increment = hma(posterior_clean - state_mean, weights)
    uncorrected_hma_increment = hma(posterior_uncorrected - state_mean, weights)
    signal_correlation = float(np.corrcoef(time_signal, corrected_hma_increment)[0, 1])
    uncorrected_offset_leakage_rms = float(np.sqrt(np.mean((uncorrected_hma_increment - clean_hma_increment) ** 2)))
    estimated_error = estimated_offsets - offsets
    identified = counts >= 30
    intercept_error = float(np.nanmax(np.abs(estimated_error[identified])))

    gates = {
        "static_package_hash_exact": sha256(STATIC) == EXPECTED_STATIC_SHA256,
        "allowed_year_axis_only": not bool(np.any((years >= 1921) & (years <= 1939))),
        "five_low_coverage_records_excluded": bool(np.array_equal(np.flatnonzero(~identified), low_coverage)),
        "offset_invariance_max_le_1e_10": offset_invariance <= 1e-10,
        "no_signal_offset_update_max_le_1e_10": no_signal_update <= 1e-10,
        "time_signal_correlation_ge_0p999": signal_correlation >= 0.999,
        "uncorrected_fixture_leakage_rms_ge_0p05": uncorrected_offset_leakage_rms >= 0.05,
        "intercept_recovery_max_le_1e_10": intercept_error <= 1e-10,
        "all_posteriors_finite": bool(
            np.all(np.isfinite(posterior_offset))
            and np.all(np.isfinite(posterior_clean))
            and np.all(np.isfinite(posterior_no_signal))
            and np.all(np.isfinite(posterior_uncorrected))
        ),
    }
    passed = all(gates.values())

    fig, axis = plt.subplots(figsize=(12.8, 5.7), constrained_layout=True)
    for start, end in ((1500, 1920), (1940, 2000)):
        period = (years >= start) & (years <= end)
        axis.plot(years[period], corrected_hma_increment[period], color="#2878B5", linewidth=2.0, label="intercept-corrected" if start == 1500 else None)
        axis.plot(years[period], uncorrected_hma_increment[period], color="#C33C2E", linewidth=1.0, alpha=0.8, label="uncorrected offset" if start == 1500 else None)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Synthetic HMA update", title="Candidate C synthetic proxy-intercept stress test")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.savefig(OUTPUT / "candidate_c_synthetic_intercept_preflight.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "synthetic_preflight_pass": passed,
        "gates": gates,
        "metrics": {
            "offset_invariance_max_abs": offset_invariance,
            "no_signal_update_max_abs": no_signal_update,
            "time_signal_correlation": signal_correlation,
            "uncorrected_offset_leakage_rms": uncorrected_offset_leakage_rms,
            "intercept_recovery_max_abs": intercept_error,
            "unique_masks": {
                "offset": masks_offset,
                "clean": masks_clean,
                "no_signal": masks_no_signal,
                "uncorrected": masks_uncorrected,
            },
        },
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "noise-and-intercept-estimation uncertainty sensitivity" if passed else "close intercept mechanism",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate C synthetic intercept preflight\n\n"
        f"Pass: `{passed}`.\n\n"
        f"Offset invariance maximum: `{offset_invariance:.3e}`.\n\n"
        f"No-signal offset update maximum: `{no_signal_update:.3e}`.\n\n"
        f"Synthetic time-signal correlation: `{signal_correlation:.6f}`.\n\n"
        f"Uncorrected offset leakage RMS: `{uncorrected_offset_leakage_rms:.6f}`.\n",
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
