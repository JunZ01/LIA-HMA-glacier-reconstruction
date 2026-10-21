from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from temperature_candidate_b2_stage_c_core_v1 import posterior_by_active_mask
from preflight_temperature_candidate_c_intercept_synthetic_v1 import estimate_intercepts, hma


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
CONTRACT = LOOP / "temp_candidate_c_intercept_prereg_v1/CANDIDATE_C_CONTRACT.md"
OUTPUT = LOOP / "temp_candidate_c_intercept_synthetic_v2"
EXPECTED_STATIC_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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
    variance = np.asarray(static["proxy6_residual_variance"], dtype=float)
    weights = np.asarray(static["hma_weights"], dtype=float)
    weights /= weights.sum()

    member_index = 217
    proxy_pattern = operator_residual[member_index] - operator_residual.mean(axis=0)
    member_hma = float((state_residual[member_index] - state_residual.mean(axis=0)) @ weights)
    time_signal = (years.astype(float) - 1674.0) / 220.0 + 0.32 * np.sin(2.0 * np.pi * (years - 1500.0) / 47.0)
    baseline = (years >= 1500) & (years <= 1848)
    time_signal -= time_signal[baseline].mean()
    expected_hma_truth = time_signal * member_hma
    offsets = 1.8 * np.sin(np.arange(operator_mean.shape[1]) * 0.73) + 0.7 * np.cos(np.arange(operator_mean.shape[1]) * 0.19)

    row = np.arange(len(years))[:, None]
    column = np.arange(operator_mean.shape[1])[None, :]
    mask = ((row + 3 * column) % 17) != 0
    low_coverage = np.arange(5)
    mask[:, low_coverage] = False
    mask[np.ix_(np.flatnonzero(baseline)[:20], low_coverage)] = True
    signal = time_signal[:, None] * proxy_pattern[None, :]

    offset_obs = operator_mean + signal + offsets[None, :]
    clean_obs = operator_mean + signal
    no_signal_obs = operator_mean + offsets[None, :]
    for array in (offset_obs, clean_obs, no_signal_obs):
        array[~mask] = np.nan

    corrected_offset, centers, counts = estimate_intercepts(years, offset_obs, operator_mean)
    corrected_clean, clean_centers, _ = estimate_intercepts(years, clean_obs, operator_mean)
    corrected_no_signal, _, _ = estimate_intercepts(years, no_signal_obs, operator_mean)
    identified = counts >= 30
    expected_centers = offsets + clean_centers

    posterior_offset, _, mask_count = posterior_by_active_mask(state_mean, state_residual, operator_mean, operator_residual, variance, corrected_offset)
    posterior_clean, _, _ = posterior_by_active_mask(state_mean, state_residual, operator_mean, operator_residual, variance, corrected_clean)
    posterior_no_signal, _, _ = posterior_by_active_mask(state_mean, state_residual, operator_mean, operator_residual, variance, corrected_no_signal)
    posterior_uncorrected, _, _ = posterior_by_active_mask(state_mean, state_residual, operator_mean, operator_residual, variance, offset_obs)

    corrected_hma = hma(posterior_offset - state_mean, weights)
    clean_hma = hma(posterior_clean - state_mean, weights)
    uncorrected_hma = hma(posterior_uncorrected - state_mean, weights)
    offset_invariance = float(np.max(np.abs(posterior_offset - posterior_clean)))
    no_signal_update = float(np.max(np.abs(posterior_no_signal - state_mean)))
    center_closure = float(np.nanmax(np.abs(centers[identified] - expected_centers[identified])))
    truth_correlation = float(np.corrcoef(expected_hma_truth, corrected_hma)[0, 1])
    response_std = float(np.std(corrected_hma))
    leakage_rms = float(np.sqrt(np.mean((uncorrected_hma - clean_hma) ** 2)))

    gates = {
        "static_package_hash_exact": sha256(STATIC) == EXPECTED_STATIC_SHA256,
        "allowed_year_axis_only": not bool(np.any((years >= 1921) & (years <= 1939))),
        "five_low_coverage_records_excluded": bool(np.array_equal(np.flatnonzero(~identified), low_coverage)),
        "offset_invariance_max_le_1e_10": offset_invariance <= 1e-10,
        "no_signal_offset_update_max_le_1e_10": no_signal_update <= 1e-10,
        "known_sampled_center_closure_le_1e_10": center_closure <= 1e-10,
        "known_time_signal_correlation_ge_0p99": truth_correlation >= 0.99,
        "nontrivial_signal_response_std_ge_0p05": response_std >= 0.05,
        "uncorrected_fixture_leakage_rms_ge_0p05": leakage_rms >= 0.05,
        "all_posteriors_finite": bool(np.all(np.isfinite(posterior_offset)) and np.all(np.isfinite(posterior_clean)) and np.all(np.isfinite(posterior_no_signal))),
    }
    passed = all(gates.values())

    fig, axis = plt.subplots(figsize=(12.8, 5.7), constrained_layout=True)
    for start, end in ((1500, 1920), (1940, 2000)):
        period = (years >= start) & (years <= end)
        axis.plot(years[period], expected_hma_truth[period], color="#111111", linewidth=1.3, label="known synthetic truth" if start == 1500 else None)
        axis.plot(years[period], corrected_hma[period], color="#2878B5", linewidth=2.0, label="intercept-corrected update" if start == 1500 else None)
        axis.plot(years[period], uncorrected_hma[period], color="#C33C2E", linewidth=0.9, alpha=0.7, label="uncorrected update" if start == 1500 else None)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Synthetic HMA update", title="Candidate C corrected synthetic intercept preflight")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.savefig(OUTPUT / "candidate_c_synthetic_intercept_preflight_v2.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "synthetic_preflight_pass": passed,
        "gates": gates,
        "metrics": {
            "offset_invariance_max_abs": offset_invariance,
            "no_signal_update_max_abs": no_signal_update,
            "known_sampled_center_closure_max_abs": center_closure,
            "known_time_signal_correlation": truth_correlation,
            "signal_response_std": response_std,
            "uncorrected_offset_leakage_rms": leakage_rms,
            "member_hma_truth_amplitude": member_hma,
            "unique_active_masks": mask_count,
        },
        "v1_failure_class": "fixture-gate-definition-error_not_mechanism_failure",
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "noise-and-intercept-estimation uncertainty sensitivity" if passed else "close intercept mechanism",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate C corrected synthetic intercept preflight\n\n"
        f"Pass: `{passed}`. Offset invariance `{offset_invariance:.3e}`; no-signal update `{no_signal_update:.3e}`; "
        f"known-truth correlation `{truth_correlation:.6f}`; uncorrected leakage RMS `{leakage_rms:.6f}`.\n",
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
