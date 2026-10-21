from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preflight_temperature_candidate_c_intercept_synthetic_v1 import estimate_intercepts
from preflight_temperature_candidate_d_decadal_v1 import block_observations, prepare_block_gains, block_updates
from preflight_temperature_candidate_d2_resolution_v1 import make_blocks
from temperature_candidate_b2_stage_c_core_v1 import posterior_by_active_mask


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
CONTRACT = LOOP / "temp_candidate_d2_resolution_prereg_v1/INTERPOLATION_CONTRACT.md"
OUTPUT = LOOP / "temp_candidate_d2_interpolation_v1"
EXPECTED_STATIC_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def interpolate_segments(
    years: np.ndarray,
    blocks: list[tuple[int, int, np.ndarray]],
    updates: np.ndarray,
) -> np.ndarray:
    output = np.full(len(years), np.nan, dtype=float)
    centers = np.asarray([(start + end) / 2.0 for start, end, _ in blocks])
    for start, end in ((1500, 1920), (1940, 2000)):
        annual = (years >= start) & (years <= end)
        nodes = (centers >= start) & (centers <= end)
        output[annual] = np.interp(years[annual].astype(float), centers[nodes], updates[nodes])
    return output


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    if sha256(STATIC) != EXPECTED_STATIC_SHA256:
        raise RuntimeError("static package hash mismatch")
    static = np.load(STATIC)
    years = np.asarray(static["years"], dtype=int)
    state_mean = np.asarray(static["balanced_state_forced_mean"], dtype=float)
    state_residual = np.asarray(static["pooled_state_residual_library"], dtype=float)
    operator_mean = np.asarray(static["proxy6_balanced_operator_forced_mean"], dtype=float)
    operator_residual = np.asarray(static["proxy6_pooled_operator_residual_library"], dtype=float)
    variance = np.asarray(static["proxy6_residual_variance"], dtype=float)
    weights = np.asarray(static["hma_weights"], dtype=float)
    weights /= weights.sum()
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("sealed years present")

    member_index = 217
    proxy_pattern = operator_residual[member_index] - operator_residual.mean(axis=0)
    time_signal = (years.astype(float) - 1674.0) / 220.0 + 0.32 * np.sin(2.0 * np.pi * (years - 1500.0) / 47.0)
    baseline = (years >= 1500) & (years <= 1848)
    time_signal -= time_signal[baseline].mean()
    signal = time_signal[:, None] * proxy_pattern[None, :]
    row = np.arange(len(years))[:, None]
    column = np.arange(operator_mean.shape[1])[None, :]
    mask = ((row + 3 * column) % 17) != 0
    mask[:, :5] = False
    mask[np.ix_(np.flatnonzero(baseline)[:20], np.arange(5))] = True
    clean = operator_mean + signal
    clean[~mask] = np.nan
    corrected, _, _ = estimate_intercepts(years, clean, operator_mean)

    annual_posterior, _, _ = posterior_by_active_mask(
        state_mean, state_residual, operator_mean, operator_residual, variance, corrected
    )
    annual_update = (annual_posterior - state_mean) @ weights

    blocks = make_blocks(years, 20)
    centers = np.asarray([(start + end) / 2.0 for start, end, _ in blocks])
    block_obs, block_operator, count = block_observations(corrected, operator_mean, blocks)
    gains = prepare_block_gains(block_obs, count, state_residual, operator_residual, variance, weights, 0.0)
    block_update = block_updates(block_obs, block_operator, gains)
    interpolated = interpolate_segments(years, blocks, block_update)
    center_closure = float(np.max(np.abs(np.interp(centers, centers, block_update) - block_update)))

    overshoot = 0.0
    for start, end in ((1500, 1920), (1940, 2000)):
        annual = (years >= start) & (years <= end)
        nodes = (centers >= start) & (centers <= end)
        low = float(block_update[nodes].min())
        high = float(block_update[nodes].max())
        overshoot = max(overshoot, float(np.maximum(low - interpolated[annual], 0).max()), float(np.maximum(interpolated[annual] - high, 0).max()))

    correlation = float(np.corrcoef(interpolated, annual_update)[0, 1])
    annual_slope = float(np.polyfit(years.astype(float), annual_update, 1)[0])
    interpolated_slope = float(np.polyfit(years.astype(float), interpolated, 1)[0])
    background = np.asarray(static["balanced_state_forced_mean"], dtype=float) @ weights
    candidate = background + interpolated
    own_baseline = candidate - candidate[(years >= 1961) & (years <= 1990)].mean()
    baseline_mean = float(own_baseline[(years >= 1961) & (years <= 1990)].mean())

    gates = {
        "block_center_closure_le_1e_12": center_closure <= 1e-12,
        "all_allowed_years_finite": bool(np.all(np.isfinite(interpolated))),
        "no_interpolation_overshoot_le_1e_12": overshoot <= 1e-12,
        "correlation_with_noise_free_annual_response_ge_0p8": correlation >= 0.8,
        "annual_response_trend_sign_retained": bool(np.sign(annual_slope) == np.sign(interpolated_slope)),
        "own_1961_1990_baseline_mean_le_1e_12": abs(baseline_mean) <= 1e-12,
    }
    passed = all(gates.values())

    fig, axis = plt.subplots(figsize=(12.8, 5.7), constrained_layout=True)
    for start, end in ((1500, 1920), (1940, 2000)):
        period = (years >= start) & (years <= end)
        nodes = (centers >= start) & (centers <= end)
        axis.plot(years[period], annual_update[period], color="#999999", linewidth=0.9, label="noise-free annual update" if start == 1500 else None)
        axis.plot(years[period], interpolated[period], color="#C33C2E", linewidth=2.2, label="20-year interpolated update" if start == 1500 else None)
        axis.scatter(centers[nodes], block_update[nodes], color="#111111", s=16, zorder=3, label="20-year block centers" if start == 1500 else None)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Synthetic HMA update", title="Candidate D2 selected 20-year interpolation")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.savefig(OUTPUT / "candidate_d2_interpolation_validation.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "interpolation_validation_pass": passed,
        "selected_resolution_years": 20,
        "gates": gates,
        "metrics": {
            "block_center_closure_max_abs": center_closure,
            "interpolation_overshoot_max_abs": overshoot,
            "correlation_with_noise_free_annual_response": correlation,
            "annual_response_slope": annual_slope,
            "interpolated_response_slope": interpolated_slope,
            "own_baseline_mean": baseline_mean,
        },
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "prepare Candidate D2 real-proxy static package and authorization boundary" if passed else "close block-mean mechanism",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate D2 interpolation validation\n\n"
        f"Pass: `{passed}`. Annual-response correlation `{correlation:.3f}`; block-center closure "
        f"`{center_closure:.3e}`; overshoot `{overshoot:.3e}`.\n",
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
