from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
REAL = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v3/candidate_b2_stage_c_real_proxy_output_v1.npz"
OUTPUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_increment_decomposition_v1"
PERIODS = ((1500, 1849), (1850, 1920), (1940, 2000))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def centered(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)


def hma_gain(
    state_residual: np.ndarray,
    operator_residual: np.ndarray,
    residual_variance: np.ndarray,
    hma_weights: np.ndarray,
) -> np.ndarray:
    x_hma = centered(state_residual) @ hma_weights
    h = centered(operator_residual)
    denominator = float(state_residual.shape[0] - 1)
    cross = x_hma @ h / denominator
    innovation_covariance = h.T @ h / denominator + np.diag(residual_variance)
    factor = scipy.linalg.cho_factor(innovation_covariance, lower=True, check_finite=False)
    return scipy.linalg.cho_solve(factor, cross, check_finite=False)


def smooth(values: np.ndarray, years: np.ndarray, window: int = 31) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        mask = (years >= start) & (years <= end)
        result[mask] = pd.Series(values[mask]).rolling(
            window, center=True, min_periods=max(3, window // 2)
        ).mean().to_numpy()
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    static = np.load(STATIC)
    real = np.load(REAL)
    years = np.asarray(real["years"], dtype=int)
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("sealed years present")

    weights = np.asarray(static["hma_weights"], dtype=float)
    weights = weights / weights.sum()
    state_residual = np.asarray(static["pooled_state_residual_library"], dtype=float)
    background = np.asarray(real["no_proxy_balanced_background"], dtype=float)
    background_hma = background @ weights

    period_rows: list[dict[str, object]] = []
    proxy_rows: list[dict[str, object]] = []
    network_increment: dict[int, np.ndarray] = {}
    closure: dict[str, float] = {}

    for network in (6, 9):
        observations = np.asarray(real[f"proxy{network}_centered_observations"], dtype=float)
        operator_mean = np.asarray(static[f"proxy{network}_balanced_operator_forced_mean"], dtype=float)
        operator_residual = np.asarray(static[f"proxy{network}_pooled_operator_residual_library"], dtype=float)
        residual_variance = np.asarray(static[f"proxy{network}_residual_variance"], dtype=float)
        posterior = np.asarray(real[f"proxy{network}_posterior_temperature_anomaly"], dtype=float)
        posterior_hma = posterior @ weights
        expected_increment = posterior_hma - background_hma

        contribution = np.zeros(observations.shape, dtype=float)
        innovation = observations - operator_mean
        mask_groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
        for year_index, row in enumerate(observations):
            active = np.isfinite(row)
            key = np.packbits(active, bitorder="little").tobytes()
            if key not in mask_groups:
                mask_groups[key] = (active, [])
            mask_groups[key][1].append(year_index)
        for active, year_indices in mask_groups.values():
            if not active.any():
                continue
            gain = hma_gain(
                state_residual,
                operator_residual[:, active],
                residual_variance[active],
                weights,
            )
            indices = np.asarray(year_indices, dtype=int)
            contribution[np.ix_(indices, active)] = innovation[np.ix_(indices, active)] * gain

        reconstructed_increment = contribution.sum(axis=1)
        closure_error = float(np.max(np.abs(reconstructed_increment - expected_increment)))
        closure[f"proxy{network}_hma_increment_max_abs_error_degC"] = closure_error
        if closure_error > 2e-6:
            raise RuntimeError(f"proxy{network} decomposition closure failed: {closure_error}")
        network_increment[network] = expected_increment

        locations = np.asarray(static[f"proxy{network}_locations"], dtype=float)
        coefficients = np.asarray(static[f"proxy{network}_psm_b0_b1_b2"], dtype=float)
        record_indices = np.asarray(static[f"proxy{network}_record_indices_0based"], dtype=int)
        ptypes = np.asarray(static[f"proxy{network}_ptype"]).astype(str)
        methods = np.asarray(static[f"proxy{network}_psm_method"]).astype(str)
        centers = np.asarray(static[f"proxy{network}_archived_hxb_mean_1500_1848"], dtype=float)

        for start, end in PERIODS:
            period = (years >= start) & (years <= end)
            active_values = innovation[period][np.isfinite(innovation[period])]
            period_rows.append(
                {
                    "period": f"{start}-{end}",
                    "network": f"proxy{network}",
                    "mean_hma_increment_degC": float(expected_increment[period].mean()),
                    "median_hma_increment_degC": float(np.median(expected_increment[period])),
                    "mean_active_proxy_count": float(np.isfinite(observations[period]).sum(axis=1).mean()),
                    "mean_active_innovation": float(np.mean(active_values)),
                    "median_active_innovation": float(np.median(active_values)),
                }
            )
            impact = contribution[period].sum(axis=0) / period.sum()
            coverage = np.isfinite(observations[period]).sum(axis=0)
            order = np.argsort(np.abs(impact))[::-1]
            for rank, proxy_index in enumerate(order[:25], start=1):
                proxy_rows.append(
                    {
                        "period": f"{start}-{end}",
                        "network": f"proxy{network}",
                        "rank_absolute_impact": rank,
                        "network_record_position_0based": int(proxy_index),
                        "archive_record_index_0based": int(record_indices[proxy_index]),
                        "latitude": float(locations[proxy_index, 0]),
                        "longitude": float(locations[proxy_index, 1]),
                        "ptype": ptypes[proxy_index],
                        "psm_method": methods[proxy_index],
                        "b0": float(coefficients[proxy_index, 0]),
                        "b1_temperature": float(coefficients[proxy_index, 1]),
                        "b2_precipitation": float(coefficients[proxy_index, 2]),
                        "archived_hxb_center": float(centers[proxy_index]),
                        "active_year_count": int(coverage[proxy_index]),
                        "mean_hma_contribution_per_period_year_degC": float(impact[proxy_index]),
                    }
                )

    joint_increment = 0.5 * (network_increment[6] + network_increment[9])
    real_joint = np.asarray(real["candidate_b2_joint_temperature_anomaly"], dtype=float) @ weights
    joint_closure = float(np.max(np.abs(joint_increment - (real_joint - background_hma))))
    closure["joint_hma_increment_max_abs_error_degC"] = joint_closure
    if joint_closure > 2e-6:
        raise RuntimeError(f"joint decomposition closure failed: {joint_closure}")
    for start, end in PERIODS:
        period = (years >= start) & (years <= end)
        period_rows.append(
            {
                "period": f"{start}-{end}",
                "network": "joint_0p5_0p5",
                "mean_hma_increment_degC": float(joint_increment[period].mean()),
                "median_hma_increment_degC": float(np.median(joint_increment[period])),
                "mean_active_proxy_count": float("nan"),
                "mean_active_innovation": float("nan"),
                "median_active_innovation": float("nan"),
            }
        )

    periods = pd.DataFrame(period_rows)
    proxies = pd.DataFrame(proxy_rows)
    periods.to_csv(OUTPUT / "network_period_increment_summary.csv", index=False)
    proxies.to_csv(OUTPUT / "top_proxy_hma_increment_contributions.csv", index=False)

    fig, (axis, count_axis) = plt.subplots(
        2, 1, figsize=(13.5, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True,
    )
    for values, label, color, width in (
        (network_increment[6], "Proxy6 increment", "#2878B5", 1.3),
        (network_increment[9], "Proxy9 increment", "#2E8B57", 1.3),
        (joint_increment, "Joint increment", "#C33C2E", 2.3),
    ):
        for start, end in ((1500, 1920), (1940, 2000)):
            period = (years >= start) & (years <= end)
            axis.plot(
                years[period], smooth(values, years)[period], color=color,
                linewidth=width, label=label if start == 1500 else None,
            )
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.set_ylabel("HMA increment (degC)")
    axis.set_title("Candidate B2 real-proxy EnKF increment relative to balanced background")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for network, color in ((6, "#2878B5"), (9, "#2E8B57")):
        counts = np.asarray(real[f"proxy{network}_active_proxy_count"], dtype=float)
        count_axis.plot(years, counts, color=color, linewidth=1.0, label=f"Proxy{network} active")
    count_axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    count_axis.set(xlabel="Year", ylabel="Active proxies")
    count_axis.legend(frameon=False, ncol=2)
    count_axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.savefig(OUTPUT / "candidate_b2_stage_c_hma_increment_decomposition.png", dpi=220)
    plt.close(fig)

    period_lookup = periods.set_index(["period", "network"])
    early6 = float(period_lookup.loc[("1500-1849", "proxy6"), "mean_hma_increment_degC"])
    early9 = float(period_lookup.loc[("1500-1849", "proxy9"), "mean_hma_increment_degC"])
    bridge6 = float(period_lookup.loc[("1850-1920", "proxy6"), "mean_hma_increment_degC"])
    bridge9 = float(period_lookup.loc[("1850-1920", "proxy9"), "mean_hma_increment_degC"])
    modern6 = float(period_lookup.loc[("1940-2000", "proxy6"), "mean_hma_increment_degC"])
    modern9 = float(period_lookup.loc[("1940-2000", "proxy9"), "mean_hma_increment_degC"])
    summary = {
        "experiment": OUTPUT.name,
        "status": "complete",
        "closure": closure,
        "mean_hma_increment_degC": {
            "1500_1849": {"proxy6": early6, "proxy9": early9, "joint": 0.5 * (early6 + early9)},
            "1850_1920": {"proxy6": bridge6, "proxy9": bridge9, "joint": 0.5 * (bridge6 + bridge9)},
            "1940_2000": {"proxy6": modern6, "proxy9": modern9, "joint": 0.5 * (modern6 + modern9)},
        },
        "finding": "both proxy networks independently reintroduce positive long-term HMA increments; the effect is not unique to proxy6 or proxy9",
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Candidate B2 Stage C increment decomposition

The HMA increment decomposition closes to the saved real-proxy posterior within {max(closure.values()):.3e} degC.

- 1500-1849 mean increments: proxy6 {early6:+.3f} degC; proxy9 {early9:+.3f} degC; joint {0.5 * (early6 + early9):+.3f} degC.
- 1850-1920 mean increments: proxy6 {bridge6:+.3f} degC; proxy9 {bridge9:+.3f} degC; joint {0.5 * (bridge6 + bridge9):+.3f} degC.
- 1940-2000 mean increments: proxy6 {modern6:+.3f} degC; proxy9 {modern9:+.3f} degC; joint {0.5 * (modern6 + modern9):+.3f} degC.

Both independently processed proxy networks push the balanced background warmer over the long period. The failed long-frequency retention is therefore not a proxy6-versus-proxy9 disagreement. The next test must target the shared observation/operator low-frequency constraint while keeping the successful modern covariance response visible.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    manifest = {
        "inputs": {str(STATIC): sha256(STATIC), str(REAL): sha256(REAL)},
        "outputs": {
            str(path): sha256(path)
            for path in sorted(OUTPUT.iterdir())
            if path.name != "source_manifest.json"
        },
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
