"""Diagnose the frozen temperature final-test failure from saved Round 86 data."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1921, 1940)
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
PRODUCTS = ("Berkeley", "CRU", "GISTEMP", "HadCRUT5", "NOAA")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def weighted_series(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=1)
    return np.nansum(values * weights[None], axis=1) / np.where(denominator == 0, np.nan, denominator)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def trend_per_decade(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return np.nan
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 10.0)


def pair_decomposition(candidate: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(candidate) & np.isfinite(observation)
    candidate = candidate[valid]
    observation = observation[valid]
    residual = candidate - observation
    bias = float(np.mean(residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    centered_residual = (candidate - np.mean(candidate)) - (observation - np.mean(observation))
    centered_rmse = float(np.sqrt(np.mean(centered_residual**2)))
    candidate_std = float(np.std(candidate, ddof=1))
    observation_std = float(np.std(observation, ddof=1))
    return {
        "candidate_mean_degC": float(np.mean(candidate)),
        "observation_mean_degC": float(np.mean(observation)),
        "mean_bias_degC": bias,
        "rmse_degC": rmse,
        "bias_fraction_of_mse": float(bias**2 / rmse**2) if rmse else np.nan,
        "centered_rmse_degC": centered_rmse,
        "candidate_std_degC": candidate_std,
        "observation_std_degC": observation_std,
        "std_ratio": float(candidate_std / observation_std) if observation_std else np.nan,
        "annual_correlation": correlation(candidate, observation),
        "candidate_trend_degC_per_decade": trend_per_decade(candidate),
        "observation_trend_degC_per_decade": trend_per_decade(observation),
        "sign_agreement_fraction": float(np.mean(np.sign(candidate) == np.sign(observation))),
    }


def draw_figure(
    output: Path,
    curves: pd.DataFrame,
    decomposition: pd.DataFrame,
    consensus: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {
        "Berkeley": "#0072b2",
        "CRU": "#d55e00",
        "GISTEMP": "#009e73",
        "HadCRUT5": "#cc79a7",
        "NOAA": "#e69f00",
    }
    for ax, region in zip(axes.ravel(), REGIONS):
        subset = curves[curves.region == region]
        first = subset[subset.observation == PRODUCTS[0]]
        ax.plot(first.year, first.five_family, color="#111111", lw=2.6, label="Five-family")
        ax.plot(first.year, first.four_family, color="#777777", lw=1.8, ls="--", label="Four-family")
        for product in PRODUCTS:
            product_rows = subset[subset.observation == product]
            ax.plot(product_rows.year, product_rows.observed, color=colors[product], lw=1.1, alpha=0.82, label=product)
        consensus_rows = consensus[consensus.region == region]
        ax.plot(consensus_rows.year, consensus_rows.observation_mean, color="#56b4e9", lw=2.0, ls=":", label="Obs mean")
        ax.axhline(0, color="#666666", lw=0.7)
        ax.set_title(region)
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature anomaly (degC)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)
    fig.suptitle("1921-1939 Frozen Temperature Failure Decomposition", fontsize=15)
    fig.savefig(output / "temperature_final_test_failure_curves.png", dpi=220)
    plt.close(fig)

    hma = decomposition[(decomposition.region == "hma") & (decomposition.candidate == "five_family")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    x = np.arange(len(PRODUCTS))
    ordered = hma.set_index("observation").loc[list(PRODUCTS)]
    axes[0].bar(x, ordered.mean_bias_degC, color="#0072b2")
    axes[0].axhline(0, color="#555555", lw=0.8)
    axes[0].set(title="Mean bias", ylabel="degC")
    axes[1].bar(x, ordered.std_ratio, color="#009e73")
    axes[1].axhline(1, color="#555555", lw=0.8)
    axes[1].set(title="Standard-deviation ratio", ylabel="Candidate / observation")
    axes[2].bar(x, ordered.annual_correlation, color="#d55e00")
    axes[2].axhline(0.30, color="#555555", lw=0.8, ls="--")
    axes[2].set(title="Annual correlation", ylabel="Correlation")
    for ax in axes:
        ax.set_xticks(x, PRODUCTS, rotation=35, ha="right")
    fig.suptitle("HMA Regional Failure Components", fontsize=14)
    fig.savefig(output / "hma_failure_components.png", dpi=220)
    plt.close(fig)


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    round86 = loop / "round_86_frozen_temperature_final_test_v1"
    output = loop / "round_87_temperature_final_test_failure_decomposition_v1"
    paths = {
        "round86_artifact": round86 / "temperature_final_test_1921_1939.nc",
        "round86_scores": round86 / "final_test_scores.csv",
        "round86_summary": round86 / "summary.json",
        "round86_source_manifest": round86 / "source_manifest.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    round86_summary = json.loads(paths["round86_summary"].read_text(encoding="utf-8"))
    if round86_summary["passes_frozen_final_test"]:
        raise RuntimeError("round86-did-not-fail")
    source_manifest = json.loads(paths["round86_source_manifest"].read_text(encoding="utf-8"))
    if not source_manifest["local_source_states_unchanged"]:
        raise RuntimeError("round86-source-integrity-failed")

    with h5py.File(paths["round86_artifact"], "r") as handle:
        years = np.asarray(handle["time"], dtype=int)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        fine_counts = np.asarray(handle["fine_hma_cell_count"], dtype=float)
        candidates = {
            "four_family": np.asarray(handle["four_family"], dtype=float),
            "five_family": np.asarray(handle["five_family"], dtype=float),
        }
        observations = {
            product: np.asarray(handle[f"observation_{product}"], dtype=float)
            for product in PRODUCTS
        }
    if not np.array_equal(years, YEARS) or len(lat) != 140:
        raise RuntimeError("round86-artifact-contract-failed")
    weights = np.cos(np.deg2rad(lat)) * fine_counts

    curve_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    for region, (lower, upper) in REGIONS.items():
        selected = (lon >= lower) & (lon < upper)
        candidate_series = {
            name: weighted_series(values[:, selected], weights[selected])
            for name, values in candidates.items()
        }
        observation_series = {
            name: weighted_series(values[:, selected], weights[selected])
            for name, values in observations.items()
        }
        obs_stack = np.stack([observation_series[name] for name in PRODUCTS])
        obs_mean = np.nanmean(obs_stack, axis=0)
        obs_spread = np.nanstd(obs_stack, axis=0, ddof=1)
        for index, year in enumerate(YEARS):
            consensus_rows.append(
                {
                    "region": region,
                    "year": int(year),
                    "observation_mean": float(obs_mean[index]),
                    "observation_product_std": float(obs_spread[index]),
                }
            )
        for product in PRODUCTS:
            obs = observation_series[product]
            for candidate_name, candidate in candidate_series.items():
                decomposition_rows.append(
                    {
                        "observation": product,
                        "region": region,
                        "candidate": candidate_name,
                        **pair_decomposition(candidate, obs),
                    }
                )
            for index, year in enumerate(YEARS):
                curve_rows.append(
                    {
                        "observation": product,
                        "region": region,
                        "year": int(year),
                        "five_family": float(candidate_series["five_family"][index]),
                        "four_family": float(candidate_series["four_family"][index]),
                        "observed": float(obs[index]),
                    }
                )
        consensus_metrics = pair_decomposition(candidate_series["five_family"], obs_mean)
        pair_correlations = [
            correlation(observation_series[left], observation_series[right])
            for left, right in combinations(PRODUCTS, 2)
        ]
        agreement_rows.append(
            {
                "region": region,
                "median_pairwise_observation_correlation": float(np.nanmedian(pair_correlations)),
                "minimum_pairwise_observation_correlation": float(np.nanmin(pair_correlations)),
                "median_annual_observation_product_std_degC": float(np.nanmedian(obs_spread)),
                **{f"candidate_vs_observation_mean_{key}": value for key, value in consensus_metrics.items()},
            }
        )

    curves = pd.DataFrame(curve_rows)
    decomposition = pd.DataFrame(decomposition_rows)
    consensus = pd.DataFrame(consensus_rows)
    agreement = pd.DataFrame(agreement_rows)
    curves.to_csv(output / "regional_annual_curves.csv", index=False)
    decomposition.to_csv(output / "candidate_observation_failure_decomposition.csv", index=False)
    consensus.to_csv(output / "observation_consensus_curves.csv", index=False)
    agreement.to_csv(output / "observation_agreement_and_consensus_fit.csv", index=False)

    five = decomposition[decomposition.candidate == "five_family"]
    hma = five[five.region == "hma"]
    hma_consensus = agreement[agreement.region == "hma"].iloc[0]
    summary = {
        "status": "complete",
        "input": "saved_round86_artifact_only",
        "locked_source_reaccessed": False,
        "round86_decision_unchanged": "reject_frozen_five_family_as_external_coarse_hma_temperature_reference",
        "hma_five_family": {
            "median_mean_bias_degC": float(hma.mean_bias_degC.median()),
            "mean_bias_range_degC": [float(hma.mean_bias_degC.min()), float(hma.mean_bias_degC.max())],
            "median_bias_fraction_of_mse": float(hma.bias_fraction_of_mse.median()),
            "median_centered_rmse_degC": float(hma.centered_rmse_degC.median()),
            "median_std_ratio": float(hma.std_ratio.median()),
            "median_annual_correlation": float(hma.annual_correlation.median()),
            "median_sign_agreement_fraction": float(hma.sign_agreement_fraction.median()),
        },
        "hma_observation_agreement": {
            "median_pairwise_correlation": float(hma_consensus.median_pairwise_observation_correlation),
            "minimum_pairwise_correlation": float(hma_consensus.minimum_pairwise_observation_correlation),
            "median_annual_product_std_degC": float(hma_consensus.median_annual_observation_product_std_degC),
        },
        "candidate_vs_equal_product_observation_mean": {
            "mean_bias_degC": float(hma_consensus.candidate_vs_observation_mean_mean_bias_degC),
            "bias_fraction_of_mse": float(hma_consensus.candidate_vs_observation_mean_bias_fraction_of_mse),
            "centered_rmse_degC": float(hma_consensus.candidate_vs_observation_mean_centered_rmse_degC),
            "std_ratio": float(hma_consensus.candidate_vs_observation_mean_std_ratio),
            "annual_correlation": float(hma_consensus.candidate_vs_observation_mean_annual_correlation),
        },
    }
    consensus_fit = summary["candidate_vs_equal_product_observation_mean"]
    if consensus_fit["bias_fraction_of_mse"] >= 0.50:
        dominant = "mean_level_bias"
    elif consensus_fit["annual_correlation"] < 0.30:
        dominant = "annual_phase_timing_mismatch"
    elif not 0.75 <= consensus_fit["std_ratio"] <= 1.25:
        dominant = "amplitude_mismatch"
    else:
        dominant = "mixed_regional_mismatch"
    summary["dominant_failure_component"] = dominant
    summary["route"] = (
        "retain_round86_rejection_and_return_to_prelocked_temperature_reconstruction_design; "
        "use the held-out result only as failure evidence, never as a tuning target"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_figure(output, curves, decomposition, consensus)
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Temperature Final-Test Failure Decomposition",
                "",
                f"- Dominant component: `{dominant}`.",
                f"- HMA candidate vs equal-product observation mean bias: `{consensus_fit['mean_bias_degC']:.3f} degC`.",
                f"- Bias share of HMA mean-observation MSE: `{consensus_fit['bias_fraction_of_mse']:.1%}`.",
                f"- HMA centered RMSE: `{consensus_fit['centered_rmse_degC']:.3f} degC`.",
                f"- HMA standard-deviation ratio: `{consensus_fit['std_ratio']:.3f}`.",
                f"- HMA annual correlation: `{consensus_fit['annual_correlation']:.3f}`.",
                f"- Observational median/minimum pairwise correlation: `{summary['hma_observation_agreement']['median_pairwise_correlation']:.3f}` / `{summary['hma_observation_agreement']['minimum_pairwise_correlation']:.3f}`.",
                "",
                "The Round 86 rejection remains final. This diagnosis does not tune or replace the candidate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
                "locked_source_reaccessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("round87-source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
