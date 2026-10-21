from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_temperature_multimodel_stage_b_pseudoproxy_v1 as core


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
PREP = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2"
PRIMARY = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_v3"
DIAGNOSTIC = LOOP / "temperature_multimodel_evolving_prior_stage_b_required_diagnostics_v1"
INPUT = PREP / "stage_b_pseudoproxy_input_v1.npz"
OUT = LOOP / "temperature_balanced_pooled_stage_b_robustness_v1"
NETWORKS = (6, 9)
SEED = 20260827
REALIZATIONS = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled_analysis(
    data: np.lib.npyio.NpzFile,
    truth: int,
    priors: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    state_mean = np.asarray(data["prior_temperature_forced_mean"], dtype=float)
    state_residual = np.asarray(data["prior_temperature_residual_library"], dtype=float)
    pooled_mean = np.mean(state_mean[priors], axis=0)
    pooled_residual = np.concatenate([state_residual[prior] for prior in priors], axis=0)
    network_fields: dict[int, np.ndarray] = {}
    network_gains: dict[int, np.ndarray] = {}
    for network in NETWORKS:
        operator_mean = np.asarray(data[f"proxy{network}_prior_operator_forced_mean"], dtype=float)
        operator_residual = np.asarray(data[f"proxy{network}_prior_operator_residual_library"], dtype=float)
        truth_operator = np.asarray(data[f"proxy{network}_truth_operator_anomaly"], dtype=float)
        variance = np.asarray(data[f"proxy{network}_residual_variance"], dtype=float)
        pooled_operator_mean = np.mean(operator_mean[priors], axis=0)
        pooled_operator_residual = np.concatenate(
            [operator_residual[prior] for prior in priors], axis=0
        )
        gain = core.kalman_gain(pooled_residual, pooled_operator_residual, variance)
        network_gains[network] = gain
        network_fields[network] = core.posterior_mean(
            pooled_mean,
            pooled_operator_mean,
            truth_operator[truth],
            gain,
        )
    return np.mean(list(network_fields.values()), axis=0), network_fields, network_gains


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "input": INPUT,
        "contract": PREP / "STAGE_B_CONTRACT.md",
        "primary_summary": PRIMARY / "summary.json",
        "primary_regional": PRIMARY / "gtng15_withheld_truth_metrics.csv",
        "pooled_hma": DIAGNOSTIC / "balanced_pooled_prior_gate_metrics.csv",
        "pooled_regional": DIAGNOSTIC / "balanced_pooled_gtng15_metrics.csv",
        "driver": Path(__file__),
        "core": Path(core.__file__),
    }
    before = {name: sha256(path) for name, path in sources.items()}
    data = np.load(INPUT, allow_pickle=False)
    labels = np.asarray(data["model_labels"]).astype(str)
    truth_fields = np.asarray(data["truth_temperature_anomaly"], dtype=float)
    hma_weights = np.asarray(data["hma_weights"], dtype=float)

    pooled_hma = pd.read_csv(DIAGNOSTIC / "balanced_pooled_prior_gate_metrics.csv")
    pooled_region = pd.read_csv(DIAGNOSTIC / "balanced_pooled_gtng15_metrics.csv")
    primary_region = pd.read_csv(PRIMARY / "gtng15_withheld_truth_metrics.csv")
    merged_region = pooled_region.merge(
        primary_region[["truth_model", "o2region", "median_single_rmse_degC"]],
        on=["truth_model", "o2region"],
        how="inner",
    )
    region_gate = (
        merged_region.groupby("o2region", as_index=False)
        .agg(
            pooled_median_rmse_degC=("pooled_rmse_degC", "median"),
            median_single_rmse_degC=("median_single_rmse_degC", "median"),
        )
    )
    region_gate["rmse_ratio"] = (
        region_gate["pooled_median_rmse_degC"] / region_gate["median_single_rmse_degC"]
    )
    region_gate["skill_vs_median_single"] = 1.0 - region_gate["rmse_ratio"]
    region_gate["improves"] = region_gate["rmse_ratio"] < 1.0

    hma_improved = int((pooled_hma["skill_vs_median_single"] > 0).sum())
    worst_hma_ratio = float(
        np.max(pooled_hma["rmse_degC"] / pooled_hma["median_single_rmse_degC"])
    )
    region_improved = int(region_gate["improves"].sum())
    noncesm = pooled_hma[pooled_hma["truth_model"] != "CESM"]
    cesm_improved = int((noncesm["skill_vs_cesm"] > 0).sum())
    worst_cesm_ratio = float(np.max(noncesm["rmse_degC"] / noncesm["cesm_rmse_degC"]))
    numeric_gates = {
        "hma_improves_at_least_3_of_4_truths": hma_improved >= 3,
        "no_hma_truth_degrades_more_than_5pct_vs_median_single": worst_hma_ratio <= 1.05,
        "at_least_10_of_15_gtng_regions_improve": region_improved >= 10,
        "candidate_beats_cesm_at_least_2_of_3_non_cesm_truths": cesm_improved >= 2,
        "no_non_cesm_truth_degrades_more_than_5pct_vs_cesm": worst_cesm_ratio <= 1.05,
    }

    full_fields = {}
    network_fields = {}
    gains = {}
    loo_rows = []
    for truth in range(4):
        priors = [prior for prior in range(4) if prior != truth]
        full, by_network, by_gain = pooled_analysis(data, truth, priors)
        full_fields[truth] = full
        network_fields[truth] = by_network
        gains[truth] = by_gain
        truth_curve = core.weighted_mean(truth_fields[truth], hma_weights)
        full_curve = core.weighted_mean(full, hma_weights)
        full_rmse = core.rmse(full_curve, truth_curve)
        for omitted in priors:
            retained = [prior for prior in priors if prior != omitted]
            loo_field, _, _ = pooled_analysis(data, truth, retained)
            loo_curve = core.weighted_mean(loo_field, hma_weights)
            loo_metric = core.curve_metrics(loo_curve, truth_curve)
            loo_rows.append(
                {
                    "truth_model": labels[truth],
                    "omitted_prior": labels[omitted],
                    "retained_priors": "+".join(labels[retained]),
                    **loo_metric,
                    "rmse_ratio_vs_full_pooled": loo_metric["rmse_degC"] / full_rmse,
                }
            )

    rng = np.random.default_rng(SEED)
    noise_rows = []
    for realization in range(REALIZATIONS):
        for truth in range(4):
            priors = [prior for prior in range(4) if prior != truth]
            state_mean = np.asarray(data["prior_temperature_forced_mean"], dtype=float)
            pooled_mean = np.mean(state_mean[priors], axis=0)
            noisy_network_fields = []
            for network in NETWORKS:
                operator_mean = np.asarray(data[f"proxy{network}_prior_operator_forced_mean"], dtype=float)
                truth_operator = np.asarray(data[f"proxy{network}_truth_operator_anomaly"], dtype=float)
                variance = np.asarray(data[f"proxy{network}_residual_variance"], dtype=float)
                pooled_operator_mean = np.mean(operator_mean[priors], axis=0)
                noise = rng.normal(
                    0.0,
                    np.sqrt(variance),
                    size=truth_operator[truth].shape,
                )
                noisy_network_fields.append(
                    core.posterior_mean(
                        pooled_mean,
                        pooled_operator_mean,
                        truth_operator[truth] + noise,
                        gains[truth][network],
                    )
                )
            noisy_field = np.mean(noisy_network_fields, axis=0)
            noisy_curve = core.weighted_mean(noisy_field, hma_weights)
            truth_curve = core.weighted_mean(truth_fields[truth], hma_weights)
            noise_rows.append(
                {
                    "realization": realization + 1,
                    "truth_model": labels[truth],
                    "rmse_degC": core.rmse(noisy_curve, truth_curve),
                    "correlation": core.correlation(noisy_curve, truth_curve),
                    "amplitude_ratio": core.amplitude_ratio(noisy_curve, truth_curve),
                }
            )

    loo = pd.DataFrame(loo_rows)
    noise = pd.DataFrame(noise_rows)
    noise_summary = (
        noise.groupby("truth_model", as_index=False)
        .agg(
            rmse_p05=("rmse_degC", lambda values: float(np.quantile(values, 0.05))),
            rmse_median=("rmse_degC", "median"),
            rmse_p95=("rmse_degC", lambda values: float(np.quantile(values, 0.95))),
            correlation_median=("correlation", "median"),
            amplitude_ratio_median=("amplitude_ratio", "median"),
        )
    )

    worst_loo_ratio = float(loo["rmse_ratio_vs_full_pooled"].max())
    median_loo_ratio = float(loo["rmse_ratio_vs_full_pooled"].median())
    pooled_numeric_pass = all(numeric_gates.values())
    robustness = {
        "fixed_seed_noise_all_finite": bool(np.all(np.isfinite(noise.select_dtypes(include=[float, int])))),
        "leave_one_prior_all_finite": bool(np.all(np.isfinite(loo.select_dtypes(include=[float, int])))),
        "worst_leave_one_prior_rmse_ratio_vs_full_pooled": worst_loo_ratio,
        "median_leave_one_prior_rmse_ratio_vs_full_pooled": median_loo_ratio,
    }

    region_gate.to_csv(OUT / "pooled_gtng15_original_gate.csv", index=False)
    loo.to_csv(OUT / "pooled_leave_one_prior_out.csv", index=False)
    noise.to_csv(OUT / "pooled_fixed_seed_noise.csv", index=False)
    noise_summary.to_csv(OUT / "pooled_fixed_seed_noise_summary.csv", index=False)

    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "primary_separate_assimilation_stage_b_pass": False,
        "balanced_pooled_secondary_passes_all_original_numeric_gates": pooled_numeric_pass,
        "numeric_gates": numeric_gates,
        "hma_truths_improved": hma_improved,
        "worst_hma_ratio_vs_median_single": worst_hma_ratio,
        "gtng_regions_improved": region_improved,
        "worst_gtng_ratio_vs_median_single": float(region_gate["rmse_ratio"].max()),
        "non_cesm_truths_beating_cesm": cesm_improved,
        "worst_non_cesm_ratio_vs_cesm": worst_cesm_ratio,
        "robustness": robustness,
        "adjudication": (
            "preregister_balanced_pooled_candidate_b2_before_any_real_data"
            if pooled_numeric_pass
            else "close_balanced_pooled_route"
        ),
        "secondary_branch_promoted_retroactively": False,
        "production_modified": False,
        "oggm_run": False,
        "sealed_1921_1939_values_used": False,
        "real_proxy_observations_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()}
    manifest["prohibited_source_accessed"] = False
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    after = {name: sha256(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-changed")


if __name__ == "__main__":
    main()
