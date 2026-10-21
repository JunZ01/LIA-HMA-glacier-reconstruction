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
INPUT = PREP / "stage_b_pseudoproxy_input_v1.npz"
OUT = LOOP / "temperature_multimodel_evolving_prior_stage_b_required_diagnostics_v1"
NETWORKS = (6, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return core.curve_metrics(np.asarray(estimate), np.asarray(truth))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "input": INPUT,
        "contract": PREP / "STAGE_B_CONTRACT.md",
        "primary_summary": PRIMARY / "summary.json",
        "primary_metrics": PRIMARY / "withheld_truth_metrics.csv",
        "primary_noise": PRIMARY / "fixed_seed_noise_sensitivity.csv",
        "driver": Path(__file__),
        "core": Path(core.__file__),
    }
    before = {name: sha256(path) for name, path in sources.items()}
    primary_summary = json.loads((PRIMARY / "summary.json").read_text(encoding="utf-8"))
    if primary_summary.get("stage_b_pass") is not False:
        raise RuntimeError("unexpected-primary-stage-b-state")

    data = np.load(INPUT, allow_pickle=False)
    models = np.asarray(data["model_keys"]).astype(str)
    labels = np.asarray(data["model_labels"]).astype(str)
    truth_fields = np.asarray(data["truth_temperature_anomaly"], dtype=float)
    prior_means = np.asarray(data["prior_temperature_forced_mean"], dtype=float)
    state_residuals = np.asarray(data["prior_temperature_residual_library"], dtype=float)
    hma_weights = np.asarray(data["hma_weights"], dtype=float)
    region_codes = np.asarray(data["gtng_region_codes"]).astype(str)
    region_weights = np.asarray(data["gtng_region_weights"], dtype=float)

    truth_operator = {
        network: np.asarray(data[f"proxy{network}_truth_operator_anomaly"], dtype=float)
        for network in NETWORKS
    }
    operator_mean = {
        network: np.asarray(data[f"proxy{network}_prior_operator_forced_mean"], dtype=float)
        for network in NETWORKS
    }
    operator_residual = {
        network: np.asarray(data[f"proxy{network}_prior_operator_residual_library"], dtype=float)
        for network in NETWORKS
    }
    variance = {
        network: np.asarray(data[f"proxy{network}_residual_variance"], dtype=float)
        for network in NETWORKS
    }

    gains = {
        (prior, network): core.kalman_gain(
            state_residuals[prior], operator_residual[network][prior], variance[network]
        )
        for prior in range(4)
        for network in NETWORKS
    }
    posterior = {
        (truth, prior, network): core.posterior_mean(
            prior_means[prior],
            operator_mean[network][prior],
            truth_operator[network][truth],
            gains[(prior, network)],
        )
        for truth in range(4)
        for prior in range(4)
        if prior != truth
        for network in NETWORKS
    }

    network_rows = []
    loo_rows = []
    pooled_rows = []
    pooled_regional_rows = []
    component_rows = []
    for truth in range(4):
        nontruth = [prior for prior in range(4) if prior != truth]
        truth_hma = core.weighted_mean(truth_fields[truth], hma_weights)
        single_combined_fields = {
            prior: np.mean([posterior[(truth, prior, network)] for network in NETWORKS], axis=0)
            for prior in nontruth
        }
        full_candidate_field = np.mean(list(single_combined_fields.values()), axis=0)
        full_candidate_curve = core.weighted_mean(full_candidate_field, hma_weights)

        for network in NETWORKS:
            candidate_network = np.mean(
                [posterior[(truth, prior, network)] for prior in nontruth], axis=0
            )
            candidate_curve = core.weighted_mean(candidate_network, hma_weights)
            network_rows.append(
                {
                    "truth_model": labels[truth],
                    "proxy_network": network,
                    "estimate": "multimodel_candidate",
                    **metrics(candidate_curve, truth_hma),
                }
            )
            for prior in nontruth:
                curve = core.weighted_mean(posterior[(truth, prior, network)], hma_weights)
                network_rows.append(
                    {
                        "truth_model": labels[truth],
                        "proxy_network": network,
                        "estimate": labels[prior],
                        **metrics(curve, truth_hma),
                    }
                )

        full_metrics = metrics(full_candidate_curve, truth_hma)
        for omitted in nontruth:
            retained = [prior for prior in nontruth if prior != omitted]
            loo_field = np.mean([single_combined_fields[prior] for prior in retained], axis=0)
            loo_curve = core.weighted_mean(loo_field, hma_weights)
            loo_metric = metrics(loo_curve, truth_hma)
            loo_rows.append(
                {
                    "truth_model": labels[truth],
                    "omitted_prior": labels[omitted],
                    "retained_priors": "+".join(labels[retained]),
                    **loo_metric,
                    "rmse_change_vs_full_candidate_degC": loo_metric["rmse_degC"] - full_metrics["rmse_degC"],
                    "rmse_fraction_change_vs_full_candidate": loo_metric["rmse_degC"] / full_metrics["rmse_degC"] - 1.0,
                }
            )

        pooled_state_residual = np.concatenate([state_residuals[prior] for prior in nontruth], axis=0)
        pooled_state_mean = np.mean(prior_means[nontruth], axis=0)
        pooled_network_fields = []
        for network in NETWORKS:
            pooled_operator_residual = np.concatenate(
                [operator_residual[network][prior] for prior in nontruth], axis=0
            )
            pooled_operator_mean = np.mean(operator_mean[network][nontruth], axis=0)
            pooled_gain = core.kalman_gain(
                pooled_state_residual, pooled_operator_residual, variance[network]
            )
            field = core.posterior_mean(
                pooled_state_mean,
                pooled_operator_mean,
                truth_operator[network][truth],
                pooled_gain,
            )
            pooled_network_fields.append(field)
            curve = core.weighted_mean(field, hma_weights)
            pooled_rows.append(
                {
                    "truth_model": labels[truth],
                    "estimate": f"pooled_proxy{network}",
                    **metrics(curve, truth_hma),
                }
            )
        pooled_field = np.mean(pooled_network_fields, axis=0)
        pooled_curve = core.weighted_mean(pooled_field, hma_weights)
        pooled_metric = metrics(pooled_curve, truth_hma)
        pooled_rows.append(
            {
                "truth_model": labels[truth],
                "estimate": "pooled_combined",
                **pooled_metric,
                "separate_candidate_rmse_degC": full_metrics["rmse_degC"],
                "skill_vs_separate_candidate": 1.0 - pooled_metric["rmse_degC"] / full_metrics["rmse_degC"],
            }
        )
        for region_index, region in enumerate(region_codes):
            truth_region = core.weighted_mean(truth_fields[truth], region_weights[region_index])
            pooled_region = core.weighted_mean(pooled_field, region_weights[region_index])
            separate_region = core.weighted_mean(full_candidate_field, region_weights[region_index])
            pooled_regional_rows.append(
                {
                    "truth_model": labels[truth],
                    "o2region": region,
                    "pooled_rmse_degC": core.rmse(pooled_region, truth_region),
                    "separate_candidate_rmse_degC": core.rmse(separate_region, truth_region),
                }
            )

        single_primary = pd.read_csv(PRIMARY / "withheld_truth_metrics.csv")
        subset = single_primary[
            (single_primary["truth_model"] == labels[truth])
            & (single_primary["estimate_type"] == "single_nontruth_prior")
        ]
        primary = single_primary[
            (single_primary["truth_model"] == labels[truth])
            & (single_primary["estimate"] == "multimodel_candidate")
        ].iloc[0]
        for component in ("rmse_degC", "lowpass_rmse_degC", "residual_rmse_degC", "field_rmse_degC"):
            comparator = float(subset[component].median())
            value = float(primary[component])
            component_rows.append(
                {
                    "truth_model": labels[truth],
                    "component": component,
                    "candidate_value": value,
                    "median_single_value": comparator,
                    "skill_vs_median_single": 1.0 - value / comparator,
                }
            )

    network_table = pd.DataFrame(network_rows)
    loo_table = pd.DataFrame(loo_rows)
    pooled_table = pd.DataFrame(pooled_rows)
    pooled_regional = pd.DataFrame(pooled_regional_rows)
    component_table = pd.DataFrame(component_rows)
    noise = pd.read_csv(PRIMARY / "fixed_seed_noise_sensitivity.csv")
    noise_summary = (
        noise.groupby("truth_model", as_index=False)
        .agg(
            rmse_p05=("hma_rmse_degC", lambda values: float(np.quantile(values, 0.05))),
            rmse_median=("hma_rmse_degC", "median"),
            rmse_p95=("hma_rmse_degC", lambda values: float(np.quantile(values, 0.95))),
            correlation_median=("correlation", "median"),
            amplitude_ratio_median=("amplitude_ratio", "median"),
        )
    )

    pooled_combined = pooled_table[pooled_table["estimate"] == "pooled_combined"].copy()
    primary_candidates = pd.read_csv(PRIMARY / "withheld_truth_metrics.csv")
    primary_candidates = primary_candidates[primary_candidates["estimate"] == "multimodel_candidate"]
    pooled_combined = pooled_combined.merge(
        primary_candidates[["truth_model", "median_single_rmse_degC", "cesm_rmse_degC"]],
        on="truth_model",
        how="left",
    )
    pooled_combined["skill_vs_median_single"] = (
        1.0 - pooled_combined["rmse_degC"] / pooled_combined["median_single_rmse_degC"]
    )
    pooled_combined["skill_vs_cesm"] = (
        1.0 - pooled_combined["rmse_degC"] / pooled_combined["cesm_rmse_degC"]
    )
    pooled_region_summary = (
        pooled_regional.groupby("o2region", as_index=False)
        .agg(
            pooled_median_rmse=("pooled_rmse_degC", "median"),
            separate_median_rmse=("separate_candidate_rmse_degC", "median"),
        )
    )
    pooled_region_summary["pooled_improves_separate"] = (
        pooled_region_summary["pooled_median_rmse"] < pooled_region_summary["separate_median_rmse"]
    )

    pooled_hma_improve_count = int((pooled_combined["skill_vs_median_single"] > 0).sum())
    pooled_worst_ratio = float(
        np.max(pooled_combined["rmse_degC"] / pooled_combined["median_single_rmse_degC"])
    )
    pooled_non_cesm = pooled_combined[pooled_combined["truth_model"] != "CESM"]
    pooled_cesm_improve_count = int((pooled_non_cesm["skill_vs_cesm"] > 0).sum())
    pooled_worst_cesm_ratio = float(
        np.max(pooled_non_cesm["rmse_degC"] / pooled_non_cesm["cesm_rmse_degC"])
    )
    pooled_region_improve_vs_separate = int(pooled_region_summary["pooled_improves_separate"].sum())

    network_table.to_csv(OUT / "network_specific_metrics.csv", index=False)
    loo_table.to_csv(OUT / "leave_one_prior_out_metrics.csv", index=False)
    pooled_table.to_csv(OUT / "balanced_pooled_prior_metrics.csv", index=False)
    pooled_combined.to_csv(OUT / "balanced_pooled_prior_gate_metrics.csv", index=False)
    pooled_regional.to_csv(OUT / "balanced_pooled_gtng15_metrics.csv", index=False)
    pooled_region_summary.to_csv(OUT / "balanced_pooled_gtng15_summary.csv", index=False)
    component_table.to_csv(OUT / "frequency_field_failure_components.csv", index=False)
    noise_summary.to_csv(OUT / "fixed_seed_noise_summary.csv", index=False)

    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "primary_stage_b_pass": False,
        "primary_rejection_unchanged": True,
        "required_diagnostics_complete": True,
        "balanced_pooled_secondary": {
            "hma_truths_improving_vs_median_single": pooled_hma_improve_count,
            "worst_hma_ratio_vs_median_single": pooled_worst_ratio,
            "non_cesm_truths_improving_vs_cesm": pooled_cesm_improve_count,
            "worst_non_cesm_ratio_vs_cesm": pooled_worst_cesm_ratio,
            "regions_improving_vs_separate_candidate": pooled_region_improve_vs_separate,
            "secondary_branch_can_replace_primary": False,
        },
        "next_action": "close_equal_multimodel_evolving_prior_route_and_identify_failure_mechanism",
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
