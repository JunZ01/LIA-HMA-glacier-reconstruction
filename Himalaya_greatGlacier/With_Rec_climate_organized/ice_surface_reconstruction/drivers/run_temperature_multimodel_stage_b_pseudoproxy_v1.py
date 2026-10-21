from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.ndimage import uniform_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
PREP = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2"
INPUT = PREP / "stage_b_pseudoproxy_input_v1.npz"
CONTRACT = PREP / "STAGE_B_CONTRACT.md"
OUT = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_v1"
NETWORKS = (6, 9)
NOISE_SEED = 20260827
NOISE_REALIZATIONS = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = weights > 0
    return np.average(values[..., valid], axis=-1, weights=weights[valid])


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2)))


def correlation(estimate: np.ndarray, truth: np.ndarray) -> float:
    if np.std(estimate) <= 1e-12 or np.std(truth) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(estimate, truth)[0, 1])


def amplitude_ratio(estimate: np.ndarray, truth: np.ndarray) -> float:
    denominator = float(np.std(truth))
    return float(np.std(estimate) / denominator) if denominator > 1e-12 else float("nan")


def lowpass(values: np.ndarray) -> np.ndarray:
    return uniform_filter1d(np.asarray(values, dtype=float), size=31, axis=0, mode="reflect")


def weighted_field_rmse(estimate: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    valid = weights > 0
    squared = (estimate[:, valid] - truth[:, valid]) ** 2
    return float(np.sqrt(np.average(squared, axis=1, weights=weights[valid]).mean()))


def kalman_gain(state_residual: np.ndarray, operator_residual: np.ndarray, variance: np.ndarray) -> np.ndarray:
    member_count = state_residual.shape[0]
    if operator_residual.shape[0] != member_count:
        raise RuntimeError("paired-member-count")
    state_members = state_residual.T
    operator_members = operator_residual.T
    cross = state_members @ operator_members.T / (member_count - 1)
    innovation_covariance = operator_members @ operator_members.T / (member_count - 1)
    innovation_covariance += np.diag(variance)
    try:
        factor = scipy.linalg.cho_factor(innovation_covariance, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(factor, cross.T, check_finite=False).T
    except np.linalg.LinAlgError:
        gain = scipy.linalg.solve(
            innovation_covariance,
            cross.T,
            assume_a="sym",
            check_finite=False,
        ).T
    if not np.all(np.isfinite(gain)):
        raise RuntimeError("nonfinite-kalman-gain")
    return gain


def posterior_mean(
    state_mean: np.ndarray,
    operator_mean: np.ndarray,
    observations: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    return state_mean + (observations - operator_mean) @ gain.T


def curve_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate_low = lowpass(estimate)
    truth_low = lowpass(truth)
    return {
        "rmse_degC": rmse(estimate, truth),
        "correlation": correlation(estimate, truth),
        "amplitude_ratio": amplitude_ratio(estimate, truth),
        "lowpass_rmse_degC": rmse(estimate_low, truth_low),
        "residual_rmse_degC": rmse(estimate - estimate_low, truth - truth_low),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "input": INPUT,
        "contract": CONTRACT,
        "preparation_summary": PREP / "summary.json",
        "preparation_manifest": PREP / "source_manifest.json",
        "driver": Path(__file__),
    }
    before = {name: sha256(path) for name, path in sources.items()}
    preparation = json.loads((PREP / "summary.json").read_text(encoding="utf-8"))
    if not preparation.get("preparation_pass") or preparation.get("enkf_run"):
        raise RuntimeError("preparation-contract")

    data = np.load(INPUT, allow_pickle=False)
    models = np.asarray(data["model_keys"]).astype(str)
    labels = np.asarray(data["model_labels"]).astype(str)
    years = np.asarray(data["evaluation_years"], dtype=int)
    truth_fields = np.asarray(data["truth_temperature_anomaly"], dtype=float)
    prior_means = np.asarray(data["prior_temperature_forced_mean"], dtype=float)
    state_residuals = np.asarray(data["prior_temperature_residual_library"], dtype=float)
    region_codes = np.asarray(data["gtng_region_codes"]).astype(str)
    region_weights = np.asarray(data["gtng_region_weights"], dtype=float)
    hma_weights = np.asarray(data["hma_weights"], dtype=float)
    if truth_fields.shape != (4, 319, 333) or prior_means.shape != (4, 319, 333):
        raise RuntimeError("state-input-shape")
    if state_residuals.shape != (4, 349, 333):
        raise RuntimeError("residual-input-shape")

    gains: dict[tuple[int, int], np.ndarray] = {}
    truth_operators: dict[int, np.ndarray] = {}
    operator_means: dict[int, np.ndarray] = {}
    operator_variances: dict[int, np.ndarray] = {}
    for network in NETWORKS:
        truth_operators[network] = np.asarray(data[f"proxy{network}_truth_operator_anomaly"], dtype=float)
        operator_means[network] = np.asarray(data[f"proxy{network}_prior_operator_forced_mean"], dtype=float)
        operator_residual = np.asarray(data[f"proxy{network}_prior_operator_residual_library"], dtype=float)
        operator_variances[network] = np.asarray(data[f"proxy{network}_residual_variance"], dtype=float)
        for prior_index in range(len(models)):
            gains[(prior_index, network)] = kalman_gain(
                state_residuals[prior_index],
                operator_residual[prior_index],
                operator_variances[network],
            )

    posterior: dict[tuple[int, int, int], np.ndarray] = {}
    for truth_index in range(len(models)):
        for prior_index in range(len(models)):
            if prior_index == truth_index:
                continue
            for network in NETWORKS:
                posterior[(truth_index, prior_index, network)] = posterior_mean(
                    prior_means[prior_index],
                    operator_means[network][prior_index],
                    truth_operators[network][truth_index],
                    gains[(prior_index, network)],
                )

    annual_rows = []
    fold_rows = []
    regional_rows = []
    fold_curves: dict[tuple[int, str], np.ndarray] = {}
    for truth_index, truth_model in enumerate(models):
        nontruth = [index for index in range(len(models)) if index != truth_index]
        truth_curve = weighted_mean(truth_fields[truth_index], hma_weights)
        single_fields = {}
        for prior_index in nontruth:
            single_fields[prior_index] = np.mean(
                [posterior[(truth_index, prior_index, network)] for network in NETWORKS],
                axis=0,
            )
        candidate_field = np.mean(list(single_fields.values()), axis=0)
        candidate_curve = weighted_mean(candidate_field, hma_weights)
        fold_curves[(truth_index, "truth")] = truth_curve
        fold_curves[(truth_index, "candidate")] = candidate_curve

        single_rmse = []
        for prior_index, field in single_fields.items():
            curve = weighted_mean(field, hma_weights)
            fold_curves[(truth_index, models[prior_index])] = curve
            metrics = curve_metrics(curve, truth_curve)
            single_rmse.append(metrics["rmse_degC"])
            fold_rows.append(
                {
                    "truth_model": labels[truth_index],
                    "estimate": labels[prior_index],
                    "estimate_type": "single_nontruth_prior",
                    **metrics,
                    "field_rmse_degC": weighted_field_rmse(field, truth_fields[truth_index], hma_weights),
                }
            )
        candidate_metrics = curve_metrics(candidate_curve, truth_curve)
        median_single = float(np.median(single_rmse))
        cesm_rmse = next(
            (row["rmse_degC"] for row in reversed(fold_rows)
             if row["truth_model"] == labels[truth_index] and row["estimate"] == "CESM"),
            float("nan"),
        )
        fold_rows.append(
            {
                "truth_model": labels[truth_index],
                "estimate": "multimodel_candidate",
                "estimate_type": "equal_three_nontruth_prior_mean",
                **candidate_metrics,
                "field_rmse_degC": weighted_field_rmse(candidate_field, truth_fields[truth_index], hma_weights),
                "median_single_rmse_degC": median_single,
                "skill_vs_median_single": 1.0 - candidate_metrics["rmse_degC"] / median_single,
                "cesm_rmse_degC": cesm_rmse,
                "skill_vs_cesm": (
                    1.0 - candidate_metrics["rmse_degC"] / cesm_rmse
                    if np.isfinite(cesm_rmse) else float("nan")
                ),
            }
        )

        for year_index, year in enumerate(years):
            annual_rows.append(
                {
                    "truth_model": labels[truth_index],
                    "year": int(year),
                    "estimate": "truth",
                    "hma_temperature_anomaly_degC": float(truth_curve[year_index]),
                }
            )
            annual_rows.append(
                {
                    "truth_model": labels[truth_index],
                    "year": int(year),
                    "estimate": "multimodel_candidate",
                    "hma_temperature_anomaly_degC": float(candidate_curve[year_index]),
                }
            )
            for prior_index in nontruth:
                annual_rows.append(
                    {
                        "truth_model": labels[truth_index],
                        "year": int(year),
                        "estimate": labels[prior_index],
                        "hma_temperature_anomaly_degC": float(
                            fold_curves[(truth_index, models[prior_index])][year_index]
                        ),
                    }
                )

        for region_index, region in enumerate(region_codes):
            truth_region = weighted_mean(truth_fields[truth_index], region_weights[region_index])
            candidate_region = weighted_mean(candidate_field, region_weights[region_index])
            single_region_rmse = []
            for prior_index, field in single_fields.items():
                single_curve = weighted_mean(field, region_weights[region_index])
                single_region_rmse.append(rmse(single_curve, truth_region))
            candidate_region_metrics = curve_metrics(candidate_region, truth_region)
            median_region_single = float(np.median(single_region_rmse))
            regional_rows.append(
                {
                    "truth_model": labels[truth_index],
                    "o2region": region,
                    **candidate_region_metrics,
                    "median_single_rmse_degC": median_region_single,
                    "skill_vs_median_single": 1.0 - candidate_region_metrics["rmse_degC"] / median_region_single,
                }
            )

    annual = pd.DataFrame(annual_rows)
    folds = pd.DataFrame(fold_rows)
    regional = pd.DataFrame(regional_rows)
    candidate_folds = folds[folds["estimate"] == "multimodel_candidate"].copy()
    regional_summary = (
        regional.groupby("o2region", as_index=False)
        .agg(
            candidate_median_rmse_degC=("rmse_degC", "median"),
            comparator_median_rmse_degC=("median_single_rmse_degC", "median"),
            median_skill_vs_single=("skill_vs_median_single", "median"),
        )
    )
    regional_summary["candidate_improves"] = (
        regional_summary["candidate_median_rmse_degC"]
        < regional_summary["comparator_median_rmse_degC"]
    )

    rng = np.random.default_rng(NOISE_SEED)
    noise_rows = []
    for realization in range(NOISE_REALIZATIONS):
        for truth_index in range(len(models)):
            nontruth = [index for index in range(len(models)) if index != truth_index]
            noisy_single_fields = []
            for prior_index in nontruth:
                network_fields = []
                for network in NETWORKS:
                    noise = rng.normal(
                        loc=0.0,
                        scale=np.sqrt(operator_variances[network]),
                        size=truth_operators[network][truth_index].shape,
                    )
                    network_fields.append(
                        posterior_mean(
                            prior_means[prior_index],
                            operator_means[network][prior_index],
                            truth_operators[network][truth_index] + noise,
                            gains[(prior_index, network)],
                        )
                    )
                noisy_single_fields.append(np.mean(network_fields, axis=0))
            noisy_candidate = np.mean(noisy_single_fields, axis=0)
            noisy_curve = weighted_mean(noisy_candidate, hma_weights)
            truth_curve = fold_curves[(truth_index, "truth")]
            noise_rows.append(
                {
                    "realization": realization + 1,
                    "truth_model": labels[truth_index],
                    "hma_rmse_degC": rmse(noisy_curve, truth_curve),
                    "correlation": correlation(noisy_curve, truth_curve),
                    "amplitude_ratio": amplitude_ratio(noisy_curve, truth_curve),
                }
            )
    noise_table = pd.DataFrame(noise_rows)

    hma_improvement_count = int((candidate_folds["skill_vs_median_single"] > 0).sum())
    worst_hma_ratio = float(
        np.max(candidate_folds["rmse_degC"] / candidate_folds["median_single_rmse_degC"])
    )
    region_improvement_count = int(regional_summary["candidate_improves"].sum())
    noncesm = candidate_folds[candidate_folds["truth_model"] != "CESM"]
    cesm_improvement_count = int((noncesm["skill_vs_cesm"] > 0).sum())
    worst_cesm_ratio = float(np.max(noncesm["rmse_degC"] / noncesm["cesm_rmse_degC"]))
    gates = {
        "hma_improves_at_least_3_of_4_truths": hma_improvement_count >= 3,
        "no_hma_truth_degrades_more_than_5pct_vs_median_single": worst_hma_ratio <= 1.05,
        "at_least_10_of_15_gtng_regions_improve": region_improvement_count >= 10,
        "candidate_beats_cesm_at_least_2_of_3_non_cesm_truths": cesm_improvement_count >= 2,
        "no_non_cesm_truth_degrades_more_than_5pct_vs_cesm": worst_cesm_ratio <= 1.05,
    }
    stage_b_pass = all(gates.values())

    annual.to_csv(OUT / "annual_hma_curves.csv", index=False)
    folds.to_csv(OUT / "withheld_truth_metrics.csv", index=False)
    regional.to_csv(OUT / "gtng15_withheld_truth_metrics.csv", index=False)
    regional_summary.to_csv(OUT / "gtng15_gate_summary.csv", index=False)
    noise_table.to_csv(OUT / "fixed_seed_noise_sensitivity.csv", index=False)

    colors = {"multimodel_candidate": "#C43C2B", "CESM": "#1B1B1B", "ModE-Sim": "#247BA0", "MPI": "#4C956C", "MIROC": "#8B5FBF"}
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11.0), constrained_layout=True)
    for axis, truth_index in zip(axes[:2].flat, range(len(models))):
        table = annual[annual["truth_model"] == labels[truth_index]]
        truth_table = table[table["estimate"] == "truth"]
        axis.plot(truth_table["year"], truth_table["hma_temperature_anomaly_degC"], color="#111111", linewidth=2.2, label="Withheld truth")
        for estimate in ["multimodel_candidate", *[labels[i] for i in range(len(models)) if i != truth_index]]:
            selected = table[table["estimate"] == estimate]
            axis.plot(selected["year"], selected["hma_temperature_anomaly_degC"], color=colors[estimate], linewidth=1.5 if estimate == "multimodel_candidate" else 0.9, alpha=1.0 if estimate == "multimodel_candidate" else 0.7, label=estimate)
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.set(title=f"Truth withheld: {labels[truth_index]}", ylabel="HMA anomaly (degC)")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)

    region_plot = regional_summary.sort_values("o2region")
    axes[2, 0].bar(region_plot["o2region"], region_plot["median_skill_vs_single"], color=np.where(region_plot["median_skill_vs_single"] > 0, "#287D56", "#C43C2B"))
    axes[2, 0].axhline(0, color="#111111", linewidth=0.8)
    axes[2, 0].tick_params(axis="x", rotation=55)
    axes[2, 0].set(title="GTN-G skill vs median single prior", ylabel="1 - candidate RMSE / comparator RMSE")

    noise_groups = [noise_table[noise_table["truth_model"] == label]["hma_rmse_degC"].to_numpy() for label in labels]
    axes[2, 1].boxplot(noise_groups, tick_labels=labels, showfliers=False)
    axes[2, 1].set(title="Fixed-seed RegEM-R noise sensitivity", ylabel="HMA RMSE (degC)")
    axes[2, 1].tick_params(axis="x", rotation=25)
    fig.savefig(OUT / "stage_b_pseudoproxy_results.png", dpi=190, facecolor="white")
    plt.close(fig)

    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage_b_pass": bool(stage_b_pass),
        "gates": gates,
        "hma_improvement_truth_count": hma_improvement_count,
        "worst_hma_rmse_ratio_vs_median_single": worst_hma_ratio,
        "gtng_region_improvement_count": region_improvement_count,
        "cesm_comparable_truth_improvement_count": cesm_improvement_count,
        "worst_non_cesm_truth_rmse_ratio_vs_cesm": worst_cesm_ratio,
        "noise_seed": NOISE_SEED,
        "noise_realizations": NOISE_REALIZATIONS,
        "next_action": "stage_c_allowed_real_data" if stage_b_pass else "reject_multimodel_evolving_prior_candidate",
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

    lines = [
        "# Multimodel evolving-prior Stage B pseudoproxy result",
        "",
        f"Stage B pass: `{stage_b_pass}`.",
        f"HMA truth folds improved: `{hma_improvement_count}/4`.",
        f"GTN-G regions improved: `{region_improvement_count}/15`.",
        f"Non-CESM truth folds beating CESM: `{cesm_improvement_count}/3`.",
        f"Worst HMA RMSE ratio versus median single prior: `{worst_hma_ratio:.4f}`.",
        f"Worst non-CESM truth RMSE ratio versus CESM: `{worst_cesm_ratio:.4f}`.",
        "",
    ]
    lines.extend([f"- `{name}`: `{value}`" for name, value in gates.items()])
    lines.extend(
        [
            "",
            "This is a leakage-free model-as-truth method test, not observational validation.",
            "No production or OGGM file was changed; no real proxy observation or sealed-year value was used.",
        ]
    )
    (OUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: sha256(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-changed")


if __name__ == "__main__":
    main()
