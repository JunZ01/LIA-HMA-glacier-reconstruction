"""Map prior temperature modes to residual-standardized PSM observability."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_round161_temperature_perfect_model_twin_v2 as round161
import run_round162_temperature_twin_information_geometry_v1 as round162
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


EXPERIMENT_ID = "round_163_temperature_prior_mode_psm_observability_v2"
BLIND_THRESHOLD = 0.20
STRUCTURAL_THRESHOLD = 0.50
ATTENUATION_RATIO_THRESHOLD = 0.60
PINV_RCOND = 1e-10


def fingerprint(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    if digest is None and stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def operator_anomalies(operator: dict[str, np.ndarray], sample: np.ndarray) -> tuple[np.ndarray, int]:
    hxb = operator["hxb"]
    variance = operator["variance"]
    active = np.isfinite(variance) & (variance > 0) & np.all(np.isfinite(hxb[:, sample]), axis=1)
    modeled = hxb[active][:, sample]
    anomaly = modeled - np.mean(modeled, axis=1, keepdims=True)
    return anomaly / np.sqrt(variance[active])[:, None], int(active.sum())


def mode_reductions(scores: np.ndarray, eigenvalues: np.ndarray, standardized_proxy: np.ndarray) -> dict[str, np.ndarray]:
    member_count = scores.shape[1]
    proxy_covariance = standardized_proxy @ standardized_proxy.T / (member_count - 1.0)
    cross_covariance = scores @ standardized_proxy.T / (member_count - 1.0)
    actual_solution = np.linalg.solve(proxy_covariance + np.eye(proxy_covariance.shape[0]), cross_covariance.T)
    actual = np.sum(cross_covariance * actual_solution.T, axis=1) / eigenvalues
    zero_solution = np.linalg.pinv(proxy_covariance, rcond=PINV_RCOND) @ cross_covariance.T
    zero = np.sum(cross_covariance * zero_solution.T, axis=1) / eigenvalues
    cross_energy = np.sum(cross_covariance**2, axis=1) / eigenvalues
    return {
        "actual": np.clip(actual, 0.0, 1.0),
        "zero": np.clip(zero, 0.0, 1.0),
        "cross_energy": cross_energy,
    }


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_improvement_loop_v2" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round163_temperature_prior_mode_psm_observability_v2_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    canonical162 = validation / "climate_reconstruction_improvement_loop_v2/round_162_temperature_twin_information_geometry_v1"
    manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round161_driver": driver / "run_round161_temperature_perfect_model_twin_v2.py",
        "round162_driver": driver / "run_round162_temperature_twin_information_geometry_v1.py",
        "round162_summary": canonical162 / "summary.json",
        "fold_manifest": fold_root / "source_hash_manifest.json",
        "operator6": fold_root / "proxy6_fold_native_operators_500.mat",
        "operator9": fold_root / "proxy9_fold_native_operators_500.mat",
        "predictor6": fold_root / "proxy6_archived_model_predictors_848.mat",
        "predictor9": fold_root / "proxy9_archived_model_predictors_848.mat",
        "allowed6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "allowed9": loop / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "native_grid": fold_root / "heldout_oof_fields.npz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path, recovered["prior"].get("sha256") if name == "prior" else None) for name, path in paths.items()}

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        sample = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    canonical = json.loads(paths["round162_summary"].read_text(encoding="utf-8"))
    truth_members = np.asarray(canonical["truth_members_0based"], dtype=int)
    if prior.shape != (336, 848) or sample.shape != (500,) or truth_members.shape != (24,):
        raise RuntimeError("frozen-dimension-contract")
    if np.intersect1d(sample, truth_members).size:
        raise RuntimeError("truth-pool-overlap")

    operators = {
        network: round161.load_full_operator(paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"])
        for network in round161.NETWORKS
    }
    if any(not np.array_equal(operator["sample_indices"], sample) for operator in operators.values()):
        raise RuntimeError("operator-sample-index-closure")
    standardized = {}
    active_counts = {}
    for network in round161.NETWORKS:
        standardized[str(network)], active_counts[str(network)] = operator_anomalies(operators[network], sample)
    standardized["joint_6_9"] = np.concatenate([standardized["6"], standardized["9"]], axis=0)
    active_counts["joint_6_9"] = active_counts["6"] + active_counts["9"]

    with np.load(paths["native_grid"]) as native:
        grid_lat = np.asarray(native["grid_lat"], dtype=float)
        grid_lon = np.asarray(native["grid_lon"], dtype=float)
        mask = np.asarray(native["hma_mask"], dtype=bool)
    if mask.size != prior.shape[0]:
        raise RuntimeError("native-grid-contract")
    weights = common.region_weights(grid_lat, grid_lon, mask)
    background = prior[:, sample]
    background_mean = np.mean(background, axis=1)
    prior_anomaly = background - background_mean[:, None]

    mode_rows = []
    truth_rows = []
    summary_rows = []
    region_data = {}
    for region, current_weights in weights.items():
        basis = round162.weighted_basis(prior_anomaly, current_weights)
        rank = int(basis["numerical_rank"])
        weighted_prior = basis["root_weight"][:, None] * prior_anomaly[basis["selected"]]
        u_full, singular_full, vt_full = np.linalg.svd(weighted_prior, full_matrices=False)
        singular = np.asarray(singular_full[:rank], dtype=float)
        u = np.asarray(u_full[:, :rank], dtype=float)
        eigenvalues = singular**2 / (len(sample) - 1.0)
        variance_fraction = eigenvalues / np.sum(eigenvalues)
        scores = singular[:, None] * vt_full[:rank]
        reductions = {
            network: mode_reductions(scores, eigenvalues, proxy_matrix)
            for network, proxy_matrix in standardized.items()
        }
        region_data[region] = {"basis": basis, "u": u, "reductions": reductions, "variance_fraction": variance_fraction}
        for network, current in reductions.items():
            weighted_actual = float(np.sum(variance_fraction * current["actual"]))
            weighted_zero = float(np.sum(variance_fraction * current["zero"]))
            blind_fraction = float(np.sum(variance_fraction[current["zero"] < BLIND_THRESHOLD]))
            summary_rows.append({
                "region": region,
                "network": network,
                "active_proxy_count": active_counts[network],
                "numerical_mode_count": rank,
                "prior_variance_weighted_actual_R_reduction": weighted_actual,
                "prior_variance_weighted_zero_R_reduction": weighted_zero,
                "actual_to_zero_R_ratio": weighted_actual / weighted_zero if weighted_zero > 0 else np.nan,
                "structurally_blind_prior_variance_fraction": blind_fraction,
            })
            for index in range(rank):
                mode_rows.append({
                    "region": region,
                    "network": network,
                    "mode": index + 1,
                    "prior_variance_fraction": float(variance_fraction[index]),
                    "cumulative_prior_variance_fraction": float(np.sum(variance_fraction[: index + 1])),
                    "actual_R_posterior_variance_reduction": float(current["actual"][index]),
                    "zero_R_structural_variance_reduction": float(current["zero"][index]),
                    "residual_weight_attenuation": float(current["zero"][index] - current["actual"][index]),
                    "standardized_cross_covariance_energy": float(current["cross_energy"][index]),
                    "structurally_blind": bool(current["zero"][index] < BLIND_THRESHOLD),
                })
        for case_index, truth_member in enumerate(truth_members):
            truth_error = round162.weighted_vector(prior[:, truth_member] - background_mean, basis)
            coefficients = u.T @ truth_error
            mode_energy = coefficients**2
            total_energy = float(truth_error @ truth_error)
            full_span_fraction = float(np.sum(mode_energy) / total_energy) if total_energy > 0 else np.nan
            for network, current in reductions.items():
                actual_resolved = float(np.sum(mode_energy * current["actual"]) / total_energy)
                zero_resolved = float(np.sum(mode_energy * current["zero"]) / total_energy)
                blind_truth = float(np.sum(mode_energy[current["zero"] < BLIND_THRESHOLD]) / total_energy)
                truth_rows.append({
                    "region": region,
                    "network": network,
                    "case_index": case_index,
                    "truth_member_0based": int(truth_member),
                    "full_prior_span_energy_fraction": full_span_fraction,
                    "actual_R_observability_weighted_truth_energy": actual_resolved,
                    "zero_R_observability_weighted_truth_energy": zero_resolved,
                    "structurally_blind_truth_energy_fraction": blind_truth,
                })

    mode_frame = pd.DataFrame(mode_rows)
    truth_frame = pd.DataFrame(truth_rows)
    summary_frame = pd.DataFrame(summary_rows)
    truth_summary = truth_frame.groupby(["region", "network"], as_index=False).agg(
        median_full_prior_span_energy_fraction=("full_prior_span_energy_fraction", "median"),
        median_actual_R_observability_weighted_truth_energy=("actual_R_observability_weighted_truth_energy", "median"),
        median_zero_R_observability_weighted_truth_energy=("zero_R_observability_weighted_truth_energy", "median"),
        median_structurally_blind_truth_energy_fraction=("structurally_blind_truth_energy_fraction", "median"),
    )
    mode_frame.to_csv(output / "prior_mode_observability.csv", index=False)
    truth_frame.to_csv(output / "heldout_truth_mode_observability.csv", index=False)
    summary_frame.to_csv(output / "prior_weighted_summary.csv", index=False)
    truth_summary.to_csv(output / "heldout_truth_summary.csv", index=False)

    primary = summary_frame[(summary_frame.region == "hma") & (summary_frame.network == "joint_6_9")].iloc[0]
    primary_truth = truth_summary[(truth_summary.region == "hma") & (truth_summary.network == "joint_6_9")].iloc[0]
    structural_primary = bool(primary.prior_variance_weighted_zero_R_reduction < STRUCTURAL_THRESHOLD)
    attenuation_primary = bool(
        primary.prior_variance_weighted_zero_R_reduction >= STRUCTURAL_THRESHOLD
        and primary.actual_to_zero_R_ratio < ATTENUATION_RATIO_THRESHOLD
    )
    if structural_primary:
        dominant_limit = "structural_psm_blindness"
        next_action = "require_new_operator_or_proxy_evidence_before_external_truth"
    elif attenuation_primary:
        dominant_limit = "residual_standardization_attenuation"
        next_action = "freeze_mode_aware_residual_model_modern_oof"
    else:
        dominant_limit = "mixed_or_moderate_mode_attenuation"
        next_action = "release_instrumental_truth_twin_without_parameter_tuning"

    hma_modes = mode_frame[(mode_frame.region == "hma") & (mode_frame.network == "joint_6_9")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    axes[0, 0].plot(hma_modes["mode"], hma_modes["cumulative_prior_variance_fraction"], color="#0072B2", linewidth=2)
    axes[0, 0].set(xlim=(1, 60), ylim=(0, 1.01), xlabel="HMA prior mode", ylabel="cumulative prior variance", title="Prior anomaly spectrum")
    axes[0, 1].plot(hma_modes["mode"], hma_modes["actual_R_posterior_variance_reduction"], color="#D55E00", label="archived R")
    axes[0, 1].plot(hma_modes["mode"], hma_modes["zero_R_structural_variance_reduction"], color="#009E73", label="R->0 bound")
    axes[0, 1].axhline(BLIND_THRESHOLD, color="#777777", linestyle="--", linewidth=1)
    axes[0, 1].set(xlim=(1, 60), ylim=(0, 1.01), xlabel="HMA prior mode", ylabel="posterior variance reduction", title="Joint-network mode observability")
    axes[0, 1].legend(frameon=False)
    networks = ["6", "9", "joint_6_9"]
    labels = ["proxy6", "proxy9", "joint"]
    hma_summary = summary_frame[summary_frame.region == "hma"].set_index("network")
    x = np.arange(len(networks))
    axes[1, 0].bar(x - 0.18, [hma_summary.loc[n].prior_variance_weighted_actual_R_reduction for n in networks], width=0.36, color="#D55E00", label="archived R")
    axes[1, 0].bar(x + 0.18, [hma_summary.loc[n].prior_variance_weighted_zero_R_reduction for n in networks], width=0.36, color="#009E73", label="R->0 bound")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set(ylim=(0, 1.01), ylabel="prior-variance-weighted reduction", title="Network information")
    axes[1, 0].legend(frameon=False)
    hma_truth = truth_frame[(truth_frame.region == "hma") & (truth_frame.network == "joint_6_9")]
    axes[1, 1].boxplot(
        [hma_truth.actual_R_observability_weighted_truth_energy, hma_truth.zero_R_observability_weighted_truth_energy, hma_truth.structurally_blind_truth_energy_fraction],
        labels=["archived R", "R->0 bound", "blind energy"], patch_artist=True,
        boxprops={"facecolor": "#56B4E9", "alpha": 0.75},
    )
    axes[1, 1].set(ylim=(0, 1.01), ylabel="held-out truth energy fraction", title="24 held-out HMA truths")
    for axis in axes.flat:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "prior_mode_psm_observability.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path, before[name].get("sha256")) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "truth_member_count": len(truth_members),
        "thresholds": {
            "structurally_blind_mode_reduction_lt": BLIND_THRESHOLD,
            "structural_primary_zero_R_weighted_reduction_lt": STRUCTURAL_THRESHOLD,
            "attenuation_primary_actual_to_zero_R_ratio_lt": ATTENUATION_RATIO_THRESHOLD,
        },
        "hma_joint_network": primary.to_dict(),
        "hma_joint_network_heldout_truth": primary_truth.to_dict(),
        "dominant_limit": dominant_limit,
        "next_action": next_action,
        "actual_proxy_observations_used": False,
        "instrumental_values_used": False,
        "locked_1901_1939_accessed": False,
        "jionglacuo_opened": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = [
        "# Round 163 Temperature Prior-Mode PSM Observability", "",
        f"- HMA joint archived-R prior-weighted variance reduction: `{primary.prior_variance_weighted_actual_R_reduction:.3f}`.",
        f"- HMA joint R->0 structural bound: `{primary.prior_variance_weighted_zero_R_reduction:.3f}`.",
        f"- Archived-R / structural-bound ratio: `{primary.actual_to_zero_R_ratio:.3f}`.",
        f"- Structurally blind HMA prior variance: `{primary.structurally_blind_prior_variance_fraction:.3f}`.",
        f"- Median held-out truth energy visible with archived R / R->0: `{primary_truth.median_actual_R_observability_weighted_truth_energy:.3f}` / `{primary_truth.median_zero_R_observability_weighted_truth_energy:.3f}`.",
        f"- Dominant limit: `{dominant_limit}`.",
        f"- Next action: `{next_action}`.",
        "- No scalar gain, R scale, proxy subset, or production reconstruction was fitted.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
