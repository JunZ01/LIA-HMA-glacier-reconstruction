"""Decompose prior rank, operator visibility, and update amplitude in the Round 161 twin."""

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
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


EXPERIMENT_ID = "round_162_temperature_twin_information_geometry_v1"
PRACTICAL_ENERGY = 0.99
RANK_ENERGIES = (0.50, 0.80, 0.90, 0.95, 0.99)


def fingerprint(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    if digest is None and stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def weighted_basis(matrix: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    selected = np.isfinite(weights) & (weights > 0)
    root_weight = np.sqrt(weights[selected] / np.mean(weights[selected]))
    weighted = root_weight[:, None] * matrix[selected]
    u, singular, _ = np.linalg.svd(weighted, full_matrices=False)
    if singular.size == 0 or singular[0] == 0:
        raise RuntimeError("empty-weighted-basis")
    tolerance = singular[0] * max(weighted.shape) * np.finfo(float).eps
    numerical_rank = int(np.sum(singular > tolerance))
    energy = singular**2
    energy = energy / np.sum(energy)
    cumulative = np.cumsum(energy)
    rank_at = {
        level: int(np.searchsorted(cumulative, level, side="left") + 1)
        for level in RANK_ENERGIES
    }
    effective_rank = float(1.0 / np.sum(energy**2))
    return {
        "selected": selected,
        "root_weight": root_weight,
        "u": u,
        "singular": singular,
        "cumulative": cumulative,
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "rank_at": rank_at,
    }


def weighted_vector(values: np.ndarray, basis: dict[str, object]) -> np.ndarray:
    return basis["root_weight"] * values[basis["selected"]]


def projection_skill(vector: np.ndarray, u: np.ndarray, rank: int) -> tuple[float, float]:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return float("nan"), float("nan")
    current = u[:, :rank]
    projected = current @ (current.T @ vector)
    residual = vector - projected
    skill = 1.0 - float(np.linalg.norm(residual)) / norm
    resolved_variance = 1.0 - float(np.dot(residual, residual)) / float(np.dot(vector, vector))
    return skill, resolved_variance


def update_geometry(truth_error: np.ndarray, increment: np.ndarray) -> dict[str, float]:
    truth_norm = float(np.linalg.norm(truth_error))
    update_norm = float(np.linalg.norm(increment))
    if truth_norm == 0:
        raise RuntimeError("zero-truth-error")
    actual_skill = 1.0 - float(np.linalg.norm(truth_error - increment)) / truth_norm
    if update_norm == 0:
        return {
            "actual_state_skill": actual_skill,
            "update_truth_norm_ratio": 0.0,
            "update_truth_cosine": float("nan"),
            "oracle_scalar": float("nan"),
            "oracle_scalar_state_skill": 0.0,
        }
    dot = float(np.dot(truth_error, increment))
    alpha = dot / float(np.dot(increment, increment))
    scaled_residual = truth_error - alpha * increment
    return {
        "actual_state_skill": actual_skill,
        "update_truth_norm_ratio": update_norm / truth_norm,
        "update_truth_cosine": dot / (truth_norm * update_norm),
        "oracle_scalar": alpha,
        "oracle_scalar_state_skill": 1.0 - float(np.linalg.norm(scaled_residual)) / truth_norm,
    }


def operator_cross_covariance(
    prior_anomaly: np.ndarray,
    sample: np.ndarray,
    truth_member: int,
    operator: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    hxb = operator["hxb"]
    variance = operator["variance"]
    active = (
        np.isfinite(hxb[:, truth_member])
        & np.isfinite(variance)
        & (variance > 0)
        & np.all(np.isfinite(hxb[:, sample]), axis=1)
    )
    modeled = hxb[active][:, sample]
    proxy_anomaly = modeled - np.mean(modeled, axis=1, keepdims=True)
    standardized = proxy_anomaly / np.sqrt(variance[active])[:, None]
    covariance = prior_anomaly @ standardized.T / (len(sample) - 1.0)
    return covariance, int(active.sum())


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_improvement_loop_v2" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round162_temperature_twin_information_geometry_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round161_driver": driver / "run_round161_temperature_perfect_model_twin_v2.py",
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
    before = {
        name: fingerprint(path, recovered["prior"].get("sha256") if name == "prior" else None)
        for name, path in paths.items()
    }

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    sample = pools[0]
    complement = np.setdiff1d(np.arange(prior.shape[1]), sample)
    truth_members = complement[np.linspace(0, len(complement) - 1, round161.TRUTH_COUNT, dtype=int)]
    if prior.shape != (336, 848) or pools.shape != (50, 500):
        raise RuntimeError("prior-pool-contract")
    if np.intersect1d(truth_members, sample).size:
        raise RuntimeError("truth-pool-overlap")

    operators = {
        network: round161.load_full_operator(
            paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"]
        )
        for network in round161.NETWORKS
    }
    if any(not np.array_equal(operator["sample_indices"], sample) for operator in operators.values()):
        raise RuntimeError("operator-sample-index-closure")

    with np.load(paths["native_grid"]) as native:
        grid_lat = np.asarray(native["grid_lat"], dtype=float)
        grid_lon = np.asarray(native["grid_lon"], dtype=float)
        mask = np.asarray(native["hma_mask"], dtype=bool)
    weights = common.region_weights(grid_lat, grid_lon, mask)
    background_ensemble = prior[:, sample]
    background_mean = np.mean(background_ensemble, axis=1)
    prior_anomaly = background_ensemble - background_mean[:, None]

    bases = {region: weighted_basis(prior_anomaly, current) for region, current in weights.items()}
    rank_rows = []
    for region, basis in bases.items():
        row = {
            "region": region,
            "state_cell_count": int(np.sum(basis["selected"])),
            "ensemble_member_count": int(len(sample)),
            "numerical_rank": int(basis["numerical_rank"]),
            "effective_rank_participation": float(basis["effective_rank"]),
        }
        row.update({f"rank_at_{int(level * 100)}pct": int(basis["rank_at"][level]) for level in RANK_ENERGIES})
        rank_rows.append(row)
    rank_frame = pd.DataFrame(rank_rows)
    rank_frame.to_csv(output / "prior_rank_metrics.csv", index=False)

    cross_covariances: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    for truth_member in truth_members:
        separate = {}
        counts = {}
        for network in round161.NETWORKS:
            covariance, count = operator_cross_covariance(prior_anomaly, sample, int(truth_member), operators[network])
            separate[str(network)] = covariance
            counts[str(network)] = count
            cross_covariances[(str(network), int(truth_member))] = (covariance, count)
        cross_covariances[("equal_6_9", int(truth_member))] = (
            np.concatenate([separate["6"], separate["9"]], axis=1),
            counts["6"] + counts["9"],
        )

    rng = np.random.default_rng(round161.SEED)
    rows = []
    for condition, repetitions in (("noise_free", 1), ("regem_noise", round161.NOISE_REPLICATES)):
        network_fields = {
            str(network): round161.reconstruct_cases(prior, sample, truth_members, operators[network], repetitions, rng)
            for network in round161.NETWORKS
        }
        combined = {
            method: np.mean([network_fields[str(network)][method] for network in round161.NETWORKS], axis=0)
            for method in ("truth", "background", "ordinary", "adaptive")
        }
        all_fields = {**network_fields, "equal_6_9": combined}
        for network, fields in all_fields.items():
            for case_index in range(fields["truth"].shape[1]):
                truth_position = case_index // repetitions
                replicate = case_index % repetitions
                truth_member = int(truth_members[truth_position])
                covariance, proxy_count = cross_covariances[(network, truth_member)]
                for region, basis in bases.items():
                    truth_error = weighted_vector(fields["truth"][:, case_index] - background_mean, basis)
                    prior_full_skill, prior_full_variance = projection_skill(
                        truth_error, basis["u"], int(basis["numerical_rank"])
                    )
                    prior_99_skill, prior_99_variance = projection_skill(
                        truth_error, basis["u"], int(basis["rank_at"][PRACTICAL_ENERGY])
                    )
                    gain_basis = weighted_basis(covariance, weights[region])
                    gain_full_skill, gain_full_variance = projection_skill(
                        truth_error, gain_basis["u"], int(gain_basis["numerical_rank"])
                    )
                    gain_99_skill, gain_99_variance = projection_skill(
                        truth_error, gain_basis["u"], int(gain_basis["rank_at"][PRACTICAL_ENERGY])
                    )
                    increment = weighted_vector(fields["ordinary"][:, case_index] - background_mean, basis)
                    geometry = update_geometry(truth_error, increment)
                    rows.append({
                        "condition": condition,
                        "network": network,
                        "region": region,
                        "case_index": case_index,
                        "truth_member_0based": truth_member,
                        "noise_replicate": replicate,
                        "active_proxy_count": proxy_count,
                        "gain_numerical_rank": int(gain_basis["numerical_rank"]),
                        "gain_effective_rank": float(gain_basis["effective_rank"]),
                        "gain_rank_at_99pct": int(gain_basis["rank_at"][PRACTICAL_ENERGY]),
                        "prior_full_oracle_skill": prior_full_skill,
                        "prior_full_resolved_variance": prior_full_variance,
                        "prior_99_oracle_skill": prior_99_skill,
                        "prior_99_resolved_variance": prior_99_variance,
                        "gain_full_oracle_skill": gain_full_skill,
                        "gain_full_resolved_variance": gain_full_variance,
                        "gain_99_oracle_skill": gain_99_skill,
                        "gain_99_resolved_variance": gain_99_variance,
                        **geometry,
                    })

    case_frame = pd.DataFrame(rows)
    case_frame.to_csv(output / "case_decomposition.csv", index=False)
    metric_columns = [
        "active_proxy_count",
        "gain_numerical_rank",
        "gain_effective_rank",
        "gain_rank_at_99pct",
        "prior_full_oracle_skill",
        "prior_full_resolved_variance",
        "prior_99_oracle_skill",
        "prior_99_resolved_variance",
        "gain_full_oracle_skill",
        "gain_full_resolved_variance",
        "gain_99_oracle_skill",
        "gain_99_resolved_variance",
        "actual_state_skill",
        "update_truth_norm_ratio",
        "update_truth_cosine",
        "oracle_scalar",
        "oracle_scalar_state_skill",
    ]
    summary_frame = case_frame.groupby(["condition", "network", "region"], as_index=False)[metric_columns].median()
    summary_frame.to_csv(output / "summary_metrics.csv", index=False)
    primary = summary_frame[
        (summary_frame.condition == "noise_free")
        & (summary_frame.network == "equal_6_9")
        & (summary_frame.region == "hma")
    ].iloc[0]

    amplitude_recoverable = bool(
        primary.update_truth_cosine >= 0.80
        and primary.update_truth_norm_ratio < 0.70
        and primary.oracle_scalar_state_skill >= 0.50
    )
    gain_limited = bool(primary.gain_99_oracle_skill < 0.50)
    prior_limited = bool(primary.prior_99_oracle_skill < 0.50)
    if prior_limited:
        dominant_limit = "practical_prior_subspace"
    elif gain_limited:
        dominant_limit = "observation_coupled_subspace"
    elif amplitude_recoverable:
        dominant_limit = "update_amplitude_attenuation"
    else:
        dominant_limit = "innovation_weighting_or_directional_mismatch"

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    hma_basis = bases["hma"]
    axes[0].plot(np.arange(1, len(hma_basis["cumulative"]) + 1), hma_basis["cumulative"], color="#0072B2", linewidth=2)
    axes[0].axhline(PRACTICAL_ENERGY, color="#777777", linestyle="--", linewidth=1)
    axes[0].set(title="HMA prior anomaly spectrum", xlabel="rank", ylabel="cumulative variance")
    labels = ["actual", "scaled actual", "gain 99%", "prior 99%", "prior full"]
    values = [
        primary.actual_state_skill,
        primary.oracle_scalar_state_skill,
        primary.gain_99_oracle_skill,
        primary.prior_99_oracle_skill,
        primary.prior_full_oracle_skill,
    ]
    axes[1].bar(np.arange(len(labels)), values, color=["#D55E00", "#E69F00", "#009E73", "#56B4E9", "#0072B2"])
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    axes[1].set(title="Noise-free HMA recovery bounds", ylabel="state RMSE skill")
    primary_cases = case_frame[
        (case_frame.condition == "noise_free")
        & (case_frame.network == "equal_6_9")
        & (case_frame.region == "hma")
    ]
    axes[2].boxplot(
        [primary_cases.update_truth_norm_ratio, primary_cases.update_truth_cosine, primary_cases.oracle_scalar],
        labels=["norm ratio", "cosine", "oracle scale"],
        patch_artist=True,
        boxprops={"facecolor": "#56B4E9", "alpha": 0.75},
    )
    axes[2].set(title="Actual increment geometry", ylabel="dimensionless")
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "information_geometry.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path, before[name].get("sha256")) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "truth_member_count": int(len(truth_members)),
        "truth_members_0based": truth_members.tolist(),
        "seed": round161.SEED,
        "primary_noise_free_equal_network_hma": {
            column: float(primary[column]) for column in metric_columns
        },
        "classification": {
            "dominant_limit": dominant_limit,
            "practical_prior_limited": prior_limited,
            "practical_gain_limited": gain_limited,
            "amplitude_recoverable": amplitude_recoverable,
        },
        "actual_proxy_observations_used": False,
        "instrumental_values_used": False,
        "locked_1921_1939_accessed": False,
        "jionglacuo_opened": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Round 162 Temperature Twin Information Geometry",
        "",
        f"- Actual noise-free equal-network HMA state skill: `{primary.actual_state_skill:.3f}`.",
        f"- Actual increment norm/truth norm: `{primary.update_truth_norm_ratio:.3f}`; cosine alignment: `{primary.update_truth_cosine:.3f}`.",
        f"- Oracle scalar: `{primary.oracle_scalar:.3f}`; scalar-rescaled skill: `{primary.oracle_scalar_state_skill:.3f}`.",
        f"- Observation-coupled 99% subspace oracle skill: `{primary.gain_99_oracle_skill:.3f}`.",
        f"- Prior 99% subspace oracle skill: `{primary.prior_99_oracle_skill:.3f}`; full prior-span oracle skill: `{primary.prior_full_oracle_skill:.3f}`.",
        f"- Dominant limitation classification: `{dominant_limit}`.",
        "- This is a perfect-model mechanism diagnostic; it does not authorize instrumental tuning or production reconstruction.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
