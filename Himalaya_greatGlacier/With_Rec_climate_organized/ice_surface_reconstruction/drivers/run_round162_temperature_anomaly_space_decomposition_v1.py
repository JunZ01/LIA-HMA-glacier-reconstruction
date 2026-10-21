"""Decompose perfect-model temperature recovery in weighted anomaly space."""

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

import run_round161_temperature_perfect_model_twin_v2 as twin
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


NETWORKS = (6, 9)
RANK_TOLERANCE = 1e-10
SEED = twin.SEED


def fingerprint(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    if digest is None and stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def weighted_vectors(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return values * np.sqrt(weights)[:, None]


def vector_metrics(estimate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    target_energy = float(target @ target)
    estimate_energy = float(estimate @ estimate)
    if target_energy <= 0:
        return {"slope": np.nan, "norm_ratio": np.nan, "cosine": np.nan, "residual_ratio": np.nan}
    dot = float(estimate @ target)
    return {
        "slope": dot / target_energy,
        "norm_ratio": np.sqrt(estimate_energy / target_energy),
        "cosine": dot / np.sqrt(estimate_energy * target_energy) if estimate_energy > 0 else np.nan,
        "residual_ratio": float(np.linalg.norm(estimate - target) / np.sqrt(target_energy)),
    }


def region_basis(pool_anomaly: np.ndarray, weights: np.ndarray):
    weighted_pool = weighted_vectors(pool_anomaly, weights)
    u, singular, _ = np.linalg.svd(weighted_pool, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * RANK_TOLERANCE)) if singular.size and singular[0] > 0 else 0
    eigenvalues = singular[:rank] ** 2 / max(pool_anomaly.shape[1] - 1, 1)
    total = float(eigenvalues.sum())
    effective_rank = float(total**2 / np.sum(eigenvalues**2)) if total > 0 else 0.0
    cumulative = np.cumsum(eigenvalues) / total if total > 0 else np.zeros_like(eigenvalues)
    return u[:, :rank], eigenvalues, cumulative, rank, effective_rank


def zero_r_analysis(prior: np.ndarray, sample: np.ndarray, truth_member: int, operator: dict[str, np.ndarray]):
    hxb = operator["hxb"]
    variance = operator["variance"]
    active = np.isfinite(hxb[:, truth_member]) & np.isfinite(variance) & (variance > 0) & np.all(np.isfinite(hxb[:, sample]), axis=1)
    modeled = hxb[active][:, sample]
    innovation = hxb[active, truth_member] - np.mean(modeled, axis=1)
    proxy_anomaly = modeled - np.mean(modeled, axis=1, keepdims=True)
    coefficients, _, _, _ = np.linalg.lstsq(proxy_anomaly, innovation, rcond=RANK_TOLERANCE)
    background = prior[:, sample]
    analysis = np.mean(background, axis=1) + (background - np.mean(background, axis=1, keepdims=True)) @ coefficients
    fitted = proxy_anomaly @ coefficients
    denominator = float(innovation @ innovation)
    represented = 1.0 - float((innovation - fitted) @ (innovation - fitted)) / denominator if denominator > 0 else np.nan
    return analysis, int(active.sum()), represented


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_improvement_loop_v2/round_162_temperature_anomaly_space_decomposition_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round162_temperature_anomaly_space_decomposition_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    round161 = validation / "climate_reconstruction_improvement_loop_v2/round_161_temperature_perfect_model_twin_v2"
    manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round161_summary": round161 / "summary.json",
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
    round161_summary = json.loads(paths["round161_summary"].read_text(encoding="utf-8"))
    truth_members = np.asarray(round161_summary["truth_members_0based"], dtype=int)
    if prior.shape != (336, 848) or sample.shape != (500,) or truth_members.shape != (24,):
        raise RuntimeError("frozen-dimension-contract")
    if np.intersect1d(sample, truth_members).size:
        raise RuntimeError("truth-pool-overlap")

    operators = {
        network: twin.load_full_operator(paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"])
        for network in NETWORKS
    }
    if any(not np.array_equal(operator["sample_indices"], sample) for operator in operators.values()):
        raise RuntimeError("operator-sample-index-closure")

    with np.load(paths["native_grid"]) as native:
        grid_lat = np.asarray(native["grid_lat"], dtype=float)
        grid_lon = np.asarray(native["grid_lon"], dtype=float)
        mask = np.asarray(native["hma_mask"], dtype=bool)
    if mask.size != prior.shape[0]:
        raise RuntimeError("native-grid-contract")
    weights = common.region_weights(grid_lat, grid_lon, mask)
    background = prior[:, sample]
    background_mean = np.mean(background, axis=1)
    pool_anomaly = background - background_mean[:, None]

    spectrum_rows = []
    bases = {}
    for region, current_weights in weights.items():
        basis, eigenvalues, cumulative, rank, effective_rank = region_basis(pool_anomaly, current_weights)
        bases[region] = basis
        for mode in range(1, rank + 1):
            spectrum_rows.append({
                "region": region,
                "mode": mode,
                "eigenvalue": float(eigenvalues[mode - 1]),
                "cumulative_variance_fraction": float(cumulative[mode - 1]),
                "numerical_rank": rank,
                "participation_effective_rank": effective_rank,
            })
    spectrum_frame = pd.DataFrame(spectrum_rows)
    spectrum_frame.to_csv(output / "prior_spectrum.csv", index=False)

    rng = np.random.default_rng(SEED)
    network_fields = {
        str(network): twin.reconstruct_cases(prior, sample, truth_members, operators[network], 1, rng)
        for network in NETWORKS
    }
    combined = {
        method: np.mean([network_fields[str(network)][method] for network in NETWORKS], axis=0)
        for method in ("truth", "background", "ordinary", "adaptive")
    }
    zero_fields = {str(network): np.empty((prior.shape[0], len(truth_members))) for network in NETWORKS}
    proxy_rows = []
    for network in NETWORKS:
        for case_index, truth_member in enumerate(truth_members):
            analysis, active_count, innovation_fraction = zero_r_analysis(prior, sample, int(truth_member), operators[network])
            zero_fields[str(network)][:, case_index] = analysis
            proxy_rows.append({"network": str(network), "case_index": case_index, "truth_member_0based": int(truth_member), "active_proxy_count": active_count, "innovation_representable_fraction": innovation_fraction})
    zero_fields["equal_6_9"] = np.mean([zero_fields[str(network)] for network in NETWORKS], axis=0)
    pd.DataFrame(proxy_rows).to_csv(output / "proxy_innovation_decomposition.csv", index=False)

    truth_rows = []
    projected_truth = {region: np.empty((prior.shape[0], len(truth_members))) for region in weights}
    for region, current_weights in weights.items():
        sqrt_weights = np.sqrt(current_weights)
        basis = bases[region]
        for case_index, truth_member in enumerate(truth_members):
            truth_anomaly = prior[:, truth_member] - background_mean
            weighted_truth = sqrt_weights * truth_anomaly
            weighted_projection = basis @ (basis.T @ weighted_truth)
            valid = sqrt_weights > 0
            projected = np.zeros_like(truth_anomaly)
            projected[valid] = weighted_projection[valid] / sqrt_weights[valid]
            projected_truth[region][:, case_index] = projected
            energy = float(weighted_truth @ weighted_truth)
            residual_energy = float((weighted_truth - weighted_projection) @ (weighted_truth - weighted_projection))
            truth_rows.append({
                "region": region,
                "case_index": case_index,
                "truth_member_0based": int(truth_member),
                "representable_variance_fraction": 1.0 - residual_energy / energy if energy > 0 else np.nan,
                "orthogonal_variance_fraction": residual_energy / energy if energy > 0 else np.nan,
            })
    truth_frame = pd.DataFrame(truth_rows)
    truth_frame.to_csv(output / "truth_prior_span_decomposition.csv", index=False)

    method_rows = []
    all_fields = {**network_fields, "equal_6_9": combined}
    for network, fields in all_fields.items():
        for region, current_weights in weights.items():
            sqrt_weights = np.sqrt(current_weights)
            for case_index, truth_member in enumerate(truth_members):
                truth_anomaly = prior[:, truth_member] - background_mean
                weighted_truth = sqrt_weights * truth_anomaly
                weighted_projected = sqrt_weights * projected_truth[region][:, case_index]
                estimates = {
                    "background": background_mean,
                    "ordinary": fields["ordinary"][:, case_index],
                    "adaptive": fields["adaptive"][:, case_index],
                    "zero_R_limit": zero_fields[network][:, case_index],
                }
                for method, estimate in estimates.items():
                    weighted_estimate = sqrt_weights * (estimate - background_mean)
                    full = vector_metrics(weighted_estimate, weighted_truth)
                    projected = vector_metrics(weighted_estimate, weighted_projected)
                    method_rows.append({
                        "network": network,
                        "region": region,
                        "case_index": case_index,
                        "truth_member_0based": int(truth_member),
                        "method": method,
                        **{f"truth_{key}": value for key, value in full.items()},
                        **{f"representable_{key}": value for key, value in projected.items()},
                    })
    method_frame = pd.DataFrame(method_rows)
    method_frame.to_csv(output / "method_anomaly_decomposition.csv", index=False)
    summary_frame = method_frame.groupby(["network", "region", "method"], as_index=False).agg(
        median_truth_slope=("truth_slope", "median"),
        median_truth_norm_ratio=("truth_norm_ratio", "median"),
        median_truth_cosine=("truth_cosine", "median"),
        median_truth_residual_ratio=("truth_residual_ratio", "median"),
        median_representable_slope=("representable_slope", "median"),
        median_representable_residual_ratio=("representable_residual_ratio", "median"),
    )
    summary_frame.to_csv(output / "method_summary.csv", index=False)

    def metric(network: str, region: str, method: str, column: str) -> float:
        row = summary_frame[(summary_frame.network == network) & (summary_frame.region == region) & (summary_frame.method == method)]
        return float(row.iloc[0][column])

    hma_repr = float(truth_frame[truth_frame.region == "hma"].representable_variance_fraction.median())
    ordinary_slope = metric("equal_6_9", "hma", "ordinary", "median_truth_slope")
    adaptive_slope = metric("equal_6_9", "hma", "adaptive", "median_truth_slope")
    zero_slope = metric("equal_6_9", "hma", "zero_R_limit", "median_truth_slope")
    ordinary_residual = metric("equal_6_9", "hma", "ordinary", "median_truth_residual_ratio")
    adaptive_residual = metric("equal_6_9", "hma", "adaptive", "median_truth_residual_ratio")
    zero_residual = metric("equal_6_9", "hma", "zero_R_limit", "median_truth_residual_ratio")
    mechanisms = {
        "prior_span_material": hma_repr < 0.75,
        "operator_observability_material": hma_repr >= 0.75 and zero_residual > 0.50,
        "regem_R_attenuation_material": (zero_slope - ordinary_slope) >= 0.20 and zero_residual < ordinary_residual,
        "adaptive_selection_attenuation_material": adaptive_slope < ordinary_slope and adaptive_residual > ordinary_residual,
    }
    if mechanisms["prior_span_material"]:
        next_action = "test_prior_span_diversification_before_external_truth"
    elif mechanisms["operator_observability_material"]:
        next_action = "audit_psm_operator_observable_modes_before_external_truth"
    elif mechanisms["regem_R_attenuation_material"]:
        next_action = "run_predeclared_R_calibration_twin_before_external_truth"
    else:
        next_action = "release_instrumental_truth_twin"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"hma": "#111111", "west": "#0072B2", "central": "#009E73", "east": "#D55E00"}
    for region in weights:
        current = spectrum_frame[spectrum_frame.region == region]
        axes[0, 0].plot(current["mode"], current["cumulative_variance_fraction"], label=region, color=colors.get(region), linewidth=1.8)
    axes[0, 0].set(xlim=(1, 60), ylim=(0, 1.01), xlabel="prior mode", ylabel="cumulative variance", title="Weighted prior spectrum")
    axes[0, 0].legend(frameon=False)
    hma_truth = truth_frame[truth_frame.region == "hma"]
    axes[0, 1].plot(np.arange(1, len(hma_truth) + 1), hma_truth.representable_variance_fraction, marker="o", color="#0072B2")
    axes[0, 1].axhline(hma_repr, color="#D55E00", linestyle="--", label=f"median {hma_repr:.2f}")
    axes[0, 1].set(xlabel="held-out truth case", ylabel="representable variance fraction", ylim=(0, 1.01), title="HMA prior-span representation")
    axes[0, 1].legend(frameon=False)
    methods = ["background", "ordinary", "adaptive", "zero_R_limit"]
    labels = ["background", "ordinary", "adaptive", "R->0 limit"]
    hma_methods = method_frame[(method_frame.network == "equal_6_9") & (method_frame.region == "hma")]
    axes[1, 0].boxplot([hma_methods[hma_methods.method == method].truth_slope.dropna() for method in methods], labels=labels, patch_artist=True)
    axes[1, 0].axhline(1, color="#777777", linestyle="--")
    axes[1, 0].set(ylabel="anomaly amplitude slope", title="Equal-network HMA amplitude")
    axes[1, 1].boxplot([hma_methods[hma_methods.method == method].truth_residual_ratio.dropna() for method in methods], labels=labels, patch_artist=True)
    axes[1, 1].set(ylabel="residual norm / truth norm", title="Equal-network HMA residual")
    for axis in axes.flat:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "anomaly_space_decomposition.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path, before[name].get("sha256")) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_162_temperature_anomaly_space_decomposition_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "truth_member_count": len(truth_members),
        "rank_tolerance": RANK_TOLERANCE,
        "hma_median_prior_representable_variance_fraction": hma_repr,
        "hma_equal_network": {
            "ordinary_median_amplitude_slope": ordinary_slope,
            "adaptive_median_amplitude_slope": adaptive_slope,
            "zero_R_limit_median_amplitude_slope": zero_slope,
            "ordinary_median_residual_ratio": ordinary_residual,
            "adaptive_median_residual_ratio": adaptive_residual,
            "zero_R_limit_median_residual_ratio": zero_residual,
        },
        "mechanisms": mechanisms,
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
        "# Round 162 Temperature Anomaly-Space Decomposition", "",
        f"- HMA median prior-representable truth variance: `{hma_repr:.3f}`.",
        f"- Equal-network ordinary amplitude slope/residual ratio: `{ordinary_slope:.3f}` / `{ordinary_residual:.3f}`.",
        f"- Equal-network adaptive amplitude slope/residual ratio: `{adaptive_slope:.3f}` / `{adaptive_residual:.3f}`.",
        f"- Equal-network R->0 limit amplitude slope/residual ratio: `{zero_slope:.3f}` / `{zero_residual:.3f}`.",
        f"- Mechanism flags: `{json.dumps(mechanisms, sort_keys=True)}`.",
        f"- Next action: `{next_action}`.",
        "- All diagnostics use anomalies relative to the fixed 500-member background mean.",
        "- The R->0 result is a theoretical observability bound, not a reconstruction candidate.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
