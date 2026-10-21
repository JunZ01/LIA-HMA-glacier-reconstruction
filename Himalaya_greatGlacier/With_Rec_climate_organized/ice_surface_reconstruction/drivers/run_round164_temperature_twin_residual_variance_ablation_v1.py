"""Ablate residual variance in the noise-free perfect-model temperature twin."""

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
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_round161_temperature_perfect_model_twin_v2 as round161
import run_round162_temperature_twin_information_geometry_v1 as round162
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


EXPERIMENT_ID = "round_164_temperature_twin_residual_variance_ablation_v1"
R_SCALES = (1.0, 0.25, 0.0625, 0.01, 0.0001)
MODE_COUNT = 15


def fingerprint(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    if digest is None and stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def reconstruct_all_truths(
    prior: np.ndarray,
    sample: np.ndarray,
    truth_members: np.ndarray,
    operator: dict[str, np.ndarray],
    variance_scale: float,
) -> np.ndarray:
    hxb = operator["hxb"]
    variance = operator["variance"]
    active = (
        np.isfinite(variance)
        & (variance > 0)
        & np.all(np.isfinite(hxb[:, sample]), axis=1)
        & np.all(np.isfinite(hxb[:, truth_members]), axis=1)
    )
    background = prior[:, sample]
    state_mean = np.mean(background, axis=1)
    state_anomaly = background - state_mean[:, None]
    modeled = hxb[active][:, sample]
    proxy_mean = np.mean(modeled, axis=1)
    proxy_anomaly = modeled - proxy_mean[:, None]
    inverse_variance = 1.0 / (variance_scale * variance[active])
    system = (len(sample) - 1.0) * np.eye(len(sample)) + proxy_anomaly.T @ (inverse_variance[:, None] * proxy_anomaly)
    innovations = hxb[active][:, truth_members] - proxy_mean[:, None]
    rhs = proxy_anomaly.T @ (inverse_variance[:, None] * innovations)
    factor = scipy.linalg.cho_factor(system, lower=True, check_finite=False)
    weights = scipy.linalg.cho_solve(factor, rhs, check_finite=False)
    return state_mean[:, None] + state_anomaly @ weights


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_improvement_loop_v2" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round164_temperature_twin_residual_variance_ablation_v1_contract.md"
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
        "round162_driver": driver / "run_round162_temperature_twin_information_geometry_v1.py",
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
    region_weights = common.region_weights(grid_lat, grid_lon, mask)
    background_mean = np.mean(prior[:, sample], axis=1)
    truth = prior[:, truth_members]

    hma_basis = round162.weighted_basis(prior[:, sample] - background_mean[:, None], region_weights["hma"])
    truth_weighted = hma_basis["root_weight"][:, None] * (truth[hma_basis["selected"]] - background_mean[hma_basis["selected"], None])
    truth_modes = hma_basis["u"][:, :MODE_COUNT].T @ truth_weighted

    case_rows = []
    modal_rows = []
    saved = {}
    for scale in R_SCALES:
        separate = {
            str(network): reconstruct_all_truths(prior, sample, truth_members, operators[network], scale)
            for network in round161.NETWORKS
        }
        fields = {**separate, "equal_6_9": np.mean([separate["6"], separate["9"]], axis=0)}
        saved[scale] = fields
        for network, estimate in fields.items():
            for region, weights in region_weights.items():
                baseline_rmse = np.asarray([
                    round161.weighted_rmse(background_mean, truth[:, index], weights)
                    for index in range(truth.shape[1])
                ])
                method_rmse = np.asarray([
                    round161.weighted_rmse(estimate[:, index], truth[:, index], weights)
                    for index in range(truth.shape[1])
                ])
                for index in range(truth.shape[1]):
                    case_rows.append({
                        "r_scale": scale,
                        "network": network,
                        "region": region,
                        "case_index": index,
                        "truth_member_0based": int(truth_members[index]),
                        "background_rmse_degC": baseline_rmse[index],
                        "analysis_rmse_degC": method_rmse[index],
                        "state_rmse_skill": 1.0 - method_rmse[index] / baseline_rmse[index],
                    })
            increment_weighted = hma_basis["root_weight"][:, None] * (
                estimate[hma_basis["selected"]] - background_mean[hma_basis["selected"], None]
            )
            analysis_modes = hma_basis["u"][:, :MODE_COUNT].T @ increment_weighted
            for mode in range(MODE_COUNT):
                target = truth_modes[mode]
                predicted = analysis_modes[mode]
                denominator = float(np.dot(target, target))
                slope = float(np.dot(target, predicted) / denominator)
                target_rmse = float(np.sqrt(np.mean(target**2)))
                transfer_rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
                modal_rows.append({
                    "r_scale": scale,
                    "network": network,
                    "mode": mode + 1,
                    "transfer_slope": slope,
                    "transfer_correlation": common.correlation(predicted, target),
                    "modal_rmse_skill": 1.0 - transfer_rmse / target_rmse,
                })

    case_frame = pd.DataFrame(case_rows)
    modal_frame = pd.DataFrame(modal_rows)
    case_frame.to_csv(output / "case_state_skill.csv", index=False)
    modal_frame.to_csv(output / "modal_transfer.csv", index=False)
    summary_frame = case_frame.groupby(["r_scale", "network", "region"], as_index=False).agg(
        median_background_rmse_degC=("background_rmse_degC", "median"),
        median_analysis_rmse_degC=("analysis_rmse_degC", "median"),
        median_case_state_skill=("state_rmse_skill", "median"),
    )
    summary_frame["ratio_of_median_rmse_skill"] = 1.0 - (
        summary_frame.median_analysis_rmse_degC / summary_frame.median_background_rmse_degC
    )
    summary_frame.to_csv(output / "summary_state_skill.csv", index=False)
    primary = summary_frame[(summary_frame.network == "equal_6_9") & (summary_frame.region == "hma")].sort_values("r_scale", ascending=False)
    archived = primary[primary.r_scale == 1.0].iloc[0]
    near_zero = primary[primary.r_scale == min(R_SCALES)].iloc[0]
    crosses = primary[primary.ratio_of_median_rmse_skill >= 0.50]
    crosses_gate = not crosses.empty
    largest_scale_crossing = float(crosses.r_scale.max()) if crosses_gate else None
    if crosses_gate:
        decision = "residual_variance_shrinkage_sufficient_in_noise_free_twin"
    else:
        decision = "operator_directionality_remains_below_gate_at_near_zero_R"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for network, color in (("6", "#0072B2"), ("9", "#D55E00"), ("equal_6_9", "#009E73")):
        current = summary_frame[(summary_frame.network == network) & (summary_frame.region == "hma")].sort_values("r_scale")
        axes[0].plot(current.r_scale, current.ratio_of_median_rmse_skill, marker="o", color=color, label=network)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].axhline(0.50, color="#777777", linestyle="--", linewidth=1)
    axes[0].set(title="Noise-free HMA state recovery", xlabel="residual-variance scale", ylabel="RMSE skill")
    axes[0].legend(frameon=False, title="network")
    equal_modes = modal_frame[modal_frame.network == "equal_6_9"].pivot(index="mode", columns="r_scale", values="transfer_slope")
    image = axes[1].imshow(equal_modes.to_numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(equal_modes.columns)), [f"{value:g}" for value in equal_modes.columns], rotation=25)
    axes[1].set_yticks(np.arange(MODE_COUNT), np.arange(1, MODE_COUNT + 1))
    axes[1].set(title="Equal-network modal transfer slope", xlabel="residual-variance scale", ylabel="prior mode")
    fig.colorbar(image, ax=axes[1], label="analysis/truth slope")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "residual_variance_ablation.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path, before[name].get("sha256")) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "r_scales": list(R_SCALES),
        "archived_R_equal_network_hma_ratio_of_median_rmse_skill": float(archived.ratio_of_median_rmse_skill),
        "near_zero_R_equal_network_hma_ratio_of_median_rmse_skill": float(near_zero.ratio_of_median_rmse_skill),
        "crosses_0_50_gate": crosses_gate,
        "largest_r_scale_crossing_gate": largest_scale_crossing,
        "decision": decision,
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
        "# Round 164 Temperature Twin Residual-Variance Ablation",
        "",
        f"- Archived-R equal-network HMA state skill: `{archived.ratio_of_median_rmse_skill:.3f}`.",
        f"- Near-zero-R equal-network HMA state skill: `{near_zero.ratio_of_median_rmse_skill:.3f}`.",
        f"- Crosses the frozen 50% gate: `{crosses_gate}`; largest tested scale crossing: `{largest_scale_crossing}`.",
        f"- Decision: `{decision}`.",
        "- The R scales are a noise-free mechanism ablation, not production candidates.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
