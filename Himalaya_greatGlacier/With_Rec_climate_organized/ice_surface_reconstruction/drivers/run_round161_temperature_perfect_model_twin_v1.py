"""Perfect-model temperature twin experiment with held-out prior members."""

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
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_round154_temperature_adaptive_background_spatial_balance_oof_v1 as round154
import run_round155_temperature_adaptive_vs_ordinary_posterior_oof_v3 as round155
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


TRUTH_COUNT = 24
NOISE_REPLICATES = 5
SEED = 16120260820
NETWORKS = (6, 9)
REGIONS = tuple(common.REGIONS)


def fingerprint(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    if digest is None and stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def load_full_operator(operator_path: Path, predictor_path: Path, allowed_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(operator_path, "r") as handle:
        coefficients = np.asarray(handle["b"], dtype=float).transpose(2, 1, 0)
        training_count = np.asarray(handle["training_count"], dtype=float).T
        branch_outer = np.asarray(handle["branch_outer"], dtype=int).reshape(-1)
        branch_inner = np.asarray(handle["branch_inner"], dtype=int).reshape(-1)
        residual_variance = np.asarray(handle["residual_variance"], dtype=float).T
        sample_indices = np.asarray(handle["sample_indices"], dtype=int).reshape(-1) - 1
    branch = np.flatnonzero((branch_outer == 0) & (branch_inner == 0))
    if len(branch) != 1:
        raise RuntimeError(f"full-branch-not-unique:{branch.tolist()}")
    branch = int(branch[0])
    predictor = scipy.io.loadmat(predictor_path, variable_names=["model_t_848", "model_p_848"])
    allowed = scipy.io.loadmat(allowed_path, variable_names=["model_type"])
    model_t = np.asarray(predictor["model_t_848"], dtype=float)
    model_p = np.asarray(predictor["model_p_848"], dtype=float)
    model_type = np.asarray(allowed["model_type"], dtype=int).reshape(-1)
    coefficient = coefficients[:, branch, :]
    hxb = coefficient[:, 0, None] + coefficient[:, 1, None] * model_t
    bivariate = model_type == 2
    hxb[bivariate] += coefficient[bivariate, 2, None] * model_p[bivariate]
    hxb[training_count[:, branch] < 20] = np.nan
    variance = residual_variance[:, branch]
    if hxb.shape[1] != 848 or hxb.shape[0] != len(variance):
        raise RuntimeError(f"operator-shape:{hxb.shape}:{variance.shape}")
    return {"hxb": hxb, "variance": variance, "sample_indices": sample_indices}


def pattern_correlation(estimate: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    valid = (weights > 0) & np.isfinite(estimate) & np.isfinite(truth)
    if int(valid.sum()) < 3:
        return float("nan")
    current_weights = weights[valid]
    left = estimate[valid] - np.average(estimate[valid], weights=current_weights)
    right = truth[valid] - np.average(truth[valid], weights=current_weights)
    return common.correlation(left, right)


def weighted_rmse(estimate: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    valid = (weights > 0) & np.isfinite(estimate) & np.isfinite(truth)
    return float(np.sqrt(np.average((estimate[valid] - truth[valid]) ** 2, weights=weights[valid])))


def reconstruct_cases(
    prior: np.ndarray,
    sample: np.ndarray,
    truth_members: np.ndarray,
    operator: dict[str, np.ndarray],
    noise_replicates: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    hxb = operator["hxb"]
    variance = operator["variance"]
    cases = len(truth_members) * noise_replicates
    result = {
        "truth": np.empty((prior.shape[0], cases)),
        "background": np.empty((prior.shape[0], cases)),
        "ordinary": np.empty((prior.shape[0], cases)),
        "adaptive": np.empty((prior.shape[0], cases)),
    }
    background = prior[:, sample]
    background_mean = np.mean(background, axis=1)
    column = 0
    for truth_member in truth_members:
        active = np.isfinite(hxb[:, truth_member]) & np.isfinite(variance) & (variance > 0) & np.all(np.isfinite(hxb[:, sample]), axis=1)
        if int(active.sum()) < 3:
            raise RuntimeError(f"active-proxy-count:{truth_member}:{int(active.sum())}")
        modeled = hxb[active][:, sample]
        for _ in range(noise_replicates):
            noise = np.zeros(int(active.sum())) if noise_replicates == 1 else rng.normal(0.0, np.sqrt(variance[active]))
            observation = hxb[active, truth_member] + noise
            ordinary = round155.ensemble_update_mean(background, modeled, observation, variance[active])
            scores = round154.correlation_score(observation, modeled)
            local = np.argsort(-scores, kind="stable")[:100]
            selected = sample[local]
            adaptive = round155.ensemble_update_mean(prior[:, selected], hxb[active][:, selected], observation, variance[active])
            result["truth"][:, column] = prior[:, truth_member]
            result["background"][:, column] = background_mean
            result["ordinary"][:, column] = ordinary
            result["adaptive"][:, column] = adaptive
            column += 1
    return result


def score(condition: str, network: str, fields: dict[str, np.ndarray], weights: dict[str, np.ndarray]):
    case_rows = []
    summary_rows = []
    truth = fields["truth"]
    for region, current_weights in weights.items():
        truth_region = common.weighted_series(truth, current_weights)
        baseline_rmse = np.asarray([
            weighted_rmse(fields["background"][:, index], truth[:, index], current_weights)
            for index in range(truth.shape[1])
        ])
        for method in ("background", "ordinary", "adaptive"):
            estimate = fields[method]
            method_rmse = np.asarray([
                weighted_rmse(estimate[:, index], truth[:, index], current_weights)
                for index in range(truth.shape[1])
            ])
            patterns = np.asarray([
                pattern_correlation(estimate[:, index], truth[:, index], current_weights)
                for index in range(truth.shape[1])
            ])
            estimate_region = common.weighted_series(estimate, current_weights)
            for index, (rmse, pattern) in enumerate(zip(method_rmse, patterns)):
                case_rows.append({
                    "condition": condition,
                    "network": network,
                    "region": region,
                    "method": method,
                    "case_index": index,
                    "state_rmse_degC": rmse,
                    "spatial_pattern_correlation": pattern,
                })
            summary_rows.append({
                "condition": condition,
                "network": network,
                "region": region,
                "method": method,
                "median_state_rmse_degC": float(np.median(method_rmse)),
                "median_state_rmse_skill_vs_background": float(1.0 - np.median(method_rmse) / np.median(baseline_rmse)),
                "median_spatial_pattern_correlation": float(np.nanmedian(patterns)),
                "regional_value_rmse_degC": common.rmse(estimate_region, truth_region),
                "regional_value_correlation": common.correlation(estimate_region, truth_region),
            })
    return case_rows, summary_rows


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_improvement_loop_v2/round_161_temperature_perfect_model_twin_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round161_temperature_perfect_model_twin_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "fold_manifest": fold_root / "source_hash_manifest.json",
        "operator6": fold_root / "proxy6_fold_native_operators_500.mat",
        "operator9": fold_root / "proxy9_fold_native_operators_500.mat",
        "predictor6": fold_root / "proxy6_archived_model_predictors_848.mat",
        "predictor9": fold_root / "proxy9_archived_model_predictors_848.mat",
        "allowed6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "allowed9": loop / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "hma_mask": Path(recovered["hma_mask"]["path"]),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path, recovered["prior"].get("sha256") if name == "prior" else None) for name, path in paths.items()}

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    sample = pools[0]
    if prior.shape != (336, 848) or pools.shape != (50, 500) or len(np.unique(sample)) != 500:
        raise RuntimeError("prior-pool-contract")
    complement = np.setdiff1d(np.arange(prior.shape[1]), sample)
    truth_members = complement[np.linspace(0, len(complement) - 1, TRUTH_COUNT, dtype=int)]
    if len(np.unique(truth_members)) != TRUTH_COUNT or np.intersect1d(truth_members, sample).size:
        raise RuntimeError("truth-member-contract")

    operators = {
        network: load_full_operator(paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"])
        for network in NETWORKS
    }
    if any(not np.array_equal(operator["sample_indices"], sample) for operator in operators.values()):
        raise RuntimeError("operator-sample-index-closure")

    grid_lat, grid_lon = common.state_coordinates()
    mask = np.load(paths["hma_mask"]).astype(bool)
    weights = common.region_weights(grid_lat, grid_lon, mask)
    rng = np.random.default_rng(SEED)
    all_cases = []
    all_summaries = []
    saved = {}
    for condition, repetitions in (("noise_free", 1), ("regem_noise", NOISE_REPLICATES)):
        network_fields = {
            str(network): reconstruct_cases(prior, sample, truth_members, operators[network], repetitions, rng)
            for network in NETWORKS
        }
        combined = {
            method: np.mean([network_fields[str(network)][method] for network in NETWORKS], axis=0)
            for method in ("truth", "background", "ordinary", "adaptive")
        }
        for network, fields in {**network_fields, "equal_6_9": combined}.items():
            rows, summaries = score(condition, network, fields, weights)
            all_cases.extend(rows)
            all_summaries.extend(summaries)
        saved[condition] = combined

    case_frame = pd.DataFrame(all_cases)
    summary_frame = pd.DataFrame(all_summaries)
    case_frame.to_csv(output / "case_metrics.csv", index=False)
    summary_frame.to_csv(output / "summary_metrics.csv", index=False)
    primary = summary_frame[(summary_frame.network == "equal_6_9") & (summary_frame.method == "ordinary")]
    def value(condition: str, region: str, column: str) -> float:
        return float(primary[(primary.condition == condition) & (primary.region == region)].iloc[0][column])
    gates = {
        "noise_free_hma_state_skill_ge_0_50": value("noise_free", "hma", "median_state_rmse_skill_vs_background") >= 0.50,
        "noise_free_hma_pattern_ge_0_50": value("noise_free", "hma", "median_spatial_pattern_correlation") >= 0.50,
        "noise_free_hma_value_correlation_ge_0_50": value("noise_free", "hma", "regional_value_correlation") >= 0.50,
        "noise_free_all_subregion_value_correlations_ge_0_30": all(value("noise_free", region, "regional_value_correlation") >= 0.30 for region in REGIONS if region != "hma"),
        "noise_hma_state_skill_ge_0_20": value("regem_noise", "hma", "median_state_rmse_skill_vs_background") >= 0.20,
        "noise_hma_pattern_ge_0_30": value("regem_noise", "hma", "median_spatial_pattern_correlation") >= 0.30,
        "noise_hma_value_correlation_ge_0_30": value("regem_noise", "hma", "regional_value_correlation") >= 0.30,
    }
    passes = bool(all(gates.values()))
    decision = "release_instrumental_truth_twin" if passes else "block_external_twin_explain_prior_operator_failure"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"background": "#777777", "ordinary": "#0072B2", "adaptive": "#D55E00"}
    subset = case_frame[(case_frame.network == "equal_6_9") & (case_frame.region == "hma")]
    positions = []
    labels = []
    cursor = 1
    for condition in ("noise_free", "regem_noise"):
        for method in ("background", "ordinary", "adaptive"):
            current = subset[(subset.condition == condition) & (subset.method == method)]
            axes[0].boxplot(current.state_rmse_degC, positions=[cursor], widths=0.55, patch_artist=True, boxprops={"facecolor": colors[method], "alpha": 0.65})
            axes[1].boxplot(current.spatial_pattern_correlation.dropna(), positions=[cursor], widths=0.55, patch_artist=True, boxprops={"facecolor": colors[method], "alpha": 0.65})
            positions.append(cursor)
            labels.append(f"{condition}\n{method}")
            cursor += 1
        cursor += 1
    for axis, title, ylabel in zip(axes, ("HMA state error", "HMA spatial recovery"), ("RMSE (degC)", "pattern correlation")):
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "perfect_model_twin_metrics.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path, before[name].get("sha256")) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_161_temperature_perfect_model_twin_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "truth_member_count": TRUTH_COUNT,
        "truth_members_0based": truth_members.tolist(),
        "truth_members_disjoint_from_reconstruction_pool": True,
        "noise_replicates": NOISE_REPLICATES,
        "seed": SEED,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "primary_metrics": primary.to_dict(orient="records"),
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
    lines = ["# Round 161 Perfect-Model Temperature Twin", ""]
    for row in primary.itertuples():
        lines.append(f"- {row.condition} {row.region}: ordinary state skill=`{row.median_state_rmse_skill_vs_background:.3f}`, pattern r=`{row.median_spatial_pattern_correlation:.3f}`, regional-value r=`{row.regional_value_correlation:.3f}`.")
    lines.extend([f"- Passes all gates: `{passes}`.", f"- Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
