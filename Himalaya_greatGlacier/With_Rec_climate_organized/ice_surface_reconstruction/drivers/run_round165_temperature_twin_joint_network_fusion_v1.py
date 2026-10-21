"""Compare posterior averaging with one stacked-network update in the perfect-model twin."""

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
import run_round164_temperature_twin_residual_variance_ablation_v1 as round164
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


EXPERIMENT_ID = "round_165_temperature_twin_joint_network_fusion_v1"
R_SCALES = (1.0, 0.0001)


def fingerprint(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    if digest is None and stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_improvement_loop_v2" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round165_temperature_twin_joint_network_fusion_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round164_driver": driver / "run_round164_temperature_twin_residual_variance_ablation_v1.py",
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
        str(network): round161.load_full_operator(
            paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"]
        )
        for network in round161.NETWORKS
    }
    if any(not np.array_equal(operator["sample_indices"], sample) for operator in operators.values()):
        raise RuntimeError("operator-sample-index-closure")
    joint = {
        "hxb": np.concatenate([operators["6"]["hxb"], operators["9"]["hxb"]], axis=0),
        "variance": np.concatenate([operators["6"]["variance"], operators["9"]["variance"]]),
    }

    with np.load(paths["native_grid"]) as native:
        grid_lat = np.asarray(native["grid_lat"], dtype=float)
        grid_lon = np.asarray(native["grid_lon"], dtype=float)
        mask = np.asarray(native["hma_mask"], dtype=bool)
    region_weights = common.region_weights(grid_lat, grid_lon, mask)
    background = np.mean(prior[:, sample], axis=1)
    truth = prior[:, truth_members]
    rows = []
    for scale in R_SCALES:
        six = round164.reconstruct_all_truths(prior, sample, truth_members, operators["6"], scale)
        nine = round164.reconstruct_all_truths(prior, sample, truth_members, operators["9"], scale)
        fields = {
            "proxy6": six,
            "proxy9": nine,
            "equal_posterior_mean": np.mean([six, nine], axis=0),
            "joint_stacked": round164.reconstruct_all_truths(prior, sample, truth_members, joint, scale),
        }
        for branch, estimate in fields.items():
            for region, weights in region_weights.items():
                background_rmse = np.asarray([
                    round161.weighted_rmse(background, truth[:, index], weights)
                    for index in range(truth.shape[1])
                ])
                analysis_rmse = np.asarray([
                    round161.weighted_rmse(estimate[:, index], truth[:, index], weights)
                    for index in range(truth.shape[1])
                ])
                for index in range(truth.shape[1]):
                    rows.append({
                        "r_scale": scale,
                        "branch": branch,
                        "region": region,
                        "case_index": index,
                        "truth_member_0based": int(truth_members[index]),
                        "background_rmse_degC": background_rmse[index],
                        "analysis_rmse_degC": analysis_rmse[index],
                        "case_state_skill": 1.0 - analysis_rmse[index] / background_rmse[index],
                    })
    case_frame = pd.DataFrame(rows)
    case_frame.to_csv(output / "case_state_skill.csv", index=False)
    summary = case_frame.groupby(["r_scale", "branch", "region"], as_index=False).agg(
        median_background_rmse_degC=("background_rmse_degC", "median"),
        median_analysis_rmse_degC=("analysis_rmse_degC", "median"),
        median_case_state_skill=("case_state_skill", "median"),
    )
    summary["ratio_of_median_rmse_skill"] = 1.0 - summary.median_analysis_rmse_degC / summary.median_background_rmse_degC
    summary.to_csv(output / "summary_state_skill.csv", index=False)

    def metric(scale: float, branch: str, region: str = "hma") -> float:
        row = summary[(summary.r_scale == scale) & (summary.branch == branch) & (summary.region == region)].iloc[0]
        return float(row.ratio_of_median_rmse_skill)

    archived_equal = metric(1.0, "equal_posterior_mean")
    archived_joint = metric(1.0, "joint_stacked")
    near_equal = metric(0.0001, "equal_posterior_mean")
    near_joint = metric(0.0001, "joint_stacked")
    if archived_joint >= 0.50:
        decision = "joint_fusion_alone_crosses_gate"
    elif near_joint >= 0.50:
        decision = "joint_fusion_and_residual_shrinkage_cross_gate_only_together"
    else:
        decision = "joint_fusion_insufficient_even_at_near_zero_R"

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)
    branches = ["proxy6", "proxy9", "equal_posterior_mean", "joint_stacked"]
    colors = ["#0072B2", "#D55E00", "#999999", "#009E73"]
    for axis, scale in zip(axes, R_SCALES):
        values = [metric(scale, branch) for branch in branches]
        axis.bar(np.arange(len(branches)), values, color=colors)
        axis.axhline(0.50, color="#777777", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(branches)), ["proxy6", "proxy9", "equal mean", "joint"], rotation=20, ha="right")
        axis.set(title=f"HMA state skill, R scale={scale:g}", ylabel="ratio-of-median RMSE skill", ylim=(0, max(0.65, max(values) + 0.05)))
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "joint_network_fusion.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path, before[name].get("sha256")) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "archived_R": {"equal_skill": archived_equal, "joint_skill": archived_joint, "joint_minus_equal": archived_joint - archived_equal},
        "near_zero_R": {"equal_skill": near_equal, "joint_skill": near_joint, "joint_minus_equal": near_joint - near_equal},
        "decision": decision,
        "joint_assumes_cross_network_residual_independence": True,
        "joint_is_mechanism_bound_not_production_candidate": True,
        "actual_proxy_observations_used": False,
        "instrumental_values_used": False,
        "locked_1921_1939_accessed": False,
        "jionglacuo_opened": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = [
        "# Round 165 Temperature Twin Joint-Network Fusion",
        "",
        f"- Archived R: equal posterior mean skill `{archived_equal:.3f}`, joint skill `{archived_joint:.3f}`.",
        f"- Near-zero R: equal posterior mean skill `{near_equal:.3f}`, joint skill `{near_joint:.3f}`.",
        f"- Decision: `{decision}`.",
        "- Joint stacking is a perfect-model mechanism bound; cross-network residual independence is not established.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
