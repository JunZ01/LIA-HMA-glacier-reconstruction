from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_ID = "climate_reconstruction_clean_gpcc_psm_rolling_pseudo_event_fixed_network_v1"
BRANCHES = ("BEA_clean_GPCC", "CRU_clean_GPCC")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def correlation(left: pd.Series, right: pd.Series) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def summarize_branch(windows: pd.DataFrame, target: pd.DataFrame, branch: str) -> dict:
    subset = windows[windows["branch"] == branch].copy()
    target_subset = target[target["branch"] == branch]
    target_actual = float(target_subset["actual_proxy_event_change_z"].mean())
    target_predicted = float(target_subset["total_component_z"].mean())
    target_transfer = target_predicted / target_actual
    magnitude_mask = np.abs(subset["actual_network_event_change_z"]) >= 0.10
    strong_mask = np.abs(subset["actual_network_event_change_z"]) >= 0.25
    strong_negative_mask = subset["actual_network_event_change_z"] <= -0.25
    comparable = subset.loc[strong_mask, "signed_amplitude_transfer"].dropna().to_numpy(dtype=float)
    negative = subset.loc[strong_negative_mask, "signed_amplitude_transfer"].dropna().to_numpy(dtype=float)
    q05 = float(np.quantile(comparable, 0.05)) if len(comparable) else None
    q50 = float(np.quantile(comparable, 0.50)) if len(comparable) else None
    q95 = float(np.quantile(comparable, 0.95)) if len(comparable) else None
    if q05 is None:
        classification = "no_comparable_pseudo_event"
    elif target_transfer < q05:
        classification = "target_transfer_below_pseudo_event_5th_percentile"
    elif target_transfer > q95:
        classification = "target_transfer_above_pseudo_event_95th_percentile"
    else:
        classification = "target_transfer_within_pseudo_event_5th_to_95th_percentiles"
    return {
        "branch": branch,
        "window_count": int(len(subset)),
        "fixed_proxy_count": int(target_subset["proxy_index"].nunique()),
        "target_1905_actual_network_change_z": target_actual,
        "target_1905_predicted_network_change_z": target_predicted,
        "target_1905_signed_amplitude_transfer": target_transfer,
        "target_1905_absolute_change_rank_percentile": float(100.0 * np.mean(np.abs(subset["actual_network_event_change_z"]) <= abs(target_actual))),
        "pseudo_event_actual_network_change_min_z": float(subset["actual_network_event_change_z"].min()),
        "pseudo_event_actual_network_change_max_z": float(subset["actual_network_event_change_z"].max()),
        "window_count_absolute_change_at_least_0.10_z": int(magnitude_mask.sum()),
        "median_signed_transfer_absolute_change_at_least_0.10_z": float(subset.loc[magnitude_mask, "signed_amplitude_transfer"].median()),
        "window_count_absolute_change_at_least_0.25_z": int(strong_mask.sum()),
        "signed_transfer_q05_absolute_change_at_least_0.25_z": q05,
        "signed_transfer_median_absolute_change_at_least_0.25_z": q50,
        "signed_transfer_q95_absolute_change_at_least_0.25_z": q95,
        "fraction_comparable_transfer_at_or_below_target": float(np.mean(comparable <= target_transfer)) if len(comparable) else None,
        "strong_negative_window_count": int(strong_negative_mask.sum()),
        "strong_negative_median_signed_transfer": float(np.median(negative)) if len(negative) else None,
        "window_actual_predicted_correlation": correlation(subset["actual_network_event_change_z"], subset["predicted_network_event_change_z"]),
        "direction_agreement_fraction": float(subset["network_direction_agreement"].mean()),
        "classification": classification,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    source = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_clean_gpcc_psm_rolling_pseudo_event_v1"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    files = {
        "rolling_sites": source / "rolling_pseudo_event_site_predictions.csv",
        "rolling_variable_windows": source / "rolling_pseudo_event_window_metrics.csv",
        "target_components": source / "clean_gpcc_selected_psm_component_decomposition.csv",
        "source_summary": source / "summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in files.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_stat(path) for name, path in files.items()}

    sites = pd.read_csv(files["rolling_sites"])
    variable_windows = pd.read_csv(files["rolling_variable_windows"])
    target = pd.read_csv(files["target_components"])
    if set(sites["branch"].unique()) != set(BRANCHES):
        raise RuntimeError("branch-contract-failed")
    expected_windows = sorted(sites["baseline_start"].unique())
    if expected_windows != list(range(1941, 1988)):
        raise RuntimeError("window-contract-failed")

    sites["finite_pair"] = np.isfinite(sites["actual_proxy_event_change_z"]) & np.isfinite(sites["predicted_event_change_z"])
    counts = sites.groupby(["branch", "proxy_index"])["finite_pair"].sum().unstack("branch", fill_value=0)
    fixed_indices = counts.index[(counts[list(BRANCHES)] == 47).all(axis=1)].to_numpy(dtype=int)
    if len(fixed_indices) < 30:
        raise RuntimeError(f"fixed-network-too-small:{len(fixed_indices)}")
    fixed_sites = sites[sites["proxy_index"].isin(fixed_indices)].copy()
    fixed_target = target[target["proxy_index"].isin(fixed_indices)].copy()
    if len(fixed_target) != len(fixed_indices) * len(BRANCHES):
        raise RuntimeError("fixed-target-count-failed")

    window_rows = []
    for (branch, baseline_start), group in fixed_sites.groupby(["branch", "baseline_start"], sort=True):
        if len(group) != len(fixed_indices) or not group["finite_pair"].all():
            raise RuntimeError(f"fixed-window-completeness-failed:{branch}:{baseline_start}")
        actual = group["actual_proxy_event_change_z"]
        predicted = group["predicted_event_change_z"]
        actual_network = float(actual.mean())
        predicted_network = float(predicted.mean())
        window_rows.append({
            "branch": branch,
            "baseline_start": int(baseline_start),
            "baseline_end": int(group["baseline_end"].iloc[0]),
            "event_start": int(group["event_start"].iloc[0]),
            "event_end": int(group["event_end"].iloc[0]),
            "fixed_proxy_count": int(len(group)),
            "actual_network_event_change_z": actual_network,
            "predicted_network_event_change_z": predicted_network,
            "network_residual_z": actual_network - predicted_network,
            "signed_amplitude_transfer": predicted_network / actual_network if abs(actual_network) > 1e-12 else float("nan"),
            "absolute_amplitude_transfer": abs(predicted_network) / abs(actual_network) if abs(actual_network) > 1e-12 else float("nan"),
            "network_direction_agreement": bool(np.sign(actual_network) == np.sign(predicted_network)),
            "site_event_change_correlation": correlation(actual, predicted),
            "site_sign_agreement_fraction": float(np.mean(np.sign(actual) == np.sign(predicted))),
        })
    windows = pd.DataFrame(window_rows)
    if len(windows) != 94 or windows["fixed_proxy_count"].nunique() != 1:
        raise RuntimeError("fixed-window-output-contract-failed")

    summaries = [summarize_branch(windows, fixed_target, branch) for branch in BRANCHES]
    summary_index = {row["branch"]: row for row in summaries}
    comparison = windows.merge(
        variable_windows[["branch", "baseline_start", "actual_network_event_change_z", "predicted_network_event_change_z"]],
        on=["branch", "baseline_start"],
        suffixes=("_fixed", "_variable"),
        validate="one_to_one",
    )
    composition_sensitivity = {
        "maximum_absolute_actual_network_difference_z": float(np.max(np.abs(comparison["actual_network_event_change_z_fixed"] - comparison["actual_network_event_change_z_variable"]))),
        "maximum_absolute_predicted_network_difference_z": float(np.max(np.abs(comparison["predicted_network_event_change_z_fixed"] - comparison["predicted_network_event_change_z_variable"]))),
    }

    output.mkdir(parents=True)
    pd.DataFrame({"proxy_index": fixed_indices}).to_csv(output / "fixed_complete_proxy_indices.csv", index=False)
    fixed_sites.drop(columns="finite_pair").to_csv(output / "fixed_network_site_predictions.csv", index=False)
    fixed_target.to_csv(output / "fixed_network_target_1905_components.csv", index=False)
    windows.to_csv(output / "fixed_network_rolling_window_metrics.csv", index=False)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "source_experiment": source.name,
        "locked_final_period": [1921, 1939],
        "locked_final_period_accessed": False,
        "fixed_proxy_count": int(len(fixed_indices)),
        "window_count_per_branch": 47,
        "network_definition": "Same proxies with finite actual and predicted changes in every one of the 47 windows and both temperature branches.",
        "branch_summaries": summaries,
        "composition_sensitivity": composition_sensitivity,
        "interpretation_limit": "This is a fixed-network sensitivity of saved diagnostic outputs, not a production PSM or climate replacement.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = {"BEA_clean_GPCC": "#009E73", "CRU_clean_GPCC": "#CC79A7"}
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    all_values = np.concatenate([
        windows["actual_network_event_change_z"].to_numpy(dtype=float),
        windows["predicted_network_event_change_z"].to_numpy(dtype=float),
        np.asarray([row["target_1905_actual_network_change_z"] for row in summaries]),
        np.asarray([row["target_1905_predicted_network_change_z"] for row in summaries]),
    ])
    limit = max(0.35, float(np.max(np.abs(all_values))) * 1.08)
    for branch, color in colors.items():
        subset = windows[windows["branch"] == branch]
        row = summary_index[branch]
        axes[0, 0].scatter(subset["actual_network_event_change_z"], subset["predicted_network_event_change_z"], color=color, alpha=0.68, s=34, label=branch.replace("_clean_GPCC", ""))
        axes[0, 0].scatter(row["target_1905_actual_network_change_z"], row["target_1905_predicted_network_change_z"], color=color, edgecolor="#111111", marker="*", s=180)
    axes[0, 0].plot([-limit, limit], [-limit, limit], color="#777777", linewidth=0.8)
    axes[0, 0].set_xlim(-limit, limit); axes[0, 0].set_ylim(-limit, limit)
    axes[0, 0].set_xlabel("Actual fixed-network event change (z)"); axes[0, 0].set_ylabel("Predicted fixed-network change (z)")
    axes[0, 0].set_title("47 held-out pseudo-events; stars are 1905-1918"); axes[0, 0].legend(frameon=False)
    for branch, color in colors.items():
        subset = windows[(windows["branch"] == branch) & (np.abs(windows["actual_network_event_change_z"]) >= 0.10)]
        axes[0, 1].scatter(subset["event_start"], subset["signed_amplitude_transfer"], color=color, alpha=0.68, s=32, label=branch.replace("_clean_GPCC", ""))
        axes[0, 1].axhline(summary_index[branch]["target_1905_signed_amplitude_transfer"], color=color, linestyle="--", linewidth=1.2)
    axes[0, 1].axhline(1, color="#777777", linewidth=0.8)
    axes[0, 1].set_xlabel("Pseudo-event start year"); axes[0, 1].set_ylabel("Predicted / actual network change")
    axes[0, 1].set_title("Amplitude transfer for |actual| >= 0.10 z"); axes[0, 1].legend(frameon=False)
    bins = np.linspace(float(windows["actual_network_event_change_z"].min()), float(windows["actual_network_event_change_z"].max()), 16)
    for branch, color in colors.items():
        subset = windows[windows["branch"] == branch]
        axes[1, 0].hist(subset["actual_network_event_change_z"], bins=bins, histtype="step", linewidth=1.8, color=color, label=branch.replace("_clean_GPCC", ""))
    axes[1, 0].axvline(summary_index["BEA_clean_GPCC"]["target_1905_actual_network_change_z"], color="#1B1B1B", linestyle="--", linewidth=1.4, label="1905-1918")
    axes[1, 0].set_xlabel("Actual fixed-network event change (z)"); axes[1, 0].set_ylabel("Window count")
    axes[1, 0].set_title("Calibration pseudo-event magnitude"); axes[1, 0].legend(frameon=False)
    for branch, color in colors.items():
        values = windows.loc[(windows["branch"] == branch) & (np.abs(windows["actual_network_event_change_z"]) >= 0.25), "signed_amplitude_transfer"].dropna().sort_values().to_numpy(dtype=float)
        if len(values):
            axes[1, 1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, linewidth=1.8, label=branch.replace("_clean_GPCC", ""))
        axes[1, 1].axvline(summary_index[branch]["target_1905_signed_amplitude_transfer"], color=color, linestyle="--", linewidth=1.2)
    axes[1, 1].axvline(1, color="#777777", linewidth=0.8)
    axes[1, 1].set_xlabel("Predicted / actual network change"); axes[1, 1].set_ylabel("Empirical cumulative fraction")
    axes[1, 1].set_title("Transfer distribution for |actual| >= 0.25 z"); axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "fixed_network_rolling_pseudo_event_amplitude_transfer.png", dpi=190)
    plt.close(figure)

    bea = summary_index["BEA_clean_GPCC"]
    cru = summary_index["CRU_clean_GPCC"]
    report = [
        "# Fixed-network Rolling Pseudo-event Result",
        "",
        f"The fixed network contains {len(fixed_indices)} proxies complete in all 47 windows and both branches.",
        "",
        f"- BEA target transfer: `{bea['target_1905_signed_amplitude_transfer']:.3f}`; strong pseudo-event median: `{bea['signed_transfer_median_absolute_change_at_least_0.25_z']:.3f}`; classification: `{bea['classification']}`.",
        f"- CRU target transfer: `{cru['target_1905_signed_amplitude_transfer']:.3f}`; strong pseudo-event median: `{cru['signed_transfer_median_absolute_change_at_least_0.25_z']:.3f}`; classification: `{cru['classification']}`.",
        f"- Target absolute event magnitude rank: `{bea['target_1905_absolute_change_rank_percentile']:.1f}%`.",
        f"- Maximum variable-versus-fixed network differences, actual/predicted: `{composition_sensitivity['maximum_absolute_actual_network_difference_z']:.4f}` / `{composition_sensitivity['maximum_absolute_predicted_network_difference_z']:.4f} z`.",
        "",
        "No values from 1921-1939 were accessed. Original inputs were not modified.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {name: source_stat(path) for name, path in files.items()}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-analysis")
    manifest = {
        "sources_before": before,
        "sources_after": after,
        "driver_sha256": digest(Path(__file__)),
        "facts_sha256": digest(root / "LIA_PROJECT_FACTS.md"),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())