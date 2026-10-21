from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_pages2k_study_batch_pseudo_event_residual_v1"
BRANCHES = ("BEA_clean_GPCC", "CRU_clean_GPCC")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def group_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    pages = frame["source_family"].eq("PAGES2k2017")
    return {
        "all_fixed": pd.Series(True, index=frame.index),
        "Cook": pages & frame["data_author"].eq("Cook, E.R."),
        "Shao": pages & frame["data_author"].eq("Shao, X.M."),
        "PAGES_consortium": pages & frame["data_author"].eq("PAGES 2k Consortium"),
        "Cook_or_Shao": pages & frame["data_author"].isin(["Cook, E.R.", "Shao, X.M."]),
        "north_west_ge35_lt80E": pages & frame["latitude"].ge(35.0) & frame["longitude"].lt(80.0),
        "north_east_ge35_90to101E": pages & frame["latitude"].ge(35.0) & frame["longitude"].ge(90.0) & frame["longitude"].lt(101.0),
        "all_PAGES_ge35N": pages & frame["latitude"].ge(35.0),
        "all_records_lt35N": frame["latitude"].lt(35.0),
    }


def group_values(frame: pd.DataFrame, actual: str, predicted: str) -> list[dict[str, object]]:
    rows = []
    for name, mask in group_masks(frame).items():
        selected = frame[mask]
        actual_mean = float(selected[actual].mean())
        predicted_mean = float(selected[predicted].mean())
        rows.append({
            "group": name,
            "record_count": int(len(selected)),
            "unique_site_count": int(selected["site_id"].nunique()),
            "actual_mean_event_change_z": actual_mean,
            "predicted_mean_event_change_z": predicted_mean,
            "residual_actual_minus_predicted_z": actual_mean - predicted_mean,
            "signed_amplitude_transfer": predicted_mean / actual_mean if actual_mean != 0 else np.nan,
        })
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    study_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_pages2k_fixed_network_study_batch_v1"
    rolling_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_clean_gpcc_psm_rolling_pseudo_event_fixed_network_v1"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "study_metadata": study_root / "fixed_network_study_metadata.csv",
        "rolling_predictions": rolling_root / "fixed_network_site_predictions.csv",
        "target_components": rolling_root / "fixed_network_target_1905_components.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    metadata = pd.read_csv(sources["study_metadata"])[
        ["proxy_index", "site_id", "latitude", "longitude", "source_family", "data_author"]
    ]
    if len(metadata) != 50 or metadata["proxy_index"].nunique() != 50:
        raise RuntimeError("study-metadata-count-contract-failed")
    target = pd.read_csv(sources["target_components"])
    rolling = pd.read_csv(sources["rolling_predictions"])
    if set(target["branch"].unique()) != set(BRANCHES) or set(rolling["branch"].unique()) != set(BRANCHES):
        raise RuntimeError("unexpected-branch-set")

    target_rows = []
    pseudo_rows = []
    for branch in BRANCHES:
        target_branch = target[target["branch"] == branch].merge(metadata, on="proxy_index", validate="one_to_one")
        if len(target_branch) != 50:
            raise RuntimeError(f"target-count-contract-failed:{branch}:{len(target_branch)}")
        branch_target_rows = group_values(
            target_branch,
            "actual_proxy_event_change_z",
            "total_component_z",
        )
        for row in branch_target_rows:
            row["branch"] = branch
            target_rows.append(row)

        branch_rolling = rolling[rolling["branch"] == branch].merge(metadata, on="proxy_index", validate="many_to_one")
        for baseline_start, window in branch_rolling.groupby("baseline_start"):
            if len(window) != 50:
                raise RuntimeError(f"rolling-window-count-contract-failed:{branch}:{baseline_start}:{len(window)}")
            values = group_values(window, "actual_proxy_event_change_z", "predicted_event_change_z")
            for row in values:
                row.update({
                    "branch": branch,
                    "baseline_start": int(baseline_start),
                    "baseline_end": int(window["baseline_end"].iloc[0]),
                    "event_start": int(window["event_start"].iloc[0]),
                    "event_end": int(window["event_end"].iloc[0]),
                })
                pseudo_rows.append(row)
    target_groups = pd.DataFrame(target_rows)
    pseudo_groups = pd.DataFrame(pseudo_rows)

    comparison_rows = []
    for target_row in target_groups.itertuples(index=False):
        pseudo = pseudo_groups[
            (pseudo_groups["branch"] == target_row.branch)
            & (pseudo_groups["group"] == target_row.group)
        ]
        residual = pseudo["residual_actual_minus_predicted_z"].to_numpy(dtype=float)
        actual = pseudo["actual_mean_event_change_z"].to_numpy(dtype=float)
        target_residual = float(target_row.residual_actual_minus_predicted_z)
        target_actual = float(target_row.actual_mean_event_change_z)
        comparison_rows.append({
            **target_row._asdict(),
            "pseudo_event_count": int(len(pseudo)),
            "pseudo_residual_mean_z": float(np.mean(residual)),
            "pseudo_residual_std_z": float(np.std(residual, ddof=1)),
            "target_residual_standardized_against_pseudo": float((target_residual - np.mean(residual)) / np.std(residual, ddof=1)),
            "target_residual_lower_tail_percentile": float(100.0 * np.mean(residual <= target_residual)),
            "target_absolute_residual_rank_percentile": float(100.0 * np.mean(np.abs(residual) <= abs(target_residual))),
            "target_absolute_actual_rank_percentile": float(100.0 * np.mean(np.abs(actual) <= abs(target_actual))),
            "pseudo_residual_min_z": float(np.min(residual)),
            "pseudo_residual_max_z": float(np.max(residual)),
            "target_residual_below_all_pseudo_events": bool(target_residual < np.min(residual)),
        })
    comparison = pd.DataFrame(comparison_rows)

    output.mkdir(parents=True)
    target_groups.to_csv(output / "target_group_actual_predicted.csv", index=False)
    pseudo_groups.to_csv(output / "rolling_pseudo_event_group_actual_predicted.csv", index=False)
    comparison.to_csv(output / "target_vs_pseudo_event_group_residual.csv", index=False)

    major = comparison[comparison["group"].isin(["Cook", "Shao"])]
    both_branches_extreme = bool((major["target_absolute_residual_rank_percentile"] >= 95.0).all())
    both_batches_below_all = bool(major.groupby("group")["target_residual_below_all_pseudo_events"].all().all())
    summary = {
        "target_definition": "mean(1915-1918) minus mean(1905-1914), scaled by each proxy's 1941-2000 standard deviation",
        "rolling_pseudo_event_definition": "10-year baseline followed by a 4-year event, 47 starts from 1941 through 1987",
        "branch_count": int(len(BRANCHES)),
        "group_count": int(target_groups["group"].nunique()),
        "pseudo_event_count_per_branch_group": int(pseudo_groups.groupby(["branch", "group"]).size().min()),
        "cook_and_shao_absolute_residual_at_least_95th_percentile_both_branches": both_branches_extreme,
        "cook_and_shao_residual_below_all_pseudo_events_both_branches": both_batches_below_all,
        "major_batch_comparison": major.to_dict(orient="records"),
        "classification": (
            "both_major_northern_study_batches_show_out_of_modern_range_proxy_instrument_residuals"
            if both_branches_extreme
            else "major_batch_residuals_overlap_modern_pseudo_event_range"
        ),
        "interpretation": (
            "An extreme residual means the fixed linear PSM driven by the selected instrumental products does not reproduce the proxy event. "
            "It does not by itself identify whether early instrumental climate, proxy chronology behavior, or PSM temporal nonstationarity is responsible."
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6), constrained_layout=True)
    colors = {"BEA_clean_GPCC": "#247BA0", "CRU_clean_GPCC": "#E07A1F"}
    for branch in BRANCHES:
        view = target_groups[target_groups["branch"] == branch]
        axes[0].scatter(
            view["actual_mean_event_change_z"], view["predicted_mean_event_change_z"],
            color=colors[branch], label=branch.replace("_clean_GPCC", ""), s=48, alpha=0.85,
        )
        for row in view[view["group"].isin(["Cook", "Shao", "all_fixed"])].itertuples(index=False):
            axes[0].annotate(row.group, (row.actual_mean_event_change_z, row.predicted_mean_event_change_z), fontsize=7, xytext=(3, 3), textcoords="offset points")
    low = min(target_groups["actual_mean_event_change_z"].min(), target_groups["predicted_mean_event_change_z"].min()) - 0.1
    high = max(target_groups["actual_mean_event_change_z"].max(), target_groups["predicted_mean_event_change_z"].max()) + 0.1
    axes[0].plot([low, high], [low, high], color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlim(low, high)
    axes[0].set_ylim(low, high)
    axes[0].set_xlabel("Proxy event change (z)")
    axes[0].set_ylabel("Instrumental PSM prediction (z)")
    axes[0].set_title("1915-1918 minus 1905-1914")
    axes[0].legend(frameon=False)

    selected_groups = ["Cook", "Shao", "north_west_ge35_lt80E", "north_east_ge35_90to101E"]
    for index, group in enumerate(selected_groups):
        pseudo = pseudo_groups[(pseudo_groups["branch"] == "BEA_clean_GPCC") & (pseudo_groups["group"] == group)]
        y = np.full(len(pseudo), index, dtype=float)
        axes[1].scatter(pseudo["residual_actual_minus_predicted_z"], y, color="#9AA0A6", s=24, alpha=0.6)
        target_value = target_groups[(target_groups["branch"] == "BEA_clean_GPCC") & (target_groups["group"] == group)]["residual_actual_minus_predicted_z"].iloc[0]
        axes[1].scatter([target_value], [index], marker="*", color="#C9483B", edgecolor="black", linewidth=0.4, s=140)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(selected_groups)), selected_groups, fontsize=8)
    axes[1].set_xlabel("Actual minus predicted event change (z)")
    axes[1].set_title("BEA pseudo-events; star is target")

    view = comparison[comparison["group"].isin(selected_groups + ["all_fixed", "all_records_lt35N"])].copy()
    positions = np.arange(view["group"].nunique())
    order = list(dict.fromkeys(view["group"].tolist()))
    for offset, branch in ((-0.12, "BEA_clean_GPCC"), (0.12, "CRU_clean_GPCC")):
        branch_view = view[view["branch"] == branch].set_index("group").loc[order]
        axes[2].scatter(
            branch_view["target_absolute_residual_rank_percentile"],
            positions + offset,
            color=colors[branch], s=42, label=branch.replace("_clean_GPCC", ""),
        )
    axes[2].axvline(95, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_yticks(positions, order, fontsize=8)
    axes[2].set_xlim(0, 102)
    axes[2].set_xlabel("Absolute residual rank percentile")
    axes[2].set_title("Target relative to 47 pseudo-events")
    axes[2].legend(frameon=False)
    fig.savefig(output / "study_batch_pseudo_event_residual.png", dpi=220)
    plt.close(fig)

    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-analysis")
    cook_bea = comparison[(comparison["group"] == "Cook") & (comparison["branch"] == "BEA_clean_GPCC")].iloc[0]
    shao_bea = comparison[(comparison["group"] == "Shao") & (comparison["branch"] == "BEA_clean_GPCC")].iloc[0]
    (output / "FINAL_RESULT.md").write_text(
        "# PAGES2k Study-batch Pseudo-event Residual v1\n\n"
        "The target is exactly mean(1915-1918) minus mean(1905-1914), not a trend over the whole 1901-1920 interval.\n\n"
        f"Cook BEA residual: {cook_bea['residual_actual_minus_predicted_z']:.3f} z, absolute pseudo-event rank "
        f"{cook_bea['target_absolute_residual_rank_percentile']:.1f}%. Shao BEA residual: "
        f"{shao_bea['residual_actual_minus_predicted_z']:.3f} z, rank {shao_bea['target_absolute_residual_rank_percentile']:.1f}%.\n\n"
        f"Classification: `{summary['classification']}`. An extreme residual does not alone identify the responsible data component.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
