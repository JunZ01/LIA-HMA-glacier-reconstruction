"""Score and plot fixed-EnKF longitude-block ablations for the other 33 records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_precipitation_other_positive_b2_longitude_block_ablation_v1"
BRANCHES = [
    "baseline",
    "west_lt80_ablation",
    "central_80_90_ablation",
    "east_90_100_ablation",
    "far_east_ge100_ablation",
    "all_other_positive_b2_ablation",
]
YEARS = np.arange(1901, 1921, dtype=int)
BASELINE_MASK = (YEARS >= 1905) & (YEARS <= 1914)
EVENT_MASK = (YEARS >= 1915) & (YEARS <= 1918)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest(path)}


def decode(handle: h5py.File, reference: h5py.Reference) -> str:
    return "".join(chr(int(value)) for value in np.asarray(handle[reference]).reshape(-1))


def trend(values: np.ndarray) -> float:
    return float(np.polyfit(YEARS, values, 1)[0] * 100.0)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def event_change(values: np.ndarray) -> float:
    return float(np.mean(values[EVENT_MASK]) - np.mean(values[BASELINE_MASK]))


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    sources = {
        "replay": output / "longitude_block_ablation.mat",
        "prepare": output / "prepare_summary.json",
        "reference_hma": result_root / "climate_reconstruction_precipitation_common_mode_reference_phase_audit_v1/common_mode_reference_hma_series_1901_1920.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}
    with sources["prepare"].open("r", encoding="utf-8") as handle:
        prepare = json.load(handle)
    with h5py.File(sources["replay"], "r") as handle:
        names = [decode(handle, reference) for reference in handle["names"][()].reshape(-1)]
        years = np.asarray(handle["years"]).reshape(-1).astype(int)
        total = np.asarray(handle["total"]).T
        baseline_closure = float(np.asarray(handle["baseline_closure"]).squeeze())
        prior_all_other_closure = float(np.asarray(handle["prior_all_other_closure"]).squeeze())
        ablation_equivalence = float(np.asarray(handle["max_zero_gain_ablation_equivalence"]).squeeze())
        update_closure = float(np.max(np.abs(np.asarray(handle["update_closure"]))))
        available_group_count = np.asarray(handle["available_group_count"]).T.astype(int)
    if names != BRANCHES or not np.array_equal(years, YEARS):
        raise RuntimeError("longitude-block-replay-contract-failed")
    if max(baseline_closure, prior_all_other_closure, ablation_equivalence, update_closure) > 2e-12:
        raise RuntimeError("longitude-block-replay-closure-failed")

    reference = pd.read_csv(sources["reference_hma"]).set_index("year").loc[YEARS]
    baseline_hma = reference["rec_result_hma_anomaly_mm_day"].to_numpy(dtype=float)
    cru = reference["cru_hma_anomaly_mm_day"].to_numpy(dtype=float)
    gpcc = reference["gpcc_hma_anomaly_mm_day"].to_numpy(dtype=float)
    baseline_increment = total[:, 0]
    annual = pd.DataFrame({"year": YEARS, "cru_hma_mm_day": cru, "gpcc_hma_mm_day": gpcc})
    score_rows: list[dict[str, object]] = []
    for index, branch in enumerate(BRANCHES):
        anchored = baseline_hma + total[:, index] - baseline_increment
        annual[f"delta_anchor_{branch}_hma_mm_day"] = anchored
        annual[f"delta_from_baseline_{branch}_mm_day"] = anchored - baseline_hma
        score_rows.append({
            "branch": branch,
            "trend_mm_day_per_century": trend(anchored),
            "event_change_1915_1918_minus_1905_1914_mm_day": event_change(anchored),
            "rmse_vs_cru_mm_day": rmse(anchored, cru),
            "rmse_vs_gpcc_mm_day": rmse(anchored, gpcc),
            "mean_two_product_rmse_mm_day": 0.5 * (rmse(anchored, cru) + rmse(anchored, gpcc)),
        })
    scores = pd.DataFrame(score_rows)
    baseline_score = scores.iloc[0]
    scores["delta_from_baseline_trend_mm_day_per_century"] = scores["trend_mm_day_per_century"] - baseline_score["trend_mm_day_per_century"]
    scores["delta_from_baseline_event_change_mm_day"] = scores["event_change_1915_1918_minus_1905_1914_mm_day"] - baseline_score["event_change_1915_1918_minus_1905_1914_mm_day"]
    scores["improves_cru_rmse"] = scores["rmse_vs_cru_mm_day"] < baseline_score["rmse_vs_cru_mm_day"]
    scores["improves_gpcc_rmse"] = scores["rmse_vs_gpcc_mm_day"] < baseline_score["rmse_vs_gpcc_mm_day"]
    scores["dual_product_improvement"] = scores["improves_cru_rmse"] & scores["improves_gpcc_rmse"]

    regional_branches = BRANCHES[1:5]
    effects = scores.set_index("branch")["delta_from_baseline_trend_mm_day_per_century"]
    full_effect = float(effects["all_other_positive_b2_ablation"])
    regional_sum = float(effects[regional_branches].sum())
    nonadditivity = full_effect - regional_sum
    dominant_branch = str(effects[regional_branches].idxmax())
    dominant_effect = float(effects[dominant_branch])
    dominant_region = dominant_branch.removesuffix("_ablation") if dominant_effect > 0 else "none"

    scores.to_csv(output / "longitude_block_delta_anchor_scores.csv", index=False)
    annual.to_csv(output / "longitude_block_delta_anchor_hma_curves_1901_1920.csv", index=False)
    colors = {
        "baseline": "#111111",
        "west_lt80_ablation": "#0072B2",
        "central_80_90_ablation": "#E69F00",
        "east_90_100_ablation": "#D55E00",
        "far_east_ge100_ablation": "#009E73",
        "all_other_positive_b2_ablation": "#CC79A7",
    }
    labels = {
        "baseline": "Baseline",
        "west_lt80_ablation": "<80E removed",
        "central_80_90_ablation": "80-90E removed",
        "east_90_100_ablation": "90-100E removed",
        "far_east_ge100_ablation": ">=100E removed",
        "all_other_positive_b2_ablation": "All other-33 removed",
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.5), constrained_layout=True)
    axes[0].plot(YEARS, cru, color="#6A3D9A", linewidth=1.6, label="CRU")
    axes[0].plot(YEARS, gpcc, color="#777777", linewidth=1.6, label="GPCC-derived")
    axes[0].fill_between(YEARS, np.minimum(cru, gpcc), np.maximum(cru, gpcc), color="#BDBDBD", alpha=0.25)
    for branch in BRANCHES:
        axes[0].plot(YEARS, annual[f"delta_anchor_{branch}_hma_mm_day"], color=colors[branch], linewidth=2.2 if branch == "baseline" else 1.8, label=labels[branch])
    axes[0].axvspan(1915, 1918, color="#E69F00", alpha=0.10)
    axes[0].set_ylabel("HMA precipitation anomaly (mm/day)")
    axes[0].set_title("Other positive-b2 longitude-block ablation")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].axhline(0, color="#777777", linewidth=0.8)
    for branch in BRANCHES[1:]:
        axes[1].plot(YEARS, annual[f"delta_from_baseline_{branch}_mm_day"], color=colors[branch], linewidth=1.9, label=labels[branch])
    axes[1].axvspan(1915, 1918, color="#E69F00", alpha=0.10)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Change from baseline (mm/day)")
    axes[1].set_title("Pure fixed-EnKF branch effect")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    fig.savefig(output / "other_positive_b2_longitude_block_hma_precipitation_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "longitude_band_definition": prepare["longitude_band_definition"],
        "group_counts": prepare["group_counts"],
        "available_group_count_by_year_and_branch": available_group_count.tolist(),
        "closure": {
            "baseline": baseline_closure,
            "prior_all_other_positive_b2": prior_all_other_closure,
            "update": update_closure,
            "explicit_ablation_vs_zero_gain": ablation_equivalence,
        },
        "scores": scores.to_dict(orient="records"),
        "regional_trend_effect_sum_mm_day_per_century": regional_sum,
        "all_other_positive_b2_trend_effect_mm_day_per_century": full_effect,
        "trend_effect_nonadditivity_mm_day_per_century": nonadditivity,
        "dominant_negative_trend_longitude_block": dominant_region,
        "dominant_block_removal_trend_shift_mm_day_per_century": dominant_effect,
        "interpretation_boundary": "This fixed-EnKF ablation uses predefined longitude bands to localize simultaneous covariance influence. It does not validate deleting regional records or modifying the production reconstruction.",
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "analysis_source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-changed-during-longitude-block-analysis")
    (output / "FINAL_RESULT.md").write_text(
        "# Other Positive-b2 Longitude-block Ablation v1\n\n"
        f"- Baseline trend: `{scores.loc[0, 'trend_mm_day_per_century']:+.3f} mm/day/century`.\n"
        f"- <80E removal shift: `{effects['west_lt80_ablation']:+.3f}`.\n"
        f"- 80-90E removal shift: `{effects['central_80_90_ablation']:+.3f}`.\n"
        f"- 90-100E removal shift: `{effects['east_90_100_ablation']:+.3f}`.\n"
        f"- >=100E removal shift: `{effects['far_east_ge100_ablation']:+.3f}`.\n"
        f"- All other-33 removal shift: `{full_effect:+.3f}`.\n"
        f"- Regional nonadditivity: `{nonadditivity:+.3f} mm/day/century`.\n\n"
        f"Dominant negative-trend longitude block: `{dominant_region}`. This is a mechanism sensitivity, not a production correction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
