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


EXPERIMENT_ID = "climate_reconstruction_precipitation_pakistan_group_weight_sensitivity_v1"
BRANCHES = ["baseline", "pakistan_ablation", "pakistan_design_effect_R"]
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
        "replay": output / "pakistan_weight_replay.mat",
        "prepare": output / "prepare_summary.json",
        "reference_hma": result_root / "climate_reconstruction_precipitation_common_mode_reference_phase_audit_v1/common_mode_reference_hma_series_1901_1920.csv",
        "positive_b2_ablation": result_root / "climate_reconstruction_precipitation_positive_b2_group_ablation_delta_anchor_v1/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    with sources["prepare"].open("r", encoding="utf-8") as handle:
        prepare = json.load(handle)
    with sources["positive_b2_ablation"].open("r", encoding="utf-8") as handle:
        positive_b2 = json.load(handle)
    with h5py.File(sources["replay"], "r") as handle:
        names = [decode(handle, reference) for reference in handle["names"][()].reshape(-1)]
        years = np.asarray(handle["years"]).reshape(-1).astype(int)
        total = np.asarray(handle["total"]).T
        baseline_closure = float(np.asarray(handle["baseline_closure"]).squeeze())
        ablation_equivalence = float(np.asarray(handle["max_zero_gain_ablation_equivalence"]).squeeze())
        update_closure = float(np.max(np.abs(np.asarray(handle["update_closure"]))))
        available_group_count = np.asarray(handle["available_group_count"]).reshape(-1).astype(int)
    if names != BRANCHES or not np.array_equal(years, YEARS) or not np.all(available_group_count == 14):
        raise RuntimeError("pakistan-replay-contract-failed")
    if max(baseline_closure, ablation_equivalence, update_closure) > 2e-12:
        raise RuntimeError("pakistan-replay-closure-failed")

    reference = pd.read_csv(sources["reference_hma"]).set_index("year").loc[YEARS]
    baseline_hma = reference["rec_result_hma_anomaly_mm_day"].to_numpy(dtype=float)
    cru = reference["cru_hma_anomaly_mm_day"].to_numpy(dtype=float)
    gpcc = reference["gpcc_hma_anomaly_mm_day"].to_numpy(dtype=float)
    baseline_increment = total[:, 0]
    annual = pd.DataFrame({"year": YEARS, "cru_hma_mm_day": cru, "gpcc_hma_mm_day": gpcc})
    score_rows = []
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
    baseline_score = scores[scores["branch"].eq("baseline")].iloc[0]
    scores["delta_from_baseline_trend_mm_day_per_century"] = scores["trend_mm_day_per_century"] - baseline_score["trend_mm_day_per_century"]
    scores["delta_from_baseline_event_change_mm_day"] = scores["event_change_1915_1918_minus_1905_1914_mm_day"] - baseline_score["event_change_1915_1918_minus_1905_1914_mm_day"]
    scores["improves_cru_rmse"] = scores["rmse_vs_cru_mm_day"] < baseline_score["rmse_vs_cru_mm_day"]
    scores["improves_gpcc_rmse"] = scores["rmse_vs_gpcc_mm_day"] < baseline_score["rmse_vs_gpcc_mm_day"]
    scores["dual_product_improvement"] = scores["improves_cru_rmse"] & scores["improves_gpcc_rmse"]

    ablation = scores[scores["branch"].eq("pakistan_ablation")].iloc[0]
    weighted = scores[scores["branch"].eq("pakistan_design_effect_R")].iloc[0]
    full_positive = next(row for row in positive_b2["scores"] if row["branch"] == "positive_b2_full_zero")
    full_positive_effect = float(full_positive["delta_from_baseline_trend_mm_day_per_century"])
    pakistan_effect = float(ablation["delta_from_baseline_trend_mm_day_per_century"])
    weighted_effect = float(weighted["delta_from_baseline_trend_mm_day_per_century"])
    pakistan_share = pakistan_effect / full_positive_effect if full_positive_effect != 0 else np.nan
    weighted_fraction = weighted_effect / pakistan_effect if pakistan_effect != 0 else np.nan
    material = bool(
        weighted["dual_product_improvement"]
        and np.sign(weighted_effect) == np.sign(pakistan_effect)
        and abs(weighted_fraction) >= 0.25
    )

    scores.to_csv(output / "pakistan_weight_delta_anchor_scores.csv", index=False)
    annual.to_csv(output / "pakistan_weight_delta_anchor_hma_curves_1901_1920.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.3), constrained_layout=True)
    axes[0].plot(YEARS, cru, color="#6A3D9A", linewidth=1.7, label="CRU")
    axes[0].plot(YEARS, gpcc, color="#777777", linewidth=1.7, label="GPCC-derived")
    axes[0].fill_between(YEARS, np.minimum(cru, gpcc), np.maximum(cru, gpcc), color="#BDBDBD", alpha=0.3)
    colors = {"baseline": "#111111", "pakistan_ablation": "#D55E00", "pakistan_design_effect_R": "#0072B2"}
    labels = {"baseline": "Baseline", "pakistan_ablation": "Pakistan removed", "pakistan_design_effect_R": "Pakistan correlation-weighted"}
    for branch in BRANCHES:
        axes[0].plot(YEARS, annual[f"delta_anchor_{branch}_hma_mm_day"], color=colors[branch], linewidth=2.3 if branch == "baseline" else 1.9, label=labels[branch])
    axes[0].axvspan(1915, 1918, color="#E69F00", alpha=0.10)
    axes[0].set_ylabel("HMA precipitation anomaly (mm/day)")
    axes[0].set_title("Pakistan proxy information-weight sensitivity")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].axhline(0, color="#777777", linewidth=0.8)
    for branch in BRANCHES[1:]:
        axes[1].plot(YEARS, annual[f"delta_from_baseline_{branch}_mm_day"], color=colors[branch], linewidth=2.0, label=labels[branch])
    axes[1].axvspan(1915, 1918, color="#E69F00", alpha=0.10)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Change from baseline (mm/day)")
    axes[1].set_title("Pure fixed-EnKF branch effect")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    fig.savefig(output / "pakistan_weight_hma_precipitation_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "pakistan_proxy_count": int(prepare["pakistan_proxy_count"]),
        "design_effect_R_multiplier": float(prepare["design_effect_R_multiplier"]),
        "classical_effective_record_count": float(prepare["classical_effective_record_count"]),
        "spectral_effective_dimension": float(prepare["spectral_effective_dimension"]),
        "closure": {
            "baseline": baseline_closure,
            "update": update_closure,
            "explicit_ablation_vs_zero_gain": ablation_equivalence,
        },
        "scores": scores.to_dict(orient="records"),
        "pakistan_ablation_share_of_full_positive_b2_ablation_trend_effect": float(pakistan_share),
        "correlation_weighting_fraction_of_pakistan_ablation_trend_effect": float(weighted_fraction),
        "classification": "material_pakistan_duplicate_information_weight_sensitivity" if material else "limited_pakistan_duplicate_information_weight_sensitivity",
        "decision_rule": "Material only when correlation weighting improves RMSE against both CRU and GPCC, shifts trend in the same direction as full Pakistan ablation, and realizes at least 25% of the ablation trend effect.",
        "interpretation_boundary": "This is a delta-anchored fixed-EnKF mechanism sensitivity. It does not validate deleting Pakistan proxies, adopting the design-effect R multiplier, or changing the production reconstruction.",
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "analysis_source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-changed-during-pakistan-weight-analysis")
    (output / "FINAL_RESULT.md").write_text(
        "# Pakistan Proxy Weight Sensitivity v1\n\n"
        f"- Design-effect R multiplier: `{prepare['design_effect_R_multiplier']:.3f}`; classical effective count `{prepare['classical_effective_record_count']:.3f}`.\n"
        f"- Pakistan ablation trend shift: `{pakistan_effect:+.3f} mm/day/century`.\n"
        f"- Correlation-weighted trend shift: `{weighted_effect:+.3f} mm/day/century`.\n"
        f"- Pakistan share of the 60-record positive-b2 ablation trend effect: `{pakistan_share:.3f}`.\n"
        f"- Weighting fraction of the Pakistan ablation effect: `{weighted_fraction:.3f}`.\n\n"
        f"Classification: `{summary['classification']}`. This is not a production correction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
