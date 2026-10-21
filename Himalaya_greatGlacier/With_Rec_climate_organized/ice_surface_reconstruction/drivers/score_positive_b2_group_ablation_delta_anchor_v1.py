"""Score a frozen positive-b2 group ablation against both disclosed products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BRANCHES = ["baseline", "positive_b2_P_zero", "positive_b2_full_zero", "positive_b2_T_zero"]
PERIOD = (1901, 1920)


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def decode(handle: h5py.File, reference: h5py.Reference) -> str:
    return "".join(chr(int(value)) for value in np.asarray(handle[reference]).reshape(-1))


def trend(years: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(years, values, 1)[0] * 100.0)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def main() -> int:
    project = root()
    result = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_group_ablation_delta_anchor_v1"
    paths = {
        "ablation": result / "group_ablation.mat",
        "prepare": result / "prepare_summary.json",
        "reference_hma": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_common_mode_reference_phase_audit_v1/common_mode_reference_hma_series_1901_1920.csv",
        "product_bounds": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_cru_gpcc_uncertainty_1901_1920_v1/cru_gpcc_annual_uncertainty_1901_1920.csv",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: digest(path) for name, path in paths.items()}
    with h5py.File(paths["ablation"], "r") as handle:
        names = [decode(handle, ref) for ref in handle["names"][()].reshape(-1)]
        years = np.asarray(handle["years"]).reshape(-1).astype(int)
        total = np.asarray(handle["total"]).T
        baseline_closure = float(np.asarray(handle["baseline_closure"]).squeeze())
        innovation_invariance = float(np.asarray(handle["innovation_invariance"]).squeeze())
        max_increment_closure = float(np.max(np.abs(np.asarray(handle["closure"]))))
    if names != BRANCHES or not np.array_equal(years, np.arange(PERIOD[0], PERIOD[1] + 1)):
        raise RuntimeError("ablation-contract-failed")
    if max(baseline_closure, innovation_invariance, max_increment_closure) > 1e-12:
        raise RuntimeError("closure-contract-failed")
    hma = pd.read_csv(paths["reference_hma"]).set_index("year").loc[years]
    bounds = pd.read_csv(paths["product_bounds"]).set_index("year").loc[years]
    baseline_hma = hma["rec_result_hma_anomaly_mm_day"].to_numpy()
    cru = hma["cru_hma_anomaly_mm_day"].to_numpy()
    gpcc = hma["gpcc_hma_anomaly_mm_day"].to_numpy()
    baseline_increment = total[:, 0]
    annual = pd.DataFrame({"year": years, "cru_hma_mm_day": cru, "gpcc_hma_mm_day": gpcc})
    rows = []
    for index, branch in enumerate(BRANCHES):
        anchored = baseline_hma + (total[:, index] - baseline_increment)
        annual[f"delta_anchor_{branch}_hma_mm_day"] = anchored
        lower = np.minimum(cru, gpcc)
        upper = np.maximum(cru, gpcc)
        rows.append({"branch": branch, "rmse_vs_cru_mm_day": rmse(anchored, cru), "rmse_vs_gpcc_mm_day": rmse(anchored, gpcc), "mean_two_product_rmse_mm_day": (rmse(anchored, cru) + rmse(anchored, gpcc)) / 2.0, "trend_mm_day_per_century": trend(years, anchored), "outside_cru_gpcc_hma_bounds_years": int(((anchored < lower) | (anchored > upper)).sum()), "delta_from_baseline_trend_mm_day_per_century": trend(years, anchored) - trend(years, baseline_hma)})
    scores = pd.DataFrame(rows).set_index("branch")
    baseline_score = scores.loc["baseline"]
    scores["improves_cru_rmse"] = scores["rmse_vs_cru_mm_day"] < baseline_score["rmse_vs_cru_mm_day"]
    scores["improves_gpcc_rmse"] = scores["rmse_vs_gpcc_mm_day"] < baseline_score["rmse_vs_gpcc_mm_day"]
    scores["dual_product_improvement"] = scores["improves_cru_rmse"] & scores["improves_gpcc_rmse"]
    scores = scores.reset_index()
    result_summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_group_ablation_delta_anchor_v1",
        "period": list(PERIOD), "locked_final_period_accessed": False,
        "anchor_definition": "rec_result HMA anomaly plus the branch-minus-baseline frozen-EnKF increment difference; sensitivity anchor only, not a new production reconstruction.",
        "closure": {"baseline": baseline_closure, "innovation_invariance": innovation_invariance, "increment": max_increment_closure},
        "scores": scores.to_dict(orient="records"),
        "acceptance_rule": "A branch is a candidate mechanism target only if it lowers RMSE versus both CRU and GPCC; the two products remain an uncertainty envelope rather than truth.",
    }
    scores.to_csv(result / "delta_anchor_dual_product_scores.csv", index=False)
    annual.to_csv(result / "delta_anchor_hma_curves_1901_1920.csv", index=False)
    (result / "summary.json").write_text(json.dumps(result_summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), constrained_layout=True)
    axes[0].plot(years, cru, color="#6A3D9A", linewidth=1.6, label="CRU")
    axes[0].plot(years, gpcc, color="#7A7A7A", linewidth=1.6, label="GPCC-derived")
    axes[0].fill_between(years, np.minimum(cru, gpcc), np.maximum(cru, gpcc), color="#BDBDBD", alpha=0.35, label="product envelope")
    colors = {"baseline": "#1b1b1b", "positive_b2_P_zero": "#0072B2", "positive_b2_full_zero": "#D55E00", "positive_b2_T_zero": "#009E73"}
    for branch in BRANCHES:
        axes[0].plot(years, annual[f"delta_anchor_{branch}_hma_mm_day"], color=colors[branch], linewidth=2 if branch == "baseline" else 1.5, label=branch)
    axes[0].set_ylabel("HMA anomaly (mm day$^{-1}$)"); axes[0].set_title("Positive-b2 group ablation: delta-anchor sensitivity"); axes[0].legend(frameon=False, ncol=3)
    score_columns = ["rmse_vs_cru_mm_day", "rmse_vs_gpcc_mm_day"]
    positions = np.arange(len(scores)); width = 0.36
    axes[1].bar(positions - width / 2, scores[score_columns[0]], width, color="#6A3D9A", label="RMSE vs CRU")
    axes[1].bar(positions + width / 2, scores[score_columns[1]], width, color="#7A7A7A", label="RMSE vs GPCC-derived")
    axes[1].set_xticks(positions, scores["branch"], rotation=15, ha="right"); axes[1].set_ylabel("RMSE (mm day$^{-1}$)"); axes[1].set_title("Dual-product score: lower is better"); axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(result / "positive_b2_group_ablation_delta_anchor.png", dpi=180); plt.close(figure)
    report = ["# Positive-b2 Group Ablation Delta-anchor Sensitivity", "", "Period: 1901-1920 only. The locked 1921-1939 final test period was not accessed.", "", "A branch is promising only when it improves RMSE against both CRU and GPCC-derived products.", ""]
    for row in scores.itertuples(index=False):
        report.append(f"- {row.branch}: CRU RMSE `{row.rmse_vs_cru_mm_day:.4f}`, GPCC RMSE `{row.rmse_vs_gpcc_mm_day:.4f}`, dual improvement `{row.dual_product_improvement}`.")
    (result / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: digest(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-scoring")
    (result / "score_source_manifest.json").write_text(json.dumps({"before": before, "after": after, "analysis_driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(result_summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
