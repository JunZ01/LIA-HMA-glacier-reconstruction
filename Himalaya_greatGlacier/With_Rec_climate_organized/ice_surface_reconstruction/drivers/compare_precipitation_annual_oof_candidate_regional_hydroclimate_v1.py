"""Compare the annual-OOF early candidate with regional hydroclimate evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_candidate_regional_hydroclimate_v1"
YEARS = np.arange(1901, 1921, dtype=int)


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": value}


def weighted_series(field: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    denominator = float(weights.sum())
    if denominator <= 0:
        raise RuntimeError("empty-regional-mask")
    return np.sum(field * weights[None], axis=(1, 2)) / denominator


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / np.std(values)


def metrics(evidence: np.ndarray, candidate: np.ndarray, production: np.ndarray) -> dict[str, object]:
    ez = zscore(evidence)
    cz = zscore(candidate)
    pz = zscore(production)
    e_trend = float(np.polyfit(YEARS, ez, 1)[0] * 100)
    c_trend = float(np.polyfit(YEARS, cz, 1)[0] * 100)
    p_trend = float(np.polyfit(YEARS, pz, 1)[0] * 100)
    return {
        "candidate_pearson_r": float(pearsonr(ez, cz).statistic),
        "production_pearson_r": float(pearsonr(ez, pz).statistic),
        "candidate_spearman_r": float(spearmanr(ez, cz).statistic),
        "production_spearman_r": float(spearmanr(ez, pz).statistic),
        "evidence_trend_z_per_century": e_trend,
        "candidate_trend_z_per_century": c_trend,
        "production_trend_z_per_century": p_trend,
        "candidate_trend_sign_match": bool(np.sign(e_trend) == np.sign(c_trend)),
        "production_trend_sign_match": bool(np.sign(e_trend) == np.sign(p_trend)),
    }


def main() -> int:
    project = root()
    result_root = project / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    evidence_root = result_root / "climate_reconstruction_improvement_loop_v2/round_102_tree_ring_hydroclimate_evidence_v4"
    sources = {
        "candidate": result_root / "climate_reconstruction_precipitation_annual_oof_early_candidate_v2/annual_oof_candidate_native_2deg_1901_1920.npz",
        "candidate_summary": result_root / "climate_reconstruction_precipitation_annual_oof_early_candidate_v2/summary.json",
        "catalog": evidence_root / "evidence_catalog_and_overlap.csv",
        "evidence_curves": evidence_root / "allowed_period_standardized_curves.csv",
        "sequence_overlap": result_root / "climate_reconstruction_improvement_loop_v2/round_103_hydroclimate_proxy_sequence_overlap_v1/summary.json",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: state(path) for name, path in sources.items()}
    with np.load(sources["candidate"]) as handle:
        years = np.asarray(handle["years"], dtype=int)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        field = np.asarray(handle["increment_mm_day"], dtype=float)
    if not np.array_equal(years, YEARS) or field.shape != (20, len(lat), len(lon)):
        raise RuntimeError("candidate-field-contract-failed")
    catalog = pd.read_csv(sources["catalog"])
    curves = pd.read_csv(sources["evidence_curves"])
    early = curves[curves["period"] == "early_1901_1920"]
    sequence = json.loads(sources["sequence_overlap"].read_text(encoding="utf-8"))
    sequence_class = {row["evidence"]: row["best_sequence_class"] for row in sequence["early_sequence_results"]}

    metric_rows = []
    curve_rows = []
    candidate_store: dict[str, np.ndarray] = {}
    evidence_store: dict[str, np.ndarray] = {}
    production_store: dict[str, np.ndarray] = {}
    for _, row in catalog.iterrows():
        name = row["evidence"]
        mask = (
            (lat[:, None] >= row["south"]) & (lat[:, None] <= row["north"])
            & (lon[None, :] >= row["west"]) & (lon[None, :] <= row["east"])
        )
        if not np.any(mask):
            iy = int(np.argmin(np.abs(lat - (row["south"] + row["north"]) / 2)))
            ix = int(np.argmin(np.abs(lon - (row["west"] + row["east"]) / 2)))
            mask[iy, ix] = True
        candidate = weighted_series(field, mask, lat)
        local = early[(early["evidence"] == name) & (early["region"] == name)]
        evidence = local[local["series"] == "evidence"].sort_values("year")["raw_value"].to_numpy(float)
        production = local[local["series"] == "project_mean"].sort_values("year")["raw_value"].to_numpy(float)
        if len(evidence) != 20 or len(production) != 20:
            raise RuntimeError(f"regional-curve-contract-failed:{name}:{len(evidence)}:{len(production)}")
        result = metrics(evidence, candidate, production)
        result.update({
            "evidence": name,
            "overlap_class": row["overlap_class"],
            "sequence_class": sequence_class.get(name, "not_assessed"),
            "candidate_minus_production_pearson_r": result["candidate_pearson_r"] - result["production_pearson_r"],
        })
        metric_rows.append(result)
        candidate_store[name] = zscore(candidate)
        evidence_store[name] = zscore(evidence)
        production_store[name] = zscore(production)
        for year, ev, ca, pr in zip(YEARS, evidence_store[name], candidate_store[name], production_store[name]):
            curve_rows.extend([
                {"evidence": name, "year": int(year), "series": "evidence", "z_value": float(ev)},
                {"evidence": name, "year": int(year), "series": "annual_oof_candidate", "z_value": float(ca)},
                {"evidence": name, "year": int(year), "series": "production_equal_six", "z_value": float(pr)},
            ])
    metrics_frame = pd.DataFrame(metric_rows)
    output.mkdir(parents=True)
    metrics_frame.to_csv(output / "regional_hydroclimate_phase_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "regional_hydroclimate_phase_curves.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    for axis, name in zip(axes.ravel(), catalog["evidence"]):
        row = metrics_frame[metrics_frame["evidence"] == name].iloc[0]
        axis.plot(YEARS, evidence_store[name], color="#0072B2", linewidth=2, label="Evidence")
        axis.plot(YEARS, production_store[name], color="#D55E00", linewidth=1.7, label="Production")
        axis.plot(YEARS, candidate_store[name], color="#009E73", linewidth=2.1, label="Annual OOF candidate")
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.set(title=f"{name}: candidate r={row['candidate_pearson_r']:+.2f}", ylabel="Standardized anomaly")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(output / "candidate_regional_hydroclimate_phase_curves.png", dpi=220)
    plt.close(fig)

    unmatched = metrics_frame[~metrics_frame["sequence_class"].isin(["strong_sequence_similarity", "near_exact_match"])]
    external_sign_matches = int(unmatched["candidate_trend_sign_match"].sum())
    external_count = int(len(unmatched))
    decision = "regional_evidence_mixed_no_proxy75_deletion" if external_sign_matches not in {0, external_count} else "regional_evidence_directionally_consistent"
    after = {name: state(path) for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "metrics": metrics_frame.to_dict(orient="records"),
        "non_strong_sequence_evidence_count": external_count,
        "candidate_trend_sign_matches_non_strong_sequence_count": external_sign_matches,
        "decision": decision,
        "all_declared_sources_unchanged": before == after,
        "interpretation_boundary": "All evidence targets are seasonal while the candidate is annual. Strong sequence matches are internal consistency, and all comparisons diagnose phase rather than precipitation amplitude.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Annual OOF Candidate Regional Hydroclimate Evidence\n\n"
        + "\n".join(
            f"- {row.evidence}: candidate/production r `{row.candidate_pearson_r:+.3f}/{row.production_pearson_r:+.3f}`, candidate trend-sign match `{row.candidate_trend_sign_match}`, sequence class `{row.sequence_class}`."
            for row in metrics_frame.itertuples()
        )
        + f"\n- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-regional-evidence-comparison")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
