"""Validate four co-located distinct annual d18O records across GPCC and CRU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("GPCC", "CRU")
EXPECTED_IDS = ("iso2k_1086", "iso2k_1344", "iso2k_503", "iso2k_556")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def standardize(values: np.ndarray) -> np.ndarray:
    return (values - np.nanmean(values)) / np.nanstd(values)


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_39_dod2k_colocated_annual_nested_cross_product_v1"
    paths = {
        "climate": output / "round39_colocated_annual_climate_v1.mat",
        "input": output / "round39_colocated_annual_input.mat",
        "manifest": output / "candidate_manifest.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "preparation_manifest": output / "preparation_source_manifest.json",
        "matlab_driver": drivers / "export_round39_colocated_annual_climate_v1.m",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    manifest = pd.read_csv(paths["manifest"])
    if tuple(manifest.datasetId) != EXPECTED_IDS:
        raise RuntimeError("manifest-contract-failed")
    with h5py.File(paths["climate"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        temperature = np.asarray(handle["temperature_formal_season"], dtype=float).T
        precipitation = {
            "GPCC": np.asarray(handle["precipitation_gpcc_formal_season"], dtype=float).T,
            "CRU": np.asarray(handle["precipitation_cru_formal_season"], dtype=float).T,
        }
    if not np.array_equal(years, np.arange(1940, 2001)) or observations.shape != (61, 4):
        raise RuntimeError("array-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-accessed")

    fold_rows: list[dict[str, object]] = []
    for record, item in enumerate(manifest.itertuples(index=False)):
        y, t = observations[:, record], temperature[:, record]
        expected_sign = 1 if item.interpretation_direction == "positive" else -1
        for fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (years >= start) & (years <= end)
            train = ~test
            for fit_product in PRODUCTS:
                p_fit = precipitation[fit_product][:, record]
                train_tp = train & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_fit)
                train_p = train & np.isfinite(y) & np.isfinite(p_fit)
                coef_t = np.linalg.lstsq(np.column_stack([np.ones(train_tp.sum()), t[train_tp]]), y[train_tp], rcond=None)[0]
                coef_tp = np.linalg.lstsq(np.column_stack([np.ones(train_tp.sum()), t[train_tp], p_fit[train_tp]]), y[train_tp], rcond=None)[0]
                mean_y = float(np.mean(y[train_p]))
                coef_p = np.linalg.lstsq(np.column_stack([np.ones(train_p.sum()), p_fit[train_p]]), y[train_p], rcond=None)[0]
                for eval_product in PRODUCTS:
                    p_eval = precipitation[eval_product][:, record]
                    test_tp = test & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_eval)
                    test_p = test & np.isfinite(y) & np.isfinite(p_eval)
                    pred_t = coef_t[0] + coef_t[1] * t[test_tp]
                    pred_tp = coef_tp[0] + coef_tp[1] * t[test_tp] + coef_tp[2] * p_eval[test_tp]
                    pred_i = np.full(test_p.sum(), mean_y)
                    pred_p = coef_p[0] + coef_p[1] * p_eval[test_p]
                    t_rmse = rmse(y[test_tp], pred_t)
                    i_rmse = rmse(y[test_p], pred_i)
                    fold_rows.append({
                        "datasetId": item.datasetId, "fold": fold, "fit_product": fit_product, "eval_product": eval_product,
                        "bivariate_incremental_improvement": float(1 - rmse(y[test_tp], pred_tp) / t_rmse),
                        "precipitation_over_intercept_improvement": float(1 - rmse(y[test_p], pred_p) / i_rmse),
                        "bivariate_precipitation_coefficient": float(coef_tp[2]),
                        "documented_coefficient_sign": expected_sign,
                    })
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "fold_nested_cross_product_skill.csv", index=False)
    path_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for item in manifest.itertuples(index=False):
        record = folds[folds.datasetId == item.datasetId]
        expected_sign = 1 if item.interpretation_direction == "positive" else -1
        unique_coef = record.drop_duplicates(["fold", "fit_product"])
        direction_fraction = float(np.mean(np.sign(unique_coef.bivariate_precipitation_coefficient) == expected_sign))
        paired = unique_coef.pivot(index="fold", columns="fit_product", values="bivariate_precipitation_coefficient").dropna()
        paired_sign = float(np.mean(np.sign(paired.GPCC) == np.sign(paired.CRU)))
        gates: dict[str, bool] = {}
        for fit_product in PRODUCTS:
            for eval_product in PRODUCTS:
                selected = record[(record.fit_product == fit_product) & (record.eval_product == eval_product)]
                incremental = selected.bivariate_incremental_improvement.to_numpy(float)
                precipitation_only = selected.precipitation_over_intercept_improvement.to_numpy(float)
                key = f"{fit_product}_fit_{eval_product}_eval"
                path_rows.append({
                    "datasetId": item.datasetId, "fit_product": fit_product, "eval_product": eval_product,
                    "median_bivariate_incremental_improvement": float(np.median(incremental)),
                    "improved_bivariate_fold_count": int(np.sum(incremental > 0)),
                    "median_precipitation_over_intercept_improvement": float(np.median(precipitation_only)),
                })
                gates[f"{key}_median_incremental_at_least_1pct"] = bool(np.median(incremental) >= 0.01)
                gates[f"{key}_at_least_4_of_5_folds_improve"] = bool(np.sum(incremental > 0) >= 4)
                if fit_product == eval_product:
                    gates[f"{key}_precipitation_only_median_at_least_1pct"] = bool(np.median(precipitation_only) >= 0.01)
        gates["documented_direction_fraction_at_least_70pct"] = direction_fraction >= 0.70
        gates["gpcc_cru_fit_sign_agreement_at_least_70pct"] = paired_sign >= 0.70
        gate_rows.append({
            "datasetId": item.datasetId, "documented_direction_fraction": direction_fraction,
            "gpcc_cru_fit_sign_agreement": paired_sign, "passes_all_record_gates": bool(all(gates.values())),
            "failed_gate_count": int(sum(not value for value in gates.values())), "gates_json": json.dumps(gates, sort_keys=True),
        })
    paths_frame = pd.DataFrame(path_rows)
    gates_frame = pd.DataFrame(gate_rows)
    paths_frame.to_csv(output / "record_path_summary.csv", index=False)
    gates_frame.to_csv(output / "record_promotion_gates.csv", index=False)
    promoted = gates_frame.loc[gates_frame.passes_all_record_gates, "datasetId"].tolist()
    summary = {
        "status": "complete", "locked_1921_1939_accessed": False, "candidate_count": 4,
        "promoted_record_count": len(promoted), "promoted_record_ids": promoted,
        "record_results": gates_frame.set_index("datasetId")[["documented_direction_fraction", "gpcc_cru_fit_sign_agreement", "passes_all_record_gates", "failed_gate_count"]].to_dict(orient="index"),
        "site_cluster_boundary": "passing records remain co-located measurement clusters, not independent spatial sites",
        "decision": "advance_passing_colocated_measurements_to_cluster_aware_network_test" if promoted else "close_colocated_annual_dod2k_psm_candidates",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(4, 2, figsize=(12.5, 12), constrained_layout=True)
    for record, item in enumerate(manifest.itertuples(index=False)):
        orientation = 1 if item.interpretation_direction == "positive" else -1
        axes[record, 0].plot(years, standardize(orientation * observations[:, record]), color="#222222", label="Proxy wetness orientation")
        axes[record, 0].plot(years, standardize(precipitation["GPCC"][:, record]), color="#0072b2", label="GPCC")
        axes[record, 0].plot(years, standardize(precipitation["CRU"][:, record]), color="#d55e00", label="CRU")
        axes[record, 0].set(title=item.datasetId, ylabel="Standardized anomaly")
        selected = paths_frame[paths_frame.datasetId == item.datasetId]
        labels = [f"{row.fit_product}->{row.eval_product}" for row in selected.itertuples(index=False)]
        skill = selected.median_bivariate_incremental_improvement.to_numpy(float) * 100
        axes[record, 1].bar(labels, skill, color=["#0072b2" if value >= 1 else "#d55e00" for value in skill])
        axes[record, 1].axhline(0, color="#555555", linewidth=0.8)
        axes[record, 1].axhline(1, color="#009e73", linestyle="--")
        axes[record, 1].set(title="Held-out incremental skill", ylabel="RMSE improvement (%)")
        axes[record, 1].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1, 0].set(xlabel="Year")
    figure.suptitle("Co-located annual d18O candidates: curves and nested skill", fontsize=14)
    figure.savefig(output / "colocated_annual_nested_cross_product.png", dpi=180)
    plt.close(figure)

    lines = ["# Co-Located Annual Moisture Nested Validation", "", f"- Candidates: `4`; passing all gates: `{len(promoted)}`."]
    for row in gates_frame.itertuples(index=False):
        lines.append(f"- {row.datasetId}: pass `{str(bool(row.passes_all_record_gates)).lower()}`; direction `{row.documented_direction_fraction:.1%}`; GPCC/CRU sign agreement `{row.gpcc_cru_fit_sign_agreement:.1%}`; failed gates `{row.failed_gate_count}`.")
    lines.extend(["", f"Decision: `{summary['decision']}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
