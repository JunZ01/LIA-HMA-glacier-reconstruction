"""Five-year blocked interpretability test for five low-frequency records."""

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


PRODUCTS = ("GPCC", "CRU")
EXPECTED_IDS = ("iso2k_1397", "sisal_111.0_64", "sisal_539.0_336", "sisal_543.0_340", "sisal_900.0_540")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    stat = path.stat(); return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def rmse(a: np.ndarray, b: np.ndarray) -> float: return float(np.sqrt(np.mean((a - b) ** 2)))


def zscore(a: np.ndarray) -> np.ndarray: return (a - np.nanmean(a)) / np.nanstd(a)


def main() -> int:
    project = root(); drivers = Path(__file__).resolve().parent
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_40_dod2k_low_frequency_interpretability_v1"
    paths = {"climate": output / "round40_low_frequency_climate_v1.mat", "input": output / "round40_low_frequency_input.mat", "manifest": output / "candidate_manifest.csv", "contract": output / "ROUND_CONTRACT.md", "preparation_manifest": output / "preparation_source_manifest.json", "matlab_driver": drivers / "export_round40_low_frequency_climate_v1.m", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}; manifest = pd.read_csv(paths["manifest"])
    with h5py.File(paths["climate"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel(); observations = np.asarray(handle["observations"], dtype=float).T
        temperature = np.asarray(handle["temperature"], dtype=float).T
        precipitation = {"GPCC": np.asarray(handle["precipitation_gpcc"], dtype=float).T, "CRU": np.asarray(handle["precipitation_cru"], dtype=float).T}
    if tuple(manifest.datasetId) != EXPECTED_IDS or not np.array_equal(years, np.arange(1940, 2001)) or observations.shape != (61, 5): raise RuntimeError("input-contract-failed")
    bin_starts = np.arange(1940, 2000, 5); bin_years = bin_starts + 2
    proxy_bins = np.full((12, 5), np.nan); temp_bins = np.full((12, 5), np.nan); precip_bins = {key: np.full((12, 5), np.nan) for key in PRODUCTS}
    for b, start in enumerate(bin_starts):
        mask = (years >= start) & (years <= start + 4)
        for record in range(5):
            finite = mask & np.isfinite(observations[:, record]); proxy_bins[b, record] = np.mean(observations[finite, record]) if finite.sum() >= 1 else np.nan
            temp_bins[b, record] = np.nanmean(temperature[mask, record])
            for product in PRODUCTS: precip_bins[product][b, record] = np.nanmean(precipitation[product][mask, record])
    fold_rows: list[dict[str, object]] = []
    for record, identity in enumerate(EXPECTED_IDS):
        y, t = proxy_bins[:, record], temp_bins[:, record]
        for fold in range(6):
            test = np.zeros(12, dtype=bool); test[2 * fold:2 * fold + 2] = True; train = ~test
            for fit_product in PRODUCTS:
                p_fit = precip_bins[fit_product][:, record]; train_tp = train & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_fit); train_p = train & np.isfinite(y) & np.isfinite(p_fit)
                if train_tp.sum() < 6: continue
                coef_t = np.linalg.lstsq(np.column_stack([np.ones(train_tp.sum()), t[train_tp]]), y[train_tp], rcond=None)[0]
                coef_tp = np.linalg.lstsq(np.column_stack([np.ones(train_tp.sum()), t[train_tp], p_fit[train_tp]]), y[train_tp], rcond=None)[0]
                coef_p = np.linalg.lstsq(np.column_stack([np.ones(train_p.sum()), p_fit[train_p]]), y[train_p], rcond=None)[0]; mean_y = float(np.mean(y[train_p]))
                for eval_product in PRODUCTS:
                    p_eval = precip_bins[eval_product][:, record]; test_tp = test & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_eval); test_p = test & np.isfinite(y) & np.isfinite(p_eval)
                    if test_tp.sum() < 1: continue
                    pred_t = coef_t[0] + coef_t[1] * t[test_tp]; pred_tp = coef_tp[0] + coef_tp[1] * t[test_tp] + coef_tp[2] * p_eval[test_tp]
                    pred_i = np.full(test_p.sum(), mean_y); pred_p = coef_p[0] + coef_p[1] * p_eval[test_p]
                    fold_rows.append({"datasetId": identity, "fold": fold + 1, "fit_product": fit_product, "eval_product": eval_product, "test_bin_count": int(test_tp.sum()), "bivariate_incremental_improvement": float(1 - rmse(y[test_tp], pred_tp) / rmse(y[test_tp], pred_t)), "precipitation_over_intercept_improvement": float(1 - rmse(y[test_p], pred_p) / rmse(y[test_p], pred_i)), "bivariate_precipitation_coefficient": float(coef_tp[2])})
    folds = pd.DataFrame(fold_rows); folds.to_csv(output / "five_year_fold_skill.csv", index=False)
    path_rows: list[dict[str, object]] = []; gate_rows: list[dict[str, object]] = []
    for record, identity in enumerate(EXPECTED_IDS):
        selected_record = folds[folds.datasetId == identity]; unique_coef = selected_record.drop_duplicates(["fold", "fit_product"])
        signs = np.sign(unique_coef.bivariate_precipitation_coefficient.to_numpy(float)); nonzero = signs[signs != 0]
        positive_fraction = float(np.mean(nonzero > 0)) if len(nonzero) else np.nan; modal_sign = 1 if positive_fraction >= 0.5 else -1
        modal_fraction = float(max(positive_fraction, 1 - positive_fraction)) if np.isfinite(positive_fraction) else 0.0
        paired = unique_coef.pivot(index="fold", columns="fit_product", values="bivariate_precipitation_coefficient").dropna(); paired_sign = float(np.mean(np.sign(paired.GPCC) == np.sign(paired.CRU))) if len(paired) else 0.0
        gates: dict[str, bool] = {"usable_five_year_bins_at_least_10": int(np.isfinite(proxy_bins[:, record]).sum()) >= 10}
        for fit_product in PRODUCTS:
            for eval_product in PRODUCTS:
                path = selected_record[(selected_record.fit_product == fit_product) & (selected_record.eval_product == eval_product)]; inc = path.bivariate_incremental_improvement.to_numpy(float); po = path.precipitation_over_intercept_improvement.to_numpy(float); key = f"{fit_product}_fit_{eval_product}_eval"
                path_rows.append({"datasetId": identity, "fit_product": fit_product, "eval_product": eval_product, "valid_fold_count": len(path), "median_bivariate_incremental_improvement": float(np.median(inc)) if len(inc) else np.nan, "improved_fold_count": int(np.sum(inc > 0)), "median_precipitation_over_intercept_improvement": float(np.median(po)) if len(po) else np.nan})
                gates[f"{key}_median_incremental_at_least_1pct"] = bool(len(inc) >= 4 and np.median(inc) >= 0.01); gates[f"{key}_at_least_4_folds_improve"] = bool(np.sum(inc > 0) >= 4)
                if fit_product == eval_product: gates[f"{key}_precipitation_only_median_at_least_1pct"] = bool(len(po) >= 4 and np.median(po) >= 0.01)
        gates["modal_coefficient_sign_fraction_at_least_70pct"] = modal_fraction >= 0.70; gates["gpcc_cru_fit_sign_agreement_at_least_70pct"] = paired_sign >= 0.70
        gate_rows.append({"datasetId": identity, "usable_five_year_bin_count": int(np.isfinite(proxy_bins[:, record]).sum()), "inferred_precipitation_direction": "positive" if modal_sign > 0 else "negative", "modal_coefficient_sign_fraction": modal_fraction, "gpcc_cru_fit_sign_agreement": paired_sign, "passes_all_gates": bool(all(gates.values())), "failed_gate_count": int(sum(not value for value in gates.values())), "gates_json": json.dumps(gates, sort_keys=True)})
    path_frame = pd.DataFrame(path_rows); gate_frame = pd.DataFrame(gate_rows); path_frame.to_csv(output / "record_path_summary.csv", index=False); gate_frame.to_csv(output / "record_promotion_gates.csv", index=False)
    promoted = gate_frame.loc[gate_frame.passes_all_gates, "datasetId"].tolist(); summary = {"status": "complete", "locked_1921_1939_accessed": False, "candidate_count": 5, "promoted_record_count": len(promoted), "promoted_record_ids": promoted, "record_results": gate_frame.set_index("datasetId")[["usable_five_year_bin_count", "inferred_precipitation_direction", "modal_coefficient_sign_fraction", "gpcc_cru_fit_sign_agreement", "passes_all_gates", "failed_gate_count"]].to_dict(orient="index"), "decision": "advance_passing_records_to_lia_50year_validation" if promoted else "close_independent_low_frequency_dod2k_records"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(5, 1, figsize=(11.5, 12), sharex=True, constrained_layout=True)
    for record, row in enumerate(gate_frame.itertuples(index=False)):
        orientation = 1 if row.inferred_precipitation_direction == "positive" else -1
        axes[record].plot(bin_years, zscore(orientation * proxy_bins[:, record]), marker="o", color="#222222", label="Proxy wetness orientation")
        axes[record].plot(bin_years, zscore(precip_bins["GPCC"][:, record]), marker="o", color="#0072b2", label="GPCC")
        axes[record].plot(bin_years, zscore(precip_bins["CRU"][:, record]), marker="o", color="#d55e00", label="CRU")
        axes[record].set(ylabel="z", title=f"{row.datasetId} | pass={bool(row.passes_all_gates)}")
    axes[0].legend(frameon=False, ncol=3); axes[-1].set(xlabel="Five-year bin center")
    figure.suptitle("Independent low-frequency proxy and precipitation curves", fontsize=14); figure.savefig(output / "low_frequency_interpretability_curves.png", dpi=180); plt.close(figure)
    lines = ["# Independent Low-Frequency Interpretability", "", f"- Candidates: `5`; passing all gates: `{len(promoted)}`."]
    for row in gate_frame.itertuples(index=False): lines.append(f"- {row.datasetId}: pass `{str(bool(row.passes_all_gates)).lower()}`; inferred `{row.inferred_precipitation_direction}`; sign stability `{row.modal_coefficient_sign_fraction:.1%}`; GPCC/CRU agreement `{row.gpcc_cru_fit_sign_agreement:.1%}`; failed gates `{row.failed_gate_count}`.")
    lines.extend(["", f"Decision: `{summary['decision']}`."]); (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after; (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
