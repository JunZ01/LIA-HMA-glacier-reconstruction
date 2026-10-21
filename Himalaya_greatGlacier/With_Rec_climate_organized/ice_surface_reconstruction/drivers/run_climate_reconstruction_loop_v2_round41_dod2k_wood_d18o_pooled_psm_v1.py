"""Test a shared precipitation slope across seven direction-oriented wood-d18O records."""

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


def bootstrap_median(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed); sampled = np.empty(draws)
    for draw in range(draws): sampled[draw] = np.median(values[rng.integers(0, len(values), size=len(values))])
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


def read_export(path: Path, manifest_path: Path, prefix: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    manifest = pd.read_csv(manifest_path).copy(); manifest["source_round"] = prefix
    with h5py.File(path, "r") as handle:
        observations = np.asarray(handle["observations"], dtype=float).T
        if prefix == "round37":
            temperature = np.asarray(handle["temperature_formal_season"], dtype=float).T
            precipitation = {"GPCC": np.asarray(handle["precipitation_gpcc_formal_season"], dtype=float).T, "CRU": np.asarray(handle["precipitation_cru_formal_season"], dtype=float).T}
        else:
            temperature = np.asarray(handle["temperature_formal_season"], dtype=float).T
            precipitation = {"GPCC": np.asarray(handle["precipitation_gpcc_formal_season"], dtype=float).T, "CRU": np.asarray(handle["precipitation_cru_formal_season"], dtype=float).T}
    if observations.shape[0] != 61 or observations.shape[1] != len(manifest): raise RuntimeError(f"shape-contract-failed:{prefix}")
    return manifest, observations, temperature, precipitation


def main() -> int:
    project = root(); loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_41_dod2k_wood_d18o_pooled_psm_v1"
    r37 = loop / "round_37_dod2k_annual_moisture_nested_cross_product_v1"; r39 = loop / "round_39_dod2k_colocated_annual_nested_cross_product_v1"
    paths = {"round37_climate": r37 / "round37_dod2k_annual_climate_v1.mat", "round37_manifest": r37 / "candidate_manifest.csv", "round39_climate": r39 / "round39_colocated_annual_climate_v1.mat", "round39_manifest": r39 / "candidate_manifest.csv", "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    m37, y37, t37, p37 = read_export(paths["round37_climate"], paths["round37_manifest"], "round37")
    m39, y39, t39, p39 = read_export(paths["round39_climate"], paths["round39_manifest"], "round39")
    manifest = pd.concat([m37, m39], ignore_index=True); observations = np.column_stack([y37, y39]); temperature = np.column_stack([t37, t39]); precipitation = {product: np.column_stack([p37[product], p39[product]]) for product in PRODUCTS}
    if len(manifest) != 7 or manifest.datasetId.duplicated().any(): raise RuntimeError("seven-record-contract-failed")
    orientations = np.where(manifest.interpretation_direction.eq("positive"), 1.0, -1.0); observations = observations * orientations[None, :]
    years = np.arange(1940, 2001); unit_rows: list[dict[str, object]] = []; coefficient_rows: list[dict[str, object]] = []
    record_count = 7
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test_year = (years >= start) & (years <= end); train_year = ~test_year
        for fit_product in PRODUCTS:
            stats: list[dict[str, float]] = []; baseline_rows: list[np.ndarray] = []; full_rows: list[np.ndarray] = []; precip_rows: list[np.ndarray] = []; target_rows: list[float] = []
            for record in range(record_count):
                valid = train_year & np.isfinite(observations[:, record]) & np.isfinite(temperature[:, record]) & np.isfinite(precipitation[fit_product][:, record])
                y = observations[valid, record]; t = temperature[valid, record]; p = precipitation[fit_product][valid, record]
                if len(y) < 20: raise RuntimeError(f"insufficient-training:{record}:{fold}:{fit_product}")
                sy = float(np.std(y)); st = float(np.std(t)); sp = float(np.std(p))
                if min(sy, st, sp) <= 0: raise RuntimeError("zero-training-scale")
                stats.append({"my": float(np.mean(y)), "sy": sy, "mt": float(np.mean(t)), "st": st, "mp": float(np.mean(p)), "sp": sp})
                for yz, tz, pz in zip((y - np.mean(y)) / sy, (t - np.mean(t)) / st, (p - np.mean(p)) / sp):
                    base = np.zeros(2 * record_count); base[record] = 1; base[record_count + record] = tz
                    baseline_rows.append(base); full_rows.append(np.concatenate([base, [pz]])); precip_base = np.zeros(record_count + 1); precip_base[record] = 1; precip_base[-1] = pz; precip_rows.append(precip_base); target_rows.append(float(yz))
            target = np.asarray(target_rows); coef_base = np.linalg.lstsq(np.vstack(baseline_rows), target, rcond=None)[0]; coef_full = np.linalg.lstsq(np.vstack(full_rows), target, rcond=None)[0]; coef_precip = np.linalg.lstsq(np.vstack(precip_rows), target, rcond=None)[0]
            coefficient_rows.append({"fold": fold, "fit_product": fit_product, "shared_bivariate_precipitation_coefficient": float(coef_full[-1]), "shared_precipitation_only_coefficient": float(coef_precip[-1])})
            for eval_product in PRODUCTS:
                for record in range(record_count):
                    s = stats[record]; valid = test_year & np.isfinite(observations[:, record]) & np.isfinite(temperature[:, record]) & np.isfinite(precipitation[eval_product][:, record])
                    yz = (observations[valid, record] - s["my"]) / s["sy"]; tz = (temperature[valid, record] - s["mt"]) / s["st"]; pz = (precipitation[eval_product][valid, record] - s["mp"]) / s["sp"]
                    base = np.zeros((len(yz), 2 * record_count)); base[:, record] = 1; base[:, record_count + record] = tz
                    full = np.column_stack([base, pz]); intercept = np.zeros((len(yz), record_count)); intercept[:, record] = 1; precip_design = np.column_stack([intercept, pz])
                    rb = rmse(yz, base @ coef_base); rf = rmse(yz, full @ coef_full); ri = rmse(yz, intercept @ coef_precip[:-1]); rp = rmse(yz, precip_design @ coef_precip)
                    unit_rows.append({"datasetId": manifest.iloc[record].datasetId, "fold": fold, "fit_product": fit_product, "eval_product": eval_product, "test_count": int(len(yz)), "bivariate_incremental_improvement": float(1 - rf / rb), "precipitation_over_intercept_improvement": float(1 - rp / ri)})
    units = pd.DataFrame(unit_rows); coefficients = pd.DataFrame(coefficient_rows); units.to_csv(output / "record_fold_skill.csv", index=False); coefficients.to_csv(output / "shared_coefficients.csv", index=False)
    record_rows: list[dict[str, object]] = []; path_summary: dict[str, dict[str, float]] = {}; all_positive = np.ones(record_count, dtype=bool); seed = 4100
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            key = f"{fit_product}_fit_{eval_product}_eval"; selected = units[(units.fit_product == fit_product) & (units.eval_product == eval_product)]
            medians = selected.groupby("datasetId").bivariate_incremental_improvement.median().reindex(manifest.datasetId).to_numpy(float); precip_medians = selected.groupby("datasetId").precipitation_over_intercept_improvement.median().reindex(manifest.datasetId).to_numpy(float)
            lower, upper = bootstrap_median(medians, seed); seed += 1
            path_summary[key] = {"median_bivariate_incremental_improvement": float(np.median(medians)), "record_cluster_bootstrap_95_lower": lower, "record_cluster_bootstrap_95_upper": upper, "median_precipitation_over_intercept_improvement": float(np.median(precip_medians)), "positive_record_fraction": float(np.mean(medians > 0))}
            all_positive &= medians > 0
            for identity, value, pvalue in zip(manifest.datasetId, medians, precip_medians): record_rows.append({"datasetId": identity, "fit_product": fit_product, "eval_product": eval_product, "median_bivariate_incremental_improvement": value, "median_precipitation_over_intercept_improvement": pvalue})
    records = pd.DataFrame(record_rows); records.to_csv(output / "record_path_skill.csv", index=False)
    positive_coefficient_fraction = float(np.mean(coefficients.shared_bivariate_precipitation_coefficient > 0)); paired = coefficients.pivot(index="fold", columns="fit_product", values="shared_bivariate_precipitation_coefficient"); paired_sign = float(np.mean(np.sign(paired.GPCC) == np.sign(paired.CRU)))
    within = ("GPCC_fit_GPCC_eval", "CRU_fit_CRU_eval")
    gates = {"all_four_medians_at_least_1pct": all(item["median_bivariate_incremental_improvement"] >= 0.01 for item in path_summary.values()), "all_four_bootstrap_lowers_positive": all(item["record_cluster_bootstrap_95_lower"] > 0 for item in path_summary.values()), "both_within_precipitation_only_medians_at_least_1pct": all(path_summary[key]["median_precipitation_over_intercept_improvement"] >= 0.01 for key in within), "positive_all_four_record_fraction_over_50pct": float(np.mean(all_positive)) > 0.5, "wet_positive_shared_coefficient_fraction_at_least_70pct": positive_coefficient_fraction >= 0.70, "gpcc_cru_shared_coefficient_sign_agreement_at_least_70pct": paired_sign >= 0.70}
    passes = bool(all(gates.values())); summary = {"status": "complete", "locked_1921_1939_accessed": False, "record_count": 7, "product_paths": path_summary, "positive_all_four_record_count": int(all_positive.sum()), "positive_all_four_record_fraction": float(np.mean(all_positive)), "wet_positive_shared_coefficient_fraction": positive_coefficient_fraction, "gpcc_cru_shared_coefficient_sign_agreement": paired_sign, "gates": gates, "passes_pooled_wood_d18o_psm": passes, "decision": "advance_to_leave_one_record_out_influence_test" if passes else "close_pooled_wood_d18o_psm"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = list(path_summary); plot_values = [records[(records.fit_product == key.split("_fit_")[0]) & (records.eval_product == key.split("_fit_")[1].replace("_eval", ""))].median_bivariate_incremental_improvement.to_numpy(float) * 100 for key in keys]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True); axes[0].boxplot(plot_values, tick_labels=[key.replace("_fit_", "->").replace("_eval", "") for key in keys], showfliers=False); axes[0].axhline(0, color="#555555"); axes[0].axhline(1, color="#009e73", linestyle="--"); axes[0].set(ylabel="Record median RMSE improvement (%)", title="Shared-slope incremental skill")
    axes[1].plot(coefficients[coefficients.fit_product == "GPCC"].fold, coefficients[coefficients.fit_product == "GPCC"].shared_bivariate_precipitation_coefficient, marker="o", label="GPCC fit"); axes[1].plot(coefficients[coefficients.fit_product == "CRU"].fold, coefficients[coefficients.fit_product == "CRU"].shared_bivariate_precipitation_coefficient, marker="o", label="CRU fit"); axes[1].axhline(0, color="#555555"); axes[1].set(xlabel="Held-out block", ylabel="Shared coefficient", title="Wet-oriented class coefficient"); axes[1].legend(frameon=False); figure.suptitle("Pooled wood-d18O moisture PSM", fontsize=14); figure.savefig(output / "pooled_wood_d18o_psm.png", dpi=180); plt.close(figure)
    lines = ["# Pooled Wood-d18O Moisture PSM", "", f"- Seven records; positive on all four paths: `{all_positive.sum()}/7`.", f"- Wet-positive coefficient fraction: `{positive_coefficient_fraction:.1%}`; GPCC/CRU sign agreement: `{paired_sign:.1%}`."]
    for key, item in path_summary.items(): lines.append(f"- {key}: `{item['median_bivariate_incremental_improvement']:.2%}` (cluster 95% `{item['record_cluster_bootstrap_95_lower']:.2%}` to `{item['record_cluster_bootstrap_95_upper']:.2%}`).")
    lines.extend(["", f"All gates pass: `{str(passes).lower()}`.", f"Decision: `{summary['decision']}`."]); (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after; (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged, "bootstrap_draws": 5000}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
