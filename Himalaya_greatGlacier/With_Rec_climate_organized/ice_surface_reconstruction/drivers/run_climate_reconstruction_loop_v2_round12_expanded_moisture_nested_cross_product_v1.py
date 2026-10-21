"""Evaluate the expanded JJA moisture network with nested GPCC/CRU controls."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]
PRODUCTS = ("GPCC", "CRU")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat(); digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    values = values[np.isfinite(values)]; rng = np.random.default_rng(seed); result = np.empty(draws)
    for draw in range(draws): result[draw] = np.median(values[rng.integers(0, len(values), size=len(values))])
    return float(np.percentile(result, 2.5)), float(np.percentile(result, 97.5))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    project = root(); drivers = Path(__file__).resolve().parent
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_12_expanded_moisture_nested_cross_product_v1"
    paths = {
        "climate_export": output / "expanded_moisture_jja_climate_v1.mat",
        "network_input": output / "expanded_moisture_network_v1.mat",
        "network_manifest": output / "network_manifest.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "preparation_manifest": output / "preparation_source_manifest.json",
        "matlab_driver": drivers / "export_round12_moisture_jja_v1.m",
        "analysis_driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    manifest = pd.read_csv(paths["network_manifest"])
    if len(manifest) != 22 or manifest[["latitude", "longitude"]].drop_duplicates().shape[0] != 21:
        raise RuntimeError("network-manifest-contract-failed")
    with h5py.File(paths["climate_export"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        temperature = np.asarray(handle["temperature_jja"], dtype=float).T
        precipitation = {
            "GPCC": np.asarray(handle["precipitation_gpcc_jja"], dtype=float).T,
            "CRU": np.asarray(handle["precipitation_cru_jja"], dtype=float).T,
        }
    if not np.array_equal(years, np.arange(1940, 2001)) or observations.shape != (61, 22):
        raise RuntimeError("array-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)): raise RuntimeError("locked-period-accessed")

    fold_rows: list[dict[str, object]] = []
    for record in range(22):
        y = observations[:, record]; t = temperature[:, record]
        for fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (years >= start) & (years <= end); train = ~test
            for fit_product in PRODUCTS:
                p_fit = precipitation[fit_product][:, record]
                valid_train_tp = train & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_fit)
                valid_train_p = train & np.isfinite(y) & np.isfinite(p_fit)
                if valid_train_tp.sum() < 20 or valid_train_p.sum() < 20: continue
                coef_t = np.linalg.lstsq(np.column_stack([np.ones(valid_train_tp.sum()), t[valid_train_tp]]), y[valid_train_tp], rcond=None)[0]
                coef_tp = np.linalg.lstsq(np.column_stack([np.ones(valid_train_tp.sum()), t[valid_train_tp], p_fit[valid_train_tp]]), y[valid_train_tp], rcond=None)[0]
                mean_p = float(np.mean(y[valid_train_p]))
                coef_p = np.linalg.lstsq(np.column_stack([np.ones(valid_train_p.sum()), p_fit[valid_train_p]]), y[valid_train_p], rcond=None)[0]
                for eval_product in PRODUCTS:
                    p_eval = precipitation[eval_product][:, record]
                    valid_test_tp = test & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_eval)
                    valid_test_p = test & np.isfinite(y) & np.isfinite(p_eval)
                    if valid_test_tp.sum() < 3 or valid_test_p.sum() < 3: continue
                    pred_t = coef_t[0] + coef_t[1] * t[valid_test_tp]
                    pred_tp = coef_tp[0] + coef_tp[1] * t[valid_test_tp] + coef_tp[2] * p_eval[valid_test_tp]
                    pred_i = np.full(valid_test_p.sum(), mean_p)
                    pred_p = coef_p[0] + coef_p[1] * p_eval[valid_test_p]
                    rmse_t, rmse_tp = rmse(y[valid_test_tp], pred_t), rmse(y[valid_test_tp], pred_tp)
                    rmse_i, rmse_p = rmse(y[valid_test_p], pred_i), rmse(y[valid_test_p], pred_p)
                    fold_rows.append({
                        "network_index_1based": record + 1, "formal_identity": manifest.iloc[record]["formal_identity"],
                        "network_origin": manifest.iloc[record]["network_origin"], "fold": fold,
                        "start_year": start, "end_year": end, "fit_product": fit_product, "eval_product": eval_product,
                        "tp_test_count": int(valid_test_tp.sum()), "p_test_count": int(valid_test_p.sum()),
                        "temperature_rmse": rmse_t, "bivariate_rmse": rmse_tp,
                        "bivariate_incremental_improvement": float(1 - rmse_tp / rmse_t),
                        "intercept_rmse": rmse_i, "precipitation_rmse": rmse_p,
                        "precipitation_over_intercept_improvement": float(1 - rmse_p / rmse_i),
                        "bivariate_precipitation_coefficient": float(coef_tp[2]),
                        "precipitation_only_coefficient": float(coef_p[1]),
                    })
    if not fold_rows: raise RuntimeError("no-valid-folds")
    write_csv(output / "fold_nested_cross_product_skill.csv", fold_rows)
    folds = pd.DataFrame(fold_rows)
    record_rows: list[dict[str, object]] = []
    for (index, identity, origin, fit_product, eval_product), group in folds.groupby(["network_index_1based", "formal_identity", "network_origin", "fit_product", "eval_product"]):
        record_rows.append({
            "network_index_1based": int(index), "formal_identity": identity, "network_origin": origin,
            "fit_product": fit_product, "eval_product": eval_product, "finite_fold_count": int(len(group)),
            "median_bivariate_incremental_improvement": float(group["bivariate_incremental_improvement"].median()),
            "median_precipitation_over_intercept_improvement": float(group["precipitation_over_intercept_improvement"].median()),
        })
    write_csv(output / "record_nested_cross_product_skill.csv", record_rows)
    records = pd.DataFrame(record_rows)
    path_summary: dict[str, dict[str, float]] = {}
    all_path_positive = np.ones(22, dtype=bool)
    seeds = 1200
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            key = f"{fit_product}_fit_{eval_product}_eval"
            selected = records[(records["fit_product"] == fit_product) & (records["eval_product"] == eval_product)].sort_values("network_index_1based")
            if len(selected) != 22: raise RuntimeError(f"record-path-count-failed:{key}:{len(selected)}")
            incremental = selected["median_bivariate_incremental_improvement"].to_numpy(float)
            precip_only = selected["median_precipitation_over_intercept_improvement"].to_numpy(float)
            lower, upper = bootstrap(incremental, seeds); seeds += 1
            path_summary[key] = {
                "median_bivariate_incremental_improvement": float(np.median(incremental)),
                "bootstrap_95_lower": lower, "bootstrap_95_upper": upper,
                "median_precipitation_over_intercept_improvement": float(np.median(precip_only)),
                "positive_bivariate_incremental_identity_fraction": float(np.mean(incremental > 0)),
            }
            all_path_positive &= incremental > 0

    coefficient = folds.pivot_table(index=["network_index_1based", "fold", "eval_product"], columns="fit_product", values="bivariate_precipitation_coefficient", aggfunc="first").reset_index()
    paired = coefficient[np.isfinite(coefficient["GPCC"]) & np.isfinite(coefficient["CRU"]) & (coefficient["GPCC"] != 0) & (coefficient["CRU"] != 0)]
    sign_agreement = float(np.mean(np.sign(paired["GPCC"]) == np.sign(paired["CRU"])))
    within = ["GPCC_fit_GPCC_eval", "CRU_fit_CRU_eval"]
    gates = {
        "all_four_incremental_medians_at_least_1pct": bool(all(item["median_bivariate_incremental_improvement"] >= 0.01 for item in path_summary.values())),
        "all_four_incremental_bootstrap_lowers_positive": bool(all(item["bootstrap_95_lower"] > 0 for item in path_summary.values())),
        "both_within_product_precipitation_only_medians_at_least_1pct": bool(all(path_summary[key]["median_precipitation_over_intercept_improvement"] >= 0.01 for key in within)),
        "positive_all_four_identity_fraction_over_50pct": bool(np.mean(all_path_positive) > 0.50),
        "gpcc_cru_bivariate_precipitation_sign_agreement_at_least_70pct": bool(sign_agreement >= 0.70),
    }
    passes = bool(all(gates.values()))
    summary = {
        "status": "complete", "locked_1921_1939_accessed": False,
        "identity_count": 22, "unique_site_count": 21, "product_paths": path_summary,
        "positive_all_four_identity_count": int(all_path_positive.sum()),
        "positive_all_four_identity_fraction": float(np.mean(all_path_positive)),
        "paired_fold_precipitation_coefficient_sign_agreement": sign_agreement,
        "gates": gates, "passes_expanded_moisture_psm": passes,
        "decision": "permit_separate_moisture_operator_test" if passes else "close_direct_linear_jja_moisture_psm_promotion",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = list(path_summary); values = [records[(records.fit_product == key.split("_fit_")[0]) & (records.eval_product == key.split("_fit_")[1].replace("_eval", ""))]["median_bivariate_incremental_improvement"].to_numpy(float) * 100 for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].boxplot(values, labels=[key.replace("_fit_", "->").replace("_eval", "") for key in keys], showfliers=False)
    axes[0].axhline(0, color="#444444"); axes[0].axhline(1, color="#9b2226", linestyle="--")
    axes[0].set(title="Bivariate incremental precipitation skill", ylabel="Median fold RMSE improvement (%)")
    axes[1].scatter(paired["GPCC"], paired["CRU"], color="#005f73", alpha=0.7)
    axes[1].axhline(0, color="#777777", linewidth=1); axes[1].axvline(0, color="#777777", linewidth=1)
    axes[1].set(title="Paired JJA precipitation coefficients", xlabel="GPCC fit", ylabel="CRU fit")
    fig.suptitle("Expanded moisture network nested cross-product validation", fontsize=14)
    fig.savefig(output / "expanded_moisture_nested_cross_product.png", dpi=180); plt.close(fig)
    lines = ["# Expanded Moisture Network Validation", "", f"- Identities/sites: `22/21`; positive on all four paths: `{all_path_positive.sum()}/22`.", f"- GPCC/CRU precipitation-coefficient sign agreement: `{sign_agreement:.3%}`."]
    for key, item in path_summary.items(): lines.append(f"- {key}: incremental `{item['median_bivariate_incremental_improvement']:.3%}` (95% `{item['bootstrap_95_lower']:.3%}` to `{item['bootstrap_95_upper']:.3%}`); precipitation-only `{item['median_precipitation_over_intercept_improvement']:.3%}`.")
    lines.extend(["", f"All gates pass: `{str(passes).lower()}`.", "", f"Decision: `{summary['decision']}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "bootstrap_draws": 5000}, indent=2), encoding="utf-8")
    if before != after: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
