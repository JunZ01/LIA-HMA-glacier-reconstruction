"""Test one-year proxy/climate alignment with nested blocked validation."""

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
LAGS = (-1, 0, 1)
TEMP_PRODUCTS = ("Berkeley", "CRU")
PRECIP_PRODUCTS = ("GPCC", "CRU")
SEASONS = {"ANN": 0, "JJA": 1, "AMJ": 2, "APR": 3}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def file_state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shifted(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if lag < 0:
        result[-lag:] = values[:lag]
    elif lag > 0:
        result[:-lag] = values[lag:]
    else:
        result[:] = values
    return result


def core_mask(years: np.ndarray, start: int, end: int) -> np.ndarray:
    return (years >= start + 1) & (years <= end - 1)


def purged_train_mask(years: np.ndarray, exclusions: list[tuple[int, int]]) -> np.ndarray:
    mask = np.ones(years.shape, dtype=bool)
    for start, end in exclusions:
        mask &= (years < start - 1) | (years > end + 1)
    return mask


def fit_model(y: np.ndarray, predictors: list[np.ndarray], mask: np.ndarray, minimum: int = 20) -> np.ndarray | None:
    valid = mask & np.isfinite(y)
    for predictor in predictors:
        valid &= np.isfinite(predictor)
    if int(valid.sum()) < minimum:
        return None
    design = np.column_stack([np.ones(int(valid.sum()))] + [p[valid] for p in predictors])
    return np.linalg.lstsq(design, y[valid], rcond=None)[0]


def evaluate_model(
    y: np.ndarray,
    predictors: list[np.ndarray],
    mask: np.ndarray,
    coefficients: np.ndarray | None,
    minimum: int = 3,
) -> tuple[float, int]:
    if coefficients is None:
        return np.nan, 0
    valid = mask & np.isfinite(y)
    for predictor in predictors:
        valid &= np.isfinite(predictor)
    count = int(valid.sum())
    if count < minimum:
        return np.nan, count
    design = np.column_stack([np.ones(count)] + [p[valid] for p in predictors])
    residual = y[valid] - design @ coefficients
    return float(np.sqrt(np.mean(residual**2))), count


def select_temperature_lag(
    years: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    outer_index: int,
) -> tuple[int, dict[int, float]]:
    outer = BLOCKS[outer_index]
    scores: dict[int, float] = {}
    for lag in LAGS:
        candidate = shifted(x, lag)
        squared_error = 0.0
        count = 0
        for inner_index, inner in enumerate(BLOCKS):
            if inner_index == outer_index:
                continue
            train = purged_train_mask(years, [outer, inner])
            coefficients = fit_model(y, [candidate], train)
            test = core_mask(years, *inner) & purged_train_mask(years, [outer])
            rmse, test_count = evaluate_model(y, [candidate], test, coefficients)
            if np.isfinite(rmse):
                squared_error += rmse**2 * test_count
                count += test_count
        scores[lag] = float(np.sqrt(squared_error / count)) if count else np.inf
    selected = min(LAGS, key=lambda lag: (scores[lag], abs(lag), lag))
    return int(selected), scores


def select_precipitation_lag(
    years: np.ndarray,
    y: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    outer_index: int,
) -> tuple[int, dict[int, float]]:
    outer = BLOCKS[outer_index]
    scores: dict[int, float] = {}
    for lag in LAGS:
        candidate = shifted(precipitation, lag)
        squared_error = 0.0
        count = 0
        for inner_index, inner in enumerate(BLOCKS):
            if inner_index == outer_index:
                continue
            train = purged_train_mask(years, [outer, inner])
            coefficients = fit_model(y, [temperature, candidate], train)
            test = core_mask(years, *inner) & purged_train_mask(years, [outer])
            rmse, test_count = evaluate_model(y, [temperature, candidate], test, coefficients)
            if np.isfinite(rmse):
                squared_error += rmse**2 * test_count
                count += test_count
        scores[lag] = float(np.sqrt(squared_error / count)) if count else np.inf
    selected = min(LAGS, key=lambda lag: (scores[lag], abs(lag), lag))
    return int(selected), scores


def bootstrap_median(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    medians = np.empty(draws)
    for draw in range(draws):
        medians[draw] = np.median(finite[rng.integers(0, len(finite), size=len(finite))])
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def summarize_family(
    folds: pd.DataFrame,
    products: tuple[str, str],
    identity_count: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    record_rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(["network_index_1based", "formal_identity", "fit_product", "eval_product"]):
        index, identity, fit_product, eval_product = keys
        record_rows.append({
            "network_index_1based": int(index),
            "formal_identity": identity,
            "fit_product": fit_product,
            "eval_product": eval_product,
            "finite_fold_count": int(len(group)),
            "median_rmse_improvement_over_zero_lag": float(group.rmse_improvement_over_zero_lag.median()),
        })
    records = pd.DataFrame(record_rows)
    path_summary: dict[str, dict[str, float]] = {}
    all_path_positive = np.ones(identity_count, dtype=bool)
    for fit_product in products:
        for eval_product in products:
            key = f"{fit_product}_fit_{eval_product}_eval"
            selected = records[(records.fit_product == fit_product) & (records.eval_product == eval_product)].sort_values("network_index_1based")
            if len(selected) != identity_count:
                raise RuntimeError(f"record-path-count-failed:{key}:{len(selected)}")
            values = selected.median_rmse_improvement_over_zero_lag.to_numpy(float)
            lower, upper = bootstrap_median(values, seed)
            seed += 1
            path_summary[key] = {
                "median_rmse_improvement_over_zero_lag": float(np.median(values)),
                "bootstrap_95_lower": lower,
                "bootstrap_95_upper": upper,
                "positive_identity_fraction": float(np.mean(values > 0)),
            }
            all_path_positive &= values > 0

    selections = folds[["network_index_1based", "fold", "fit_product", "selected_lag"]].drop_duplicates()
    pivot = selections.pivot(index=["network_index_1based", "fold"], columns="fit_product", values="selected_lag").reset_index()
    paired_agreement = float(np.mean(pivot[products[0]] == pivot[products[1]]))
    nonzero = {
        product: float(np.mean(selections[selections.fit_product == product].selected_lag != 0))
        for product in products
    }
    gates = {
        "all_four_path_medians_at_least_1pct": bool(all(item["median_rmse_improvement_over_zero_lag"] >= 0.01 for item in path_summary.values())),
        "all_four_bootstrap_lowers_positive": bool(all(item["bootstrap_95_lower"] > 0 for item in path_summary.values())),
        "positive_all_four_identity_fraction_over_50pct": bool(np.mean(all_path_positive) > 0.5),
        "paired_product_selected_lag_agreement_at_least_70pct": bool(paired_agreement >= 0.7),
        "both_fit_products_nonzero_selection_fraction_over_50pct": bool(all(value > 0.5 for value in nonzero.values())),
    }
    result = {
        "identity_count": identity_count,
        "product_paths": path_summary,
        "positive_all_four_identity_count": int(all_path_positive.sum()),
        "positive_all_four_identity_fraction": float(np.mean(all_path_positive)),
        "paired_product_selected_lag_agreement": paired_agreement,
        "nonzero_selected_lag_fraction": nonzero,
        "selected_lag_counts": {
            product: {str(lag): int(np.sum((selections.fit_product == product) & (selections.selected_lag == lag))) for lag in LAGS}
            for product in products
        },
        "gates": gates,
        "passes": bool(all(gates.values())),
    }
    return result, records


def main() -> int:
    project = project_root()
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / "round_18_nested_year_alignment_v1"
    paths = {
        "contract": output / "ROUND_CONTRACT.md",
        "temperature_export": base / "round_14_temperature_product_cross_validation_v1/temperature_products_v1.mat",
        "temperature_manifest": base / "round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv",
        "moisture_export": base / "round_12_expanded_moisture_nested_cross_product_v1/expanded_moisture_jja_climate_v1.mat",
        "moisture_manifest": base / "round_12_expanded_moisture_nested_cross_product_v1/network_manifest.csv",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: file_state(path) for name, path in paths.items()}

    temperature_manifest = pd.read_csv(paths["temperature_manifest"])
    moisture_manifest = pd.read_csv(paths["moisture_manifest"])
    with h5py.File(paths["temperature_export"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        temperature_observations = np.asarray(handle["observations"], dtype=float).T
        temperature_products = np.asarray(handle["temperature"], dtype=float).transpose(3, 2, 1, 0)
    with h5py.File(paths["moisture_export"], "r") as handle:
        moisture_years = np.asarray(handle["years"], dtype=int).ravel()
        moisture_observations = np.asarray(handle["observations"], dtype=float).T
        moisture_temperature = np.asarray(handle["temperature_jja"], dtype=float).T
        moisture_precipitation = {
            "GPCC": np.asarray(handle["precipitation_gpcc_jja"], dtype=float).T,
            "CRU": np.asarray(handle["precipitation_cru_jja"], dtype=float).T,
        }
    if not np.array_equal(years, np.arange(1940, 2001)) or not np.array_equal(moisture_years, years):
        raise RuntimeError("year-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-accessed")
    if temperature_observations.shape != (61, 214) or moisture_observations.shape != (61, 22):
        raise RuntimeError("network-shape-failed")

    selected_temperature = np.full((61, 214, 2), np.nan)
    for record in range(214):
        season = SEASONS[str(temperature_manifest.iloc[record].season_code)]
        selected_temperature[:, record, :] = temperature_products[:, record, season, :]

    temperature_rows: list[dict[str, object]] = []
    for record in range(214):
        y = temperature_observations[:, record]
        for fold, outer in enumerate(BLOCKS):
            train = purged_train_mask(years, [outer])
            test = core_mask(years, *outer)
            for fit_index, fit_product in enumerate(TEMP_PRODUCTS):
                fit_series = selected_temperature[:, record, fit_index]
                selected_lag, inner_scores = select_temperature_lag(years, y, fit_series, fold)
                adaptive_fit = shifted(fit_series, selected_lag)
                adaptive_coefficients = fit_model(y, [adaptive_fit], train)
                baseline_coefficients = fit_model(y, [fit_series], train)
                for eval_index, eval_product in enumerate(TEMP_PRODUCTS):
                    eval_series = selected_temperature[:, record, eval_index]
                    adaptive_eval = shifted(eval_series, selected_lag)
                    adaptive_rmse, count = evaluate_model(y, [adaptive_eval], test, adaptive_coefficients)
                    baseline_rmse, baseline_count = evaluate_model(y, [eval_series], test, baseline_coefficients)
                    if count != baseline_count or not np.isfinite(adaptive_rmse) or not np.isfinite(baseline_rmse):
                        continue
                    temperature_rows.append({
                        "network_index_1based": record + 1,
                        "formal_identity": temperature_manifest.iloc[record].formal_identity,
                        "fold": fold + 1,
                        "fit_product": fit_product,
                        "eval_product": eval_product,
                        "selected_lag": selected_lag,
                        "inner_rmse_lag_minus1": inner_scores[-1],
                        "inner_rmse_lag_zero": inner_scores[0],
                        "inner_rmse_lag_plus1": inner_scores[1],
                        "test_count": count,
                        "adaptive_rmse": adaptive_rmse,
                        "zero_lag_rmse": baseline_rmse,
                        "rmse_improvement_over_zero_lag": float(1 - adaptive_rmse / baseline_rmse),
                    })

    moisture_rows: list[dict[str, object]] = []
    for record in range(22):
        y = moisture_observations[:, record]
        temperature = moisture_temperature[:, record]
        for fold, outer in enumerate(BLOCKS):
            train = purged_train_mask(years, [outer])
            test = core_mask(years, *outer)
            for fit_product in PRECIP_PRODUCTS:
                fit_precipitation = moisture_precipitation[fit_product][:, record]
                selected_lag, inner_scores = select_precipitation_lag(years, y, temperature, fit_precipitation, fold)
                adaptive_fit = shifted(fit_precipitation, selected_lag)
                adaptive_coefficients = fit_model(y, [temperature, adaptive_fit], train)
                baseline_coefficients = fit_model(y, [temperature, fit_precipitation], train)
                for eval_product in PRECIP_PRODUCTS:
                    eval_precipitation = moisture_precipitation[eval_product][:, record]
                    adaptive_eval = shifted(eval_precipitation, selected_lag)
                    adaptive_rmse, count = evaluate_model(y, [temperature, adaptive_eval], test, adaptive_coefficients)
                    baseline_rmse, baseline_count = evaluate_model(y, [temperature, eval_precipitation], test, baseline_coefficients)
                    if count != baseline_count or not np.isfinite(adaptive_rmse) or not np.isfinite(baseline_rmse):
                        continue
                    moisture_rows.append({
                        "network_index_1based": record + 1,
                        "formal_identity": moisture_manifest.iloc[record].formal_identity,
                        "fold": fold + 1,
                        "fit_product": fit_product,
                        "eval_product": eval_product,
                        "selected_lag": selected_lag,
                        "inner_rmse_lag_minus1": inner_scores[-1],
                        "inner_rmse_lag_zero": inner_scores[0],
                        "inner_rmse_lag_plus1": inner_scores[1],
                        "test_count": count,
                        "adaptive_rmse": adaptive_rmse,
                        "zero_lag_rmse": baseline_rmse,
                        "rmse_improvement_over_zero_lag": float(1 - adaptive_rmse / baseline_rmse),
                    })

    write_csv(output / "temperature_fold_year_alignment.csv", temperature_rows)
    write_csv(output / "moisture_fold_year_alignment.csv", moisture_rows)
    temperature_folds = pd.DataFrame(temperature_rows)
    moisture_folds = pd.DataFrame(moisture_rows)
    temperature_summary, temperature_records = summarize_family(temperature_folds, TEMP_PRODUCTS, 214, 1801)
    moisture_summary, moisture_records = summarize_family(moisture_folds, PRECIP_PRODUCTS, 22, 1811)
    temperature_records.to_csv(output / "temperature_record_year_alignment.csv", index=False)
    moisture_records.to_csv(output / "moisture_record_year_alignment.csv", index=False)

    any_passes = bool(temperature_summary["passes"] or moisture_summary["passes"])
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "candidate_climate_year_offsets": list(LAGS),
        "temperature": temperature_summary,
        "moisture": moisture_summary,
        "any_family_passes": any_passes,
        "decision": "permit_separate_year_alignment_operator_test" if any_passes else "close_simple_one_year_misalignment_as_network_repair",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, records, products, title in (
        (axes[0], temperature_records, TEMP_PRODUCTS, "Temperature"),
        (axes[1], moisture_records, PRECIP_PRODUCTS, "Moisture"),
    ):
        labels: list[str] = []
        values: list[np.ndarray] = []
        for fit_product in products:
            for eval_product in products:
                subset = records[(records.fit_product == fit_product) & (records.eval_product == eval_product)]
                labels.append(f"{fit_product}->{eval_product}")
                values.append(subset.median_rmse_improvement_over_zero_lag.to_numpy(float) * 100)
        axis.boxplot(values, labels=labels, showfliers=False)
        axis.axhline(0, color="#444444", linewidth=1)
        axis.axhline(1, color="#9b2226", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("RMSE improvement over zero lag (%)")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Nested one-year climate/proxy alignment test", fontsize=14)
    fig.savefig(output / "nested_year_alignment.png", dpi=180)
    plt.close(fig)

    lines = ["# Nested Year-Alignment Test", ""]
    for name, item in (("Temperature", temperature_summary), ("Moisture", moisture_summary)):
        lines.append(f"## {name}")
        lines.append(f"- Paired product lag agreement: `{item['paired_product_selected_lag_agreement']:.3%}`.")
        lines.append(f"- Positive on all four paths: `{item['positive_all_four_identity_count']}/{item['identity_count']}`.")
        for key, path in item["product_paths"].items():
            lines.append(f"- {key}: `{path['median_rmse_improvement_over_zero_lag']:.3%}` (95% `{path['bootstrap_95_lower']:.3%}` to `{path['bootstrap_95_upper']:.3%}`).")
        lines.append(f"- Passes all gates: `{str(item['passes']).lower()}`.")
        lines.append("")
    lines.append(f"Decision: `{summary['decision']}`.")
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: file_state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({
        "sources_before": before,
        "sources_after": after,
        "source_states_unchanged": before == after,
        "bootstrap_draws": 5000,
        "outer_blocks": BLOCKS,
        "candidate_lags": LAGS,
    }, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
