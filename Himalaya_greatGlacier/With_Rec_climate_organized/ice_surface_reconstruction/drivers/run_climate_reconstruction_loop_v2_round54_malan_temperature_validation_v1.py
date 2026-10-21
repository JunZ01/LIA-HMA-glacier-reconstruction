"""Cross-product nested validation of Malan Glacier d18O against JJA temperature."""

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


YEARS = np.arange(1940, 2001)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("Berkeley", "CRU")


def project_root() -> Path:
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
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1]) if valid.sum() >= 3 else np.nan


def fit_slope(response: np.ndarray, predictor: np.ndarray, selected: np.ndarray) -> float:
    valid = selected & np.isfinite(response) & np.isfinite(predictor)
    if valid.sum() < 10:
        return np.nan
    design = np.column_stack([np.ones(valid.sum()), predictor[valid]])
    return float(np.linalg.lstsq(design, response[valid], rcond=None)[0][1])


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_54_malan_icecore_temperature_nested_validation_v1"
    drivers = Path(__file__).resolve().parent
    paths = {
        "climate_export": output / "malan_temperature_products_v1.mat",
        "network_manifest": output / "network_manifest.csv",
        "preparation_manifest": output / "preparation_source_manifest.json",
        "contract": output / "ROUND_CONTRACT.md",
        "matlab_exporter": drivers / "export_round14_temperature_products_v1.m",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    manifest = pd.read_csv(paths["network_manifest"])
    if len(manifest) != 1 or manifest.iloc[0].datasetId != "iso2k_1151":
        raise RuntimeError("malan-manifest-contract-failed")
    with h5py.File(paths["climate_export"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        temperature = np.asarray(handle["temperature"], dtype=float).transpose(3, 2, 1, 0)
    if observations.shape != (61, 1) or temperature.shape != (61, 1, 4, 2):
        raise RuntimeError(
            f"array-contract-failed:{observations.shape}:{temperature.shape}"
        )
    if not np.array_equal(years, YEARS) or np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("year-contract-failed")
    response = observations[:, 0]
    predictors = temperature[:, 0, 1, :]
    early = (YEARS >= 1940) & (YEARS <= 1969)
    late = (YEARS >= 1970) & (YEARS <= 2000)
    full = np.ones(len(YEARS), dtype=bool)

    stability: dict[str, dict[str, float]] = {}
    for index, product in enumerate(PRODUCTS):
        stability[product] = {
            "full_correlation": correlation(response, predictors[:, index]),
            "full_slope": fit_slope(response, predictors[:, index], full),
            "early_slope": fit_slope(response, predictors[:, index], early),
            "late_slope": fit_slope(response, predictors[:, index], late),
        }

    fold_rows: list[dict[str, object]] = []
    for fit_index, fit_product in enumerate(PRODUCTS):
        x_fit = predictors[:, fit_index]
        for fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (YEARS >= start) & (YEARS <= end)
            train = ~test
            valid_train = train & np.isfinite(response) & np.isfinite(x_fit)
            if valid_train.sum() < 20:
                raise RuntimeError(f"insufficient-training:{fit_product}:{fold}")
            design = np.column_stack([np.ones(valid_train.sum()), x_fit[valid_train]])
            coefficient = np.linalg.lstsq(design, response[valid_train], rcond=None)[0]
            train_mean = float(np.mean(response[valid_train]))
            for eval_index, eval_product in enumerate(PRODUCTS):
                x_eval = predictors[:, eval_index]
                valid_test = test & np.isfinite(response) & np.isfinite(x_eval)
                prediction = coefficient[0] + coefficient[1] * x_eval[valid_test]
                model_rmse = float(
                    np.sqrt(np.mean((response[valid_test] - prediction) ** 2))
                )
                intercept_rmse = float(
                    np.sqrt(np.mean((response[valid_test] - train_mean) ** 2))
                )
                fold_rows.append(
                    {
                        "fold": fold,
                        "start_year": start,
                        "end_year": end,
                        "fit_product": fit_product,
                        "eval_product": eval_product,
                        "test_count": int(valid_test.sum()),
                        "fit_slope": float(coefficient[1]),
                        "model_rmse": model_rmse,
                        "intercept_rmse": intercept_rmse,
                        "rmse_improvement": float(1.0 - model_rmse / intercept_rmse),
                    }
                )
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "malan_temperature_fold_skill.csv", index=False)
    path_results: dict[str, dict[str, object]] = {}
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            key = f"{fit_product}_fit_{eval_product}_eval"
            selected = folds[
                (folds.fit_product == fit_product) & (folds.eval_product == eval_product)
            ].sort_values("fold")
            improvements = selected.rmse_improvement.to_numpy(float)
            path_results[key] = {
                "median_rmse_improvement": float(np.median(improvements)),
                "improving_fold_count": int(np.sum(improvements > 0)),
                "maximum_fold_regression": float(np.max(-improvements)),
            }

    gates = {
        "both_full_correlations_at_least_0_30": all(
            stability[product]["full_correlation"] >= 0.30 for product in PRODUCTS
        ),
        "both_full_slopes_positive": all(
            stability[product]["full_slope"] > 0 for product in PRODUCTS
        ),
        "both_early_and_late_slopes_positive": all(
            stability[product][period] > 0
            for product in PRODUCTS
            for period in ("early_slope", "late_slope")
        ),
        "all_paths_median_improvement_at_least_1pct": all(
            result["median_rmse_improvement"] >= 0.01
            for result in path_results.values()
        ),
        "all_paths_improve_at_least_4_of_5_folds": all(
            result["improving_fold_count"] >= 4 for result in path_results.values()
        ),
        "no_path_fold_regression_worse_than_10pct": all(
            result["maximum_fold_regression"] <= 0.10
            for result in path_results.values()
        ),
        "all_training_fold_slopes_positive": bool((folds.fit_slope > 0).all()),
    }
    passes = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "datasetId": "iso2k_1151",
        "locked_1921_1939_accessed": False,
        "stability": stability,
        "product_paths": path_results,
        "gates": gates,
        "passes_malan_temperature_validation": passes,
        "decision": (
            "advance_malan_as_independent_temperature_evidence"
            if passes
            else "close_malan_local_linear_temperature_psm"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"Berkeley": "#0072b2", "CRU": "#d55e00"}
    for index, product in enumerate(PRODUCTS):
        axes[0].scatter(
            predictors[:, index],
            response,
            s=24,
            alpha=0.72,
            color=colors[product],
            label=product,
        )
    axes[0].set(
        title="Malan d18O and JJA temperature",
        xlabel="Temperature anomaly",
        ylabel="d18O",
    )
    axes[0].legend(frameon=False)
    labels = list(path_results)
    values = [
        folds[
            (folds.fit_product == label.split("_fit_")[0])
            & (folds.eval_product == label.split("_fit_")[1].replace("_eval", ""))
        ].rmse_improvement.to_numpy(float)
        * 100.0
        for label in labels
    ]
    axes[1].boxplot(
        values,
        labels=[label.replace("_fit_", "->").replace("_eval", "") for label in labels],
        showfliers=False,
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].axhline(1, color="#b2182b", linestyle="--", linewidth=1.0)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Five-block cross-product skill", ylabel="RMSE improvement (%)")
    figure.savefig(output / "malan_temperature_nested_validation.png", dpi=180)
    plt.close(figure)

    lines = ["# Malan Ice-Core Temperature Validation", ""]
    for product, item in stability.items():
        lines.append(
            f"- {product}: correlation `{item['full_correlation']:.3f}`; full/early/late slopes "
            f"`{item['full_slope']:.3f}/{item['early_slope']:.3f}/{item['late_slope']:.3f}`."
        )
    for key, item in path_results.items():
        lines.append(
            f"- {key}: median skill `{item['median_rmse_improvement']:.2%}`, "
            f"improving folds `{item['improving_fold_count']}/5`, "
            f"worst regression `{item['maximum_fold_regression']:.2%}`."
        )
    lines.extend(
        [
            "",
            f"All gates pass: `{str(passes).lower()}`.",
            "",
            f"Decision: `{summary['decision']}`.",
        ]
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
