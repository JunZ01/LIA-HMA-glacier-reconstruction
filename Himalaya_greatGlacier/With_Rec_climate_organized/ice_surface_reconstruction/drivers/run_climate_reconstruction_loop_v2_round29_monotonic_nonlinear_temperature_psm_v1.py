"""Test whether a documented-positive nonlinear temperature PSM transfers in time."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("Berkeley", "CRU")
SEASONS = {"ANN": 0, "JJA": 1, "AMJ": 2, "APR": 3}
BOOTSTRAP_DRAWS = 5000


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


def bootstrap_median(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = np.median(values[rng.integers(0, len(values), size=len(values))])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = project_root()
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / "round_29_monotonic_nonlinear_temperature_psm_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")

    sources = {
        "temperature_products": base / "round_14_temperature_product_cross_validation_v1/temperature_products_v1.mat",
        "network_manifest": base / "round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv",
        "round14_folds": base / "round_14_temperature_product_cross_validation_v1/fold_temperature_product_skill.csv",
        "round14_summary": base / "round_14_temperature_product_cross_validation_v1/summary.json",
        "analysis_driver": Path(__file__).resolve(),
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}

    contract = """# Round 29 Contract: Monotonic Nonlinear Temperature PSM

## Question

Can a flexible but physically documented-positive nonlinear proxy-temperature response rescue transferable temperature information after linear, regularized-network, low-frequency, memory, and product-choice tests failed?

## Frozen Inputs and Folds

- The exact Round 14 set of 214 source-native LMR temperature identities, ordering, formal seasons, proxy observations, Berkeley Earth and CRU temperatures.
- The exact five blocked 1940-2000 outer folds.
- Every fold uses identical finite support for the intercept, linear control, and nonlinear model.

## Model

- Fit proxy-on-temperature isotonic regression with a predeclared increasing response and clipped extrapolation.
- This permits thresholds, plateaus, and arbitrary monotonic curvature while preserving the formal positive temperature direction.
- Evaluate all four product paths and compare against both the training-mean intercept and the Round 14 linear control.

## Frozen Rescue Gates

- Both within-product paths must have median blocked RMSE improvement over intercept of at least 1%, record-bootstrap 95% lower bound above zero, more than 50% improving identities, median improvement over linear of at least 0.5%, and its bootstrap lower bound above zero.
- Both cross-product paths must have positive median improvement over intercept and positive bootstrap lower bounds.
- All gates must pass. No record filtering, data-driven sign selection, or post-hoc tuning is allowed.

## Exclusions

- No 1921-1939 access, precipitation, EnKF, localization, OGGM, or production reconstruction.
- Original and prior-round files remain read-only.
"""
    (output / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")

    manifest = pd.read_csv(sources["network_manifest"])
    with h5py.File(sources["temperature_products"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        temperature = np.asarray(handle["temperature"], dtype=float).transpose(3, 2, 1, 0)
    if observations.shape != (61, 214) or temperature.shape != (61, 214, 4, 2):
        raise RuntimeError("round14-array-contract-failed")
    if not np.array_equal(years, np.arange(1940, 2001)):
        raise RuntimeError("year-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-accessed")

    predictors = np.full((61, 214, 2), np.nan)
    for record in range(214):
        season = str(manifest.iloc[record].season_code)
        predictors[:, record, :] = temperature[:, record, SEASONS[season], :]

    fold_rows: list[dict[str, object]] = []
    for record in range(214):
        response = observations[:, record]
        for fit_index, fit_product in enumerate(PRODUCTS):
            fit_predictor = predictors[:, record, fit_index]
            for fold, (start, end) in enumerate(BLOCKS, start=1):
                test = (years >= start) & (years <= end)
                train = ~test
                valid_train = train & np.isfinite(response) & np.isfinite(fit_predictor)
                if valid_train.sum() < 20 or np.unique(fit_predictor[valid_train]).size < 3:
                    continue
                response_train = response[valid_train]
                predictor_train = fit_predictor[valid_train]
                mean = float(np.mean(response_train))
                linear_coef = np.linalg.lstsq(
                    np.column_stack([np.ones(valid_train.sum()), predictor_train]),
                    response_train,
                    rcond=None,
                )[0]
                nonlinear = IsotonicRegression(increasing=True, out_of_bounds="clip")
                nonlinear.fit(predictor_train, response_train)
                for eval_index, eval_product in enumerate(PRODUCTS):
                    eval_predictor = predictors[:, record, eval_index]
                    valid_test = test & np.isfinite(response) & np.isfinite(eval_predictor)
                    if valid_test.sum() < 3:
                        continue
                    actual = response[valid_test]
                    x_test = eval_predictor[valid_test]
                    intercept_prediction = np.full(actual.shape, mean)
                    linear_prediction = linear_coef[0] + linear_coef[1] * x_test
                    nonlinear_prediction = nonlinear.predict(x_test)
                    rmse_intercept = float(np.sqrt(np.mean((actual - intercept_prediction) ** 2)))
                    rmse_linear = float(np.sqrt(np.mean((actual - linear_prediction) ** 2)))
                    rmse_nonlinear = float(np.sqrt(np.mean((actual - nonlinear_prediction) ** 2)))
                    fold_rows.append(
                        {
                            "network_index_1based": record + 1,
                            "formal_identity": manifest.iloc[record].formal_identity,
                            "fold": fold,
                            "fit_product": fit_product,
                            "eval_product": eval_product,
                            "test_count": int(valid_test.sum()),
                            "intercept_rmse": rmse_intercept,
                            "linear_rmse": rmse_linear,
                            "nonlinear_rmse": rmse_nonlinear,
                            "nonlinear_vs_intercept_improvement": float(1 - rmse_nonlinear / rmse_intercept),
                            "linear_vs_intercept_improvement": float(1 - rmse_linear / rmse_intercept),
                            "nonlinear_vs_linear_improvement": float(1 - rmse_nonlinear / rmse_linear),
                        }
                    )

    write_csv(output / "fold_monotonic_nonlinear_skill.csv", fold_rows)
    folds = pd.DataFrame(fold_rows)
    record_rows: list[dict[str, object]] = []
    group_columns = ["network_index_1based", "formal_identity", "fit_product", "eval_product"]
    for keys, group in folds.groupby(group_columns):
        record_rows.append(
            {
                "network_index_1based": int(keys[0]),
                "formal_identity": keys[1],
                "fit_product": keys[2],
                "eval_product": keys[3],
                "finite_fold_count": int(len(group)),
                "median_nonlinear_vs_intercept_improvement": float(group.nonlinear_vs_intercept_improvement.median()),
                "median_linear_vs_intercept_improvement": float(group.linear_vs_intercept_improvement.median()),
                "median_nonlinear_vs_linear_improvement": float(group.nonlinear_vs_linear_improvement.median()),
            }
        )
    write_csv(output / "record_monotonic_nonlinear_skill.csv", record_rows)
    records = pd.DataFrame(record_rows)

    archived = pd.read_csv(sources["round14_folds"])
    joined = folds.merge(
        archived,
        on=["network_index_1based", "formal_identity", "fold", "fit_product", "eval_product", "test_count"],
        how="inner",
        suffixes=("_new", "_round14"),
        validate="one_to_one",
    )
    if len(joined) != len(folds) or len(joined) != len(archived):
        raise RuntimeError("round14-linear-closure-row-mismatch")
    linear_rmse_closure = float(np.max(np.abs(joined.linear_rmse - joined.model_rmse)))
    intercept_rmse_closure = float(np.max(np.abs(joined.intercept_rmse_new - joined.intercept_rmse_round14)))
    if max(linear_rmse_closure, intercept_rmse_closure) > 1e-12:
        raise RuntimeError("round14-linear-closure-failed")

    path_summary: dict[str, dict[str, object]] = {}
    seed = 2901
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            key = f"{fit_product}_fit_{eval_product}_eval"
            selected = records[
                (records.fit_product == fit_product) & (records.eval_product == eval_product)
            ].sort_values("network_index_1based")
            versus_intercept = selected.median_nonlinear_vs_intercept_improvement.to_numpy(float)
            versus_linear = selected.median_nonlinear_vs_linear_improvement.to_numpy(float)
            intercept_ci = bootstrap_median(versus_intercept, seed)
            seed += 1
            linear_ci = bootstrap_median(versus_linear, seed)
            seed += 1
            path_summary[key] = {
                "identity_count": int(len(selected)),
                "median_nonlinear_vs_intercept_improvement": float(np.median(versus_intercept)),
                "nonlinear_vs_intercept_bootstrap_95_lower": intercept_ci[0],
                "nonlinear_vs_intercept_bootstrap_95_upper": intercept_ci[1],
                "positive_vs_intercept_identity_fraction": float(np.mean(versus_intercept > 0)),
                "median_nonlinear_vs_linear_improvement": float(np.median(versus_linear)),
                "nonlinear_vs_linear_bootstrap_95_lower": linear_ci[0],
                "nonlinear_vs_linear_bootstrap_95_upper": linear_ci[1],
                "positive_vs_linear_identity_fraction": float(np.mean(versus_linear > 0)),
            }

    within_gates: dict[str, dict[str, object]] = {}
    for product in PRODUCTS:
        item = path_summary[f"{product}_fit_{product}_eval"]
        gates = {
            "median_vs_intercept_at_least_1pct": item["median_nonlinear_vs_intercept_improvement"] >= 0.01,
            "vs_intercept_bootstrap_lower_positive": item["nonlinear_vs_intercept_bootstrap_95_lower"] > 0,
            "positive_vs_intercept_identity_fraction_over_50pct": item["positive_vs_intercept_identity_fraction"] > 0.5,
            "median_vs_linear_at_least_half_pct": item["median_nonlinear_vs_linear_improvement"] >= 0.005,
            "vs_linear_bootstrap_lower_positive": item["nonlinear_vs_linear_bootstrap_95_lower"] > 0,
        }
        within_gates[product] = {"gates": gates, "passes": bool(all(gates.values()))}
    cross_gates: dict[str, bool] = {}
    for fit_product, eval_product in (("Berkeley", "CRU"), ("CRU", "Berkeley")):
        key = f"{fit_product}_fit_{eval_product}_eval"
        item = path_summary[key]
        cross_gates[f"{key}_median_positive"] = item["median_nonlinear_vs_intercept_improvement"] > 0
        cross_gates[f"{key}_bootstrap_lower_positive"] = item["nonlinear_vs_intercept_bootstrap_95_lower"] > 0
    rescued = bool(all(item["passes"] for item in within_gates.values()) and all(cross_gates.values()))

    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "identity_count": 214,
        "fold_row_count": int(len(folds)),
        "round14_linear_rmse_max_abs_closure": linear_rmse_closure,
        "round14_intercept_rmse_max_abs_closure": intercept_rmse_closure,
        "model": "documented-positive isotonic regression with clipped extrapolation",
        "product_paths": path_summary,
        "within_product_rescue": within_gates,
        "cross_product_gates": cross_gates,
        "monotonic_nonlinearity_rescues_temperature_psm": rescued,
        "decision": "permit_nonlinear_temperature_operator_reconstruction_proof" if rescued else "close_monotonic_nonlinear_temperature_psm_family",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    keys = list(path_summary)
    labels = [key.replace("_fit_", "->").replace("_eval", "") for key in keys]
    intercept_values = []
    linear_values = []
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            selected = records[(records.fit_product == fit_product) & (records.eval_product == eval_product)]
            intercept_values.append(selected.median_nonlinear_vs_intercept_improvement.to_numpy(float) * 100)
            linear_values.append(selected.median_nonlinear_vs_linear_improvement.to_numpy(float) * 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].boxplot(intercept_values, tick_labels=labels, showfliers=False)
    axes[0].axhline(0, color="#444444")
    axes[0].axhline(1, color="#9b2226", linestyle="--")
    axes[0].set(title="Nonlinear vs intercept", ylabel="Median blocked RMSE improvement (%)")
    axes[1].boxplot(linear_values, tick_labels=labels, showfliers=False)
    axes[1].axhline(0, color="#444444")
    axes[1].axhline(0.5, color="#9b2226", linestyle="--")
    axes[1].set(title="Nonlinear vs linear", ylabel="Median blocked RMSE improvement (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    fig.suptitle("Documented-positive nonlinear temperature PSM", fontsize=14)
    fig.savefig(output / "monotonic_nonlinear_temperature_psm.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Monotonic Nonlinear Temperature PSM",
        "",
        f"- Round 14 linear closure: `{linear_rmse_closure:.3e}`; intercept closure: `{intercept_rmse_closure:.3e}`.",
    ]
    for key, item in path_summary.items():
        lines.append(
            f"- {key}: nonlinear vs intercept `{item['median_nonlinear_vs_intercept_improvement']:.3%}` "
            f"(95% `{item['nonlinear_vs_intercept_bootstrap_95_lower']:.3%}` to `{item['nonlinear_vs_intercept_bootstrap_95_upper']:.3%}`); "
            f"vs linear `{item['median_nonlinear_vs_linear_improvement']:.3%}`."
        )
    lines.extend(
        [
            "",
            f"Monotonic nonlinearity rescues the temperature PSM: `{str(rescued).lower()}`.",
            "",
            f"Decision: `{summary['decision']}`.",
        ]
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: file_state(path) for name, path in sources.items()}
    source_unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": source_unchanged,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not source_unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
