"""Nested GPCC/CRU validation of Round 36 annual moisture candidates."""

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


BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("GPCC", "CRU")
EXPECTED_IDS = ("iso2k_1867", "iso2k_272", "iso2k_745")


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standardize(values: np.ndarray) -> np.ndarray:
    return (values - np.nanmean(values)) / np.nanstd(values)


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_37_dod2k_annual_moisture_nested_cross_product_v1"
    paths = {
        "climate_export": output / "round37_dod2k_annual_climate_v1.mat",
        "input": output / "round37_dod2k_annual_moisture_input.mat",
        "candidate_manifest": output / "candidate_manifest.csv",
        "round12_records": loop / "round_12_expanded_moisture_nested_cross_product_v1/record_nested_cross_product_skill.csv",
        "round12_summary": loop / "round_12_expanded_moisture_nested_cross_product_v1/summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "preparation_manifest": output / "preparation_source_manifest.json",
        "matlab_driver": drivers / "export_round37_dod2k_annual_climate_v1.m",
        "analysis_driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    manifest = pd.read_csv(paths["candidate_manifest"])
    if tuple(manifest.datasetId) != EXPECTED_IDS:
        raise RuntimeError("candidate-manifest-contract-failed")

    with h5py.File(paths["climate_export"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        temperature = np.asarray(handle["temperature_formal_season"], dtype=float).T
        precipitation = {
            "GPCC": np.asarray(handle["precipitation_gpcc_formal_season"], dtype=float).T,
            "CRU": np.asarray(handle["precipitation_cru_formal_season"], dtype=float).T,
        }
    if not np.array_equal(years, np.arange(1940, 2001)) or observations.shape != (61, 3):
        raise RuntimeError("array-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-accessed")

    fold_rows: list[dict[str, object]] = []
    for record, identity in enumerate(EXPECTED_IDS):
        y = observations[:, record]
        t = temperature[:, record]
        for fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (years >= start) & (years <= end)
            train = ~test
            for fit_product in PRODUCTS:
                p_fit = precipitation[fit_product][:, record]
                train_tp = train & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_fit)
                train_p = train & np.isfinite(y) & np.isfinite(p_fit)
                if train_tp.sum() < 20 or train_p.sum() < 20:
                    raise RuntimeError(f"insufficient-training:{identity}:{fold}:{fit_product}")
                coef_t = np.linalg.lstsq(np.column_stack([np.ones(train_tp.sum()), t[train_tp]]), y[train_tp], rcond=None)[0]
                coef_tp = np.linalg.lstsq(np.column_stack([np.ones(train_tp.sum()), t[train_tp], p_fit[train_tp]]), y[train_tp], rcond=None)[0]
                mean_y = float(np.mean(y[train_p]))
                coef_p = np.linalg.lstsq(np.column_stack([np.ones(train_p.sum()), p_fit[train_p]]), y[train_p], rcond=None)[0]
                for eval_product in PRODUCTS:
                    p_eval = precipitation[eval_product][:, record]
                    test_tp = test & np.isfinite(y) & np.isfinite(t) & np.isfinite(p_eval)
                    test_p = test & np.isfinite(y) & np.isfinite(p_eval)
                    if test_tp.sum() < 3 or test_p.sum() < 3:
                        raise RuntimeError(f"insufficient-test:{identity}:{fold}:{fit_product}:{eval_product}")
                    predicted_t = coef_t[0] + coef_t[1] * t[test_tp]
                    predicted_tp = coef_tp[0] + coef_tp[1] * t[test_tp] + coef_tp[2] * p_eval[test_tp]
                    predicted_i = np.full(test_p.sum(), mean_y)
                    predicted_p = coef_p[0] + coef_p[1] * p_eval[test_p]
                    temperature_rmse = rmse(y[test_tp], predicted_t)
                    bivariate_rmse = rmse(y[test_tp], predicted_tp)
                    intercept_rmse = rmse(y[test_p], predicted_i)
                    precipitation_rmse = rmse(y[test_p], predicted_p)
                    fold_rows.append({
                        "datasetId": identity,
                        "fold": fold,
                        "start_year": start,
                        "end_year": end,
                        "fit_product": fit_product,
                        "eval_product": eval_product,
                        "test_count": int(test_tp.sum()),
                        "temperature_rmse": temperature_rmse,
                        "bivariate_rmse": bivariate_rmse,
                        "bivariate_incremental_improvement": float(1 - bivariate_rmse / temperature_rmse),
                        "intercept_rmse": intercept_rmse,
                        "precipitation_rmse": precipitation_rmse,
                        "precipitation_over_intercept_improvement": float(1 - precipitation_rmse / intercept_rmse),
                        "bivariate_precipitation_coefficient": float(coef_tp[2]),
                        "precipitation_only_coefficient": float(coef_p[1]),
                    })
    write_csv(output / "fold_nested_cross_product_skill.csv", fold_rows)
    folds = pd.DataFrame(fold_rows)

    path_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    for identity in EXPECTED_IDS:
        record = folds[folds.datasetId == identity]
        coefficient = record.drop_duplicates(["fold", "fit_product"])
        documented_sign_fraction = float(np.mean(coefficient.bivariate_precipitation_coefficient < 0))
        paired = coefficient.pivot(index="fold", columns="fit_product", values="bivariate_precipitation_coefficient").dropna()
        paired_sign = float(np.mean(np.sign(paired.GPCC) == np.sign(paired.CRU)))
        gates: dict[str, bool] = {}
        for fit_product in PRODUCTS:
            for eval_product in PRODUCTS:
                selected = record[(record.fit_product == fit_product) & (record.eval_product == eval_product)]
                key = f"{fit_product}_fit_{eval_product}_eval"
                incremental = selected.bivariate_incremental_improvement.to_numpy(float)
                precipitation_only = selected.precipitation_over_intercept_improvement.to_numpy(float)
                path_rows.append({
                    "datasetId": identity,
                    "fit_product": fit_product,
                    "eval_product": eval_product,
                    "median_bivariate_incremental_improvement": float(np.median(incremental)),
                    "improved_bivariate_fold_count": int(np.sum(incremental > 0)),
                    "median_precipitation_over_intercept_improvement": float(np.median(precipitation_only)),
                    "improved_precipitation_only_fold_count": int(np.sum(precipitation_only > 0)),
                })
                gates[f"{key}_median_incremental_at_least_1pct"] = bool(np.median(incremental) >= 0.01)
                gates[f"{key}_at_least_4_of_5_folds_improve"] = bool(np.sum(incremental > 0) >= 4)
                if fit_product == eval_product:
                    gates[f"{key}_precipitation_only_median_at_least_1pct"] = bool(np.median(precipitation_only) >= 0.01)
        gates["documented_negative_coefficient_fraction_at_least_70pct"] = documented_sign_fraction >= 0.70
        gates["gpcc_cru_fit_coefficient_sign_agreement_at_least_70pct"] = paired_sign >= 0.70
        passes = bool(all(gates.values()))
        record_rows.append({
            "datasetId": identity,
            "documented_negative_coefficient_fraction": documented_sign_fraction,
            "gpcc_cru_fit_coefficient_sign_agreement": paired_sign,
            "passes_all_record_gates": passes,
            "failed_gate_count": int(sum(not value for value in gates.values())),
            "gates_json": json.dumps(gates, sort_keys=True),
        })
    paths_frame = pd.DataFrame(path_rows)
    records = pd.DataFrame(record_rows)
    paths_frame.to_csv(output / "record_path_summary.csv", index=False)
    records.to_csv(output / "record_promotion_gates.csv", index=False)

    control = pd.read_csv(paths["round12_records"])
    control_summary = json.loads(paths["round12_summary"].read_text(encoding="utf-8"))
    network_rows: list[dict[str, object]] = []
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            key = f"{fit_product}_fit_{eval_product}_eval"
            old = control[(control.fit_product == fit_product) & (control.eval_product == eval_product)].median_bivariate_incremental_improvement.to_numpy(float)
            new = paths_frame[(paths_frame.fit_product == fit_product) & (paths_frame.eval_product == eval_product)].median_bivariate_incremental_improvement.to_numpy(float)
            archived = float(control_summary["product_paths"][key]["median_bivariate_incremental_improvement"])
            if abs(float(np.median(old)) - archived) > 1e-12:
                raise RuntimeError(f"round12-control-closure-failed:{key}")
            network_rows.append({
                "fit_product": fit_product,
                "eval_product": eval_product,
                "round12_22_record_median": float(np.median(old)),
                "expanded_25_record_median": float(np.median(np.concatenate([old, new]))),
                "new_3_record_median": float(np.median(new)),
            })
    network = pd.DataFrame(network_rows)
    network.to_csv(output / "network_comparison.csv", index=False)

    promoted = records.loc[records.passes_all_record_gates, "datasetId"].tolist()
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "candidate_count": 3,
        "promoted_record_count": len(promoted),
        "promoted_record_ids": promoted,
        "record_results": records.set_index("datasetId")[["documented_negative_coefficient_fraction", "gpcc_cru_fit_coefficient_sign_agreement", "passes_all_record_gates", "failed_gate_count"]].to_dict(orient="index"),
        "network_path_comparison": network.to_dict(orient="records"),
        "decision": "advance_passing_records_to_expanded_network_test" if promoted else "close_new_annual_dod2k_psm_candidates",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(3, 2, figsize=(12.5, 10), constrained_layout=True)
    for record, identity in enumerate(EXPECTED_IDS):
        axes[record, 0].plot(years, standardize(-observations[:, record]), label="Proxy wetness orientation", color="#222222", linewidth=1.4)
        axes[record, 0].plot(years, standardize(precipitation["GPCC"][:, record]), label="GPCC", color="#0072b2", alpha=0.85)
        axes[record, 0].plot(years, standardize(precipitation["CRU"][:, record]), label="CRU", color="#d55e00", alpha=0.85)
        axes[record, 0].axhline(0, color="#888888", linewidth=0.7)
        axes[record, 0].set(title=identity, ylabel="Standardized anomaly")
        selected = paths_frame[paths_frame.datasetId == identity]
        labels = [f"{row.fit_product}->{row.eval_product}" for row in selected.itertuples(index=False)]
        values = selected.median_bivariate_incremental_improvement.to_numpy(float) * 100
        axes[record, 1].bar(labels, values, color=["#0072b2" if value >= 1 else "#d55e00" for value in values])
        axes[record, 1].axhline(0, color="#555555", linewidth=0.8)
        axes[record, 1].axhline(1, color="#009e73", linestyle="--", linewidth=1)
        axes[record, 1].set(title="Median held-out incremental skill", ylabel="RMSE improvement (%)")
        axes[record, 1].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1, 0].set(xlabel="Year")
    figure.suptitle("DoD2k annual moisture candidates: observed curves and nested skill", fontsize=14)
    figure.savefig(output / "dod2k_annual_moisture_nested_cross_product.png", dpi=180)
    plt.close(figure)

    lines = [
        "# DoD2k Annual Moisture Nested Validation",
        "",
        f"- Candidates: `3`; passing all frozen gates: `{len(promoted)}`.",
    ]
    for row in records.itertuples(index=False):
        lines.append(f"- {row.datasetId}: pass `{str(bool(row.passes_all_record_gates)).lower()}`; negative-coefficient fraction `{row.documented_negative_coefficient_fraction:.1%}`; GPCC/CRU sign agreement `{row.gpcc_cru_fit_coefficient_sign_agreement:.1%}`; failed gates `{row.failed_gate_count}`.")
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
