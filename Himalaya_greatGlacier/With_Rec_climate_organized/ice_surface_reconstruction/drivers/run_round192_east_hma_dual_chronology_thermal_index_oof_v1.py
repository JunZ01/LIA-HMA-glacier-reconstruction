from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 1999)]
YEARS = np.arange(1940, 2000)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def zscore(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    scale = float(np.std(train, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("nonpositive-training-scale")
    return (values - float(np.mean(train))) / scale


def load_series(path: Path, value_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame.year.between(1940, 1999), ["year", *value_columns]].copy()
    if frame.year.tolist() != YEARS.tolist() or not np.isfinite(frame[value_columns].to_numpy(float)).all():
        raise RuntimeError(f"invalid-modern-series:{path}")
    return frame


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_192_east_hma_dual_chronology_thermal_index_oof_v1"
    output.mkdir(parents=True, exist_ok=False)

    r180 = loop / "round_180_hengduan_chronology_modern_oof_v2"
    r183 = loop / "round_183_haizi_903y_chronology_modern_oof_v2"
    paths = {
        "hengduan_proxy": r180 / "modern_candidate_series.csv",
        "haizi_proxy": r183 / "modern_candidate_series.csv",
        "hengduan_target": r180 / "modern_oof_curves.csv",
        "haizi_target": r183 / "modern_oof_curves.csv",
        "round190": loop / "round_190_wang2015_source_overlap_audit_v1/summary.json",
        "round191": loop / "round_191_chin041_hengduan_sample_identity_v1/summary.json",
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    hp = load_series(paths["hengduan_proxy"], ["chronology"])
    zp = load_series(paths["haizi_proxy"], ["chronology"])
    ht = load_series(paths["hengduan_target"], ["BEA_observed", "CRU_observed"])
    zt = load_series(paths["haizi_target"], ["BEA_observed", "CRU_observed"])
    hengduan_proxy = hp.chronology.to_numpy(float)
    haizi_proxy = zp.chronology.to_numpy(float)

    curves = pd.DataFrame({"year": YEARS, "hengduan_chronology": hengduan_proxy, "haizi_chronology": haizi_proxy})
    fold_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    predictions: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    predictor_full: dict[str, np.ndarray] = {}

    for product in ["BEA", "CRU"]:
        hengduan_target = ht[f"{product}_observed"].to_numpy(float)
        haizi_target = zt[f"{product}_observed"].to_numpy(float)
        prediction = np.full(60, np.nan)
        target_index = np.full(60, np.nan)
        proxy_index = np.full(60, np.nan)
        climatology = np.full(60, np.nan)
        for fold_index, (start, end) in enumerate(BLOCKS, start=1):
            test = (YEARS >= start) & (YEARS <= end)
            train = ~test
            proxy_train = 0.5 * zscore(hengduan_proxy[train], hengduan_proxy[train]) + 0.5 * zscore(haizi_proxy[train], haizi_proxy[train])
            proxy_test = 0.5 * zscore(hengduan_proxy[train], hengduan_proxy[test]) + 0.5 * zscore(haizi_proxy[train], haizi_proxy[test])
            target_train = 0.5 * zscore(hengduan_target[train], hengduan_target[train]) + 0.5 * zscore(haizi_target[train], haizi_target[train])
            target_test = 0.5 * zscore(hengduan_target[train], hengduan_target[test]) + 0.5 * zscore(haizi_target[train], haizi_target[test])
            coefficients = np.linalg.lstsq(
                np.column_stack((np.ones(int(train.sum())), proxy_train)), target_train, rcond=None
            )[0]
            prediction[test] = coefficients[0] + coefficients[1] * proxy_test
            target_index[test] = target_test
            proxy_index[test] = proxy_test
            climatology[test] = float(np.mean(target_train))
            fold_rows.append(
                {
                    "product": product,
                    "fold": fold_index,
                    "test_years": f"{start}-{end}",
                    "fit_intercept": float(coefficients[0]),
                    "fit_slope": float(coefficients[1]),
                    "test_correlation": correlation(target_test, prediction[test]),
                    "test_rmse_index_units": float(np.sqrt(np.mean((target_test - prediction[test]) ** 2))),
                }
            )
        product_folds = pd.DataFrame([row for row in fold_rows if row["product"] == product])
        candidate_sse = float(np.sum((target_index - prediction) ** 2))
        baseline_sse = float(np.sum((target_index - climatology) ** 2))
        early = YEARS <= 1969
        late = YEARS >= 1970
        summary_rows.append(
            {
                "product": product,
                "oof_correlation": correlation(target_index, prediction),
                "oof_skill_vs_training_fold_climatology": 1 - candidate_sse / baseline_sse,
                "oof_rmse_index_units": float(np.sqrt(candidate_sse / len(YEARS))),
                "positive_fold_correlation_count": int((product_folds.test_correlation > 0).sum()),
                "positive_fit_slope_count": int((product_folds.fit_slope > 0).sum()),
                "early_split_proxy_target_correlation": correlation(proxy_index[early], target_index[early]),
                "late_split_proxy_target_correlation": correlation(proxy_index[late], target_index[late]),
            }
        )
        predictions[product] = prediction
        observed[product] = target_index
        predictor_full[product] = proxy_index
        curves[f"{product}_proxy_index_fold_standardized"] = proxy_index
        curves[f"{product}_observed_index_fold_standardized"] = target_index
        curves[f"{product}_oof_index"] = prediction

    oof = pd.DataFrame(summary_rows)
    product_prediction_correlation = correlation(predictions["BEA"], predictions["CRU"])
    gate = {
        "both_oof_correlations_ge_0_30": bool((oof.oof_correlation >= 0.30).all()),
        "both_oof_skills_positive": bool((oof.oof_skill_vs_training_fold_climatology > 0).all()),
        "both_products_four_of_five_fold_correlations_positive": bool((oof.positive_fold_correlation_count >= 4).all()),
        "all_ten_fit_slopes_positive": bool((oof.positive_fit_slope_count == 5).all()),
        "both_products_early_late_split_correlations_positive": bool(
            ((oof.early_split_proxy_target_correlation > 0) & (oof.late_split_proxy_target_correlation > 0)).all()
        ),
        "bea_cru_oof_prediction_correlation_ge_0_80": bool(product_prediction_correlation >= 0.80),
    }
    passed = all(gate.values())
    pd.DataFrame(fold_rows).to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    oof.to_csv(output / "blocked_oof_summary.csv", index=False)
    curves.to_csv(output / "modern_oof_curves.csv", index=False)
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("source-hash-changed")
    decision = (
        "advance_to_separately_contracted_historical_shape_test_as_mechanism_candidate"
        if passed
        else "close_dual_chronology_thermal_index_on_modern_oof"
    )
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "predictor": "equal_weight_training_fold_standardized_hengduan_haizi_chronology_index",
        "target": "equal_weight_training_fold_standardized_documented_local_temperature_index",
        "units": "dimensionless_index_not_degC_field",
        "oof": summary_rows,
        "bea_cru_oof_prediction_correlation": product_prediction_correlation,
        "gate": gate,
        "passed": passed,
        "paleo_values_inspected_or_exported": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"sources": hashes_after}, indent=2), encoding="ascii")
    lines = ["# Round 192 East-HMA Dual-Chronology Thermal Index OOF", ""]
    for row in summary_rows:
        lines.append(
            f"- {row['product']}: r=`{row['oof_correlation']:.3f}`, skill=`{row['oof_skill_vs_training_fold_climatology']:.3f}`, positive folds=`{row['positive_fold_correlation_count']}/5`."
        )
    lines += ["", f"Passed frozen modern gate: `{str(passed).lower()}`.", f"Decision: `{decision}`."]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

