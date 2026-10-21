"""Modern blocked-OOF test of the published Xinzhu NGS precipitation series."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safe_two_column_year_first_v1 import parse_two_column_allowed_years


YEARS = np.arange(1941, 2001, dtype=int)
SOURCE_YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1941, 1952), (1953, 1964), (1965, 1976), (1977, 1988), (1989, 2000))
PRODUCTS = ("GPCC", "CRU")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 100_000_000:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def aggregate_previous_november_current_february(monthly: np.ndarray) -> np.ndarray:
    if monthly.shape[0] != len(SOURCE_YEARS) * 12:
        raise RuntimeError(f"unexpected-monthly-length:{monthly.shape}")
    cube = monthly.reshape(len(SOURCE_YEARS), 12, monthly.shape[1])
    result = np.full((len(YEARS), monthly.shape[1]), np.nan)
    for row, year in enumerate(YEARS):
        source_row = int(year - SOURCE_YEARS[0])
        values = np.concatenate((cube[source_row - 1, 10:12], cube[source_row, 0:2]), axis=0)
        complete = np.all(np.isfinite(values), axis=0)
        result[row, complete] = np.sum(values[:, complete], axis=0)
    return result


def blocked_oof(predictor: np.ndarray, target: np.ndarray):
    prediction = np.full(len(YEARS), np.nan)
    baseline = np.full(len(YEARS), np.nan)
    fold_rows: list[dict[str, object]] = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        train_valid = train & np.isfinite(predictor) & np.isfinite(target)
        test_valid = test & np.isfinite(predictor) & np.isfinite(target)
        if int(train_valid.sum()) < 30 or int(test_valid.sum()) < 3:
            raise RuntimeError(f"insufficient-fold-data:{start}-{end}")
        design = np.column_stack((np.ones(int(train_valid.sum())), predictor[train_valid]))
        coefficients = np.linalg.lstsq(design, target[train_valid], rcond=None)[0]
        prediction[test_valid] = coefficients[0] + coefficients[1] * predictor[test_valid]
        baseline[test_valid] = np.mean(target[train_valid])
        fold_rows.append({
            "fold": fold,
            "test_period": f"{start}-{end}",
            "train_count": int(train_valid.sum()),
            "test_count": int(test_valid.sum()),
            "fit_intercept_mm_season": float(coefficients[0]),
            "fit_slope": float(coefficients[1]),
            "test_correlation": correlation(target[test_valid], prediction[test_valid]),
            "test_rmse_mm_season": float(np.sqrt(np.mean((target[test_valid] - prediction[test_valid]) ** 2))),
        })
    return prediction, baseline, fold_rows


def pooled_metrics(target: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(baseline)
    model_sse = float(np.sum((target[valid] - prediction[valid]) ** 2))
    baseline_sse = float(np.sum((target[valid] - baseline[valid]) ** 2))
    return {
        "oof_count": int(valid.sum()),
        "oof_correlation": correlation(target[valid], prediction[valid]),
        "oof_skill_vs_training_fold_mean": float(1.0 - model_sse / baseline_sse),
        "oof_rmse_mm_season": float(np.sqrt(model_sse / valid.sum())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-modern-only", action="store_true")
    parser.add_argument("--xinzhu", type=Path, required=True)
    parser.add_argument("--expected-xinzhu-sha256", required=True)
    parser.add_argument("--gpcc", type=Path, required=True)
    parser.add_argument("--cru", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_modern_only:
        raise RuntimeError("explicit-modern-only-flag-required")
    if args.output.exists():
        raise FileExistsError(args.output)

    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    contract = driver / "round271_xinzhu_ngs_precipitation_modern_oof_v1_contract.md"
    r118_path = driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "xinzhu": args.xinzhu,
        "gpcc": args.gpcc,
        "cru": args.cru,
        "monthly_helper": r118_path,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["xinzhu"].get("sha256", "").lower() != args.expected_xinzhu_sha256.lower():
        raise RuntimeError("xinzhu-hash-conflict")

    series, parse_metadata = parse_two_column_allowed_years(
        args.xinzhu,
        minimum_year=1941,
        maximum_year=2000,
        forbidden_intervals=((1921, 1939),),
    )
    predictor = series.reindex(YEARS).to_numpy(dtype=float)
    if not np.isfinite(predictor).all():
        raise RuntimeError("xinzhu-modern-gap")

    r118 = load_module("round271_r118", r118_path)
    latitude = np.asarray([27.25], dtype=float)
    longitude = np.asarray([99.43], dtype=float)
    monthly = {
        "GPCC": r118.read_netcdf3_points(args.gpcc, "precip", 1891, latitude, longitude),
        "CRU": r118.read_netcdf3_points(args.cru, "pre", 1901, latitude, longitude),
    }
    targets = {
        product: aggregate_previous_november_current_february(values)[:, 0]
        for product, values in monthly.items()
    }
    if any(not np.isfinite(values).all() for values in targets.values()):
        raise RuntimeError("modern-target-gap")

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        prediction, baseline, rows = blocked_oof(predictor, targets[product])
        for row in rows:
            fold_rows.append({"product": product, **row})
        metrics = pooled_metrics(targets[product], prediction, baseline)
        early = YEARS <= 1970
        late = YEARS >= 1971
        summary_rows.append({
            "product": product,
            **metrics,
            "positive_fold_correlation_count": int(sum(row["test_correlation"] > 0 for row in rows)),
            "positive_fit_slope_count": int(sum(row["fit_slope"] > 0 for row in rows)),
            "early_raw_correlation_1941_1970": correlation(predictor[early], targets[product][early]),
            "late_raw_correlation_1971_2000": correlation(predictor[late], targets[product][late]),
        })
        predictions[product] = prediction

    fold_frame = pd.DataFrame(fold_rows)
    summary_frame = pd.DataFrame(summary_rows)
    observed_product_correlation = correlation(targets["GPCC"], targets["CRU"])
    gates = {
        "both_oof_correlations_at_least_0p30": bool((summary_frame.oof_correlation >= 0.30).all()),
        "both_oof_skills_positive": bool((summary_frame.oof_skill_vs_training_fold_mean > 0).all()),
        "both_products_at_least_four_positive_fold_correlations": bool((summary_frame.positive_fold_correlation_count >= 4).all()),
        "all_ten_fit_slopes_positive": bool((fold_frame.fit_slope > 0).all()),
        "both_products_early_late_raw_correlations_positive": bool(
            (summary_frame.early_raw_correlation_1941_1970 > 0).all()
            and (summary_frame.late_raw_correlation_1971_2000 > 0).all()
        ),
        "observed_gpcc_cru_correlation_at_least_0p70": bool(observed_product_correlation >= 0.70),
    }
    passes = bool(all(gates.values()))
    decision = (
        "advance_xinzhu_to_second_spatial_mode_feasibility"
        if passes
        else "close_xinzhu_as_generalizable_precipitation_evidence"
    )

    args.output.mkdir(parents=True)
    shutil.copy2(contract, args.output / "ROUND_CONTRACT.md")
    fold_frame.to_csv(args.output / "blocked_oof_fold_metrics.csv", index=False)
    summary_frame.to_csv(args.output / "blocked_oof_summary.csv", index=False)
    curves = pd.DataFrame({"year": YEARS, "xinzhu_reconstruction_mm_season": predictor})
    for product in PRODUCTS:
        curves[f"{product.lower()}_observed_mm_season"] = targets[product]
        curves[f"{product.lower()}_oof_mm_season"] = predictions[product]
    curves.to_csv(args.output / "modern_oof_curves.csv", index=False)

    colors = {"GPCC": "#0072B2", "CRU": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), constrained_layout=True, sharex=True)
    axes[0].plot(YEARS, predictor, color="#111111", linewidth=1.5, label="Xinzhu published reconstruction")
    axes[0].set_ylabel("Xinzhu precipitation (mm/season)")
    axes[0].legend(frameon=False)
    for product in PRODUCTS:
        row = summary_frame[summary_frame["product"] == product].iloc[0]
        axes[1].plot(YEARS, targets[product], color=colors[product], linewidth=1.4, label=f"{product} observed")
        axes[1].plot(YEARS, predictions[product], color=colors[product], linewidth=1.1, linestyle="--", label=f"{product} OOF r={row.oof_correlation:.3f}")
    for start, _ in BLOCKS[1:]:
        axes[1].axvline(start - 0.5, color="#AAAAAA", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Nov-Feb precipitation (mm/season)")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "xinzhu_ngs_precipitation_modern_oof.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_271_xinzhu_ngs_precipitation_modern_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "years_decoded": [int(YEARS[0]), int(YEARS[-1])],
        "target": "previous_November_current_February_total_precipitation",
        "site": {"name": "Xinzhu", "latitude": 27.25, "longitude": 99.43},
        "parse_metadata": parse_metadata,
        "products": summary_rows,
        "observed_gpcc_cru_correlation": observed_product_correlation,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "interpretation_limit": "Modern transport and stability diagnostic; not independent paleo validation.",
        "locked_1921_1939_values_accessed": False,
        "production_reconstruction_run": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    lines = [
        "# Round 271 Xinzhu NGS precipitation modern OOF",
        "",
        f"- Decision: `{decision}`.",
        f"- Passes all gates: `{passes}`.",
        f"- Observed GPCC/CRU correlation: `{observed_product_correlation:.3f}`.",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['product']}: OOF r `{row['oof_correlation']:.3f}`, skill "
            f"`{row['oof_skill_vs_training_fold_mean']:.3f}`, positive folds "
            f"`{row['positive_fold_correlation_count']}/5`."
        )
    (args.output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

