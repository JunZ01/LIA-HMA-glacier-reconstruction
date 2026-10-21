"""Purged 11-year modern OOF test for the source-distinct BT021 chronology."""

from __future__ import annotations

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


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1959), (1960, 1979), (1980, 2000))
HALF_WINDOW = 5
PRODUCTS = ("BEA", "CRU")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def smooth_11(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(HALF_WINDOW, len(values) - HALF_WINDOW):
        result[index] = float(
            np.mean(values[index - HALF_WINDOW : index + HALF_WINDOW + 1])
        )
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def read_bt021_modern(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if ((frame.year >= 1921) & (frame.year <= 1939)).any():
        raise RuntimeError("locked-year-in-bt021-allowed")
    lookup = dict(zip(frame.year.astype(int), frame.bt021_trsgi.astype(float)))
    values = np.asarray([lookup.get(int(year), np.nan) for year in YEARS], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("bt021-modern-gap")
    return values


def purged_oof(
    proxy: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    prediction = np.full(len(YEARS), np.nan, dtype=float)
    baseline = np.full(len(YEARS), np.nan, dtype=float)
    low_target = smooth_11(target)
    rows: list[dict[str, object]] = []
    positions = np.arange(len(YEARS))

    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test_centers = (YEARS >= start + HALF_WINDOW) & (YEARS <= end - HALF_WINDOW)
        train_centers = (
            ((YEARS + HALF_WINDOW < start) | (YEARS - HALF_WINDOW > end))
            & (positions >= HALF_WINDOW)
            & (positions < len(YEARS) - HALF_WINDOW)
        )
        raw_training_support = np.zeros(len(YEARS), dtype=bool)
        for center in np.flatnonzero(train_centers):
            raw_training_support[center - HALF_WINDOW : center + HALF_WINDOW + 1] = True
        proxy_mean = float(np.mean(proxy[raw_training_support]))
        proxy_std = float(np.std(proxy[raw_training_support]))
        if not np.isfinite(proxy_std) or proxy_std <= 0:
            raise RuntimeError(f"invalid-proxy-training-scale:{fold}")
        standardized_proxy = (proxy - proxy_mean) / proxy_std
        low_proxy = smooth_11(standardized_proxy)
        train = train_centers & np.isfinite(low_proxy) & np.isfinite(low_target)
        test = test_centers & np.isfinite(low_proxy) & np.isfinite(low_target)
        coefficients = np.linalg.lstsq(
            np.column_stack((np.ones(int(train.sum())), low_proxy[train])),
            low_target[train],
            rcond=None,
        )[0]
        fitted = coefficients[0] + coefficients[1] * low_proxy[test]
        fold_baseline = np.full(int(test.sum()), float(np.mean(low_target[train])))
        prediction[test] = fitted
        baseline[test] = fold_baseline
        model_rmse = rmse(low_target[test], fitted)
        baseline_rmse = rmse(low_target[test], fold_baseline)
        rows.append(
            {
                "fold": fold,
                "start_year": start,
                "end_year": end,
                "training_center_count": int(train.sum()),
                "test_center_count": int(test.sum()),
                "fit_intercept": float(coefficients[0]),
                "fit_slope": float(coefficients[1]),
                "test_correlation": correlation(low_target[test], fitted),
                "test_rmse": model_rmse,
                "baseline_rmse": baseline_rmse,
                "test_skill": float(1.0 - model_rmse / baseline_rmse),
            }
        )
    return prediction, baseline, rows


def main() -> int:
    root = project_root()
    driver_dir = Path(__file__).resolve().parent
    identity = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_bt021_identity_closure_20260819_v2"
    )
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_bt021_low_frequency_modern_oof_20260819_v1"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver_dir / "round158_bt021_low_frequency_modern_oof_v1_contract.md"
    shutil.copyfile(contract, output / "ROUND_CONTRACT.md")
    r118 = load_module(
        "round158_monthly_readers",
        driver_dir
        / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "identity_summary": identity / "summary.json",
        "bt021_allowed": identity / "bt021_allowed.csv",
        "bea": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "monthly_reader": driver_dir
        / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in paths.items()}
    identity_summary = json.loads(paths["identity_summary"].read_text(encoding="utf-8"))
    if identity_summary.get("decision") != "advance_bt021_as_new_east_fitting_candidate_to_modern_oof":
        raise RuntimeError("round157-decision-mismatch")
    if identity_summary.get("locked_1921_1939_values_accessed") is not False:
        raise RuntimeError("round157-boundary-failed")

    proxy = read_bt021_modern(paths["bt021_allowed"])
    latitude = np.asarray([27.4223])
    longitude = np.asarray([90.9675])
    bea_monthly = r118.read_berkeley_points(paths["bea"], latitude, longitude)
    cru_monthly = r118.read_netcdf3_points(paths["cru"], "tmp", 1901, latitude, longitude)
    months = [tuple((6, 7, 8))]
    targets = {
        "BEA": r118.aggregate_temperature(bea_monthly, months)[:, 0],
        "CRU": r118.aggregate_temperature(cru_monthly, months)[:, 0],
    }
    for product, values in targets.items():
        if values.shape != (61,) or not np.isfinite(values).all():
            raise RuntimeError(f"temperature-target-failed:{product}")

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {"year": YEARS, "bt021_trsgi": proxy}
    predictions: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        prediction, baseline, rows = purged_oof(proxy, targets[product])
        predictions[product] = prediction
        low_target = smooth_11(targets[product])
        valid = np.isfinite(prediction) & np.isfinite(low_target) & np.isfinite(baseline)
        model_rmse = rmse(low_target[valid], prediction[valid])
        baseline_rmse = rmse(low_target[valid], baseline[valid])
        pooled_correlation = correlation(low_target[valid], prediction[valid])
        product_rows = pd.DataFrame(rows)
        summary_rows.append(
            {
                "product": product,
                "pooled_oof_correlation": pooled_correlation,
                "pooled_oof_skill": float(1.0 - model_rmse / baseline_rmse),
                "pooled_oof_rmse": model_rmse,
                "positive_test_correlation_count": int((product_rows.test_correlation > 0).sum()),
                "positive_fit_slope_count": int((product_rows.fit_slope > 0).sum()),
                "first_block_test_correlation": float(product_rows.iloc[0].test_correlation),
                "third_block_test_correlation": float(product_rows.iloc[2].test_correlation),
            }
        )
        fold_rows.extend({"product": product, **row} for row in rows)
        curves[f"{product}_target_lp11"] = low_target
        curves[f"{product}_oof"] = prediction
        curves[f"{product}_training_mean"] = baseline

    summary_frame = pd.DataFrame(summary_rows)
    product_prediction_correlation = correlation(predictions["BEA"], predictions["CRU"])
    gates = {
        "both_pooled_correlations_at_least_0p30": bool(
            (summary_frame.pooled_oof_correlation >= 0.30).all()
        ),
        "both_pooled_skills_at_least_0p05": bool(
            (summary_frame.pooled_oof_skill >= 0.05).all()
        ),
        "both_products_two_of_three_test_correlations_positive": bool(
            (summary_frame.positive_test_correlation_count >= 2).all()
        ),
        "all_six_fit_slopes_positive": bool(
            (summary_frame.positive_fit_slope_count == 3).all()
        ),
        "both_first_block_test_correlations_positive": bool(
            (summary_frame.first_block_test_correlation > 0).all()
        ),
        "both_third_block_test_correlations_positive": bool(
            (summary_frame.third_block_test_correlation > 0).all()
        ),
        "bea_cru_oof_prediction_correlation_at_least_0p80": bool(
            product_prediction_correlation >= 0.80
        ),
    }
    passes = bool(all(gates.values()))
    decision = (
        "advance_bt021_to_incremental_east_network_low_frequency_oof"
        if passes
        else "close_bt021_as_standalone_low_frequency_temperature_operator"
    )

    pd.DataFrame(fold_rows).to_csv(output / "blocked_fold_metrics.csv", index=False)
    summary_frame.to_csv(output / "product_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "modern_low_frequency_oof_curves.csv", index=False)

    fig, axis = plt.subplots(figsize=(12.0, 5.8), constrained_layout=True)
    colors = {"BEA": "#0072B2", "CRU": "#D55E00"}
    for product in PRODUCTS:
        row = summary_frame[summary_frame["product"] == product].iloc[0]
        axis.plot(YEARS, curves[f"{product}_target_lp11"], color=colors[product], alpha=0.45, linewidth=1.3, label=f"{product} JJA target")
        axis.plot(YEARS, curves[f"{product}_oof"], color=colors[product], linewidth=2.0, label=f"{product} OOF r={row.pooled_oof_correlation:.3f}")
    for _, end in BLOCKS[:-1]:
        axis.axvline(end + 0.5, color="#888888", linestyle=":", linewidth=0.8)
    axis.axhline(0.0, color="#555555", linewidth=0.7)
    axis.set_title("BT021 purged 11-year JJA temperature OOF")
    axis.set_xlabel("11-year center year")
    axis.set_ylabel("JJA temperature anomaly")
    axis.legend(frameon=False, ncol=2)
    fig.savefig(output / "bt021_low_frequency_modern_oof.png", dpi=200)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round158_bt021_low_frequency_modern_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "target": "JJA_temperature_at_27.4223N_90.9675E",
        "filter_window_years": 11,
        "outer_blocks": BLOCKS,
        "products": summary_rows,
        "bea_cru_oof_prediction_correlation": product_prediction_correlation,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "paleo_trajectory_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    lines = ["# BT021 Low-Frequency Modern OOF", ""]
    for row in summary_rows:
        lines.append(
            f"- {row['product']}: pooled r `{row['pooled_oof_correlation']:.3f}`, skill `{row['pooled_oof_skill']:.3%}`, positive blocks `{row['positive_test_correlation_count']}/3`."
        )
    lines.extend(
        [
            f"- BEA/CRU OOF prediction correlation: `{product_prediction_correlation:.3f}`.",
            f"- Passes all gates: `{str(passes).lower()}`.",
            f"- Decision: `{decision}`.",
        ]
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
