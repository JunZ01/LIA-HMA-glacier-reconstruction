from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safe_rwl_year_first_v2 import parse_rwl_allowed_years


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("BEA", "CRU")
SITE_LATITUDE = 28.25
SITE_LONGITUDE = 85.38
MAM_MONTHS = (3, 4, 5)


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
    digest = None
    if stat.st_size <= 100_000_000:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    if float(np.std(left[valid])) <= 1e-12 or float(np.std(right[valid])) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def detrend_chronology(dplpy, raw: pd.DataFrame) -> tuple[pd.Series, pd.Series, list[str]]:
    retained = raw.loc[:, raw.notna().sum(axis=0) >= 20]
    if retained.empty:
        raise RuntimeError("no-series-with-20-modern-values")
    detrended = []
    fallbacks = []
    for sample in retained.columns:
        try:
            series = dplpy.detrend(
                retained[sample], fit="ModNegEx", method="residual", plot=False
            )
        except RuntimeError:
            series = dplpy.detrend(
                retained[sample], fit="linear", method="residual", plot=False
            )
            fallbacks.append(str(sample))
        detrended.append(series.rename(sample))
    rwi = pd.concat(detrended, axis=1).reindex(YEARS)
    chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
    return (
        chronology["Mean RWI"].astype(float).reindex(YEARS),
        chronology["Sample depth"].astype(float).reindex(YEARS),
        fallbacks,
    )


def blocked_oof(proxy: np.ndarray, target: np.ndarray, product: str):
    prediction = np.full(len(YEARS), np.nan)
    climatology = np.full(len(YEARS), np.nan)
    rows: list[dict[str, object]] = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        valid_train = train & np.isfinite(proxy) & np.isfinite(target)
        valid_test = test & np.isfinite(proxy) & np.isfinite(target)
        if int(valid_train.sum()) < 30 or int(valid_test.sum()) < 8:
            raise RuntimeError(f"insufficient-fold-data:{product}:{start}-{end}")
        design = np.column_stack((np.ones(int(valid_train.sum())), proxy[valid_train]))
        coefficient = np.linalg.lstsq(design, target[valid_train], rcond=None)[0]
        prediction[valid_test] = coefficient[0] + coefficient[1] * proxy[valid_test]
        climatology[valid_test] = float(np.mean(target[valid_train]))
        rows.append(
            {
                "product": product,
                "fold": fold,
                "test_years": f"{start}-{end}",
                "fit_intercept": float(coefficient[0]),
                "fit_slope": float(coefficient[1]),
                "test_count": int(valid_test.sum()),
                "test_correlation": correlation(target[valid_test], prediction[valid_test]),
                "test_rmse_degC": float(
                    np.sqrt(np.mean((target[valid_test] - prediction[valid_test]) ** 2))
                ),
            }
        )
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(climatology)
    candidate_sse = float(np.sum((target[valid] - prediction[valid]) ** 2))
    baseline_sse = float(np.sum((target[valid] - climatology[valid]) ** 2))
    early = valid & (YEARS <= 1969)
    late = valid & (YEARS >= 1970)
    summary = {
        "product": product,
        "oof_count": int(valid.sum()),
        "oof_correlation": correlation(target[valid], prediction[valid]),
        "oof_skill_vs_training_fold_climatology": 1.0 - candidate_sse / baseline_sse,
        "oof_rmse_degC": float(np.sqrt(candidate_sse / int(valid.sum()))),
        "positive_fold_correlation_count": int(
            sum(float(row["test_correlation"]) > 0 for row in rows)
        ),
        "negative_fit_slope_count": int(sum(float(row["fit_slope"]) < 0 for row in rows)),
        "early_raw_proxy_target_correlation": correlation(proxy[early], target[early]),
        "late_raw_proxy_target_correlation": correlation(proxy[late], target[late]),
    }
    return prediction, climatology, rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-modern-oof", action="store_true")
    parser.add_argument("--nepa043", type=Path, required=True)
    parser.add_argument("--expected-nepa043-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_modern_oof:
        raise RuntimeError("explicit-modern-oof-authorization-flag-required")

    root = project_root()
    driver_dir = Path(__file__).resolve().parent
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver_dir / "round267_nepa043_mam_temperature_modern_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "nepa043": args.nepa043,
        "berkeley": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
        "monthly_helper": driver_dir
        / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    if args.output.exists():
        raise FileExistsError(args.output)
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["nepa043"]["sha256"].lower() != args.expected_nepa043_sha256.lower():
        raise RuntimeError("nepa043-hash-conflict")

    raw, parse_metadata = parse_rwl_allowed_years(
        paths["nepa043"],
        minimum_year=1940,
        maximum_year=2000,
        forbidden_intervals=((1921, 1939),),
    )
    import dplpy

    chronology, depth, fallbacks = detrend_chronology(dplpy, raw)
    proxy = chronology.to_numpy(dtype=float)
    sample_depth = depth.to_numpy(dtype=float)
    if not np.isfinite(proxy).all() or not np.isfinite(sample_depth).all():
        raise RuntimeError("modern-chronology-gap")

    helper = load_module("round267_monthly_helper", paths["monthly_helper"])
    latitudes = np.asarray([SITE_LATITUDE])
    longitudes = np.asarray([SITE_LONGITUDE])
    bea_monthly = helper.read_berkeley_points(paths["berkeley"], latitudes, longitudes)
    cru_monthly = helper.read_netcdf3_points(paths["cru"], "tmp", 1901, latitudes, longitudes)
    climate = {
        "BEA": helper.aggregate_temperature(bea_monthly, [MAM_MONTHS])[:, 0],
        "CRU": helper.aggregate_temperature(cru_monthly, [MAM_MONTHS])[:, 0],
    }
    if any(values.shape != (61,) or not np.isfinite(values).all() for values in climate.values()):
        raise RuntimeError("mam-temperature-target-gap")

    predictions: dict[str, np.ndarray] = {}
    fold_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for product in PRODUCTS:
        prediction, _, rows, summary = blocked_oof(proxy, climate[product], product)
        predictions[product] = prediction
        fold_rows.extend(rows)
        summaries.append(summary)
    oof = pd.DataFrame(summaries)
    prediction_correlation = correlation(predictions["BEA"], predictions["CRU"])
    gates = {
        "both_oof_correlations_at_least_0p30": bool((oof.oof_correlation >= 0.30).all()),
        "both_oof_skills_positive": bool((oof.oof_skill_vs_training_fold_climatology > 0).all()),
        "both_products_at_least_four_positive_fold_correlations": bool(
            (oof.positive_fold_correlation_count >= 4).all()
        ),
        "all_ten_fit_slopes_negative": bool((oof.negative_fit_slope_count == 5).all()),
        "both_products_early_late_raw_correlations_negative": bool(
            (
                (oof.early_raw_proxy_target_correlation < 0)
                & (oof.late_raw_proxy_target_correlation < 0)
            ).all()
        ),
        "bea_cru_oof_prediction_correlation_at_least_0p80": bool(
            prediction_correlation >= 0.80
        ),
        "minimum_modern_sample_depth_at_least_10": bool(np.min(sample_depth) >= 10),
    }
    passes = bool(all(gates.values()))
    decision = (
        "admit_nepa043_as_modern_validated_mam_temperature_fitting_candidate"
        if passes
        else "close_nepa043_as_temperature_fitting_evidence"
    )

    after = {name: fingerprint(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-fingerprint-changed")
    args.output.mkdir(parents=True)
    pd.DataFrame(fold_rows).to_csv(args.output / "blocked_oof_fold_metrics.csv", index=False)
    oof.to_csv(args.output / "blocked_oof_summary.csv", index=False)
    curves = pd.DataFrame({"year": YEARS, "nepa043_std_chronology": proxy, "sample_depth": sample_depth})
    for product in PRODUCTS:
        curves[f"{product}_MAM_observed"] = climate[product]
        curves[f"{product}_MAM_oof"] = predictions[product]
    curves.to_csv(args.output / "modern_oof_curves.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True, constrained_layout=True)
    colors = {"BEA": "#0072B2", "CRU": "#D55E00"}
    for axis, product in zip(axes, PRODUCTS):
        row = oof[oof["product"] == product].iloc[0]
        axis.plot(YEARS, climate[product], color=colors[product], linewidth=1.4, label=f"{product} MAM observed")
        axis.plot(YEARS, predictions[product], color="#222222", linestyle="--", linewidth=1.2, label=f"NEPA043 OOF, r={row.oof_correlation:.3f}")
        for _, end in BLOCKS[:-1]:
            axis.axvline(end + 0.5, color="#999999", linestyle=":", linewidth=0.7)
        axis.set_ylabel("MAM temperature (degC)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xlabel("Year")
    figure.savefig(args.output / "nepa043_mam_temperature_modern_oof.png", dpi=220)
    plt.close(figure)

    result = {
        "experiment_id": "round_267_nepa043_mam_temperature_modern_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "chronology_method": {
            "library": "dplpy",
            "version": dplpy.__version__,
            "primary_detrend": "ModNegEx",
            "fallback_detrend": "linear",
            "mean": "Tukey_biweight",
            "prewhiten": False,
        },
        "parse_metadata": parse_metadata,
        "retained_series_count": int(raw.loc[:, raw.notna().sum(axis=0) >= 20].shape[1]),
        "fallback_series_count": len(fallbacks),
        "minimum_modern_sample_depth": int(np.min(sample_depth)),
        "target": "same_year_March_May_mean_temperature",
        "documented_response_direction": "negative",
        "oof": summaries,
        "bea_cru_oof_prediction_correlation": prediction_correlation,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "locked_1921_1939_values_accessed": False,
        "nepa043_may_be_used_for_locked_final_validation": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (args.output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    gate_lines = "\n".join(f"- `{name}`: `{value}`" for name, value in gates.items())
    (args.output / "FINAL_RESULT.md").write_text(
        "# Round 267 NEPA043 MAM temperature modern OOF\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Passes all gates: `{passes}`.\n"
        f"- Minimum sample depth: `{int(np.min(sample_depth))}`.\n"
        f"- Berkeley/CRU OOF prediction correlation: `{prediction_correlation:.3f}`.\n\n"
        "## Gates\n\n" + gate_lines + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
