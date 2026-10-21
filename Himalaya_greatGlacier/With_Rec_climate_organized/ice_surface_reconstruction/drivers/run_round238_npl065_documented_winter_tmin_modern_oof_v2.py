from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITE_LAT = 27.89
SITE_LON = 86.37
LOCKED_START = 1921
LOCKED_END = 1939
MATRIX_YEARS = np.arange(1941, 2001, dtype=int)
CALIBRATION_YEARS = np.arange(1960, 2001, dtype=int)
BLOCKS = ((1960, 1967), (1968, 1975), (1976, 1983), (1984, 1991), (1992, 2000))
PRODUCTS = ("CRU_TMIN", "BEA_TMEAN")
EXPECTED_RWL_SHA256 = "d7f9d588e1362745857ac2aea3ca7c0686cee8b6ecc410e20af23a97d81d7bcd"


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


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def parse_modern_rwl(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: dict[str, dict[int, float]] = {}
    locked_slots_skipped = 0
    paleo_slots_skipped = 0
    post_2000_slots_skipped = 0
    decoded_slots = 0
    marker_counts = {"999": 0, "-9999": 0, "9999": 0}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip(b"\r\n")
            if not line.strip() or line == b"\x1a":
                continue
            if len(line) < 12:
                raise RuntimeError(f"short-rwl-line:{line_number}")
            sample = line[:8].strip().decode("ascii", errors="strict")
            year_text = line[8:12].strip()
            if not year_text.lstrip(b"-").isdigit():
                raise RuntimeError(f"invalid-rwl-year:{line_number}:{year_text}")
            start_year = int(year_text)
            series = records.setdefault(sample, {})
            for offset, token in enumerate(line[12:].split()):
                token_text = token.decode("ascii", errors="strict")
                if token_text in marker_counts:
                    marker_counts[token_text] += 1
                    break
                year = start_year + offset
                if LOCKED_START <= year <= LOCKED_END:
                    locked_slots_skipped += 1
                    continue
                if year < int(MATRIX_YEARS[0]):
                    paleo_slots_skipped += 1
                    continue
                if year > int(MATRIX_YEARS[-1]):
                    post_2000_slots_skipped += 1
                    continue
                value = int(token_text)
                decoded_slots += 1
                series[year] = float(value)

    frame = pd.DataFrame(
        index=pd.Index(MATRIX_YEARS, name="Year"),
        columns=sorted(records),
        dtype=float,
    )
    for sample, values in records.items():
        for year, value in values.items():
            frame.at[year, sample] = value
    source_count = int(frame.shape[1])
    frame = frame.dropna(axis=1, how="all")
    return frame, {
        "source_sample_count": source_count,
        "modern_nonempty_sample_count": int(frame.shape[1]),
        "modern_empty_sample_count_skipped": source_count - int(frame.shape[1]),
        "decoded_1941_2000_value_count": int(decoded_slots),
        "locked_slots_skipped_before_numeric_conversion": int(locked_slots_skipped),
        "pre_1941_slots_skipped_before_numeric_conversion": int(paleo_slots_skipped),
        "post_2000_slots_skipped_before_numeric_conversion": int(post_2000_slots_skipped),
        "end_marker_counts": marker_counts,
        "raw_integer_scale_retained": True,
        "locked_numeric_values_decoded": False,
        "paleo_numeric_values_decoded": False,
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    if np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def nearest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def read_cru_tmin_point(
    manifest_path: Path,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], bool], dict[str, Any], list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("forbidden_decades_not_requested") != ["1921.1930", "1931.1940"]:
        raise RuntimeError("round208-boundary-contract-changed")
    monthly: dict[tuple[int, int], float] = {}
    support: dict[tuple[int, int], bool] = {}
    files: list[Path] = []
    checks: list[dict[str, Any]] = []
    selected_lat = None
    selected_lon = None
    for record in manifest["files"]:
        start, end = [int(value) for value in record["allowed_year_range"].split("-")]
        if start < 1941:
            continue
        if start in (1921, 1931) or end > 2000:
            raise RuntimeError(f"unexpected-cru-decade:{start}-{end}")
        path = Path(record["netcdf_path"])
        current = fingerprint(path)
        if current["sha256"] != record["netcdf_sha256"]:
            raise RuntimeError(f"cru-hash-changed:{path.name}")
        files.append(path)
        with netcdf_file(path, "r", mmap=True) as dataset:
            latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
            longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
            iy = nearest(latitudes, SITE_LAT)
            ix = nearest(longitudes, SITE_LON)
            if abs(latitudes[iy] - SITE_LAT) > 0.51 or abs(longitudes[ix] - SITE_LON) > 0.51:
                raise RuntimeError("cru-nearest-cell-too-distant")
            tmn = np.asarray(dataset.variables["tmn"].data[:, iy, ix], dtype=float).copy()
            stn = np.asarray(dataset.variables["stn"].data[:, iy, ix], dtype=float).copy()
            fill = float(dataset.variables["tmn"]._attributes["_FillValue"])
            tmn[np.isclose(tmn, fill) | (tmn > 1e20)] = np.nan
            selected_lat = float(latitudes[iy])
            selected_lon = float(longitudes[ix])
        periods = pd.period_range(f"{start}-01", periods=12 * (end - start + 1), freq="M")
        if len(periods) != len(tmn):
            raise RuntimeError(f"cru-month-count-changed:{path.name}")
        for period, value, station_count in zip(periods, tmn, stn):
            key = (int(period.year), int(period.month))
            if key in monthly:
                raise RuntimeError(f"duplicate-cru-month:{key}")
            monthly[key] = float(value)
            support[key] = bool(station_count > 0)
        checks.append({"file": path.name, "range": [start, end], "month_count": len(periods)})
    return monthly, support, {
        "grid_latitude": selected_lat,
        "grid_longitude": selected_lon,
        "files": checks,
    }, files


def read_bea_point(path: Path) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    monthly: dict[tuple[int, int], float] = {}
    with h5py.File(path, "r") as handle:
        latitudes = np.asarray(handle["latitude"], dtype=float)
        longitudes = np.asarray(handle["longitude"], dtype=float)
        iy = nearest(latitudes, SITE_LAT)
        ix = nearest(longitudes, SITE_LON)
        start = (1940 - 1850) * 12
        stop = (2001 - 1850) * 12
        values = np.asarray(handle["temperature"][start:stop, iy, ix], dtype=float)
        values[(values <= -9000) | (values >= 1e20)] = np.nan
        selected_lat = float(latitudes[iy])
        selected_lon = float(longitudes[ix])
    periods = pd.period_range("1940-01", "2000-12", freq="M")
    if len(periods) != len(values):
        raise RuntimeError("bea-month-count-changed")
    for period, value in zip(periods, values):
        monthly[(int(period.year), int(period.month))] = float(value)
    return monthly, {"grid_latitude": selected_lat, "grid_longitude": selected_lon}


def djf_series(
    monthly: dict[tuple[int, int], float],
    years: np.ndarray,
) -> np.ndarray:
    result = np.full(len(years), np.nan)
    for index, year in enumerate(years):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        values = np.asarray([monthly.get(key, np.nan) for key in keys], dtype=float)
        if np.isfinite(values).all():
            result[index] = float(np.mean(values))
    return result


def djf_support_fraction(
    support: dict[tuple[int, int], bool],
    years: np.ndarray,
) -> float:
    slots = [
        support.get(key, False)
        for year in years
        for key in ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
    ]
    return float(np.mean(slots))


def blocked_oof(
    proxy: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    n = len(CALIBRATION_YEARS)
    if proxy.shape != (n,) or target.shape != (n,):
        raise RuntimeError("oof-shape-changed")
    if not np.isfinite(proxy).all() or not np.isfinite(target).all():
        raise RuntimeError("incomplete-oof-input")
    prediction = np.full(n, np.nan)
    climatology = np.full(n, np.nan)
    folds: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test = (CALIBRATION_YEARS >= start) & (CALIBRATION_YEARS <= end)
        train = ~test
        coefficients = np.linalg.lstsq(
            np.column_stack((np.ones(int(train.sum())), proxy[train])),
            target[train],
            rcond=None,
        )[0]
        prediction[test] = coefficients[0] + coefficients[1] * proxy[test]
        climatology[test] = float(np.mean(target[train]))
        folds.append({
            "fold": fold,
            "test_years": f"{start}-{end}",
            "test_year_count": int(test.sum()),
            "fit_intercept": float(coefficients[0]),
            "fit_slope": float(coefficients[1]),
            "test_correlation": correlation(target[test], prediction[test]),
            "test_rmse_degC": float(np.sqrt(np.mean((target[test] - prediction[test]) ** 2))),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("incomplete-oof-prediction")
    sse = float(np.sum((target - prediction) ** 2))
    baseline_sse = float(np.sum((target - climatology) ** 2))
    fold_frame = pd.DataFrame(folds)
    early = CALIBRATION_YEARS <= 1979
    late = CALIBRATION_YEARS >= 1980
    return {
        "oof_correlation": correlation(target, prediction),
        "oof_skill_vs_training_fold_climatology": 1 - sse / baseline_sse,
        "oof_rmse_degC": float(np.sqrt(sse / n)),
        "positive_fold_correlation_count": int((fold_frame.test_correlation > 0).sum()),
        "positive_fit_slope_count": int((fold_frame.fit_slope > 0).sum()),
        "early_split_proxy_target_correlation": correlation(proxy[early], target[early]),
        "late_split_proxy_target_correlation": correlation(proxy[late], target[late]),
    }, prediction, folds


def null_statistic(proxy: np.ndarray, targets: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    metrics = {product: blocked_oof(proxy, targets[product])[0] for product in PRODUCTS}
    correlations = {product: float(metrics[product]["oof_correlation"]) for product in PRODUCTS}
    all_slopes_positive = all(metrics[product]["positive_fit_slope_count"] == 5 for product in PRODUCTS)
    return (min(correlations.values()) if all_slopes_positive else 0.0), correlations


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_238_npl065_documented_winter_tmin_modern_oof_v2"
    output.mkdir(parents=True, exist_ok=False)
    round208 = loop / "round_208_setp_cru_tmin_annual_target_audit_v1"
    round237 = loop / "round_237_npl065_source_acquisition_preflight_v1"
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy

    paths: dict[str, Path] = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round238_npl065_documented_winter_tmin_modern_oof_v2_contract.md",
        "driver": Path(__file__).resolve(),
        "round237_summary": round237 / "summary.json",
        "rwl": round237 / "raw_unopened/npl065.rwl",
        "cru_manifest": round208 / "cru_tmn_acquisition_manifest.json",
        "bea": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["rwl"]["sha256"] != EXPECTED_RWL_SHA256:
        raise RuntimeError("npl065-source-hash-changed")
    round237_summary = json.loads(paths["round237_summary"].read_text(encoding="ascii"))
    if round237_summary.get("decision") != "advance_npl065_to_boundary_safe_modern_oof":
        raise RuntimeError("round237-admission-changed")

    raw, parse_metadata = parse_modern_rwl(paths["rwl"])
    rwi = dplpy.detrend(raw, fit="ModNegEx", method="residual", plot=False)
    chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
    std = chronology["Mean RWI"].astype(float).reindex(MATRIX_YEARS)
    depth = chronology["Sample depth"].astype(float).reindex(MATRIX_YEARS)
    if not np.isfinite(std.to_numpy()).all() or not np.isfinite(depth.to_numpy()).all():
        raise RuntimeError("modern-chronology-gap")
    calibration_index = pd.Index(MATRIX_YEARS).get_indexer(CALIBRATION_YEARS)
    if (calibration_index < 0).any():
        raise RuntimeError("calibration-year-index-failed")
    proxy = std.to_numpy(float)[calibration_index]
    calibration_depth = depth.to_numpy(float)[calibration_index]

    cru_monthly, cru_support, cru_check, cru_files = read_cru_tmin_point(paths["cru_manifest"])
    bea_monthly, bea_check = read_bea_point(paths["bea"])
    targets = {
        "CRU_TMIN": djf_series(cru_monthly, CALIBRATION_YEARS),
        "BEA_TMEAN": djf_series(bea_monthly, CALIBRATION_YEARS),
    }
    station_support = djf_support_fraction(cru_support, CALIBRATION_YEARS)
    if not all(np.isfinite(target).all() for target in targets.values()):
        raise RuntimeError("incomplete-temperature-target")

    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        metrics, prediction, folds = blocked_oof(proxy, targets[product])
        predictions[product] = prediction
        metric_rows.append({"product": product, **metrics})
        fold_rows.extend({"product": product, **row} for row in folds)
    metrics_frame = pd.DataFrame(metric_rows).set_index("product")
    prediction_correlation = correlation(predictions["CRU_TMIN"], predictions["BEA_TMEAN"])

    observed_statistic, observed_correlations = null_statistic(proxy, targets)
    null_rows: list[dict[str, Any]] = []
    for shift in range(1, len(CALIBRATION_YEARS)):
        statistic, correlations = null_statistic(np.roll(proxy, shift), targets)
        null_rows.append({
            "circular_shift_years": shift,
            "dual_product_statistic": statistic,
            "CRU_TMIN_oof_correlation": correlations["CRU_TMIN"],
            "BEA_TMEAN_oof_correlation": correlations["BEA_TMEAN"],
        })
    null_frame = pd.DataFrame(null_rows)
    empirical_p = float(
        (1 + int((null_frame.dual_product_statistic >= observed_statistic - 1e-15).sum()))
        / len(CALIBRATION_YEARS)
    )

    gates = {
        "complete_chronology_and_min_depth_ge_6": bool(np.min(calibration_depth) >= 6),
        "complete_cru_tmin_and_bea_tmean_targets": bool(all(np.isfinite(value).all() for value in targets.values())),
        "cru_station_supported_month_cell_fraction_ge_0_75": bool(station_support >= 0.75),
        "both_oof_correlations_ge_0_30": bool((metrics_frame.oof_correlation >= 0.30).all()),
        "both_oof_skills_positive": bool((metrics_frame.oof_skill_vs_training_fold_climatology > 0).all()),
        "both_products_four_of_five_fold_correlations_positive": bool((metrics_frame.positive_fold_correlation_count >= 4).all()),
        "all_ten_fit_slopes_positive": bool((metrics_frame.positive_fit_slope_count == 5).all()),
        "both_products_early_late_split_correlations_positive": bool(
            ((metrics_frame.early_split_proxy_target_correlation > 0)
             & (metrics_frame.late_split_proxy_target_correlation > 0)).all()
        ),
        "cru_bea_oof_prediction_correlation_ge_0_80": bool(prediction_correlation >= 0.80),
        "circular_shift_empirical_p_le_0_05": bool(empirical_p <= 0.05),
    }
    passes = bool(all(gates.values()))
    if passes:
        decision = "advance_npl065_to_exact_method_and_historical_transport"
    elif all(value for key, value in gates.items() if "station_supported" not in key) and not gates["cru_station_supported_month_cell_fraction_ge_0_75"]:
        decision = "hold_npl065_signal_due_to_inadequate_cru_station_support"
    else:
        decision = "close_npl065_documented_winter_temperature_modern_oof"

    chronology_frame = pd.DataFrame({
        "year": MATRIX_YEARS,
        "modern_only_std_chronology": std.to_numpy(float),
        "sample_depth": depth.to_numpy(float),
    })
    chronology_frame.to_csv(output / "modern_only_chronology.csv", index=False)
    curve_frame = pd.DataFrame({
        "year": CALIBRATION_YEARS,
        "chronology": proxy,
        "sample_depth": calibration_depth,
        "cru_djf_tmin_observed": targets["CRU_TMIN"],
        "cru_djf_tmin_oof": predictions["CRU_TMIN"],
        "bea_djf_tmean_observed": targets["BEA_TMEAN"],
        "bea_djf_tmean_oof": predictions["BEA_TMEAN"],
    })
    curve_frame.to_csv(output / "modern_oof_curves.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "oof_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    null_frame.to_csv(output / "circular_shift_null.csv", index=False)

    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True, constrained_layout=True)
    axes[0].plot(CALIBRATION_YEARS, proxy, color="#333333", linewidth=1.5)
    axes[0].set_ylabel("RWI")
    axes[0].set_title(f"NPL065 modern-only STD chronology; minimum depth {int(np.min(calibration_depth))}")
    colors = {"CRU_TMIN": "#006D77", "BEA_TMEAN": "#C44E52"}
    labels = {"CRU_TMIN": "CRU DJF Tmin", "BEA_TMEAN": "Berkeley DJF Tmean"}
    for axis, product in zip(axes[1:], PRODUCTS):
        row = metrics_frame.loc[product]
        axis.plot(CALIBRATION_YEARS, targets[product], color=colors[product], linewidth=1.5, label="Observed")
        axis.plot(CALIBRATION_YEARS, predictions[product], color="#222222", linewidth=1.25, linestyle="--", label=f"Blocked OOF, r={row.oof_correlation:.3f}")
        axis.set_ylabel("degC")
        axis.set_title(labels[product])
        axis.legend(frameon=False, ncol=2, fontsize=9)
    for axis in axes:
        for _, end in BLOCKS[:-1]:
            axis.axvline(end + 0.5, color="#999999", linestyle=":", linewidth=0.7)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Ring year")
    figure.savefig(output / "npl065_documented_winter_temperature_oof_curves.png", dpi=220)
    plt.close(figure)

    paths.update({f"cru_{index:02d}": path for index, path in enumerate(cru_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")

    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "question": "Does NPL065 reproduce documented positive DJF Tmin signal under blocked OOF?",
        "chronology_method": {
            "library": "dplpy",
            "version": dplpy.__version__,
            "input_years": [1941, 2000],
            "detrend_fit": "ModNegEx",
            "detrend_method": "residual",
            "chronology_mean": "Tukey_biweight",
            "prewhiten": False,
            "published_chronology_equivalence_claimed": False,
        },
        "parse_metadata": parse_metadata,
        "calibration_years": [1960, 2000],
        "folds": [list(block) for block in BLOCKS],
        "targets": {
            "CRU_TMIN": {"season": "previous_December_current_January_February", **cru_check},
            "BEA_TMEAN": {"season": "previous_December_current_January_February", **bea_check},
        },
        "cru_station_supported_month_cell_fraction": station_support,
        "minimum_sample_depth": int(np.min(calibration_depth)),
        "metrics": metric_rows,
        "cross_product_oof_prediction_correlation": prediction_correlation,
        "circular_shift_test": {
            "nonzero_shift_count": len(null_rows),
            "observed_statistic": observed_statistic,
            "observed_product_correlations": observed_correlations,
            "empirical_p_value": empirical_p,
        },
        "gates": gates,
        "passes": passes,
        "decision": decision,
        "paleo_trajectory_exported_or_inspected": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="ascii",
    )
    lines = [
        "# Round 238 NPL065 documented winter-temperature modern OOF",
        "",
        f"- CRU DJF Tmin OOF r: `{metrics_frame.at['CRU_TMIN', 'oof_correlation']:.3f}`; skill: `{metrics_frame.at['CRU_TMIN', 'oof_skill_vs_training_fold_climatology']:.3f}`.",
        f"- Berkeley DJF Tmean OOF r: `{metrics_frame.at['BEA_TMEAN', 'oof_correlation']:.3f}`; skill: `{metrics_frame.at['BEA_TMEAN', 'oof_skill_vs_training_fold_climatology']:.3f}`.",
        f"- Cross-product OOF prediction correlation: `{prediction_correlation:.3f}`.",
        f"- CRU station-supported month-cell fraction: `{station_support:.1%}`.",
        f"- Circular-shift p-value: `{empirical_p:.4f}`.",
        f"- Minimum chronology depth: `{int(np.min(calibration_depth))}`.",
        "",
        f"Decision: `{decision}`.",
        "",
        "The chronology is a modern-only exploratory STD reconstruction, not the paper's exact production chronology.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
