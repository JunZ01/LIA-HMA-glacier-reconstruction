from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MATRIX_YEARS = np.arange(1941, 2014, dtype=int)
SHORT_YEARS = np.arange(1960, 2001, dtype=int)
FULL_YEARS = np.arange(1960, 2014, dtype=int)
ENDPOINT_YEARS = np.arange(2001, 2014, dtype=int)
SHORT_BLOCKS = ((1960, 1967), (1968, 1975), (1976, 1983), (1984, 1991), (1992, 2000))
FULL_BLOCKS = ((1960, 1968), (1969, 1977), (1978, 1986), (1987, 1995), (1996, 2004), (2005, 2013))
LAT_RANGE = (27.5, 28.5)
LON_RANGE = (86.0, 87.0)
LOCKED_START = 1921
LOCKED_END = 1939
EXPECTED_RWL_SHA256 = "d7f9d588e1362745857ac2aea3ca7c0686cee8b6ecc410e20af23a97d81d7bcd"
PUBLISHED_R = 0.781


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def parse_modern_rwl(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: dict[str, dict[int, float]] = {}
    skipped = {"locked": 0, "pre_1941": 0, "post_2013": 0}
    decoded = 0
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
                continue
            start_year = int(year_text)
            series = records.setdefault(sample, {})
            for offset, token in enumerate(line[12:].split()):
                if token in (b"999", b"-9999", b"9999"):
                    break
                year = start_year + offset
                if LOCKED_START <= year <= LOCKED_END:
                    skipped["locked"] += 1
                    continue
                if year < int(MATRIX_YEARS[0]):
                    skipped["pre_1941"] += 1
                    continue
                if year > int(MATRIX_YEARS[-1]):
                    skipped["post_2013"] += 1
                    continue
                series[year] = float(int(token.decode("ascii", errors="strict")))
                decoded += 1
    frame = pd.DataFrame(index=pd.Index(MATRIX_YEARS, name="Year"), columns=sorted(records), dtype=float)
    for sample, values in records.items():
        for year, value in values.items():
            frame.at[year, sample] = value
    frame = frame.dropna(axis=1, how="all")
    return frame, {
        "sample_count": int(frame.shape[1]),
        "decoded_1941_2013_value_count": decoded,
        "locked_slots_skipped_before_numeric_conversion": skipped["locked"],
        "pre_1941_slots_skipped_before_numeric_conversion": skipped["pre_1941"],
        "post_2013_slots_skipped_before_numeric_conversion": skipped["post_2013"],
        "locked_numeric_values_decoded": False,
        "paleo_numeric_values_decoded": False,
    }


def detrend_with_linear_fallback(dplpy: Any, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = []
    fallbacks = []
    for sample in raw.columns:
        try:
            series = dplpy.detrend(raw[sample], fit="ModNegEx", method="residual", plot=False)
        except RuntimeError:
            series = dplpy.detrend(raw[sample], fit="linear", method="residual", plot=False)
            fallbacks.append(str(sample))
        columns.append(series.rename(sample))
    return pd.concat(columns, axis=1).reindex(raw.index), fallbacks


def area_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def read_cru_box(manifests: list[Path]) -> tuple[np.ndarray, float, list[Path], dict[str, Any]]:
    monthly: dict[tuple[int, int], float] = {}
    support: dict[tuple[int, int], float] = {}
    files: list[Path] = []
    selected_lats = None
    selected_lons = None
    ranges = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            start, end = [int(value) for value in record["allowed_year_range"].split("-")]
            if end < 1959 or start > 2013:
                continue
            if start <= LOCKED_END:
                raise RuntimeError(f"forbidden-cru-range:{start}-{end}")
            path = Path(record["netcdf_path"])
            if fingerprint(path)["sha256"] != record["netcdf_sha256"]:
                raise RuntimeError(f"cru-hash-changed:{path.name}")
            files.append(path)
            with netcdf_file(path, "r", mmap=True) as dataset:
                latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
                longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
                iy = np.flatnonzero((latitudes >= LAT_RANGE[0]) & (latitudes <= LAT_RANGE[1]))
                ix = np.flatnonzero((longitudes >= LON_RANGE[0]) & (longitudes <= LON_RANGE[1]))
                if len(iy) != 2 or len(ix) != 2:
                    raise RuntimeError(f"published-box-cell-count-changed:{len(iy)}x{len(ix)}")
                tmn = np.asarray(dataset.variables["tmn"].data[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], dtype=float).copy()
                stn = np.asarray(dataset.variables["stn"].data[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], dtype=float).copy()
                fill = float(dataset.variables["tmn"]._attributes["_FillValue"])
                tmn[np.isclose(tmn, fill) | (tmn > 1e20)] = np.nan
                selected_lats = latitudes[iy].astype(float)
                selected_lons = longitudes[ix].astype(float)
            periods = pd.period_range(f"{start}-01", periods=12 * (end - start + 1), freq="M")
            values = area_mean(tmn, selected_lats)
            fractions = np.mean(stn > 0, axis=(1, 2))
            if len(periods) != len(values):
                raise RuntimeError(f"cru-month-count-changed:{path.name}")
            for period, value, fraction in zip(periods, values, fractions):
                key = (int(period.year), int(period.month))
                if key in monthly:
                    raise RuntimeError(f"duplicate-cru-month:{key}")
                monthly[key] = float(value)
                support[key] = float(fraction)
            ranges.append([start, end])
    result = np.full(len(FULL_YEARS), np.nan)
    support_slots = []
    for index, year in enumerate(FULL_YEARS):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        values = np.asarray([monthly.get(key, np.nan) for key in keys], dtype=float)
        if np.isfinite(values).all():
            result[index] = float(np.mean(values))
        support_slots.extend(support.get(key, 0.0) for key in keys)
    if not np.isfinite(result).all():
        missing = FULL_YEARS[~np.isfinite(result)].tolist()
        raise RuntimeError(f"incomplete-cru-target:{missing}")
    return result, float(np.mean(support_slots)), files, {
        "latitude_centers": selected_lats.tolist(),
        "longitude_centers": selected_lons.tolist(),
        "cell_count": 4,
        "weighting": "cosine_latitude",
        "source_ranges": ranges,
    }


def blocked_oof(years: np.ndarray, proxy: np.ndarray, target: np.ndarray, blocks: tuple[tuple[int, int], ...]):
    prediction = np.full(len(years), np.nan)
    climatology = np.full(len(years), np.nan)
    folds = []
    for fold, (start, end) in enumerate(blocks, start=1):
        test = (years >= start) & (years <= end)
        train = ~test
        beta = np.linalg.lstsq(
            np.column_stack((np.ones(int(train.sum())), proxy[train])), target[train], rcond=None
        )[0]
        prediction[test] = beta[0] + beta[1] * proxy[test]
        climatology[test] = float(np.mean(target[train]))
        folds.append({
            "fold": fold,
            "test_start": start,
            "test_end": end,
            "fit_slope": float(beta[1]),
            "test_correlation": correlation(target[test], prediction[test]),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("incomplete-oof-prediction")
    sse = float(np.sum((target - prediction) ** 2))
    baseline_sse = float(np.sum((target - climatology) ** 2))
    metrics = {
        "year_start": int(years[0]),
        "year_end": int(years[-1]),
        "year_count": len(years),
        "in_sample_correlation": correlation(proxy, target),
        "oof_correlation": correlation(prediction, target),
        "oof_skill_vs_training_fold_climatology": 1.0 - sse / baseline_sse,
        "oof_rmse_degC": float(np.sqrt(sse / len(years))),
        "positive_fit_slope_count": int(sum(row["fit_slope"] > 0 for row in folds)),
        "positive_fold_correlation_count": int(sum(row["test_correlation"] > 0 for row in folds)),
    }
    return metrics, prediction, folds


def standardize(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / np.std(values, ddof=1)


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_246_npl065_calibration_endpoint_attribution_v2"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round246_npl065_calibration_endpoint_attribution_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "round238_chronology": loop / "round_238_npl065_documented_winter_tmin_modern_oof_v4/modern_only_chronology.csv",
        "pre2001_manifest": loop / "round_208_setp_cru_tmin_annual_target_audit_v1/cru_tmn_acquisition_manifest.json",
        "post2000_manifest": loop / "round_245_cru_ts401_tmin_post2000_acquisition_v1/acquisition_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["rwl"]["sha256"] != EXPECTED_RWL_SHA256:
        raise RuntimeError("npl065-source-hash-changed")

    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy

    raw, parse_metadata = parse_modern_rwl(paths["rwl"])
    rwi, fallbacks = detrend_with_linear_fallback(dplpy, raw)
    chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
    proxy_all = chronology["Mean RWI"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
    depth_all = chronology["Sample depth"].astype(float).reindex(MATRIX_YEARS).to_numpy(float)
    if not np.isfinite(proxy_all).all() or not np.isfinite(depth_all).all():
        raise RuntimeError("modern-chronology-gap")

    target_full, support, cru_files, box = read_cru_box([paths["pre2001_manifest"], paths["post2000_manifest"]])
    paths.update({f"cru_{index:02d}": path for index, path in enumerate(cru_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})
    proxy_full = proxy_all[pd.Index(MATRIX_YEARS).get_indexer(FULL_YEARS)]
    short_mask = FULL_YEARS <= 2000
    endpoint_mask = FULL_YEARS >= 2001
    short_metrics, short_prediction, short_folds = blocked_oof(
        SHORT_YEARS, proxy_full[short_mask], target_full[short_mask], SHORT_BLOCKS
    )
    full_metrics, full_prediction, full_folds = blocked_oof(
        FULL_YEARS, proxy_full, target_full, FULL_BLOCKS
    )
    endpoint_r = correlation(proxy_full[endpoint_mask], target_full[endpoint_mask])

    old = pd.read_csv(paths["round238_chronology"])
    overlap = old[(old.year >= 1941) & (old.year <= 2000)]
    rebuilt_overlap = proxy_all[pd.Index(MATRIX_YEARS).get_indexer(overlap.year.to_numpy(int))]
    chronology_overlap_r = correlation(overlap.modern_only_std_chronology.to_numpy(float), rebuilt_overlap)
    chronology_overlap_rmse = float(np.sqrt(np.mean((overlap.modern_only_std_chronology.to_numpy(float) - rebuilt_overlap) ** 2)))

    metrics_frame = pd.DataFrame([
        {"period": "1960-2000", **short_metrics},
        {"period": "1960-2013", **full_metrics},
    ])
    metrics_frame.to_csv(output / "calibration_period_metrics.csv", index=False)
    pd.DataFrame(
        [{"period": "1960-2000", **row} for row in short_folds]
        + [{"period": "1960-2013", **row} for row in full_folds]
    ).to_csv(output / "blocked_fold_metrics.csv", index=False)
    curves = pd.DataFrame({
        "year": FULL_YEARS,
        "chronology": proxy_full,
        "chronology_z": standardize(proxy_full),
        "cru_box_djf_tmin": target_full,
        "cru_box_djf_tmin_z": standardize(target_full),
        "oof_1960_2000": np.r_[short_prediction, np.full(len(FULL_YEARS) - len(short_prediction), np.nan)],
        "oof_1960_2013": full_prediction,
    })
    curves.to_csv(output / "calibration_endpoint_curves.csv", index=False)
    pd.DataFrame({
        "year": MATRIX_YEARS,
        "modern_only_std_chronology": proxy_all,
        "sample_depth": depth_all,
    }).to_csv(output / "modern_only_chronology_1941_2013.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(FULL_YEARS, standardize(target_full), color="#006D77", linewidth=1.6, label="CRU TS4.01 box DJF Tmin")
    axes[0].plot(FULL_YEARS, standardize(proxy_full), color="#C44E52", linewidth=1.4, label="Approximate NPL065 chronology")
    axes[0].axvline(2000.5, color="#555555", linestyle=":", linewidth=1)
    axes[0].set_ylabel("standardized anomaly")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title(f"1960-2000 r={short_metrics['in_sample_correlation']:.3f}; 1960-2013 r={full_metrics['in_sample_correlation']:.3f}; published r={PUBLISHED_R:.3f}")
    axes[1].plot(FULL_YEARS, target_full, color="#006D77", linewidth=1.5, label="Observed")
    axes[1].plot(FULL_YEARS, full_prediction, color="#222222", linestyle="--", linewidth=1.25, label=f"Six-block OOF, r={full_metrics['oof_correlation']:.3f}")
    axes[1].axvspan(2000.5, 2013.5, color="#E9C46A", alpha=0.22, label="Added endpoint")
    axes[1].set_ylabel("degC")
    axes[1].set_xlabel("Ring year")
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "npl065_calibration_endpoint_attribution.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    in_sample_gain = full_metrics["in_sample_correlation"] - short_metrics["in_sample_correlation"]
    oof_gain = full_metrics["oof_correlation"] - short_metrics["oof_correlation"]
    published_gap_remaining = PUBLISHED_R - full_metrics["in_sample_correlation"]
    if published_gap_remaining <= 0.05:
        decision = "calibration_endpoint_and_reporting_metric_explain_published_skill_gap"
    elif in_sample_gain >= 0.10:
        decision = "post2000_endpoint_material_but_exact_chronology_still_required"
    else:
        decision = "post2000_endpoint_not_primary_exact_chronology_still_required"
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "chronology_method": {
            "input_years": [1941, 2013],
            "method": "dplpy_ModNegEx_residual_or_linear_fallback_Tukey_biweight_no_prewhiten",
            "linear_fallbacks": fallbacks,
            "published_chronology_equivalence_claimed": False,
        },
        "parse_metadata": parse_metadata,
        "target": {"dataset": "CRU_TS4.01_tmn", "season": "DJF", "box": box},
        "station_supported_cell_month_fraction": support,
        "metrics": {"1960_2000": short_metrics, "1960_2013": full_metrics, "2001_2013_in_sample_correlation": endpoint_r},
        "attribution": {
            "in_sample_correlation_gain_from_adding_2001_2013": in_sample_gain,
            "oof_correlation_gain_from_adding_2001_2013": oof_gain,
            "published_correlation": PUBLISHED_R,
            "remaining_gap_to_published_in_sample_correlation": published_gap_remaining,
            "chronology_1941_2000_overlap_correlation_round238_vs_rebuilt": chronology_overlap_r,
            "chronology_1941_2000_overlap_rmse": chronology_overlap_rmse,
        },
        "decision": decision,
        "locked_1921_1939_numeric_values_decoded": False,
        "paleo_proxy_values_decoded": False,
        "jionglacuo_opened_or_hashed": False,
        "binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 246 NPL065 calibration-endpoint attribution\n\n"
        f"- 1960-2000 in-sample r: `{short_metrics['in_sample_correlation']:.3f}`; blocked OOF r: `{short_metrics['oof_correlation']:.3f}`.\n"
        f"- 1960-2013 in-sample r: `{full_metrics['in_sample_correlation']:.3f}`; blocked OOF r: `{full_metrics['oof_correlation']:.3f}`.\n"
        f"- Adding 2001-2013 changes in-sample r by `{in_sample_gain:+.3f}` and OOF r by `{oof_gain:+.3f}`.\n"
        f"- 2001-2013 endpoint-only r: `{endpoint_r:.3f}`.\n"
        f"- Remaining gap to published r=0.781: `{published_gap_remaining:.3f}`.\n"
        f"- Round238/rebuilt overlap chronology r: `{chronology_overlap_r:.3f}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
