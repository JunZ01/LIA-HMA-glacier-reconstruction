from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCKED_START = 1921
LOCKED_END = 1939
MATRIX_YEARS = np.arange(1940, 2001, dtype=int)
PROXY_YEARS = np.arange(1941, 2001, dtype=int)
TARGET_YEARS = np.arange(1940, 2000, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 1999))
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
    digest = None
    if stat.st_size <= 100_000_000:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def parse_rwl_without_locked_values(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    records: dict[str, dict[int, float]] = {}
    locked_tokens_skipped = 0
    decoded_value_count = 0
    for raw_line in path.read_bytes().splitlines():
        tokens = raw_line.split()
        if len(tokens) < 3:
            continue
        try:
            start_year = int(tokens[1])
        except ValueError:
            continue
        sample = tokens[0].decode("ascii", errors="strict")
        series = records.setdefault(sample, {})
        for offset, raw_value in enumerate(tokens[2:]):
            year = start_year + offset
            if LOCKED_START <= year <= LOCKED_END:
                locked_tokens_skipped += 1
                continue
            value = int(raw_value)
            decoded_value_count += 1
            if value in (999, -9999):
                break
            series[year] = value / 100.0
    years = sorted({year for series in records.values() for year in series})
    frame = pd.DataFrame(index=pd.Index(years, name="Year"), columns=sorted(records), dtype=float)
    for sample, series in records.items():
        for year, value in series.items():
            frame.at[year, sample] = value
    if any(LOCKED_START <= int(year) <= LOCKED_END for year in frame.index):
        raise RuntimeError("locked-year-entered-rwl-frame")
    metadata = {
        "sample_count": int(frame.shape[1]),
        "allowed_year_min": int(frame.index.min()),
        "allowed_year_max": int(frame.index.max()),
        "allowed_value_count": int(frame.notna().sum().sum()),
        "locked_tokens_skipped_before_numeric_conversion": int(locked_tokens_skipped),
        "decoded_value_count": int(decoded_value_count),
        "locked_numeric_values_decoded": False,
    }
    return frame, metadata


def parse_two_column_without_locked_values(path: Path) -> pd.Series:
    values: dict[int, float] = {}
    for raw_line in path.read_bytes().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        first = re.split(rb"\s+", stripped, maxsplit=1)[0]
        try:
            year = int(first)
        except ValueError:
            continue
        if LOCKED_START <= year <= LOCKED_END:
            continue
        text = stripped.decode("ascii", errors="strict")
        columns = re.split(r"\s+", text)
        if len(columns) < 2:
            continue
        values[year] = float(columns[1])
    result = pd.Series(values, dtype=float).sort_index()
    if any(LOCKED_START <= int(year) <= LOCKED_END for year in result.index):
        raise RuntimeError("locked-year-entered-two-column-series")
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def affine_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    count = int(valid.sum())
    if count < 3:
        return {"overlap_count": count, "correlation": np.nan, "maximum_abs_affine_residual": np.nan}
    left = candidate[valid]
    right = reference[valid]
    design = np.column_stack((right, np.ones(count)))
    coefficient = np.linalg.lstsq(design, left, rcond=None)[0]
    residual = float(np.max(np.abs(left - design @ coefficient)))
    return {
        "overlap_count": count,
        "correlation": correlation(left, right),
        "maximum_abs_affine_residual": residual,
    }


def main() -> int:
    root = project_root()
    driver_dir = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_167_chin074_qamdo2_identity_recovery_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    vendor = output / "vendor"
    sys.path.insert(0, str(vendor))
    import dplpy

    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    matrix_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = output / "raw_unopened/mainz_qamdo_archive/Qamdo"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver_dir / "round167_qamdo2_std_chronology_modern_oof_v2_contract.md",
        "driver": Path(__file__).resolve(),
        "raw_rwl": archive / "Qamdo.txt",
        "published_reconstruction": archive / "Qamdo_Recon.txt",
        "source_zip": output / "raw_unopened/mainz_Qamdo.zip",
        "sample_identity": output / "sample_identity_summary.json",
        "proxy6": matrix_dir / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy9": matrix_dir / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "proxy9_inventory": loop / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "bea": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    raw, raw_metadata = parse_rwl_without_locked_values(paths["raw_rwl"])
    if raw_metadata["sample_count"] != 144:
        raise RuntimeError(f"unexpected-sample-count:{raw_metadata['sample_count']}")
    rwi = dplpy.detrend(raw, fit="ModNegEx", method="residual", plot=False)
    chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
    std = chronology["Mean RWI"].astype(float)
    depth = chronology["Sample depth"].astype(float)
    published = parse_two_column_without_locked_values(paths["published_reconstruction"])

    std_matrix = std.reindex(MATRIX_YEARS).to_numpy(float)
    published_matrix = published.reindex(MATRIX_YEARS).to_numpy(float)
    if not np.isfinite(std_matrix).all() or not np.isfinite(published_matrix).all():
        raise RuntimeError("modern-candidate-gap")
    lineage_correlation = correlation(std_matrix, published_matrix)

    proxy9_inventory = pd.read_csv(paths["proxy9_inventory"])
    proxy9_identity = {}
    for index in range(1, 298):
        rows = proxy9_inventory[proxy9_inventory.record_index_1based == index]
        proxy9_identity[index] = str(rows.iloc[0].identity_key) if not rows.empty else ""
    alias_rows = []
    for network in ("proxy6", "proxy9"):
        observations = np.asarray(loadmat(paths[network], variable_names=["observations"])["observations"], dtype=float)
        if observations.shape[0] != len(MATRIX_YEARS):
            raise RuntimeError(f"{network}-year-shape:{observations.shape}")
        for source_name, candidate in (("std_chronology", std_matrix), ("published_reconstruction_lineage_only", published_matrix)):
            for column in range(observations.shape[1]):
                metrics = affine_metrics(candidate, observations[:, column])
                corr = float(metrics["correlation"])
                residual = float(metrics["maximum_abs_affine_residual"])
                exact = bool(metrics["overlap_count"] >= 50 and abs(corr) >= 0.999999 and residual <= 1e-8)
                near = bool(metrics["overlap_count"] >= 50 and abs(corr) >= 0.995)
                alias_rows.append({
                    "network": network,
                    "candidate_source": source_name,
                    "column_1based": column + 1,
                    "proxy9_identity_if_available": proxy9_identity.get(column + 1, "") if network == "proxy9" else "",
                    **metrics,
                    "exact_affine_alias": exact,
                    "near_alias": near,
                })
    alias = pd.DataFrame(alias_rows)
    alias.to_csv(output / "proxy6_proxy9_alias_metrics.csv", index=False)
    top_alias = alias.reindex(alias.correlation.abs().sort_values(ascending=False).index).groupby(["network", "candidate_source"], as_index=False).head(5)
    top_alias.to_csv(output / "proxy6_proxy9_top_aliases.csv", index=False)
    alias_rejected = bool(alias.exact_affine_alias.any() or alias.near_alias.any())

    modern = pd.DataFrame({
        "ring_year": MATRIX_YEARS,
        "std_chronology": std_matrix,
        "sample_depth": depth.reindex(MATRIX_YEARS).to_numpy(float),
        "published_reconstruction_lineage_only": published_matrix,
    })
    modern.to_csv(output / "qamdo_modern_candidate_series.csv", index=False)

    oof_rows = []
    fold_rows = []
    gate = None
    decision = "close_qamdo2_proxy6_proxy9_alias" if alias_rejected else "pending_modern_oof"
    predictions: dict[str, np.ndarray] = {}
    climate: dict[str, np.ndarray] = {}
    proxy_oof = std.reindex(PROXY_YEARS).to_numpy(float)
    if not alias_rejected:
        if not np.isfinite(proxy_oof).all():
            raise RuntimeError("proxy-oof-gap")
        r118 = load_module("round167_r118", driver_dir / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py")
        latitudes = np.asarray([31.0])
        longitudes = np.asarray([97.0])
        bea_monthly = r118.read_berkeley_points(paths["bea"], latitudes, longitudes)
        cru_monthly = r118.read_netcdf3_points(paths["cru"], "tmp", 1901, latitudes, longitudes)
        annual_months = [tuple(range(1, 13))]
        for product, monthly in (("BEA", bea_monthly), ("CRU", cru_monthly)):
            annual = r118.aggregate_temperature(monthly, annual_months)
            if annual.shape != (61, 1) or not np.isfinite(annual).all():
                raise RuntimeError(f"temperature-target-failed:{product}:{annual.shape}")
            climate[product] = annual[:60, 0]
            predictions[product] = np.full(60, np.nan)
            climatology = np.full(60, np.nan)
            for fold_index, (start, end) in enumerate(BLOCKS, start=1):
                test = (TARGET_YEARS >= start) & (TARGET_YEARS <= end)
                train = ~test
                design = np.column_stack((np.ones(int(train.sum())), proxy_oof[train]))
                coefficients = np.linalg.lstsq(design, climate[product][train], rcond=None)[0]
                predictions[product][test] = coefficients[0] + coefficients[1] * proxy_oof[test]
                climatology[test] = np.mean(climate[product][train])
                fold_rows.append({
                    "product": product,
                    "fold": fold_index,
                    "target_test_years": f"{start}-{end}",
                    "ring_test_years": f"{start + 1}-{end + 1}",
                    "fit_intercept": float(coefficients[0]),
                    "fit_slope": float(coefficients[1]),
                    "test_correlation": correlation(climate[product][test], predictions[product][test]),
                    "test_rmse_degC": float(np.sqrt(np.mean((climate[product][test] - predictions[product][test]) ** 2))),
                })
            folds = pd.DataFrame([row for row in fold_rows if row["product"] == product])
            candidate_sse = float(np.sum((climate[product] - predictions[product]) ** 2))
            baseline_sse = float(np.sum((climate[product] - climatology) ** 2))
            early = TARGET_YEARS <= 1969
            late = TARGET_YEARS >= 1970
            oof_rows.append({
                "product": product,
                "oof_correlation": correlation(climate[product], predictions[product]),
                "oof_skill_vs_training_fold_climatology": 1 - candidate_sse / baseline_sse,
                "oof_rmse_degC": float(np.sqrt(candidate_sse / len(TARGET_YEARS))),
                "positive_fold_correlation_count": int((folds.test_correlation > 0).sum()),
                "positive_fit_slope_count": int((folds.fit_slope > 0).sum()),
                "early_split_proxy_target_correlation": correlation(proxy_oof[early], climate[product][early]),
                "late_split_proxy_target_correlation": correlation(proxy_oof[late], climate[product][late]),
            })
        oof = pd.DataFrame(oof_rows)
        product_prediction_correlation = correlation(predictions["BEA"], predictions["CRU"])
        gate = {
            "both_oof_correlations_ge_0_30": bool((oof.oof_correlation >= 0.30).all()),
            "both_oof_skills_positive": bool((oof.oof_skill_vs_training_fold_climatology > 0).all()),
            "both_products_four_of_five_fold_correlations_positive": bool((oof.positive_fold_correlation_count >= 4).all()),
            "all_ten_fit_slopes_positive": bool((oof.positive_fit_slope_count == 5).all()),
            "both_products_early_late_split_correlations_positive": bool(((oof.early_split_proxy_target_correlation > 0) & (oof.late_split_proxy_target_correlation > 0)).all()),
            "bea_cru_oof_prediction_correlation_ge_0_80": bool(product_prediction_correlation >= 0.80),
        }
        passes = bool(all(gate.values()))
        decision = "advance_qamdo2_to_signal_free_rcs_replication" if passes else "close_qamdo2_modern_temperature_oof"
        pd.DataFrame(fold_rows).to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
        oof.to_csv(output / "blocked_oof_summary.csv", index=False)
        curves = pd.DataFrame({"target_year": TARGET_YEARS, "ring_year": PROXY_YEARS, "std_chronology": proxy_oof})
        for product in PRODUCTS:
            curves[f"{product}_observed"] = climate[product]
            curves[f"{product}_oof"] = predictions[product]
        curves.to_csv(output / "modern_oof_curves.csv", index=False)

        fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True, constrained_layout=True)
        colors = {"BEA": "#0072B2", "CRU": "#D55E00"}
        for axis, product in zip(axes, PRODUCTS):
            row = oof[oof["product"] == product].iloc[0]
            axis.plot(TARGET_YEARS, climate[product], color=colors[product], linewidth=1.5, label=f"{product} observed")
            axis.plot(TARGET_YEARS, predictions[product], color="#222222", linestyle="--", linewidth=1.2, label=f"Qamdo STD OOF, r={row.oof_correlation:.3f}")
            for _, end in BLOCKS[:-1]:
                axis.axvline(end + 0.5, color="#999999", linestyle=":", linewidth=0.7)
            axis.set_ylabel("annual temperature (degC)")
            axis.grid(axis="y", color="#dddddd", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, ncol=2, fontsize=9)
        axes[-1].set_xlabel("target climate year (ring year = target + 1)")
        fig.savefig(output / "qamdo_modern_oof_curves.png", dpi=220)
        plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    top_records = top_alias.replace({np.nan: None}).to_dict(orient="records")
    result = {
        "experiment_id": "round_167_chin074_qamdo2_identity_recovery_v1",
        "analysis_version": "std_chronology_modern_oof_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "chronology_method": {"library": "dplpy", "version": dplpy.__version__, "detrend_fit": "ModNegEx", "detrend_method": "residual", "chronology_mean": "Tukey_biweight", "prewhiten": False},
        "raw_parse": raw_metadata,
        "modern_sample_depth_min": int(np.nanmin(depth.reindex(MATRIX_YEARS).to_numpy(float))),
        "std_vs_published_reconstruction_modern_correlation": lineage_correlation,
        "sample_id_reuse_fraction_vs_chin046_chin047": json.loads(paths["sample_identity"].read_text(encoding="utf-8"))["chin074_ids_reused_fraction"],
        "proxy_alias_rejected": alias_rejected,
        "exact_alias_count": int(alias.exact_affine_alias.sum()),
        "near_alias_count": int(alias.near_alias.sum()),
        "top_aliases": top_records,
        "oof": oof_rows,
        "modern_gate": gate,
        "decision": decision,
        "paleo_trajectory_exported_or_inspected": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "published_reconstruction_used_as_oof_predictor": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = [
        "# Round 167 Qamdo 2 Identity and Modern OOF",
        "",
        f"- Raw CHIN074 samples: `{raw_metadata['sample_count']}`; locked numeric values decoded: `false`.",
        f"- Minimum 1940-2000 sample depth: `{result['modern_sample_depth_min']}`.",
        f"- STD chronology versus published reconstruction modern correlation: `{lineage_correlation:.3f}`.",
        f"- Proxy6/proxy9 exact aliases: `{result['exact_alias_count']}`; near aliases: `{result['near_alias_count']}`.",
    ]
    for row in oof_rows:
        lines.append(f"- {row['product']}: OOF r=`{row['oof_correlation']:.3f}`, skill=`{row['oof_skill_vs_training_fold_climatology']:.3f}`, positive folds=`{row['positive_fold_correlation_count']}/5`.")
    lines.extend(["", f"Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
