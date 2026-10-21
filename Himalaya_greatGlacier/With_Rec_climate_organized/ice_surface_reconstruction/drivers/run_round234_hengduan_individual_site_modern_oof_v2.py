from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCKED_START = 1921
LOCKED_END = 1939
MATRIX_YEARS = np.arange(1940, 2001, dtype=int)
TARGET_YEARS = np.arange(1940, 2000, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 1999))
PRODUCTS = ("BEA", "CRU")
SITES = {
    "DLL": (27 + 47 / 60, 98 + 28 / 60),
    "FGH": (27 + 11 / 60, 98 + 43 / 60),
    "KQH": (28 + 4 / 60, 98 + 45 / 60),
    "XYS": (26 + 26 / 60, 98 + 46 / 60),
    "YPL": (27 + 10 / 60, 98 + 43 / 60),
}


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
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def parse_modern_rwl(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: dict[str, dict[int, float]] = {}
    locked_slots_skipped = 0
    paleo_slots_skipped = 0
    decoded_modern_slots = 0
    with path.open("r", encoding="ascii", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip() or line == "\x1a":
                continue
            if len(line) < 12:
                raise RuntimeError(f"short-rwl-line:{path.name}:{line_number}")
            sample = line[:8].strip()
            year_text = line[8:12].strip()
            if not year_text.lstrip("-").isdigit():
                raise RuntimeError(f"invalid-rwl-year:{path.name}:{line_number}:{year_text}")
            start_year = int(year_text)
            series = records.setdefault(sample, {})
            for offset, token in enumerate(line[12:].split()):
                year = start_year + offset
                if LOCKED_START <= year <= LOCKED_END:
                    locked_slots_skipped += 1
                    continue
                if year < 1940 or year > 2000:
                    paleo_slots_skipped += int(year < 1940)
                    continue
                value = int(token)
                decoded_modern_slots += 1
                if value in {-9999, 9999}:
                    continue
                series[year] = value / 1000.0
    frame = pd.DataFrame(index=pd.Index(MATRIX_YEARS, name="Year"), columns=sorted(records), dtype=float)
    for sample, values in records.items():
        for year, value in values.items():
            frame.at[year, sample] = value
    return frame, {
        "sample_count": int(frame.shape[1]),
        "decoded_modern_value_count": int(frame.notna().sum().sum()),
        "locked_slots_skipped_before_numeric_conversion": int(locked_slots_skipped),
        "paleo_slots_skipped_before_numeric_conversion": int(paleo_slots_skipped),
        "decoded_modern_slots_including_missing_markers": int(decoded_modern_slots),
        "locked_numeric_values_decoded": False,
        "paleo_numeric_values_decoded": False,
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def blocked_oof(proxy: np.ndarray, target: np.ndarray) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    if proxy.shape != (60,) or target.shape != (60,) or not np.isfinite(proxy).all() or not np.isfinite(target).all():
        raise RuntimeError("incomplete-oof-input")
    prediction = np.full(60, np.nan)
    climatology = np.full(60, np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (start, end) in enumerate(BLOCKS, start=1):
        test = (TARGET_YEARS >= start) & (TARGET_YEARS <= end)
        train = ~test
        design = np.column_stack((np.ones(int(train.sum())), proxy[train]))
        coefficients = np.linalg.lstsq(design, target[train], rcond=None)[0]
        prediction[test] = coefficients[0] + coefficients[1] * proxy[test]
        climatology[test] = np.mean(target[train])
        fold_rows.append({
            "fold": fold_index,
            "test_years": f"{start}-{end}",
            "fit_intercept": float(coefficients[0]),
            "fit_slope": float(coefficients[1]),
            "test_correlation": correlation(target[test], prediction[test]),
            "test_rmse_degC": float(np.sqrt(np.mean((target[test] - prediction[test]) ** 2))),
        })
    candidate_sse = float(np.sum((target - prediction) ** 2))
    baseline_sse = float(np.sum((target - climatology) ** 2))
    early = TARGET_YEARS <= 1969
    late = TARGET_YEARS >= 1970
    folds = pd.DataFrame(fold_rows)
    metrics = {
        "oof_correlation": correlation(target, prediction),
        "oof_skill_vs_training_fold_climatology": 1 - candidate_sse / baseline_sse,
        "oof_rmse_degC": float(np.sqrt(candidate_sse / len(TARGET_YEARS))),
        "positive_fold_correlation_count": int((folds.test_correlation > 0).sum()),
        "positive_fit_slope_count": int((folds.fit_slope > 0).sum()),
        "early_split_proxy_target_correlation": correlation(proxy[early], target[early]),
        "late_split_proxy_target_correlation": correlation(proxy[late], target[late]),
    }
    return metrics, prediction, fold_rows


def family_score(
    proxies: dict[str, np.ndarray],
    targets: dict[str, dict[str, np.ndarray]],
) -> tuple[float, str]:
    scores: dict[str, float] = {}
    for site, proxy in proxies.items():
        metrics = {product: blocked_oof(proxy, targets[site][product])[0] for product in PRODUCTS}
        all_slopes_positive = all(metrics[product]["positive_fit_slope_count"] == 5 for product in PRODUCTS)
        scores[site] = (
            min(metrics["BEA"]["oof_correlation"], metrics["CRU"]["oof_correlation"])
            if all_slopes_positive
            else 0.0
        )
    best = max(scores, key=scores.get)
    return float(scores[best]), best


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_234_hengduan_individual_site_modern_oof_v2"
    output.mkdir(parents=True, exist_ok=False)
    raw_root = loop / "round_178_mendeley_east_hma_temperature_candidates_v1/raw_unopened/889n6ccx4z_v1"
    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy

    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths: dict[str, Path] = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round234_hengduan_individual_site_modern_oof_v2_contract.md",
        "driver": Path(__file__).resolve(),
        "round180": loop / "round_180_hengduan_chronology_modern_oof_v2/summary.json",
        "round196": loop / "round_196_hengduan_four_site_reproducibility_decision_v1/summary.json",
        "bea": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
    }
    paths.update({f"rwl_{site}": raw_root / site for site in SITES})
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    latitudes = np.asarray([SITES[site][0] for site in SITES], dtype=float)
    longitudes = np.asarray([SITES[site][1] for site in SITES], dtype=float)
    r118 = load_module(
        "round234_r118",
        drivers / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    monthly = {
        "BEA": r118.read_berkeley_points(paths["bea"], latitudes, longitudes),
        "CRU": r118.read_netcdf3_points(paths["cru"], "tmp", 1901, latitudes, longitudes),
    }
    targets: dict[str, dict[str, np.ndarray]] = {site: {} for site in SITES}
    for product in PRODUCTS:
        annual = r118.aggregate_temperature(monthly[product], [tuple(range(5, 10))] * len(SITES))
        if annual.shape != (61, 5) or not np.isfinite(annual).all():
            raise RuntimeError(f"temperature-target-failed:{product}:{annual.shape}")
        for site_index, site in enumerate(SITES):
            targets[site][product] = annual[:60, site_index].astype(float)

    proxies: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    parse_metadata: dict[str, Any] = {}
    chronology_rows: list[pd.DataFrame] = []
    for site in SITES:
        raw, metadata = parse_modern_rwl(paths[f"rwl_{site}"])
        parse_metadata[site] = metadata
        rwi = dplpy.detrend(raw, fit="ModNegEx", method="residual", plot=False)
        chronology = dplpy.chron(rwi, biweight=True, prewhiten=False, plot=False)
        std = chronology["Mean RWI"].astype(float).reindex(MATRIX_YEARS)
        depth = chronology["Sample depth"].astype(float).reindex(MATRIX_YEARS)
        if not np.isfinite(std.to_numpy()).all() or not np.isfinite(depth.to_numpy()).all():
            raise RuntimeError(f"modern-chronology-gap:{site}")
        proxies[site] = std.iloc[:60].to_numpy(float)
        depths[site] = depth.to_numpy(float)
        chronology_rows.append(pd.DataFrame({
            "site": site,
            "year": MATRIX_YEARS,
            "std_chronology": std.to_numpy(float),
            "sample_depth": depth.to_numpy(float),
        }))
    pd.concat(chronology_rows, ignore_index=True).to_csv(output / "modern_site_chronologies.csv", index=False)

    site_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    predictions: dict[str, dict[str, np.ndarray]] = {site: {} for site in SITES}
    site_gates: dict[str, dict[str, bool]] = {}
    for site in SITES:
        product_metrics: dict[str, dict[str, Any]] = {}
        site_curve = pd.DataFrame({"site": site, "year": TARGET_YEARS, "chronology": proxies[site]})
        for product in PRODUCTS:
            metrics, prediction, folds = blocked_oof(proxies[site], targets[site][product])
            product_metrics[product] = metrics
            predictions[site][product] = prediction
            site_curve[f"{product}_observed"] = targets[site][product]
            site_curve[f"{product}_oof"] = prediction
            site_rows.append({"site": site, "product": product, **metrics})
            fold_rows.extend({"site": site, "product": product, **row} for row in folds)
        min_depth = int(np.min(depths[site]))
        prediction_correlation = correlation(predictions[site]["BEA"], predictions[site]["CRU"])
        frame = pd.DataFrame([row for row in site_rows if row["site"] == site])
        gates = {
            "complete_modern_chronology_and_min_depth_ge_6": bool(min_depth >= 6),
            "both_oof_correlations_ge_0_30": bool((frame.oof_correlation >= 0.30).all()),
            "both_oof_skills_positive": bool((frame.oof_skill_vs_training_fold_climatology > 0).all()),
            "both_products_four_of_five_fold_correlations_positive": bool((frame.positive_fold_correlation_count >= 4).all()),
            "all_ten_fit_slopes_positive": bool((frame.positive_fit_slope_count == 5).all()),
            "both_products_early_late_split_correlations_positive": bool(
                ((frame.early_split_proxy_target_correlation > 0) & (frame.late_split_proxy_target_correlation > 0)).all()
            ),
            "bea_cru_oof_prediction_correlation_ge_0_80": bool(prediction_correlation >= 0.80),
        }
        site_gates[site] = gates
        site_curve["sample_depth"] = depths[site][:60]
        curve_rows.append(site_curve)
    pd.DataFrame(site_rows).to_csv(output / "site_oof_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(output / "modern_oof_curves.csv", index=False)

    observed_score, observed_best_site = family_score(proxies, targets)
    null_rows: list[dict[str, Any]] = []
    for shift in range(1, 60):
        shifted = {site: np.roll(proxy, shift) for site, proxy in proxies.items()}
        score, best_site = family_score(shifted, targets)
        null_rows.append({"circular_shift_years": shift, "family_score": score, "best_site": best_site})
    null = pd.DataFrame(null_rows)
    family_p = float((1 + int((null.family_score >= observed_score - 1e-15).sum())) / 60)
    family_gate = bool(family_p <= 0.05)
    any_site_gate = bool(any(all(gates.values()) for gates in site_gates.values()))
    passing_sites = [site for site, gates in site_gates.items() if all(gates.values())]
    decision = (
        "advance_best_hengduan_site_to_lineage_and_exact_method_validation"
        if family_gate and any_site_gate
        else "close_hengduan_individual_site_modern_oof"
    )
    null.to_csv(output / "familywise_circular_shift_null.csv", index=False)

    metrics_frame = pd.DataFrame(site_rows)
    best_plot_site = observed_best_site
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True, constrained_layout=True)
    colors = {"BEA": "#006D77", "CRU": "#C44E52"}
    for axis, product in zip(axes, PRODUCTS):
        observed = targets[best_plot_site][product]
        predicted = predictions[best_plot_site][product]
        row = metrics_frame[(metrics_frame.site == best_plot_site) & (metrics_frame["product"] == product)].iloc[0]
        axis.plot(TARGET_YEARS, observed, color=colors[product], lw=1.4, label=f"{product} observed")
        axis.plot(TARGET_YEARS, predicted, color="#222222", lw=1.2, ls="--", label=f"OOF prediction, r={row.oof_correlation:.3f}")
        for _, end in BLOCKS[:-1]:
            axis.axvline(end + 0.5, color="#999999", ls=":", lw=0.7)
        axis.set_ylabel("May-Sep temp (degC)")
        axis.grid(axis="y", color="#dddddd", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=9)
    axes[0].set_title(f"Hengduan site {best_plot_site}: blocked modern OOF")
    axes[-1].set_xlabel("Year")
    fig.savefig(output / "best_site_modern_oof_curves.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "modern_only_individual_site_discovery_screen",
        "chronology_method": {
            "library": "dplpy",
            "version": dplpy.__version__,
            "detrend_fit": "ModNegEx",
            "detrend_method": "residual",
            "chronology_mean": "Tukey_biweight",
            "prewhiten": False,
            "published_RCSigFree_equivalence_claimed": False,
        },
        "parse_metadata": parse_metadata,
        "site_metrics": site_rows,
        "site_gates": site_gates,
        "passing_sites_before_family_gate": passing_sites,
        "familywise_test": {
            "statistic": "max_site_min_dual_product_oof_r_with_all_positive_slopes_else_zero",
            "joint_nonzero_circular_shift_count": 59,
            "observed_score": observed_score,
            "observed_best_site": observed_best_site,
            "empirical_p_value": family_p,
            "passes_p_le_0_05": family_gate,
        },
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
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="ascii"
    )
    lines = [
        "# Round 234 Hengduan Individual-Site Modern OOF",
        "",
        f"- Best family statistic site: `{observed_best_site}`; score: `{observed_score:.3f}`.",
        f"- Family-wise circular-shift p-value: `{family_p:.4f}`.",
        f"- Sites passing all site gates before family correction: `{', '.join(passing_sites) if passing_sites else 'none'}`.",
    ]
    for site in SITES:
        rows = metrics_frame[metrics_frame.site == site].set_index("product")
        lines.append(
            f"- {site}: BEA/CRU OOF r=`{rows.at['BEA', 'oof_correlation']:.3f}`/"
            f"`{rows.at['CRU', 'oof_correlation']:.3f}`; minimum depth=`{int(np.min(depths[site]))}`."
        )
    lines.extend([
        "",
        f"Decision: `{decision}`.",
        "",
        "This modern dplpy screen is not an RCSigFree chronology and is not production-eligible.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

