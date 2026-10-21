"""Validate the frozen central-JJA long-core shape against Chongce."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCKS = [(1940, 1959), (1960, 1979), (1980, 2000)]
PALEO_WINDOWS = [(start, start + 49) for start in range(1550, 1900, 50)]
HALF_WINDOW = 5


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_year(token: str) -> int:
    value = float(token)
    year = int(round(value))
    if not math.isfinite(value) or abs(value - year) > 1e-6:
        raise ValueError(f"invalid-year-token:{token!r}")
    return year


def decode_value(token: str) -> float:
    text = token.strip()
    if not text:
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def smooth_11(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(HALF_WINDOW, len(values) - HALF_WINDOW):
        result[index] = float(
            np.mean(values[index - HALF_WINDOW : index + HALF_WINDOW + 1])
        )
    return result


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def unique_site_mean(
    field: np.ndarray, locations: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    selected = []
    seen = set()
    for index in indices:
        key = (
            round(float(locations[index, 0]), 6),
            round(float(locations[index, 1]), 6),
        )
        if key not in seen:
            seen.add(key)
            selected.append(int(index))
    return np.nanmean(field[:, selected], axis=1)


def purged_oof(
    years: np.ndarray, predictors: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    prediction = np.full(target.shape, np.nan)
    baseline = np.full(target.shape, np.nan)
    rows = []
    positions = np.arange(len(years))
    low_target = smooth_11(target)
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test_centers = (years >= start + HALF_WINDOW) & (years <= end - HALF_WINDOW)
        train_centers = (
            ((years + HALF_WINDOW < start) | (years - HALF_WINDOW > end))
            & (positions >= HALF_WINDOW)
            & (positions < len(years) - HALF_WINDOW)
        )
        raw_support = np.zeros(len(years), dtype=bool)
        for center in np.flatnonzero(train_centers):
            raw_support[center - HALF_WINDOW : center + HALF_WINDOW + 1] = True
        counts = np.sum(np.isfinite(predictors[raw_support]), axis=0)
        means = np.nanmean(predictors[raw_support], axis=0)
        stds = np.nanstd(predictors[raw_support], axis=0)
        keep = (counts >= 10) & np.isfinite(means) & np.isfinite(stds) & (stds > 0)
        if int(keep.sum()) < 5:
            raise RuntimeError(f"insufficient-oof-features:fold={fold}")
        standardized = (predictors[:, keep] - means[keep]) / stds[keep]
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        composite = smooth_11(np.mean(standardized, axis=1))
        train = train_centers & np.isfinite(composite) & np.isfinite(low_target)
        test = test_centers & np.isfinite(composite) & np.isfinite(low_target)
        coefficients = np.linalg.lstsq(
            np.column_stack([np.ones(int(train.sum())), composite[train]]),
            low_target[train],
            rcond=None,
        )[0]
        prediction[test] = coefficients[0] + coefficients[1] * composite[test]
        baseline[test] = float(np.mean(low_target[train]))
        rows.append(
            {
                "fold": fold,
                "start_year": start,
                "end_year": end,
                "retained_proxy_count": int(keep.sum()),
                "slope": float(coefficients[1]),
                "test_correlation": correlation(low_target[test], prediction[test]),
            }
        )
    return prediction, baseline, rows


def fit_full_mapping(
    predictors: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    counts = np.sum(np.isfinite(predictors), axis=0)
    means = np.nanmean(predictors, axis=0)
    stds = np.nanstd(predictors, axis=0)
    keep = (counts >= 40) & np.isfinite(means) & np.isfinite(stds) & (stds > 0)
    if int(keep.sum()) < 5:
        raise RuntimeError("insufficient-full-fit-features")
    standardized = (predictors[:, keep] - means[keep]) / stds[keep]
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    composite = smooth_11(np.mean(standardized, axis=1))
    low_target = smooth_11(target)
    valid = np.isfinite(composite) & np.isfinite(low_target)
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(int(valid.sum())), composite[valid]]),
        low_target[valid],
        rcond=None,
    )[0]
    return keep, means, stds, float(coefficients[0]), float(coefficients[1])


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / np.std(values)


def main() -> None:
    root = project_root()
    base = root / "validation_data/results/paleo_surface_validation_v0"
    loop = base / "climate_reconstruction_improvement_loop_v2"
    drivers = (
        root
        / "Himalaya_greatGlacier/With_Rec_climate_organized"
        / "ice_surface_reconstruction/drivers"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=loop
        / "round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=loop
        / "round_14_temperature_product_cross_validation_v1/temperature_products_v1.mat",
    )
    parser.add_argument(
        "--long-core",
        type=Path,
        default=loop
        / "round_185_central_jja_historical_support_preflight_v1/coverage_only_long_core.csv",
    )
    parser.add_argument(
        "--lmr-csv",
        type=Path,
        default=Path(
            "D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master/"
            "proxy-data/LMR_NCDC_database_v.0.2.0/lmr_data.csv"
        ),
    )
    parser.add_argument(
        "--chongce-windows",
        type=Path,
        default=base
        / "climate_reconstruction_temperature_chongce_independent_blind_validation_v1"
        / "chongce_50year_windows.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=loop / "round_186_central_jja_chongce_shape_validation_v1",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    contract = drivers / "round186_central_jja_chongce_shape_validation_v1_contract.md"
    shutil.copyfile(contract, args.output / "ROUND_CONTRACT.md")
    inputs = {
        "network_manifest": args.manifest,
        "temperature_products": args.products,
        "long_core": args.long_core,
        "lmr_csv": args.lmr_csv,
        "chongce_windows": args.chongce_windows,
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in inputs.items()}

    metadata = pd.read_csv(args.manifest)
    core = pd.read_csv(args.long_core)
    if len(core) != 14 or not core["coverage_only_long_core"].all():
        raise RuntimeError("long-core-contract-failed")
    identities = core["formal_identity"].tolist()
    network_indices = np.array(
        [
            int(metadata.loc[metadata.formal_identity == identity, "network_index_1based"].iloc[0]) - 1
            for identity in identities
        ],
        dtype=int,
    )
    central_indices = np.flatnonzero(
        (metadata["season_code"].to_numpy() == "JJA")
        & (metadata["longitude"].to_numpy(float) >= 80.0)
        & (metadata["longitude"].to_numpy(float) < 90.0)
    )
    if len(central_indices) != 41:
        raise RuntimeError("central-network-count-changed")

    with h5py.File(args.products, "r") as handle:
        modern_years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        locations = np.asarray(handle["locations"], dtype=float).T
        temperature = np.asarray(handle["temperature"], dtype=float).transpose(3, 2, 1, 0)
    if not np.array_equal(modern_years, np.arange(1940, 2001)):
        raise RuntimeError("modern-year-contract-failed")

    historical_years = np.arange(1545, 1905)
    needed_years = set(historical_years.tolist()) | set(modern_years.tolist())
    raw_by_year: dict[int, np.ndarray] = {}
    locked_rows_skipped = 0
    with args.lmr_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: index for index, name in enumerate(header)}
        if any(identity not in positions for identity in identities):
            raise RuntimeError("historical-header-match-failed")
        columns = [positions[identity] for identity in identities]
        for row in reader:
            if not row:
                continue
            year = decode_year(row[0])
            if 1921 <= year <= 1939:
                locked_rows_skipped += 1
                continue
            if year not in needed_years:
                continue
            if year in raw_by_year:
                raise RuntimeError(f"duplicate-year:{year}")
            raw_by_year[year] = np.array([decode_value(row[column]) for column in columns])

    historical = np.vstack(
        [raw_by_year.get(int(year), np.full(len(identities), np.nan)) for year in historical_years]
    )
    raw_modern = np.vstack(
        [raw_by_year.get(int(year), np.full(len(identities), np.nan)) for year in modern_years]
    )
    mat_modern = observations[:, network_indices]
    common = np.isfinite(raw_modern) & np.isfinite(mat_modern)
    if int(common.sum()) == 0:
        raise RuntimeError("no-modern-csv-mat-overlap")
    modern_max_abs_error = float(np.max(np.abs(raw_modern[common] - mat_modern[common])))
    if modern_max_abs_error > 1e-12:
        raise RuntimeError(f"csv-mat-modern-mismatch:{modern_max_abs_error}")

    chongce = pd.read_csv(args.chongce_windows)
    if list(zip(chongce.window_start, chongce.window_end)) != PALEO_WINDOWS:
        raise RuntimeError("chongce-window-contract-failed")
    chongce_values = chongce["chongce_degC"].to_numpy(float)
    project_values = chongce["project_degC"].to_numpy(float)
    project_correlation = correlation(project_values, chongce_values)

    product_names = ["Berkeley", "CRU"]
    window_output = chongce[["window_start", "window_end", "project_degC", "chongce_degC"]].copy()
    modern_rows = []
    paleo_metrics: dict[str, dict[str, float | int]] = {}
    full_fit: dict[str, dict[str, float | int]] = {}

    for product_index, product in enumerate(product_names):
        target = unique_site_mean(
            temperature[:, :, 1, product_index], locations, central_indices
        )
        oof_prediction, oof_baseline, fold_rows = purged_oof(
            modern_years, mat_modern, target
        )
        low_target = smooth_11(target)
        valid = np.isfinite(low_target) & np.isfinite(oof_prediction) & np.isfinite(oof_baseline)
        oof_corr = correlation(low_target[valid], oof_prediction[valid])
        oof_skill = float(
            1.0
            - rmse(low_target[valid], oof_prediction[valid])
            / rmse(low_target[valid], oof_baseline[valid])
        )
        all_slopes_positive = bool(all(float(row["slope"]) > 0 for row in fold_rows))
        modern_rows.append(
            {
                "product": product,
                "pooled_oof_correlation": oof_corr,
                "absolute_skill": oof_skill,
                "all_fold_slopes_positive": all_slopes_positive,
                "fold_metrics": fold_rows,
            }
        )

        keep, means, stds, intercept, slope = fit_full_mapping(mat_modern, target)
        historical_standardized = (historical[:, keep] - means[keep]) / stds[keep]
        historical_standardized = np.where(
            np.isfinite(historical_standardized), historical_standardized, 0.0
        )
        historical_composite = smooth_11(np.mean(historical_standardized, axis=1))
        historical_candidate = intercept + slope * historical_composite
        candidate_windows = []
        valid_center_counts = []
        for start, end in PALEO_WINDOWS:
            mask = (
                (historical_years >= start)
                & (historical_years <= end)
                & np.isfinite(historical_candidate)
            )
            valid_center_counts.append(int(mask.sum()))
            candidate_windows.append(float(np.mean(historical_candidate[mask])))
        candidate_windows_array = np.asarray(candidate_windows)
        window_output[f"central_{product.lower()}_degC"] = candidate_windows_array
        candidate_corr = correlation(candidate_windows_array, chongce_values)
        sign_agreement = int(
            np.sum(
                np.sign(np.diff(candidate_windows_array))
                == np.sign(np.diff(chongce_values))
            )
        )
        paleo_metrics[product] = {
            "chongce_window_correlation": candidate_corr,
            "improvement_over_project_correlation": candidate_corr - project_correlation,
            "adjacent_change_sign_agreement_of_6": sign_agreement,
        }
        full_fit[product] = {
            "retained_proxy_count": int(keep.sum()),
            "intercept": intercept,
            "slope": slope,
            "minimum_valid_centers_per_50year_window": min(valid_center_counts),
        }

    berkeley = window_output["central_berkeley_degC"].to_numpy(float)
    cru = window_output["central_cru_degC"].to_numpy(float)
    interproduct_corr = correlation(berkeley, cru)
    modern_gates = {
        f"{row['product']}_oof_correlation_at_least_0p30": bool(
            row["pooled_oof_correlation"] >= 0.30
        )
        for row in modern_rows
    }
    modern_gates.update(
        {
            f"{row['product']}_absolute_skill_at_least_0p05": bool(
                row["absolute_skill"] >= 0.05
            )
            for row in modern_rows
        }
    )
    modern_gates.update(
        {
            f"{row['product']}_all_fold_slopes_positive": bool(
                row["all_fold_slopes_positive"]
            )
            for row in modern_rows
        }
    )
    paleo_gates = {
        "both_chongce_correlations_at_least_0p10": bool(
            all(paleo_metrics[p]["chongce_window_correlation"] >= 0.10 for p in product_names)
        ),
        "both_improve_project_correlation_by_at_least_0p20": bool(
            all(
                paleo_metrics[p]["improvement_over_project_correlation"] >= 0.20
                for p in product_names
            )
        ),
        "berkeley_cru_window_correlation_at_least_0p80": bool(interproduct_corr >= 0.80),
        "both_adjacent_sign_agreement_at_least_4_of_6": bool(
            all(
                paleo_metrics[p]["adjacent_change_sign_agreement_of_6"] >= 4
                for p in product_names
            )
        ),
        "both_full_fit_slopes_positive": bool(
            all(full_fit[p]["slope"] > 0 for p in product_names)
        ),
    }
    passes_modern = bool(all(modern_gates.values()))
    passes_paleo = bool(all(paleo_gates.values()))
    passes_all = passes_modern and passes_paleo

    window_output.to_csv(args.output / "central_jja_chongce_50year_windows.csv", index=False)
    pd.DataFrame(
        [
            {
                "product": row["product"],
                "pooled_oof_correlation": row["pooled_oof_correlation"],
                "absolute_skill": row["absolute_skill"],
                "all_fold_slopes_positive": row["all_fold_slopes_positive"],
            }
            for row in modern_rows
        ]
    ).to_csv(args.output / "modern_oof_summary.csv", index=False)

    x = (window_output.window_start + window_output.window_end) / 2
    fig, axis = plt.subplots(figsize=(10.5, 5.3))
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.plot(x, zscore(chongce_values), color="#111111", marker="o", linewidth=2.2, label="Chongce annual")
    axis.plot(x, zscore(project_values), color="#d55e00", marker="s", linewidth=1.8, label="Current project local")
    axis.plot(x, zscore(berkeley), color="#0072b2", marker="^", linewidth=1.8, label="Central JJA - Berkeley")
    axis.plot(x, zscore(cru), color="#009e73", marker="D", linewidth=1.8, label="Central JJA - CRU")
    axis.set(
        title="Central JJA long-core versus Chongce: 50-year low-frequency shape",
        xlabel="50-year window midpoint",
        ylabel="Standardized seven-window anomaly",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "central_jja_chongce_shape.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("source-fingerprint-changed")
    if passes_all:
        decision = "advance_central_jja_shape_mechanism_not_amplitude_or_production"
    elif not passes_modern:
        decision = "close_14_record_long_core_on_modern_transport_failure"
    else:
        decision = "close_central_jja_shape_on_independent_chongce_failure"
    summary = {
        "status": "complete",
        "decision": decision,
        "long_core_record_count": len(core),
        "original_central_target_site_record_count": len(central_indices),
        "csv_mat_modern_max_abs_error": modern_max_abs_error,
        "locked_rows_skipped_before_proxy_conversion": locked_rows_skipped,
        "locked_1921_1939_proxy_value_accessed": False,
        "annual_paleo_candidate_exported": False,
        "project_chongce_window_correlation": project_correlation,
        "modern_results": modern_rows,
        "paleo_results": paleo_metrics,
        "full_fit": full_fit,
        "berkeley_cru_window_correlation": interproduct_corr,
        "modern_gates": modern_gates,
        "paleo_gates": paleo_gates,
        "passes_modern_gates": passes_modern,
        "passes_paleo_gates": passes_paleo,
        "passes_all_frozen_gates": passes_all,
        "production_release_authorized": False,
        "sources_unchanged": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "source_manifest.json").write_text(
        json.dumps(
            {
                "inputs": {name: str(path) for name, path in inputs.items()},
                "sha256_before": before,
                "sha256_after": after,
                "sources_unchanged": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result_text = (
        "# Round 186 Result\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Modern gates pass: `{passes_modern}`; paleo-shape gates pass: `{passes_paleo}`.\n"
        f"- Project-Chongce correlation: `{project_correlation:.3f}`.\n"
        f"- Central candidate correlations: Berkeley `{paleo_metrics['Berkeley']['chongce_window_correlation']:.3f}`, CRU `{paleo_metrics['CRU']['chongce_window_correlation']:.3f}`.\n"
        f"- Berkeley-CRU candidate correlation: `{interproduct_corr:.3f}`.\n"
        "- No production reconstruction, amplitude correction, EnKF, or OGGM run was authorized.\n"
    )
    (args.output / "FINAL_RESULT.md").write_text(result_text, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

