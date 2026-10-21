"""Test the frozen 1600-coverage central-JJA cohort against Chongce."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_round186_central_jja_chongce_shape_validation_v1 import (
    correlation,
    decode_value,
    decode_year,
    fit_full_mapping,
    sha256,
    smooth_11,
    unique_site_mean,
    zscore,
)


WINDOWS = [(start, start + 49) for start in range(1600, 1900, 50)]


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


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
        "--coverage",
        type=Path,
        default=loop
        / "round_185_central_jja_historical_support_preflight_v1/record_coverage.csv",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=loop
        / "round_14_temperature_product_cross_validation_v1/temperature_products_v1.mat",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        default=loop
        / "round_187_central_jja_coverage_skill_ladder_v1/coverage_skill_ladder.csv",
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
        default=loop / "round_188_central_jja_1600_chongce_shape_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    contract = drivers / "round188_central_jja_1600_chongce_shape_v1_contract.md"
    helper = drivers / "run_round186_central_jja_chongce_shape_validation_v1.py"
    shutil.copyfile(contract, args.output / "ROUND_CONTRACT.md")
    inputs = {
        "manifest": args.manifest,
        "coverage": args.coverage,
        "temperature_products": args.products,
        "coverage_skill_ladder": args.ladder,
        "lmr_csv": args.lmr_csv,
        "chongce_windows": args.chongce_windows,
        "contract": contract,
        "round186_helper": helper,
    }
    before = {name: sha256(path) for name, path in inputs.items()}

    metadata = pd.read_csv(args.manifest)
    coverage = pd.read_csv(args.coverage)
    ladder = pd.read_csv(args.ladder)
    modern_branch = ladder.loc[ladder.cohort == "coverage_from_1600"]
    if len(modern_branch) != 2 or not modern_branch.passes_product_gate.all():
        raise RuntimeError("round187-modern-gate-not-passed")
    required = [f"finite_{start}_{start + 49}" for start in range(1600, 1900, 50)]
    eligible = coverage.loc[
        (coverage[required] >= 40).all(axis=1)
        & (coverage["finite_1940_2000"] >= 40),
        "formal_identity",
    ].tolist()
    if len(eligible) != 30:
        raise RuntimeError(f"unexpected-1600-cohort-count:{len(eligible)}")
    network_indices = np.array(
        [
            int(metadata.loc[metadata.formal_identity == identity, "network_index_1based"].iloc[0]) - 1
            for identity in eligible
        ],
        dtype=int,
    )
    central_indices = np.flatnonzero(
        (metadata["season_code"].to_numpy() == "JJA")
        & (metadata["longitude"].to_numpy(float) >= 80.0)
        & (metadata["longitude"].to_numpy(float) < 90.0)
    )

    with h5py.File(args.products, "r") as handle:
        modern_years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        locations = np.asarray(handle["locations"], dtype=float).T
        temperature = np.asarray(handle["temperature"], dtype=float).transpose(3, 2, 1, 0)
    if not np.array_equal(modern_years, np.arange(1940, 2001)):
        raise RuntimeError("modern-year-contract-failed")

    historical_years = np.arange(1595, 1905)
    needed_years = set(historical_years.tolist()) | set(modern_years.tolist())
    values_by_year: dict[int, np.ndarray] = {}
    locked_rows_skipped = 0
    with args.lmr_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: index for index, name in enumerate(header)}
        if any(identity not in positions for identity in eligible):
            raise RuntimeError("historical-header-match-failed")
        columns = [positions[identity] for identity in eligible]
        for row in reader:
            if not row:
                continue
            year = decode_year(row[0])
            if 1921 <= year <= 1939:
                locked_rows_skipped += 1
                continue
            if year in needed_years:
                values_by_year[year] = np.array(
                    [decode_value(row[column]) for column in columns]
                )
    historical = np.vstack(
        [values_by_year.get(int(year), np.full(len(eligible), np.nan)) for year in historical_years]
    )
    raw_modern = np.vstack(
        [values_by_year.get(int(year), np.full(len(eligible), np.nan)) for year in modern_years]
    )
    mat_modern = observations[:, network_indices]
    common = np.isfinite(raw_modern) & np.isfinite(mat_modern)
    modern_max_abs_error = float(np.max(np.abs(raw_modern[common] - mat_modern[common])))
    if modern_max_abs_error > 1e-12:
        raise RuntimeError(f"csv-mat-modern-mismatch:{modern_max_abs_error}")

    chongce = pd.read_csv(args.chongce_windows)
    chongce = chongce.loc[chongce.window_start >= 1600].reset_index(drop=True)
    if list(zip(chongce.window_start, chongce.window_end)) != WINDOWS:
        raise RuntimeError("chongce-window-contract-failed")
    chongce_values = chongce.chongce_degC.to_numpy(float)
    project_values = chongce.project_degC.to_numpy(float)
    project_corr = correlation(project_values, chongce_values)
    output = chongce[["window_start", "window_end", "project_degC", "chongce_degC"]].copy()

    metrics = {}
    fit_rows = []
    for product_index, product in enumerate(["Berkeley", "CRU"]):
        target = unique_site_mean(
            temperature[:, :, 1, product_index], locations, central_indices
        )
        keep, means, stds, intercept, slope = fit_full_mapping(mat_modern, target)
        standardized = (historical[:, keep] - means[keep]) / stds[keep]
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        candidate = intercept + slope * smooth_11(np.mean(standardized, axis=1))
        window_values = []
        counts = []
        for start, end in WINDOWS:
            mask = (
                (historical_years >= start)
                & (historical_years <= end)
                & np.isfinite(candidate)
            )
            counts.append(int(mask.sum()))
            window_values.append(float(np.mean(candidate[mask])))
        vector = np.asarray(window_values)
        output[f"central_{product.lower()}_degC"] = vector
        corr = correlation(vector, chongce_values)
        sign_agreement = int(
            np.sum(np.sign(np.diff(vector)) == np.sign(np.diff(chongce_values)))
        )
        metrics[product] = {
            "chongce_window_correlation": corr,
            "improvement_over_project_correlation": corr - project_corr,
            "adjacent_change_sign_agreement_of_5": sign_agreement,
        }
        fit_rows.append(
            {
                "product": product,
                "retained_proxy_count": int(keep.sum()),
                "intercept": intercept,
                "slope": slope,
                "minimum_valid_centers_per_window": min(counts),
            }
        )

    berkeley = output.central_berkeley_degC.to_numpy(float)
    cru = output.central_cru_degC.to_numpy(float)
    interproduct_corr = correlation(berkeley, cru)
    gates = {
        "round187_modern_oof_passes_both_products": True,
        "both_chongce_correlations_at_least_0p10": bool(
            all(metrics[p]["chongce_window_correlation"] >= 0.10 for p in metrics)
        ),
        "both_improve_project_correlation_by_at_least_0p20": bool(
            all(metrics[p]["improvement_over_project_correlation"] >= 0.20 for p in metrics)
        ),
        "both_adjacent_sign_agreement_at_least_3_of_5": bool(
            all(metrics[p]["adjacent_change_sign_agreement_of_5"] >= 3 for p in metrics)
        ),
        "berkeley_cru_window_correlation_at_least_0p80": bool(interproduct_corr >= 0.80),
        "both_full_fit_slopes_positive": bool(all(row["slope"] > 0 for row in fit_rows)),
    }
    passes = bool(all(gates.values()))
    decision = (
        "advance_1600_1899_central_jja_shape_mechanism_only"
        if passes
        else "close_1600_1899_central_jja_shape_on_chongce"
    )
    output.to_csv(args.output / "central_jja_1600_chongce_windows.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.output / "full_fit_coefficients.csv", index=False)

    x = (output.window_start + output.window_end) / 2
    fig, axis = plt.subplots(figsize=(10.2, 5.2))
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.plot(x, zscore(chongce_values), color="#111111", marker="o", linewidth=2.2, label="Chongce annual")
    axis.plot(x, zscore(project_values), color="#d55e00", marker="s", linewidth=1.8, label="Current project local")
    axis.plot(x, zscore(berkeley), color="#0072b2", marker="^", linewidth=1.8, label="Central JJA - Berkeley")
    axis.plot(x, zscore(cru), color="#009e73", marker="D", linewidth=1.8, label="Central JJA - CRU")
    axis.set(
        title="Frozen 1600-coverage central JJA cohort versus Chongce",
        xlabel="50-year window midpoint",
        ylabel="Standardized six-window anomaly",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "central_jja_1600_chongce_shape.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "status": "complete",
        "decision": decision,
        "cohort_record_count": len(eligible),
        "validation_windows": WINDOWS,
        "project_chongce_window_correlation": project_corr,
        "candidate_metrics": metrics,
        "berkeley_cru_window_correlation": interproduct_corr,
        "full_fit": fit_rows,
        "gates": gates,
        "passes_all_frozen_gates": passes,
        "csv_mat_modern_max_abs_error": modern_max_abs_error,
        "locked_rows_skipped_before_proxy_conversion": locked_rows_skipped,
        "locked_1921_1939_proxy_value_accessed": False,
        "annual_paleo_candidate_exported": False,
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
    (args.output / "FINAL_RESULT.md").write_text(
        "# Round 188 Result\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Project-Chongce correlation: `{project_corr:.3f}`.\n"
        f"- Candidate correlations: Berkeley `{metrics['Berkeley']['chongce_window_correlation']:.3f}`, CRU `{metrics['CRU']['chongce_window_correlation']:.3f}`.\n"
        "- No amplitude correction or production release was authorized.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

