"""Locate when coverage-constrained central-JJA cohorts recover modern skill."""

from __future__ import annotations

import argparse
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
    purged_oof,
    rmse,
    sha256,
    smooth_11,
    unique_site_mean,
)


CUTOFFS = list(range(1550, 1900, 50))


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
        "--output",
        type=Path,
        default=loop / "round_187_central_jja_coverage_skill_ladder_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    contract = drivers / "round187_central_jja_coverage_skill_ladder_v1_contract.md"
    shutil.copyfile(contract, args.output / "ROUND_CONTRACT.md")
    helper = drivers / "run_round186_central_jja_chongce_shape_validation_v1.py"
    inputs = {
        "manifest": args.manifest,
        "coverage": args.coverage,
        "temperature_products": args.products,
        "contract": contract,
        "round186_helper": helper,
    }
    before = {name: sha256(path) for name, path in inputs.items()}

    metadata = pd.read_csv(args.manifest)
    coverage = pd.read_csv(args.coverage)
    with h5py.File(args.products, "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        locations = np.asarray(handle["locations"], dtype=float).T
        temperature = np.asarray(handle["temperature"], dtype=float).transpose(3, 2, 1, 0)
    if not np.array_equal(years, np.arange(1940, 2001)):
        raise RuntimeError("modern-year-contract-failed")
    central_indices = np.flatnonzero(
        (metadata["season_code"].to_numpy() == "JJA")
        & (metadata["longitude"].to_numpy(float) >= 80.0)
        & (metadata["longitude"].to_numpy(float) < 90.0)
    )
    if len(central_indices) != 41 or len(coverage) != 41:
        raise RuntimeError("central-network-contract-failed")

    cohorts: list[tuple[str, int, np.ndarray]] = []
    previous: set[str] = set()
    for cutoff in CUTOFFS:
        required = [f"finite_{start}_{start + 49}" for start in range(cutoff, 1900, 50)]
        eligible = coverage.loc[
            (coverage[required] >= 40).all(axis=1)
            & (coverage["finite_1940_2000"] >= 40),
            "formal_identity",
        ].tolist()
        current = set(eligible)
        if not previous.issubset(current):
            raise RuntimeError(f"cohort-not-nested:{cutoff}")
        previous = current
        indices = np.array(
            [
                int(metadata.loc[metadata.formal_identity == identity, "network_index_1based"].iloc[0]) - 1
                for identity in eligible
            ],
            dtype=int,
        )
        cohorts.append((f"coverage_from_{cutoff}", cutoff, indices))
    cohorts.append(("modern_all_41", 1900, central_indices))

    rows = []
    cohort_results = []
    for label, plot_year, indices in cohorts:
        product_results = []
        for product_index, product in enumerate(["Berkeley", "CRU"]):
            target = unique_site_mean(
                temperature[:, :, 1, product_index], locations, central_indices
            )
            prediction, baseline, folds = purged_oof(
                years, observations[:, indices], target
            )
            low_target = smooth_11(target)
            valid = np.isfinite(low_target) & np.isfinite(prediction) & np.isfinite(baseline)
            corr = correlation(low_target[valid], prediction[valid])
            skill = float(
                1.0
                - rmse(low_target[valid], prediction[valid])
                / rmse(low_target[valid], baseline[valid])
            )
            slopes_positive = bool(all(float(row["slope"]) > 0 for row in folds))
            passes = bool(corr >= 0.30 and skill >= 0.05 and slopes_positive)
            row = {
                "cohort": label,
                "coverage_start_or_plot_year": plot_year,
                "record_count": len(indices),
                "product": product,
                "pooled_oof_correlation": corr,
                "absolute_skill": skill,
                "all_fold_slopes_positive": slopes_positive,
                "passes_product_gate": passes,
            }
            rows.append(row)
            product_results.append(row)
        cohort_results.append(
            {
                "cohort": label,
                "coverage_start_or_plot_year": plot_year,
                "record_count": len(indices),
                "passes_both_products": bool(
                    all(row["passes_product_gate"] for row in product_results)
                ),
            }
        )

    result = pd.DataFrame(rows)
    cohort_table = pd.DataFrame(cohort_results)
    passing_historical = cohort_table.loc[
        cohort_table.cohort.str.startswith("coverage_from_")
        & cohort_table.passes_both_products
    ]
    earliest_passing_cutoff = (
        int(passing_historical.coverage_start_or_plot_year.min())
        if len(passing_historical)
        else None
    )
    result.to_csv(args.output / "coverage_skill_ladder.csv", index=False)
    cohort_table.to_csv(args.output / "cohort_gate_summary.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True)
    colors = {"Berkeley": "#0072b2", "CRU": "#009e73"}
    markers = {"Berkeley": "o", "CRU": "s"}
    for product in ["Berkeley", "CRU"]:
        subset = result.loc[result["product"] == product]
        axes[0].plot(
            subset.coverage_start_or_plot_year,
            subset.pooled_oof_correlation,
            color=colors[product],
            marker=markers[product],
            label=product,
        )
        axes[1].plot(
            subset.coverage_start_or_plot_year,
            subset.absolute_skill,
            color=colors[product],
            marker=markers[product],
            label=product,
        )
    axes[0].axhline(0.30, color="#555555", linestyle="--", linewidth=1)
    axes[1].axhline(0.05, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Pooled OOF correlation")
    axes[1].set_ylabel("Absolute OOF skill")
    axes[1].set_xlabel("Required historical coverage start; 1900 = all 41 modern records")
    for axis in axes:
        axis.axhline(0, color="#999999", linewidth=0.7)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Central JJA modern skill versus historical coverage depth")
    fig.tight_layout()
    fig.savefig(args.output / "coverage_skill_ladder.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "status": "complete",
        "decision": (
            "modern_signal_not_transportable_before_first_passing_cutoff"
            if earliest_passing_cutoff is not None
            else "no_coverage_constrained_historical_cohort_recovers_modern_signal"
        ),
        "earliest_passing_coverage_cutoff": earliest_passing_cutoff,
        "cohorts": cohort_results,
        "locked_1921_1939_proxy_values_accessed": False,
        "historical_proxy_values_reopened": False,
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

