"""Modern identity and precipitation OOF qualification of the 2025 NE-TP record."""

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
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1941, 2001, dtype=int)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
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


def aggregate_crossyear(monthly: np.ndarray) -> np.ndarray:
    source_years = np.arange(1940, 2001, dtype=int)
    if monthly.shape[0] != len(source_years) * 12:
        raise RuntimeError(f"monthly-shape:{monthly.shape}")
    cube = monthly.reshape(len(source_years), 12, monthly.shape[1])
    result = np.full((len(YEARS), monthly.shape[1]), np.nan)
    for row, year in enumerate(YEARS):
        source_row = int(year - source_years[0])
        values = np.concatenate((cube[source_row - 1, 8:12], cube[source_row, 0:8]), axis=0)
        complete = np.all(np.isfinite(values), axis=0)
        result[row, complete] = np.sum(values[:, complete], axis=0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-modern-only", action="store_true")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--proxy6", type=Path, required=True)
    parser.add_argument("--proxy9", type=Path, required=True)
    parser.add_argument("--gpcc", type=Path, required=True)
    parser.add_argument("--cru", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_modern_only:
        raise RuntimeError("explicit-modern-only-flag-required")
    if args.output.exists():
        raise FileExistsError(args.output)

    driver = Path(__file__).resolve().parent
    r118_path = driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py"
    r271_path = driver / "run_round271_xinzhu_ngs_precipitation_modern_oof_v1.py"
    r273_path = driver / "run_round273_hengduan_new_chronology_identity_v1.py"
    contract = driver / "round276_ne_tp_d18o_precipitation_modern_qualification_v1_contract.md"
    paths = {
        "candidate": args.candidate,
        "proxy6": args.proxy6,
        "proxy9": args.proxy9,
        "gpcc": args.gpcc,
        "cru": args.cru,
        "r118_driver": r118_path,
        "r271_driver": r271_path,
        "r273_driver": r273_path,
        "contract": contract,
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["candidate"].get("sha256", "").lower() != args.expected_candidate_sha256.lower():
        raise RuntimeError("candidate-hash-conflict")

    r118 = load_module("round276_r118", r118_path)
    r271 = load_module("round276_r271", r271_path)
    r273 = load_module("round276_r273", r273_path)
    candidate_sheets, parse_audit = r273.parse_modern_chronologies(args.candidate)
    if set(candidate_sheets) != {"Fig. 3"}:
        raise RuntimeError(f"candidate-sheet-inventory:{set(candidate_sheets)}")
    source_years = r273.YEARS
    candidate_all = candidate_sheets["Fig. 3"]
    candidate = candidate_all[(source_years >= YEARS[0]) & (source_years <= YEARS[-1])]
    if not np.isfinite(candidate).all():
        raise RuntimeError("candidate-modern-gap")

    alias_rows = []
    for network, path in (("proxy6", args.proxy6), ("proxy9", args.proxy9)):
        observations = np.asarray(loadmat(path, variable_names=["observations"])["observations"], dtype=float)
        for column in range(observations.shape[1]):
            metric = r273.metrics(candidate_all, observations[:, column])
            alias_rows.append({"network": network, "network_column": column, **metric})
    alias_frame = pd.DataFrame(alias_rows)
    exact_alias_count = int(alias_frame.exact_affine_alias.sum())
    near_alias_count = int(alias_frame.near_alias.sum())

    latitudes = np.repeat(np.arange(35.5, 38.01, 0.5), 6)
    longitudes = np.tile(np.arange(96.5, 99.01, 0.5), 6)
    weights = np.cos(np.deg2rad(latitudes))
    monthly = {
        "GPCC": r118.read_netcdf3_points(args.gpcc, "precip", 1891, latitudes, longitudes),
        "CRU": r118.read_netcdf3_points(args.cru, "pre", 1901, latitudes, longitudes),
    }
    targets = {
        product: np.average(aggregate_crossyear(values), axis=1, weights=weights)
        for product, values in monthly.items()
    }
    if any(not np.isfinite(values).all() for values in targets.values()):
        raise RuntimeError("target-gap")

    fold_rows = []
    summary_rows = []
    predictions = {}
    for product in ("GPCC", "CRU"):
        prediction, baseline, rows = r271.blocked_oof(candidate, targets[product])
        predictions[product] = prediction
        for row in rows:
            fold_rows.append({"product": product, **row})
        metrics = r271.pooled_metrics(targets[product], prediction, baseline)
        early = YEARS <= 1970
        late = YEARS >= 1971
        summary_rows.append({
            "product": product,
            **metrics,
            "positive_fold_correlation_count": int(sum(row["test_correlation"] > 0 for row in rows)),
            "positive_fit_slope_count": int(sum(row["fit_slope"] > 0 for row in rows)),
            "early_raw_correlation_1941_1970": r271.correlation(candidate[early], targets[product][early]),
            "late_raw_correlation_1971_2000": r271.correlation(candidate[late], targets[product][late]),
        })

    fold_frame = pd.DataFrame(fold_rows)
    summary_frame = pd.DataFrame(summary_rows)
    product_correlation = r271.correlation(targets["GPCC"], targets["CRU"])
    gates = {
        "zero_exact_or_near_aliases": exact_alias_count == 0 and near_alias_count == 0,
        "both_oof_correlations_at_least_0p30": bool((summary_frame.oof_correlation >= 0.30).all()),
        "both_oof_skills_positive": bool((summary_frame.oof_skill_vs_training_fold_mean > 0).all()),
        "both_products_at_least_four_positive_fold_correlations": bool((summary_frame.positive_fold_correlation_count >= 4).all()),
        "all_ten_fit_slopes_positive": bool((fold_frame.fit_slope > 0).all()),
        "both_products_early_late_raw_correlations_positive": bool(
            (summary_frame.early_raw_correlation_1941_1970 > 0).all()
            and (summary_frame.late_raw_correlation_1971_2000 > 0).all()
        ),
        "observed_gpcc_cru_correlation_at_least_0p70": product_correlation >= 0.70,
    }
    passes = bool(all(gates.values()))
    decision = (
        "advance_ne_tp_d18o_to_historical_netp_external_validation"
        if passes
        else "close_ne_tp_d18o_as_modern_generalizable_precipitation_evidence"
    )

    args.output.mkdir(parents=True)
    alias_frame.to_csv(args.output / "network_identity_comparisons.csv", index=False)
    fold_frame.to_csv(args.output / "blocked_oof_fold_metrics.csv", index=False)
    summary_frame.to_csv(args.output / "blocked_oof_summary.csv", index=False)
    curves = pd.DataFrame({"year": YEARS, "published_precipitation_mm": candidate})
    for product in ("GPCC", "CRU"):
        curves[f"{product.lower()}_observed_mm"] = targets[product]
        curves[f"{product.lower()}_oof_mm"] = predictions[product]
    curves.to_csv(args.output / "modern_oof_curves.csv", index=False)

    colors = {"GPCC": "#0072B2", "CRU": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), constrained_layout=True, sharex=True)
    axes[0].plot(YEARS, candidate, color="#111111", linewidth=1.5, label="published precipitation reconstruction")
    axes[0].set_ylabel("Precipitation (mm)")
    axes[0].legend(frameon=False)
    for product in ("GPCC", "CRU"):
        row = summary_frame[summary_frame["product"] == product].iloc[0]
        axes[1].plot(YEARS, targets[product], color=colors[product], linewidth=1.4, label=f"{product} observed")
        axes[1].plot(YEARS, predictions[product], color=colors[product], linestyle="--", linewidth=1.1, label=f"{product} OOF r={row.oof_correlation:.3f}")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Sep-Aug precipitation (mm)")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "ne_tp_d18o_precipitation_modern_oof.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_276_ne_tp_d18o_precipitation_modern_qualification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "years": [int(YEARS[0]), int(YEARS[-1])],
        "exact_alias_count": exact_alias_count,
        "near_alias_count": near_alias_count,
        "maximum_network_absolute_correlation": float(alias_frame.absolute_correlation.max()),
        "products": summary_rows,
        "observed_gpcc_cru_correlation": product_correlation,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "interpretation_role": "independent_NETP_temporal_amplitude_evidence_not_new_spatial_support",
        "parse_audit": parse_audit["Fig. 3"],
        "locked_1921_1939_values_accessed": False,
        "production_reconstruction_run": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (args.output / "FINAL_RESULT.md").write_text(
        "# Round 276 NE-TP d18O precipitation modern qualification\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Exact/near aliases: `{exact_alias_count}` / `{near_alias_count}`.\n"
        f"- GPCC OOF r/skill: `{summary_rows[0]['oof_correlation']:.3f}` / `{summary_rows[0]['oof_skill_vs_training_fold_mean']:.3f}`.\n"
        f"- CRU OOF r/skill: `{summary_rows[1]['oof_correlation']:.3f}` / `{summary_rows[1]['oof_skill_vs_training_fold_mean']:.3f}`.\n"
        f"- Passes all gates: `{passes}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

