"""Analyze Round 130 frozen EnKF addition branches."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BRANCHES = ["baseline", "xinpu_bea", "xinpu_cru", "keylong_bea", "keylong_cru", "both_bea", "both_cru"]
PAIRS = {"xinpu": (1, 2), "keylong": (3, 4), "both": (5, 6)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rolling(values: np.ndarray, window: int = 31) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=window // 2).mean().to_numpy()


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1]) if int(valid.sum()) >= 3 else float("nan")


def trend(years: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    use = (years >= start) & (years <= end) & np.isfinite(values)
    return float(np.polyfit(years[use], values[use], 1)[0] * 100) if int(use.sum()) >= 10 else float("nan")


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    base = load_module("round130_analysis_base", driver / "run_round129_new_d18o_seasonal_prior_coupling_v2.py")
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_130_xinpu_keylong_frozen_enkf_addition_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver / "round130_xinpu_keylong_frozen_enkf_addition_v1_contract.md",
        "preparation_driver": driver / "prepare_round130_xinpu_keylong_frozen_enkf_addition_v1.py",
        "matlab_runner": driver / "run_round130_xinpu_keylong_frozen_enkf_addition_v3.m",
        "analysis_driver": Path(__file__).resolve(),
        "preparation_summary": output / "preparation_summary.json",
        "input": output / "round130_input.mat",
        "branches": output / "enkf_branches.mat",
        "exact_prior": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_129_new_d18o_exact_seasonal_prior_coupling_v3/exact_seasonal_prior_fields.mat",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing:" + ";".join(missing))
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: base.fingerprint(path) for name, path in paths.items()}
    prep = json.loads(paths["preparation_summary"].read_text(encoding="utf-8"))
    if prep["locked_1921_1939_accessed"] or prep["locked_1921_1939_values_decoded"]:
        raise RuntimeError("locked-boundary-failed")

    with h5py.File(paths["exact_prior"], "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()
    with h5py.File(paths["branches"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        posterior_mean = np.asarray(handle["posterior_mean"], dtype=float).transpose(2, 1, 0)
        posterior_sigma = np.asarray(handle["posterior_sigma"], dtype=float).transpose(2, 1, 0)
        active_new = np.asarray(handle["active_new_count"], dtype=float).T
        innovation_z = np.asarray(handle["innovation_z"], dtype=float).transpose(2, 1, 0)
        baseline_closure = float(np.asarray(handle["baseline_max_abs_closure"]).ravel()[0])
    if posterior_mean.shape != (336, 921, 7) or not np.array_equal(years, np.arange(1000, 1921)):
        raise RuntimeError(f"output-shape-or-year-failed:{posterior_mean.shape}")
    if baseline_closure > 0.05:
        raise RuntimeError(f"baseline-approved-gate-failed:{baseline_closure}")

    weights = np.tile(np.cos(np.deg2rad(lat)), len(lon))
    hma_mean = np.average(posterior_mean, axis=0, weights=weights)
    hma_sigma = np.average(posterior_sigma, axis=0, weights=weights)
    increments = hma_mean - hma_mean[:, [0]]
    baseline_std = float(np.std(hma_mean[:, 0], ddof=1))
    curve_rows = []
    for branch_index, branch in enumerate(BRANCHES):
        smooth_mean = rolling(hma_mean[:, branch_index])
        smooth_increment = rolling(increments[:, branch_index])
        for year, value, smooth, increment, smooth_inc, sigma, active in zip(
            years, hma_mean[:, branch_index], smooth_mean, increments[:, branch_index], smooth_increment,
            hma_sigma[:, branch_index], active_new[branch_index],
        ):
            curve_rows.append({"year": int(year), "branch": branch, "hma_mean_mm_day": value, "hma_mean_31yr": smooth, "increment_mm_day": increment, "increment_31yr": smooth_inc, "hma_sigma_mm_day": sigma, "active_new_proxy_count": int(active)})
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(output / "hma_branch_curves.csv", index=False)

    metric_rows = []
    for family, (bea_index, cru_index) in PAIRS.items():
        active = (active_new[bea_index] > 0) & (active_new[cru_index] > 0)
        bea = increments[:, bea_index]
        cru = increments[:, cru_index]
        bea_smooth, cru_smooth = rolling(bea), rolling(cru)
        smooth_valid = active & np.isfinite(bea_smooth) & np.isfinite(cru_smooth)
        sign_valid = smooth_valid & ((np.abs(bea_smooth) > 1e-12) | (np.abs(cru_smooth) > 1e-12))
        metric_rows.append({
            "family": family,
            "active_year_count": int(active.sum()),
            "annual_increment_product_correlation": correlation(bea[active], cru[active]),
            "smoothed_increment_product_correlation": correlation(bea_smooth[smooth_valid], cru_smooth[smooth_valid]),
            "smoothed_sign_agreement_fraction": float(np.mean(np.sign(bea_smooth[sign_valid]) == np.sign(cru_smooth[sign_valid]))),
            "bea_median_abs_increment_mm_day": float(np.median(np.abs(bea[active]))),
            "cru_median_abs_increment_mm_day": float(np.median(np.abs(cru[active]))),
            "bea_max_abs_increment_mm_day": float(np.max(np.abs(bea[active]))),
            "cru_max_abs_increment_mm_day": float(np.max(np.abs(cru[active]))),
            "bea_max_abs_increment_over_baseline_std": float(np.max(np.abs(bea[active])) / baseline_std),
            "cru_max_abs_increment_over_baseline_std": float(np.max(np.abs(cru[active])) / baseline_std),
            "bea_1800_1900_increment_trend_mm_day_per_century": trend(years, bea, 1800, 1900),
            "cru_1800_1900_increment_trend_mm_day_per_century": trend(years, cru, 1800, 1900),
            "bea_1901_1920_increment_trend_mm_day_per_century": trend(years, bea, 1901, 1920),
            "cru_1901_1920_increment_trend_mm_day_per_century": trend(years, cru, 1901, 1920),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "branch_robustness_metrics.csv", index=False)

    innovation_rows = []
    for proxy_index, site in enumerate(("xinpu", "keylong")):
        for product_index, product in enumerate(("BEA_GPCC", "CRU")):
            values = innovation_z[proxy_index, :, product_index]
            values = values[np.isfinite(values)]
            innovation_rows.append({"site": site, "product": product, "count": len(values), "median_abs_z": float(np.median(np.abs(values))), "p95_abs_z": float(np.percentile(np.abs(values), 95)), "maximum_abs_z": float(np.max(np.abs(values))), "fraction_abs_z_gt_3": float(np.mean(np.abs(values) > 3))})
    innovations = pd.DataFrame(innovation_rows)
    innovations.to_csv(output / "new_proxy_innovation_diagnostics.csv", index=False)

    nonadditivity_rows = []
    for product, xi, ki, bi in (("BEA_GPCC", 1, 3, 5), ("CRU", 2, 4, 6)):
        active = active_new[bi] == 2
        residual = increments[:, bi] - increments[:, xi] - increments[:, ki]
        nonadditivity_rows.append({"product": product, "both_active_year_count": int(active.sum()), "rmse_mm_day": float(np.sqrt(np.mean(residual[active] ** 2))), "maximum_abs_mm_day": float(np.max(np.abs(residual[active]))), "rmse_over_both_increment_std": float(np.sqrt(np.mean(residual[active] ** 2)) / np.std(increments[active, bi], ddof=1))})
    nonadditivity = pd.DataFrame(nonadditivity_rows)
    nonadditivity.to_csv(output / "two_proxy_nonadditivity.csv", index=False)

    both = metrics[metrics.family == "both"].iloc[0]
    innovation_gate = bool((innovations.median_abs_z <= 2).all() and (innovations.p95_abs_z <= 4).all())
    robustness_gates = {
        "annual_increment_product_correlation_ge_0_80": bool(both.annual_increment_product_correlation >= 0.80),
        "smoothed_increment_product_correlation_ge_0_80": bool(both.smoothed_increment_product_correlation >= 0.80),
        "smoothed_sign_agreement_ge_0_75": bool(both.smoothed_sign_agreement_fraction >= 0.75),
        "new_proxy_innovations_bounded": innovation_gate,
        "both_products_max_increment_below_half_baseline_std": bool(max(both.bea_max_abs_increment_over_baseline_std, both.cru_max_abs_increment_over_baseline_std) <= 0.50),
    }
    passes = bool(all(robustness_gates.values()))
    decision = "advance_as_candidate_reconstruction_branch_for_independent_validation" if passes else "do_not_advance_combined_new_proxy_branch"

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), constrained_layout=True, sharex=True)
    axes[0, 0].plot(years, rolling(hma_mean[:, 0]), color="#222222", linewidth=1.5, label="baseline")
    axes[0, 0].plot(years, rolling(hma_mean[:, 5]), color="#0072B2", linewidth=1.2, label="both BEA+GPCC")
    axes[0, 0].plot(years, rolling(hma_mean[:, 6]), color="#D55E00", linewidth=1.2, label="both CRU")
    axes[0, 0].set_title("HMA annual precipitation posterior (31-year mean)")
    axes[0, 0].set_ylabel("mm/day")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis, (family, (bea_index, cru_index)) in zip(axes.flat[1:], PAIRS.items()):
        axis.plot(years, rolling(increments[:, bea_index]), color="#0072B2", linewidth=1.2, label="BEA+GPCC")
        axis.plot(years, rolling(increments[:, cru_index]), color="#D55E00", linewidth=1.2, label="CRU")
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set_title(f"{family} increment (31-year mean)")
        axis.set_ylabel("mm/day")
    for axis in axes.flat:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("year")
    figure.savefig(output / "round130_hma_increment_curves.png", dpi=220)
    plt.close(figure)

    after = {name: base.fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed-during-round130-analysis")
    summary = {
        "experiment_id": "round130_xinpu_keylong_frozen_enkf_addition_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "years": [1000, 1920],
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "baseline_max_abs_closure_mm_day": baseline_closure,
        "baseline_approved_gate_mm_day": 0.05,
        "baseline_hma_standard_deviation_mm_day": baseline_std,
        "robustness_gates": robustness_gates,
        "passes_robustness_gates": passes,
        "decision": decision,
        "source_fingerprints_unchanged": unchanged,
        "interpretation_boundary": "This establishes influence and BEA/CRU robustness only. Accuracy must be tested with independent validation evidence before production promotion.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 130 Xinpu-Keylong Frozen EnKF Addition\n\n"
        f"- Baseline closure: `{baseline_closure:.8f} mm/day` (approved gate `0.05`).\n"
        f"- Combined annual increment product correlation: `{both.annual_increment_product_correlation:.3f}`.\n"
        f"- Combined 31-year increment product correlation: `{both.smoothed_increment_product_correlation:.3f}`.\n"
        f"- Combined 31-year sign agreement: `{both.smoothed_sign_agreement_fraction:.1%}`.\n"
        f"- Innovation gate passed: `{str(innovation_gate).lower()}`.\n"
        f"- Decision: `{decision}`.\n\n"
        "This is an influence and robustness result, not an accuracy validation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
