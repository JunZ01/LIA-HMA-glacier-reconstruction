"""Analyze the Round 131 robust new-proxy innovation sensitivity."""

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


def module(path: Path):
    spec = importlib.util.spec_from_file_location("round131_base", path)
    item = importlib.util.module_from_spec(spec); spec.loader.exec_module(item); return item


def corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def smooth(a):
    return pd.Series(a).rolling(31, center=True, min_periods=15).mean().to_numpy()


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    base = module(driver / "run_round129_new_d18o_seasonal_prior_coupling_v2.py")
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    r130 = loop / "round_130_xinpu_keylong_frozen_enkf_addition_v1"
    output = loop / "round_131_new_proxy_huber_innovation_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver / "round131_new_proxy_huber_innovation_v1_contract.md",
        "runner": driver / "run_round131_new_proxy_huber_innovation_v1.m",
        "analysis": Path(__file__).resolve(),
        "round130_summary": r130 / "summary.json",
        "round130_branches": r130 / "enkf_branches.mat",
        "huber_branches": output / "huber_branches.mat",
        "exact_prior": loop / "round_129_new_d18o_exact_seasonal_prior_coupling_v3/exact_seasonal_prior_fields.mat",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]: raise FileNotFoundError(missing)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: base.fingerprint(path) for name, path in paths.items()}
    with h5py.File(paths["exact_prior"], "r") as f:
        lat = np.asarray(f["lat"], float).ravel(); lon = np.asarray(f["lon"], float).ravel()
    with h5py.File(paths["round130_branches"], "r") as f:
        standard = np.asarray(f["posterior_mean"], float).transpose(2, 1, 0)
        years = np.asarray(f["years"], int).ravel()
    with h5py.File(paths["huber_branches"], "r") as f:
        robust = np.asarray(f["posterior_mean"], float).transpose(2, 1, 0)
        clipped = np.asarray(f["clipped"], bool).transpose(2, 1, 0)
        threshold = float(np.asarray(f["clip_threshold"]).ravel()[0])
    weights = np.tile(np.cos(np.deg2rad(lat)), len(lon))
    baseline = np.average(standard[:, :, 0], axis=0, weights=weights)
    standard_hma = np.average(standard[:, :, [5, 6]], axis=0, weights=weights)
    robust_hma = np.average(robust, axis=0, weights=weights)
    standard_inc = standard_hma - baseline[:, None]
    robust_inc = robust_hma - baseline[:, None]
    baseline_std = float(np.std(baseline, ddof=1))
    active = years >= 1352
    rows = []
    for p, product in enumerate(("BEA_GPCC", "CRU")):
        s31, r31 = smooth(standard_inc[:, p]), smooth(robust_inc[:, p])
        valid = active & np.isfinite(s31) & np.isfinite(r31)
        rows.append({
            "product": product,
            "standard_max_abs_increment_mm_day": float(np.max(np.abs(standard_inc[active, p]))),
            "robust_max_abs_increment_mm_day": float(np.max(np.abs(robust_inc[active, p]))),
            "robust_max_abs_over_baseline_std": float(np.max(np.abs(robust_inc[active, p])) / baseline_std),
            "standard_vs_robust_annual_correlation": corr(standard_inc[active, p], robust_inc[active, p]),
            "standard_vs_robust_31yr_correlation": corr(s31[valid], r31[valid]),
            "standard_vs_robust_31yr_rmse_mm_day": float(np.sqrt(np.mean((s31[valid] - r31[valid]) ** 2))),
        })
    metrics = pd.DataFrame(rows)
    product_corr = corr(robust_inc[active, 0], robust_inc[active, 1])
    product_smooth_corr = corr(smooth(robust_inc[:, 0])[active], smooth(robust_inc[:, 1])[active])
    clip_rows = []
    for j, site in enumerate(("xinpu", "keylong")):
        for p, product in enumerate(("BEA_GPCC", "CRU")):
            years_clipped = years[clipped[j, :, p]]
            clip_rows.append({"site": site, "product": product, "clipped_count": len(years_clipped), "clipped_fraction_of_record": float(len(years_clipped) / (569 if site == "xinpu" else 775)), "first_clipped_year": int(years_clipped.min()) if len(years_clipped) else np.nan, "last_clipped_year": int(years_clipped.max()) if len(years_clipped) else np.nan})
    clipping = pd.DataFrame(clip_rows)
    metrics.to_csv(output / "huber_impact_metrics.csv", index=False)
    clipping.to_csv(output / "huber_clipping_counts.csv", index=False)
    curve_rows = []
    for p, product in enumerate(("BEA_GPCC", "CRU")):
        for year, s, r, s31, r31 in zip(years, standard_inc[:, p], robust_inc[:, p], smooth(standard_inc[:, p]), smooth(robust_inc[:, p])):
            curve_rows.append({"year": int(year), "product": product, "standard_increment": s, "robust_increment": r, "standard_increment_31yr": s31, "robust_increment_31yr": r31})
    pd.DataFrame(curve_rows).to_csv(output / "standard_and_huber_increment_curves.csv", index=False)
    gates = {
        "robust_product_annual_correlation_ge_0_80": product_corr >= 0.80,
        "robust_product_31yr_correlation_ge_0_80": product_smooth_corr >= 0.80,
        "both_standard_vs_robust_31yr_correlation_ge_0_95": bool((metrics.standard_vs_robust_31yr_correlation >= 0.95).all()),
        "both_robust_max_increment_below_half_baseline_std": bool((metrics.robust_max_abs_over_baseline_std <= 0.50).all()),
    }
    passed = bool(all(gates.values()))
    decision = "advance_huber_new_proxy_branch_to_independent_validation" if passed else "close_three_sigma_huber_as_insufficient"
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.0), constrained_layout=True, sharex=True)
    for p, (axis, product) in enumerate(zip(axes, ("BEA+GPCC", "CRU"))):
        axis.plot(years, smooth(standard_inc[:, p]), color="#777777", linewidth=1.2, label="standard Gaussian")
        axis.plot(years, smooth(robust_inc[:, p]), color="#0072B2", linewidth=1.4, label="3-sigma Huber")
        axis.axhline(0, color="#333333", linewidth=0.7)
        axis.set_title(product)
        axis.set_ylabel("HMA increment (mm/day)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False); axes[-1].set_xlabel("year")
    fig.savefig(output / "round131_standard_vs_huber_increment_curves.png", dpi=220); plt.close(fig)
    after = {name: base.fingerprint(path) for name, path in paths.items()}; unchanged = before == after
    if not unchanged: raise RuntimeError("source-changed")
    summary = {
        "experiment_id": "round131_new_proxy_huber_innovation_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "years": [1000, 1920], "locked_1921_1939_accessed": False, "external_validation_reserve_opened": False,
        "clip_threshold_sigma": threshold, "robust_product_annual_increment_correlation": product_corr, "robust_product_31yr_increment_correlation": product_smooth_corr,
        "gates": gates, "passes_all_gates": passed, "decision": decision, "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text("# Round 131 New-Proxy Huber Innovation\n\n" + f"- Robust BEA/CRU annual correlation: `{product_corr:.3f}`.\n- Robust BEA/CRU 31-year correlation: `{product_smooth_corr:.3f}`.\n- Decision: `{decision}`.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
