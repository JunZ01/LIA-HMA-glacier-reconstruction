from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PUBLISHED = {
    "correlation": 0.781,
    "adjusted_r2": 0.6022,
    "loo_correlation": 0.761,
    "first_difference_correlation": 0.372,
    "slope": 3.964,
    "intercept": -21.512,
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def fit(proxy: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack((np.ones(len(proxy)), proxy)), target, rcond=None)[0]


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_257_npl065_published_diagnostic_replication_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "round256_summary": loop / "round_256_npl065_modern_signal_free_sensitivity_v1/summary.json",
        "round256_curves": loop / "round_256_npl065_modern_signal_free_sensitivity_v1/signal_free_curves.csv",
        "round256_trace": loop / "round_256_npl065_modern_signal_free_sensitivity_v1/signal_free_convergence_trace.csv",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    summary = json.loads(paths["round256_summary"].read_text(encoding="ascii"))
    if summary.get("decision") != "modern_signal_free_explains_most_published_skill_gap":
        raise RuntimeError("round256-decision-changed")
    curves = pd.read_csv(paths["round256_curves"])
    years = curves.year.to_numpy(int)
    if not np.array_equal(years, np.arange(1960, 2014)):
        raise RuntimeError("year-axis-changed")
    proxy = curves.ssf_positive_end_constrained_final.to_numpy(float)
    target = curves.cru_box_djf_tmin.to_numpy(float)
    beta = fit(proxy, target)
    fitted = beta[0] + beta[1] * proxy
    r = correlation(proxy, target)
    r2 = r ** 2
    n = len(proxy)
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
    loo = np.full(n, np.nan)
    for index in range(n):
        train = np.ones(n, dtype=bool)
        train[index] = False
        fold_beta = fit(proxy[train], target[train])
        loo[index] = fold_beta[0] + fold_beta[1] * proxy[index]
    loo_r = correlation(loo, target)
    first_difference_r = correlation(np.diff(proxy), np.diff(target))
    published_model = PUBLISHED["intercept"] + PUBLISHED["slope"] * proxy
    metrics = {
        "correlation": r,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "loo_correlation": loo_r,
        "first_difference_correlation": first_difference_r,
        "fit_intercept": float(beta[0]),
        "fit_slope": float(beta[1]),
        "fit_rmse_degC": float(np.sqrt(np.mean((target - fitted) ** 2))),
        "published_equation_rmse_degC": float(np.sqrt(np.mean((target - published_model) ** 2))),
    }
    deltas = {
        "correlation": r - PUBLISHED["correlation"],
        "adjusted_r2": adjusted_r2 - PUBLISHED["adjusted_r2"],
        "loo_correlation": loo_r - PUBLISHED["loo_correlation"],
        "first_difference_correlation": first_difference_r - PUBLISHED["first_difference_correlation"],
        "slope": float(beta[1]) - PUBLISHED["slope"],
        "intercept": float(beta[0]) - PUBLISHED["intercept"],
    }
    gates = {
        "correlation_within_0_01": abs(deltas["correlation"]) <= 0.01,
        "adjusted_r2_within_0_02": abs(deltas["adjusted_r2"]) <= 0.02,
        "loo_correlation_within_0_03": abs(deltas["loo_correlation"]) <= 0.03,
        "first_difference_correlation_within_0_05": abs(deltas["first_difference_correlation"]) <= 0.05,
        "slope_within_10_percent": abs(deltas["slope"]) <= 0.10 * abs(PUBLISHED["slope"]),
        "intercept_within_1_degC": abs(deltas["intercept"]) <= 1.0,
    }
    core_pass = all(gates[key] for key in (
        "correlation_within_0_01", "adjusted_r2_within_0_02",
        "loo_correlation_within_0_03", "first_difference_correlation_within_0_05",
    ))
    equation_pass = gates["slope_within_10_percent"] and gates["intercept_within_1_degC"]
    if core_pass and equation_pass:
        decision = "replicate_published_npl065_modern_diagnostics"
    elif core_pass:
        decision = "replicate_published_skill_but_not_absolute_equation_scaling"
    else:
        decision = "partial_published_diagnostic_replication"
    pd.DataFrame({
        "year": years,
        "chronology": proxy,
        "observed_cru_djf_tmin": target,
        "fitted_tmin": fitted,
        "loo_tmin": loo,
        "published_equation_tmin": published_model,
    }).to_csv(output / "published_diagnostic_curves.csv", index=False)
    pd.DataFrame([
        {"metric": key, "replicated": metrics[key], "published": PUBLISHED.get(key), "delta": deltas.get(key)}
        for key in ("correlation", "adjusted_r2", "loo_correlation", "first_difference_correlation")
    ] + [
        {"metric": "slope", "replicated": metrics["fit_slope"], "published": PUBLISHED["slope"], "delta": deltas["slope"]},
        {"metric": "intercept", "replicated": metrics["fit_intercept"], "published": PUBLISHED["intercept"], "delta": deltas["intercept"]},
    ]).to_csv(output / "published_diagnostic_comparison.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True, constrained_layout=True)
    axes[0].plot(years, target, color="#006D77", linewidth=1.6, label="CRU DJF Tmin")
    axes[0].plot(years, fitted, color="#D55E00", linewidth=1.4, label=f"In-sample fit, r={r:.3f}")
    axes[0].plot(years, loo, color="#333333", linewidth=1.15, linestyle="--", label=f"LOO, r={loo_r:.3f}")
    axes[0].set_ylabel("degC")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_title("NPL065 published-diagnostic replication")
    axes[1].plot(years[1:], np.diff(target), color="#006D77", linewidth=1.3, label="Observed first difference")
    scaled_diff = np.diff(fitted)
    axes[1].plot(years[1:], scaled_diff, color="#D55E00", linewidth=1.2, label=f"Chronology first difference, r={first_difference_r:.3f}")
    axes[1].set_ylabel("degC / year")
    axes[1].set_xlabel("Ring year")
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "npl065_published_diagnostic_replication.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "published": PUBLISHED,
        "replicated": metrics,
        "replicated_minus_published": deltas,
        "gates": gates,
        "core_skill_pass": core_pass,
        "equation_scaling_pass": equation_pass,
        "decision": decision,
        "locked_1921_1939_values_accessed": False,
        "paleo_proxy_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "rcsigfree_binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 257 NPL065 published-diagnostic replication\n\n"
        f"- Correlation: replicated `{r:.3f}`, published `{PUBLISHED['correlation']:.3f}`.\n"
        f"- Adjusted R2: replicated `{adjusted_r2:.4f}`, published `{PUBLISHED['adjusted_r2']:.4f}`.\n"
        f"- LOO r: replicated `{loo_r:.3f}`, published `{PUBLISHED['loo_correlation']:.3f}`.\n"
        f"- First-difference r: replicated `{first_difference_r:.3f}`, published `{PUBLISHED['first_difference_correlation']:.3f}`.\n"
        f"- Equation: replicated `T = {beta[1]:.3f}*RWI {beta[0]:+.3f}`; published `T = 3.964*RWI - 21.512`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
