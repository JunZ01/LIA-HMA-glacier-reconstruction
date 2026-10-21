"""Round 133: independent low-frequency validation of new precipitation branches."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BRANCHES = ("baseline", "both_bea", "both_cru", "product_marginalized")
PERIODS = ((1500, 1920), (1800, 1920))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": sha256(path)}


def rolling(values: np.ndarray, window: int, minimum: int | None = None) -> np.ndarray:
    minimum = window if minimum is None else minimum
    return pd.Series(values).rolling(window, center=True, min_periods=minimum).mean().to_numpy()


def standardize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    center = float(np.nanmean(values[mask]))
    scale = float(np.nanstd(values[mask], ddof=1))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("standardization-scale-failed")
    result[np.isfinite(values)] = (values[np.isfinite(values)] - center) / scale
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def slope_per_century(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(years) & np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def load_state(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(paths["grid"], "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()
    with h5py.File(paths["round130"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        posterior = np.asarray(handle["posterior_mean"], dtype=float).transpose(2, 1, 0)
        closure = float(np.asarray(handle["baseline_max_abs_closure"]).ravel()[0])
    with h5py.File(paths["round132"], "r") as handle:
        marginalized = np.asarray(handle["posterior_mean"], dtype=float).T
        years132 = np.asarray(handle["years"], dtype=int).ravel()
    if posterior.shape != (336, 921, 7) or marginalized.shape != (336, 921):
        raise RuntimeError(f"unexpected-state-shape:{posterior.shape}:{marginalized.shape}")
    if not np.array_equal(years, np.arange(1000, 1921)) or not np.array_equal(years, years132):
        raise RuntimeError("year-axis-failed")
    if closure > 0.05:
        raise RuntimeError(f"baseline-closure-gate-failed:{closure}")
    fields = {
        "baseline": posterior[:, :, 0],
        "both_bea": posterior[:, :, 5],
        "both_cru": posterior[:, :, 6],
        "product_marginalized": marginalized,
    }
    return years, lat, lon, fields


def area_curve(field: np.ndarray, state_lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.average(field[mask, :], axis=0, weights=np.cos(np.deg2rad(state_lat[mask])))


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_133_independent_precipitation_low_frequency_validation_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver / "round133_independent_precipitation_low_frequency_validation_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "grid": loop / "round_129_new_d18o_exact_seasonal_prior_coupling_v3/exact_seasonal_prior_fields.mat",
        "round130": loop / "round_130_xinpu_keylong_frozen_enkf_addition_v1/enkf_branches.mat",
        "round132": loop / "round_132_product_marginalized_new_proxy_operator_v1/marginalized_branch.mat",
        "external": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_external_low_frequency_benchmark_20260818_v2/external_benchmark_allowed_values.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    years, lat, lon, fields = load_state(paths)
    state_lat = np.tile(lat, len(lon))
    state_lon = np.repeat(lon, len(lat))
    netp = (state_lat >= 35) & (state_lat <= 42) & (state_lon >= 92) & (state_lon <= 108)
    distance2 = (state_lat - 28.93) ** 2 + ((state_lon - 103.10) * np.cos(np.deg2rad(28.93))) ** 2
    shenqi_index = int(np.argmin(distance2))
    if int(netp.sum()) != 32 or (state_lat[shenqi_index], state_lon[shenqi_index]) != (29.0, 103.0):
        raise RuntimeError("spatial-mask-preflight-failed")

    curves: dict[str, dict[str, np.ndarray]] = {"netp": {}, "shenqi": {}}
    for branch, field in fields.items():
        curves["netp"][branch] = area_curve(field, state_lat, netp)
        curves["shenqi"][branch] = field[shenqi_index, :]

    external = pd.read_csv(paths["external"])
    if ((external["year"] >= 1921) & (external["year"] <= 1939)).any():
        raise RuntimeError("locked-years-present-in-external-input")
    qt = external.loc[external["benchmark"] == "northwestern_qtp_mjj_precipitation", ["year", "value"]].copy()
    sq = external.loc[external["benchmark"] == "shenqi_cave_precipitation_index", ["year", "value"]].copy()
    qt = qt.loc[(qt.year >= 1500) & (qt.year <= 1920)].sort_values("year")
    sq = sq.loc[(sq.year >= 1500) & (sq.year <= 1920)].sort_values("year")
    if not np.array_equal(qt.year.to_numpy(dtype=int), np.arange(1500, 1921)):
        raise RuntimeError("qt-annual-axis-failed")
    if not np.allclose(sq.year, np.rint(sq.year)):
        raise RuntimeError("shenqi-noninteger-date")

    qt_values = qt.value.to_numpy(dtype=float)
    qt_smooth = rolling(qt_values, 31, 31)
    sq_years = sq.year.to_numpy(dtype=int)
    sq_values = sq.value.to_numpy(dtype=float)
    sq_smooth = rolling(sq_values, 7, 5)
    model_year_index = {int(year): i for i, year in enumerate(years)}
    sq_indices = np.asarray([model_year_index[int(year)] for year in sq_years], dtype=int)

    metric_rows: list[dict[str, object]] = []
    plotted: dict[str, dict[str, np.ndarray]] = {"netp": {}, "shenqi": {}}
    for branch in BRANCHES:
        netp_smooth = rolling(curves["netp"][branch], 31, 31)
        sq_model_smooth = rolling(curves["shenqi"][branch], 31, 31)[sq_indices]
        plotted["netp"][branch] = netp_smooth
        plotted["shenqi"][branch] = sq_model_smooth
        for benchmark, obs_years, obs_values, model_values in (
            ("qt_prec", qt.year.to_numpy(dtype=int), qt_smooth, netp_smooth[years >= 1500]),
            ("shenqi", sq_years, sq_smooth, sq_model_smooth),
        ):
            for start, end in PERIODS:
                use = (obs_years >= start) & (obs_years <= end) & np.isfinite(obs_values) & np.isfinite(model_values)
                obs_z = standardize(obs_values, use)
                model_z = standardize(model_values, use)
                obs_trend = slope_per_century(obs_years[use], obs_z[use])
                model_trend = slope_per_century(obs_years[use], model_z[use])
                metric_rows.append({
                    "benchmark": benchmark,
                    "period": f"{start}_{end}",
                    "branch": branch,
                    "sample_count": int(use.sum()),
                    "low_frequency_correlation": correlation(obs_z[use], model_z[use]),
                    "external_standardized_trend_per_century": obs_trend,
                    "model_standardized_trend_per_century": model_trend,
                    "trend_sign_agreement": bool(np.sign(obs_trend) == np.sign(model_trend)),
                    "absolute_standardized_trend_error": abs(model_trend - obs_trend),
                })
    metrics = pd.DataFrame(metric_rows)

    baseline = metrics.loc[metrics.branch == "baseline"].set_index(["benchmark", "period"])
    candidate_rows = []
    for branch in BRANCHES[1:]:
        candidate = metrics.loc[metrics.branch == branch].set_index(["benchmark", "period"])
        delta = candidate.low_frequency_correlation - baseline.low_frequency_correlation
        qt_delta = delta.loc["qt_prec"]
        sq_delta = delta.loc["shenqi"]
        baseline_correct = baseline.trend_sign_agreement.astype(bool)
        candidate_correct = candidate.trend_sign_agreement.astype(bool)
        lost_correct_sign = bool((baseline_correct & ~candidate_correct).any())
        gate = {
            "qt_mean_correlation_improvement_ge_0_05": bool(qt_delta.mean() >= 0.05),
            "qt_no_period_degradation_gt_0_02": bool(qt_delta.min() >= -0.02),
            "shenqi_mean_correlation_degradation_le_0_05": bool(sq_delta.mean() >= -0.05),
            "no_loss_of_baseline_correct_trend_sign": not lost_correct_sign,
        }
        candidate_rows.append({
            "branch": branch,
            "qt_mean_correlation_delta": float(qt_delta.mean()),
            "qt_min_correlation_delta": float(qt_delta.min()),
            "shenqi_mean_correlation_delta": float(sq_delta.mean()),
            "lost_baseline_correct_trend_sign": lost_correct_sign,
            **gate,
            "passes_all_gates": bool(all(gate.values())),
        })
    decisions = pd.DataFrame(candidate_rows)
    supported = decisions.loc[decisions.passes_all_gates, "branch"].tolist()
    decision = "retain_supported_low_frequency_candidate" if supported else "close_current_new_proxy_precipitation_assimilation_route"

    metric_lookup = metrics.set_index(["benchmark", "period", "branch"])
    colors = {"baseline": "#111111", "both_bea": "#0072B2", "both_cru": "#D55E00", "product_marginalized": "#009E73"}
    labels = {"baseline": "baseline", "both_bea": "Xinpu+Keylong BEA", "both_cru": "Xinpu+Keylong CRU", "product_marginalized": "product-marginalized"}
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), constrained_layout=True)
    qt_plot_use = (qt.year >= 1500) & (qt.year <= 1920)
    qt_z = standardize(qt_smooth, qt_plot_use.to_numpy())
    axes[0].plot(qt.year, qt_z, color="#CC79A7", linewidth=2.0, label="QT-Prec MJJ external")
    period_model = (years >= 1500) & (years <= 1920)
    for branch in BRANCHES:
        values = plotted["netp"][branch]
        z = standardize(values, period_model)
        corr_value = metric_lookup.loc[("qt_prec", "1500_1920", branch), "low_frequency_correlation"]
        axes[0].plot(years[period_model], z[period_model], color=colors[branch], linewidth=1.25, alpha=0.95, label=f"{labels[branch]} (r={corr_value:.3f})")
    axes[0].set(title="NETP precipitation, 31-year means", ylabel="standardized anomaly")

    sq_use = (sq_years >= 1500) & (sq_years <= 1920) & np.isfinite(sq_smooth)
    sq_z = standardize(sq_smooth, sq_use)
    axes[1].plot(sq_years[sq_use], sq_z[sq_use], color="#CC79A7", linewidth=1.8, marker="o", markersize=2.4, label="Shenqi PI external (~35-year)")
    for branch in BRANCHES:
        values = plotted["shenqi"][branch]
        z = standardize(values, sq_use)
        corr_value = metric_lookup.loc[("shenqi", "1500_1920", branch), "low_frequency_correlation"]
        axes[1].plot(sq_years[sq_use], z[sq_use], color=colors[branch], linewidth=1.2, alpha=0.95, label=f"{labels[branch]} (r={corr_value:.3f})")
    axes[1].set(title="Shenqi Cave nearest-grid precipitation", xlabel="year", ylabel="standardized anomaly")
    for axis in axes:
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output / "round133_external_low_frequency_branch_curves.png", dpi=220)
    plt.close(fig)

    curve_rows = []
    for branch in BRANCHES:
        for year, value in zip(years, plotted["netp"][branch]):
            if 1500 <= year <= 1920:
                curve_rows.append({"benchmark": "qt_prec", "branch": branch, "year": int(year), "model_low_frequency": value})
        for year, value in zip(sq_years, plotted["shenqi"][branch]):
            if 1500 <= year <= 1920:
                curve_rows.append({"benchmark": "shenqi", "branch": branch, "year": int(year), "model_low_frequency": value})
    pd.DataFrame(curve_rows).to_csv(output / "branch_low_frequency_curves.csv", index=False)
    metrics.to_csv(output / "external_validation_metrics.csv", index=False)
    decisions.to_csv(output / "candidate_decision_gates.csv", index=False)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_133_independent_precipitation_low_frequency_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "years": [1500, 1920],
        "locked_1921_1939_accessed": False,
        "external_validation_reserve_opened": False,
        "oggm_run": False,
        "netp_box": {"lat": [35, 42], "lon": [92, 108], "cell_count": int(netp.sum())},
        "shenqi_site": {"published_lat": 28.93, "published_lon": 103.10, "grid_lat": float(state_lat[shenqi_index]), "grid_lon": float(state_lon[shenqi_index])},
        "supported_candidates": supported,
        "decision": decision,
        "source_fingerprints_unchanged": unchanged,
        "candidate_gates": candidate_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    best = decisions.sort_values("qt_mean_correlation_delta", ascending=False).iloc[0]
    (output / "FINAL_RESULT.md").write_text(
        "# Round 133 Independent Precipitation Low-Frequency Validation\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Supported candidates: `{supported}`.\n"
        f"- Largest mean QT-Prec correlation change: `{best['branch']}` = `{best['qt_mean_correlation_delta']:+.3f}`.\n"
        "- Spatial correction: QT-Prec uses 35-42 N, 92-108 E; Shenqi uses the nearest 29 N, 103 E state cell.\n"
        "- This test compares standardized low-frequency phase, not raw annual-versus-seasonal precipitation amplitude.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
