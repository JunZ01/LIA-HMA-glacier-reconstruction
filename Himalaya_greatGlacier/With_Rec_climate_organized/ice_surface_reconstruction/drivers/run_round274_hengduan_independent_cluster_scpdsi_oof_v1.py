"""Blocked OOF scPDSI test using only nine new Hengduan chronologies."""

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
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1954, 2001, dtype=int)
BLOCKS = ((1954, 1962), (1963, 1971), (1972, 1980), (1981, 1989), (1990, 2000))
CLUSTERS = {
    "north": ("HLS", "JYC", "YZC"),
    "south": ("BLX", "CYS", "FNPTS", "GDCTS", "GDCYS", "LGQ"),
}
BOXES = {
    "north": {"lat": (28.0, 32.0), "lon": (96.0, 103.0)},
    "south": {"lat": (25.0, 28.0), "lon": (96.0, 102.0)},
}


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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def read_hydrological_year_targets(path: Path) -> dict[str, np.ndarray]:
    with Dataset(path, "r") as handle:
        latitude = np.asarray(handle.variables["latitude"][:], dtype=float)
        longitude = np.asarray(handle.variables["longitude"][:], dtype=float)
        source = handle.variables["scpdsi"]
        targets: dict[str, np.ndarray] = {}
        for cluster, box in BOXES.items():
            iy = np.flatnonzero((latitude >= box["lat"][0]) & (latitude <= box["lat"][1]))
            ix = np.flatnonzero((longitude >= box["lon"][0]) & (longitude <= box["lon"][1]))
            if len(iy) == 0 or len(ix) == 0:
                raise RuntimeError(f"empty-box:{cluster}")
            first_month = (1953 - 1901) * 12 + 4
            last_month_exclusive = (2000 - 1901) * 12 + 4
            monthly = np.asarray(source[first_month:last_month_exclusive, iy, ix], dtype=float)
            monthly[(monthly <= -9000) | (monthly >= 1e20)] = np.nan
            weights = np.cos(np.deg2rad(latitude[iy]))[:, None]
            valid = np.isfinite(monthly)
            weighted_sum = np.nansum(monthly * weights[None, :, :], axis=(1, 2))
            weight_sum = np.sum(valid * weights[None, :, :], axis=(1, 2))
            regional_monthly = np.divide(
                weighted_sum,
                weight_sum,
                out=np.full(monthly.shape[0], np.nan),
                where=weight_sum > 0,
            )
            if len(regional_monthly) != len(YEARS) * 12:
                raise RuntimeError(f"hydrological-month-count:{cluster}:{len(regional_monthly)}")
            annual = regional_monthly.reshape(len(YEARS), 12).mean(axis=1)
            if not np.isfinite(annual).all():
                raise RuntimeError(f"target-gap:{cluster}")
            targets[cluster] = annual
    return targets


def fit_fold_composite(proxy: np.ndarray, target: np.ndarray, train: np.ndarray):
    center = np.nanmean(proxy[train], axis=0)
    scale = np.nanstd(proxy[train], axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise RuntimeError("proxy-scale-failed")
    standardized = (proxy - center[None, :]) / scale[None, :]
    correlations = np.asarray([correlation(standardized[train, column], target[train]) for column in range(proxy.shape[1])])
    signs = np.sign(correlations)
    if np.any(~np.isfinite(signs)) or np.any(signs == 0):
        raise RuntimeError("proxy-direction-failed")
    composite = np.nanmean(standardized * signs[None, :], axis=1)
    valid = train & np.isfinite(composite) & np.isfinite(target)
    coefficients = np.linalg.lstsq(
        np.column_stack((np.ones(int(valid.sum())), composite[valid])),
        target[valid],
        rcond=None,
    )[0]
    return composite, coefficients, correlations


def blocked_oof(proxy: np.ndarray, target: np.ndarray):
    prediction = np.full(len(YEARS), np.nan)
    baseline = np.full(len(YEARS), np.nan)
    rows: list[dict[str, object]] = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        composite, coefficients, proxy_correlations = fit_fold_composite(proxy, target, train)
        prediction[test] = coefficients[0] + coefficients[1] * composite[test]
        baseline[test] = np.mean(target[train])
        rows.append({
            "fold": fold,
            "test_period": f"{start}-{end}",
            "test_count": int(test.sum()),
            "fit_slope": float(coefficients[1]),
            "fit_minimum_proxy_correlation": float(np.min(proxy_correlations)),
            "fit_maximum_proxy_correlation": float(np.max(proxy_correlations)),
            "test_correlation": correlation(target[test], prediction[test]),
            "test_rmse_scpdsi": float(np.sqrt(np.mean((target[test] - prediction[test]) ** 2))),
        })
    return prediction, baseline, rows


def pooled(target: np.ndarray, prediction: np.ndarray, baseline: np.ndarray):
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(baseline)
    model_sse = float(np.sum((target[valid] - prediction[valid]) ** 2))
    baseline_sse = float(np.sum((target[valid] - baseline[valid]) ** 2))
    return {
        "count": int(valid.sum()),
        "correlation": correlation(target[valid], prediction[valid]),
        "skill_vs_training_fold_mean": float(1.0 - model_sse / baseline_sse),
        "rmse_scpdsi": float(np.sqrt(model_sse / valid.sum())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-modern-only", action="store_true")
    parser.add_argument("--chronologies", type=Path, required=True)
    parser.add_argument("--expected-chronology-sha256", required=True)
    parser.add_argument("--scpdsi", type=Path, required=True)
    parser.add_argument("--round273-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_modern_only:
        raise RuntimeError("explicit-modern-only-flag-required")
    if args.output.exists():
        raise FileExistsError(args.output)

    driver = Path(__file__).resolve().parent
    r273_path = driver / "run_round273_hengduan_new_chronology_identity_v1.py"
    contract = driver / "round274_hengduan_independent_cluster_scpdsi_oof_v1_contract.md"
    paths = {
        "chronologies": args.chronologies,
        "scpdsi": args.scpdsi,
        "round273_summary": args.round273_summary,
        "round273_driver": r273_path,
        "contract": contract,
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["chronologies"].get("sha256", "").lower() != args.expected_chronology_sha256.lower():
        raise RuntimeError("chronology-hash-conflict")
    prior = json.loads(args.round273_summary.read_text(encoding="utf-8"))
    if prior.get("decision") != "advance_independent_hengduan_clusters_to_modern_scpdsi_oof":
        raise RuntimeError("round273-decision-mismatch")

    r273 = load_module("round274_r273", r273_path)
    parsed, parse_audit = r273.parse_modern_chronologies(args.chronologies)
    source_years = r273.YEARS
    selection = (source_years >= YEARS[0]) & (source_years <= YEARS[-1])
    proxy = {
        cluster: np.column_stack([parsed[name][selection] for name in names])
        for cluster, names in CLUSTERS.items()
    }
    if any(not np.isfinite(values).all() for values in proxy.values()):
        raise RuntimeError("new-chronology-modern-gap")
    targets = read_hydrological_year_targets(args.scpdsi)

    predictions: dict[str, np.ndarray] = {}
    baselines: dict[str, np.ndarray] = {}
    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cluster in CLUSTERS:
        prediction, baseline, rows = blocked_oof(proxy[cluster], targets[cluster])
        predictions[cluster] = prediction
        baselines[cluster] = baseline
        for row in rows:
            fold_rows.append({"cluster": cluster, **row})
        metrics = pooled(targets[cluster], prediction, baseline)
        summary_rows.append({
            "cluster": cluster,
            "proxy_count": proxy[cluster].shape[1],
            **metrics,
            "positive_fold_correlation_count": int(sum(row["test_correlation"] > 0 for row in rows)),
            "positive_fit_slope_count": int(sum(row["fit_slope"] > 0 for row in rows)),
        })

    observed_contrast = targets["north"] - targets["south"]
    predicted_contrast = predictions["north"] - predictions["south"]
    baseline_contrast = baselines["north"] - baselines["south"]
    contrast_metrics = pooled(observed_contrast, predicted_contrast, baseline_contrast)
    fold_frame = pd.DataFrame(fold_rows)
    summary_frame = pd.DataFrame(summary_rows)
    target_correlation = correlation(targets["north"], targets["south"])
    gates = {
        "both_regional_oof_correlations_at_least_0p30": bool((summary_frame.correlation >= 0.30).all()),
        "both_regional_oof_skills_positive": bool((summary_frame.skill_vs_training_fold_mean > 0).all()),
        "both_regions_at_least_four_positive_fold_correlations": bool((summary_frame.positive_fold_correlation_count >= 4).all()),
        "all_ten_fit_slopes_positive": bool((fold_frame.fit_slope > 0).all()),
        "contrast_oof_correlation_at_least_0p20": bool(contrast_metrics["correlation"] >= 0.20),
        "contrast_oof_skill_positive": bool(contrast_metrics["skill_vs_training_fold_mean"] > 0),
        "observed_north_south_correlation_below_0p90": bool(target_correlation < 0.90),
    }
    passes = bool(all(gates.values()))
    decision = (
        "advance_hengduan_independent_dipole_to_second_mode_feasibility"
        if passes
        else "retain_hengduan_as_external_hydroclimate_evidence_not_second_mode_operator"
    )

    args.output.mkdir(parents=True)
    fold_frame.to_csv(args.output / "blocked_oof_fold_metrics.csv", index=False)
    summary_frame.to_csv(args.output / "blocked_oof_summary.csv", index=False)
    curves = pd.DataFrame({"year": YEARS})
    for cluster in CLUSTERS:
        curves[f"{cluster}_observed_scpdsi"] = targets[cluster]
        curves[f"{cluster}_oof_scpdsi"] = predictions[cluster]
    curves["contrast_observed_north_minus_south"] = observed_contrast
    curves["contrast_oof_north_minus_south"] = predicted_contrast
    curves.to_csv(args.output / "modern_oof_curves.csv", index=False)

    colors = {"north": "#0072B2", "south": "#D55E00", "contrast": "#111111"}
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10), constrained_layout=True, sharex=True)
    for axis, cluster in zip(axes[:2], ("north", "south")):
        row = summary_frame[summary_frame.cluster == cluster].iloc[0]
        axis.plot(YEARS, targets[cluster], color=colors[cluster], linewidth=1.5, label=f"{cluster} observed")
        axis.plot(YEARS, predictions[cluster], color=colors[cluster], linestyle="--", linewidth=1.2, label=f"OOF r={row.correlation:.3f}, skill={row.skill_vs_training_fold_mean:.3f}")
        axis.set_ylabel("scPDSI")
        axis.legend(frameon=False)
    axes[2].plot(YEARS, observed_contrast, color="#666666", linewidth=1.4, label="observed north-south")
    axes[2].plot(YEARS, predicted_contrast, color=colors["contrast"], linestyle="--", linewidth=1.2, label=f"OOF r={contrast_metrics['correlation']:.3f}, skill={contrast_metrics['skill_vs_training_fold_mean']:.3f}")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("scPDSI difference")
    axes[2].legend(frameon=False)
    for axis in axes:
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#BBBBBB", linestyle=":", linewidth=0.8)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "hengduan_independent_cluster_scpdsi_oof.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_274_hengduan_independent_cluster_scpdsi_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "years": [int(YEARS[0]), int(YEARS[-1])],
        "target": "previous_May_current_April_mean_scpdsi",
        "regional_metrics": summary_rows,
        "contrast_metrics": contrast_metrics,
        "observed_north_south_target_correlation": target_correlation,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "parse_audit": {name: parse_audit[name] for names in CLUSTERS.values() for name in names},
        "locked_1921_1939_values_accessed": False,
        "published_paleo_reconstruction_read": False,
        "old_itrdb_chronologies_used": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (args.output / "FINAL_RESULT.md").write_text(
        "# Round 274 independent Hengduan cluster scPDSI OOF\n\n"
        f"- Decision: `{decision}`.\n"
        f"- North OOF r/skill: `{summary_rows[0]['correlation']:.3f}` / `{summary_rows[0]['skill_vs_training_fold_mean']:.3f}`.\n"
        f"- South OOF r/skill: `{summary_rows[1]['correlation']:.3f}` / `{summary_rows[1]['skill_vs_training_fold_mean']:.3f}`.\n"
        f"- North-minus-south OOF r/skill: `{contrast_metrics['correlation']:.3f}` / `{contrast_metrics['skill_vs_training_fold_mean']:.3f}`.\n"
        f"- Passes all gates: `{passes}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

