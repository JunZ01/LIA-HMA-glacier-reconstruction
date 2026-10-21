"""Diagnose weak versus temporally unstable documented-temperature PSMs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def season_code(value: object) -> str:
    candidates: list[str] = []
    for part in str(value).strip().split("||"):
        normalized = part.strip().replace("[", "").replace("]", "").replace(",", " ")
        if normalized == "Summer":
            candidates.append("JJA")
            continue
        numbers = [item for item in normalized.split() if item.isdigit()]
        if numbers == ["6", "7", "8"]:
            candidates.append("JJA")
        elif numbers == ["4", "5", "6"]:
            candidates.append("AMJ")
        elif numbers == ["4"]:
            candidates.append("APR")
        elif numbers == [str(month) for month in range(1, 13)]:
            candidates.append("ANN")
        else:
            raise RuntimeError(f"unsupported-documented-season:{value}")
    if len(set(candidates)) != 1:
        raise RuntimeError(f"conflicting-documented-season:{value}")
    return candidates[0]


def fit_stats(response: np.ndarray, predictor: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(response) & np.isfinite(predictor)
    if valid.sum() < 10:
        return {key: np.nan for key in ("n", "slope", "correlation", "rmse_intercept", "rmse_model", "improvement")}
    y = response[valid]
    x = predictor[valid]
    design = np.column_stack([np.ones(valid.sum()), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = intercept + slope * x
    rmse_intercept = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    rmse_model = float(np.sqrt(np.mean((y - prediction) ** 2)))
    correlation = float(np.corrcoef(y, x)[0, 1]) if np.std(y) > 0 and np.std(x) > 0 else np.nan
    return {
        "n": int(valid.sum()),
        "slope": float(slope),
        "correlation": correlation,
        "rmse_intercept": rmse_intercept,
        "rmse_model": rmse_model,
        "improvement": float(1.0 - rmse_model / rmse_intercept) if rmse_intercept > 0 else np.nan,
    }


def blocked_stats(response: np.ndarray, predictor: np.ndarray, years: np.ndarray) -> dict[str, object]:
    model_prediction = np.full(response.shape, np.nan)
    intercept_prediction = np.full(response.shape, np.nan)
    slopes = np.full(len(BLOCKS), np.nan)
    for fold, (start, end) in enumerate(BLOCKS):
        test = (years >= start) & (years <= end)
        train = ~test
        valid_train = train & np.isfinite(response) & np.isfinite(predictor)
        valid_test = test & np.isfinite(response) & np.isfinite(predictor)
        if valid_train.sum() < 20 or valid_test.sum() < 3:
            continue
        coefficient = np.linalg.lstsq(
            np.column_stack([np.ones(valid_train.sum()), predictor[valid_train]]),
            response[valid_train],
            rcond=None,
        )[0]
        slopes[fold] = coefficient[1]
        model_prediction[valid_test] = coefficient[0] + coefficient[1] * predictor[valid_test]
        intercept_prediction[valid_test] = np.mean(response[valid_train])
    valid = np.isfinite(response) & np.isfinite(model_prediction) & np.isfinite(intercept_prediction)
    if valid.sum() < 10:
        return {"n": int(valid.sum()), "improvement": np.nan, "correlation": np.nan, "slopes": slopes}
    rmse_model = float(np.sqrt(np.mean((response[valid] - model_prediction[valid]) ** 2)))
    rmse_intercept = float(np.sqrt(np.mean((response[valid] - intercept_prediction[valid]) ** 2)))
    correlation = (
        float(np.corrcoef(response[valid], model_prediction[valid])[0, 1])
        if np.std(response[valid]) > 0 and np.std(model_prediction[valid]) > 0 else np.nan
    )
    return {
        "n": int(valid.sum()),
        "improvement": float(1.0 - rmse_model / rmse_intercept) if rmse_intercept > 0 else np.nan,
        "correlation": correlation,
        "slopes": slopes,
    }


def sign_agrees(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second) or first == 0 or second == 0:
        return np.nan
    return float(np.sign(first) == np.sign(second))


def documented_sign(value: object) -> float:
    parts = {part.strip().lower() for part in str(value).split("||")}
    if parts == {"positive"}:
        return 1.0
    if parts == {"negative"}:
        return -1.0
    return np.nan


def bootstrap_median(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    result = np.empty(draws)
    for draw in range(draws):
        result[draw] = np.median(values[rng.integers(0, len(values), size=len(values))])
    return float(np.percentile(result, 2.5)), float(np.percentile(result, 97.5))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / "round_09_temperature_proxy_association_stability_v1"
    paths = {
        "inventory": base / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "season_export": base / "round_08_documented_temperature_seasonality_v1/temperature_seasons_v1.mat",
        "round08_summary": base / "round_08_documented_temperature_seasonality_v1/summary.json",
        "proxy9": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "analysis_driver": drivers / Path(__file__).name,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    inventory = pd.read_csv(paths["inventory"])
    records = (
        inventory[inventory["network_role"] == "documented_temperature"]
        .sort_values("record_index_1based")
        .drop_duplicates("identity_key")
    )
    if len(records) != 214:
        raise RuntimeError("temperature-identity-count-failed")
    observations = np.asarray(scipy.io.loadmat(paths["proxy9"], variable_names=["observations"])["observations"], dtype=float)
    with h5py.File(paths["season_export"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        seasonal = {
            "ANN": np.asarray(handle["annual"], dtype=float).T,
            "JJA": np.asarray(handle["jja"], dtype=float).T,
            "AMJ": np.asarray(handle["amj"], dtype=float).T,
            "APR": np.asarray(handle["april"], dtype=float).T,
        }
    if not np.array_equal(years, np.arange(1940, 2001)) or observations.shape != (61, 297):
        raise RuntimeError("array-contract-failed")

    early = (years >= 1940) & (years <= 1969)
    late = (years >= 1970) & (years <= 2000)
    rows: list[dict[str, object]] = []
    for record in records.itertuples(index=False):
        index = int(record.record_index_1based) - 1
        code = season_code(record.documented_seasonality)
        response = observations[:, index]
        predictor = seasonal[code][:, index]
        full = fit_stats(response, predictor, np.ones(years.shape, dtype=bool))
        first = fit_stats(response, predictor, early)
        second = fit_stats(response, predictor, late)
        blocked = blocked_stats(response, predictor, years)
        archived_b1 = float(record.psm_b1_temperature)
        direction = documented_sign(record.documented_direction)
        rows.append({
            "record_index_1based": int(record.record_index_1based),
            "identity_key": record.identity_key,
            "season_code": code,
            "archive_type": record.archive_type,
            "measurement": record.measurement,
            "full_n": full["n"],
            "full_slope": full["slope"],
            "full_correlation": full["correlation"],
            "full_abs_correlation": abs(full["correlation"]),
            "full_insample_rmse_improvement": full["improvement"],
            "blocked_n": blocked["n"],
            "blocked_rmse_improvement": blocked["improvement"],
            "blocked_prediction_correlation": blocked["correlation"],
            "early_slope": first["slope"],
            "late_slope": second["slope"],
            "early_late_slope_sign_agreement": sign_agrees(first["slope"], second["slope"]),
            "archived_b1": archived_b1,
            "full_archived_slope_sign_agreement": sign_agrees(full["slope"], archived_b1),
            "documented_direction_sign": direction,
            "full_documented_direction_agreement": sign_agrees(full["slope"], direction),
            "finite_fold_slope_count": int(np.isfinite(blocked["slopes"]).sum()),
        })
    write_csv(output / "record_temperature_relationship_diagnostic.csv", rows)
    frame = pd.DataFrame(rows)

    group_rows: list[dict[str, object]] = []
    for dimension in ("season_code", "archive_type", "measurement"):
        for value, group in frame.groupby(dimension, dropna=False):
            if len(group) < 10:
                continue
            group_rows.append({
                "group_dimension": dimension,
                "group_value": str(value),
                "identity_count": int(len(group)),
                "median_abs_full_correlation": float(group["full_abs_correlation"].median()),
                "median_insample_rmse_improvement": float(group["full_insample_rmse_improvement"].median()),
                "median_blocked_rmse_improvement": float(group["blocked_rmse_improvement"].median()),
                "early_late_slope_sign_agreement_fraction": float(group["early_late_slope_sign_agreement"].mean()),
                "archived_slope_sign_agreement_fraction": float(group["full_archived_slope_sign_agreement"].mean()),
                "documented_direction_agreement_fraction": float(group["full_documented_direction_agreement"].mean()),
            })
    write_csv(output / "group_temperature_relationship_diagnostic.csv", group_rows)

    abs_correlation = frame["full_abs_correlation"].to_numpy(float)
    insample = frame["full_insample_rmse_improvement"].to_numpy(float)
    blocked = frame["blocked_rmse_improvement"].to_numpy(float)
    corr_ci = bootstrap_median(abs_correlation, 901)
    in_ci = bootstrap_median(insample, 902)
    out_ci = bootstrap_median(blocked, 903)
    median_abs_correlation = float(np.nanmedian(abs_correlation))
    median_insample = float(np.nanmedian(insample))
    median_blocked = float(np.nanmedian(blocked))
    early_late_agreement = float(frame["early_late_slope_sign_agreement"].mean())
    archived_agreement = float(frame["full_archived_slope_sign_agreement"].mean())
    direction_agreement = float(frame["full_documented_direction_agreement"].mean())
    weak = bool(median_abs_correlation < 0.20 or median_insample < 0.01)
    unstable = bool(early_late_agreement < 0.70 or median_blocked <= 0)
    if weak and unstable:
        decision = "current_temperature_psm_is_weak_and_temporally_unstable"
    elif weak:
        decision = "current_temperature_psm_has_weak_local_association"
    elif unstable:
        decision = "current_temperature_psm_has_temporal_instability"
    else:
        decision = "association_is_stable_but_operator_skill_requires_separate_test"
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "temperature_identity_count": int(len(frame)),
        "median_absolute_full_period_correlation": median_abs_correlation,
        "median_absolute_full_period_correlation_bootstrap_95": list(corr_ci),
        "median_full_period_insample_rmse_improvement": median_insample,
        "median_full_period_insample_rmse_improvement_bootstrap_95": list(in_ci),
        "median_blocked_rmse_improvement": median_blocked,
        "median_blocked_rmse_improvement_bootstrap_95": list(out_ci),
        "early_late_slope_sign_agreement_fraction": early_late_agreement,
        "full_slope_archived_b1_sign_agreement_fraction": archived_agreement,
        "full_slope_documented_direction_agreement_fraction": direction_agreement,
        "weak_association_flag": weak,
        "temporal_instability_flag": unstable,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    axes[0, 0].hist(abs_correlation[np.isfinite(abs_correlation)], bins=30, color="#005f73", edgecolor="white")
    axes[0, 0].axvline(0.20, color="#9b2226", linestyle="--")
    axes[0, 0].set(title="Full-period association", xlabel="Absolute correlation", ylabel="Identities")
    axes[0, 1].scatter(insample * 100, blocked * 100, s=17, alpha=0.65, color="#0a9396")
    axes[0, 1].axhline(0, color="#9b2226", linewidth=1)
    axes[0, 1].axvline(1, color="#9b2226", linestyle="--", linewidth=1)
    axes[0, 1].set(title="Calibration versus blocked skill", xlabel="In-sample RMSE improvement (%)", ylabel="Blocked RMSE improvement (%)")
    axes[1, 0].bar(["Early/late", "Archived b1", "Documented"], [early_late_agreement, archived_agreement, direction_agreement], color=["#ee9b00", "#005f73", "#94d2bd"])
    axes[1, 0].axhline(0.70, color="#9b2226", linestyle="--")
    axes[1, 0].set_ylim(0, 1); axes[1, 0].set(title="Slope-sign agreement", ylabel="Fraction")
    measurements = [row for row in group_rows if row["group_dimension"] == "measurement"]
    axes[1, 1].bar([row["group_value"] for row in measurements], [100 * row["median_blocked_rmse_improvement"] for row in measurements], color="#ca6702")
    axes[1, 1].axhline(0, color="#9b2226", linewidth=1)
    axes[1, 1].set(title="Blocked skill by measurement", ylabel="Median RMSE improvement (%)")
    fig.suptitle("Documented temperature proxy relationship diagnosis", fontsize=14)
    fig.savefig(output / "temperature_proxy_association_stability.png", dpi=180)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Proxy Association and Stability\n\n"
        f"- Identities: `{len(frame)}`.\n"
        f"- Median absolute full-period correlation: `{median_abs_correlation:.3f}` (95% CI `{corr_ci[0]:.3f}` to `{corr_ci[1]:.3f}`).\n"
        f"- Median in-sample RMSE improvement: `{median_insample:.3%}` (95% CI `{in_ci[0]:.3%}` to `{in_ci[1]:.3%}`).\n"
        f"- Median blocked RMSE improvement: `{median_blocked:.3%}` (95% CI `{out_ci[0]:.3%}` to `{out_ci[1]:.3%}`).\n"
        f"- Early/late slope-sign agreement: `{early_late_agreement:.3%}`.\n"
        f"- Full slope sign agreement with archived b1: `{archived_agreement:.3%}`; documented direction: `{direction_agreement:.3%}`.\n"
        f"- Weak association flag: `{str(weak).lower()}`; temporal instability flag: `{str(unstable).lower()}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "bootstrap_draws": 5000}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
