from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
SOURCE_DIR = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_four_background_scheme_evaluation_v2"
)
SOURCE_CSV = SOURCE_DIR / "modern_station_year_scheme_values.csv"
OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_four_background_instrumental_metrics_v1"
)

BRANCHES = {
    "current_CESM": "current_CESM_degC",
    "old_50_50": "old_50_50_degC",
    "equal_four": "equal_four_degC",
    "skill_independence": "skill_independence_oof_degC",
}
OBSERVATIONS = {"qcf": "qcf_station_degC", "qcu": "qcu_station_degC"}
BLOCKS = [(1951, 1960), (1961, 1970), (1971, 1980), (1981, 1990), (1991, 2000)]
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260825


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs = observed[valid]
    pred = predicted[valid]
    if len(obs) == 0:
        return {key: np.nan for key in (
            "n", "rmse_degC", "mae_degC", "bias_degC", "correlation_r",
            "correlation_p", "variance_ratio", "re_climatology_zero", "ce_evaluation_mean",
        )}
    error = pred - obs
    obs_sd = float(np.std(obs, ddof=1)) if len(obs) > 1 else np.nan
    pred_sd = float(np.std(pred, ddof=1)) if len(pred) > 1 else np.nan
    if len(obs) >= 3 and obs_sd > 0 and pred_sd > 0:
        correlation = pearsonr(obs, pred)
        r = float(correlation.statistic)
        p = float(correlation.pvalue)
    else:
        r = np.nan
        p = np.nan
    sse = float(np.sum(np.square(error)))
    re_denominator = float(np.sum(np.square(obs)))
    ce_denominator = float(np.sum(np.square(obs - np.mean(obs))))
    return {
        "n": int(len(obs)),
        "rmse_degC": float(np.sqrt(np.mean(np.square(error)))),
        "mae_degC": float(np.mean(np.abs(error))),
        "bias_degC": float(np.mean(error)),
        "correlation_r": r,
        "correlation_p": p,
        "variance_ratio": pred_sd / obs_sd if obs_sd > 0 else np.nan,
        "re_climatology_zero": 1.0 - sse / re_denominator if re_denominator > 0 else np.nan,
        "ce_evaluation_mean": 1.0 - sse / ce_denominator if ce_denominator > 0 else np.nan,
    }


def slope_per_century(year: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(year) & np.isfinite(values)
    if int(valid.sum()) < 10:
        return np.nan
    return float(linregress(year[valid], values[valid]).slope * 100.0)


def station_metrics(frame: pd.DataFrame, observation: str, branch: str, column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station_id, group in frame.groupby("station_id"):
        valid = group[[observation, column]].notna().all(axis=1)
        selected = group.loc[valid]
        if len(selected) < 10:
            continue
        metrics = paired_metrics(selected[observation].to_numpy(float), selected[column].to_numpy(float))
        observed_trend = slope_per_century(
            selected["year"].to_numpy(float), selected[observation].to_numpy(float)
        )
        predicted_trend = slope_per_century(
            selected["year"].to_numpy(float), selected[column].to_numpy(float)
        )
        rows.append({
            "station_id": station_id,
            "station_name": selected["station_name"].iloc[0],
            "country": selected["country"].iloc[0],
            "longitude": float(selected["longitude"].iloc[0]),
            "latitude": float(selected["latitude"].iloc[0]),
            "observation": observation.removesuffix("_station_degC"),
            "branch": branch,
            **metrics,
            "observed_trend_degC_per_century": observed_trend,
            "predicted_trend_degC_per_century": predicted_trend,
            "abs_trend_error_degC_per_century": abs(predicted_trend - observed_trend),
        })
    return pd.DataFrame(rows)


def decadal_pairs(frame: pd.DataFrame, observation: str, column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start, end in BLOCKS:
        subset = frame[frame["year"].between(start, end)]
        for station_id, group in subset.groupby("station_id"):
            valid = group[[observation, column]].notna().all(axis=1)
            if int(valid.sum()) < 6:
                continue
            rows.append({
                "station_id": station_id,
                "start_year": start,
                "end_year": end,
                "observed_degC": float(group.loc[valid, observation].mean()),
                "predicted_degC": float(group.loc[valid, column].mean()),
                "valid_year_count": int(valid.sum()),
            })
    return pd.DataFrame(rows)


def fisher_mean(values: pd.Series) -> float:
    finite = values[np.isfinite(values)].clip(-0.999999, 0.999999)
    if len(finite) == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(finite))))


def aggregate_metrics(
    frame: pd.DataFrame,
    station: pd.DataFrame,
    decades: pd.DataFrame,
    observation: str,
    branch: str,
    column: str,
) -> dict[str, object]:
    valid = frame[[observation, column]].notna().all(axis=1)
    annual = paired_metrics(
        frame.loc[valid, observation].to_numpy(float), frame.loc[valid, column].to_numpy(float)
    )
    decadal = paired_metrics(
        decades["observed_degC"].to_numpy(float), decades["predicted_degC"].to_numpy(float)
    )
    regional = (
        frame.loc[valid]
        .groupby("year")[[observation, column]]
        .mean()
        .dropna()
    )
    regional_metrics = paired_metrics(
        regional[observation].to_numpy(float), regional[column].to_numpy(float)
    )
    return {
        "observation": observation.removesuffix("_station_degC"),
        "branch": branch,
        "station_count": int(station["station_id"].nunique()),
        "annual_pair_count": int(annual["n"]),
        "annual_pooled_rmse_degC": annual["rmse_degC"],
        "annual_pooled_mae_degC": annual["mae_degC"],
        "annual_pooled_bias_degC": annual["bias_degC"],
        "annual_pooled_correlation_r": annual["correlation_r"],
        "annual_pooled_correlation_p": annual["correlation_p"],
        "annual_pooled_variance_ratio": annual["variance_ratio"],
        "annual_pooled_re": annual["re_climatology_zero"],
        "annual_pooled_ce": annual["ce_evaluation_mean"],
        "median_station_rmse_degC": float(station["rmse_degC"].median()),
        "median_station_mae_degC": float(station["mae_degC"].median()),
        "equal_station_mean_bias_degC": float(station["bias_degC"].mean()),
        "median_station_correlation_r": float(station["correlation_r"].median()),
        "fisher_equal_station_correlation_r": fisher_mean(station["correlation_r"]),
        "median_station_variance_ratio": float(station["variance_ratio"].median()),
        "median_station_re": float(station["re_climatology_zero"].median()),
        "median_station_ce": float(station["ce_evaluation_mean"].median()),
        "median_abs_trend_error_degC_per_century": float(
            station["abs_trend_error_degC_per_century"].median()
        ),
        "decadal_pair_count": int(decadal["n"]),
        "decadal_rmse_degC": decadal["rmse_degC"],
        "decadal_mae_degC": decadal["mae_degC"],
        "decadal_bias_degC": decadal["bias_degC"],
        "decadal_correlation_r": decadal["correlation_r"],
        "decadal_correlation_p": decadal["correlation_p"],
        "decadal_variance_ratio": decadal["variance_ratio"],
        "decadal_re": decadal["re_climatology_zero"],
        "decadal_ce": decadal["ce_evaluation_mean"],
        "matched_station_mean_year_count": int(regional_metrics["n"]),
        "matched_station_mean_annual_rmse_degC": regional_metrics["rmse_degC"],
        "matched_station_mean_annual_correlation_r": regional_metrics["correlation_r"],
        "matched_station_mean_annual_variance_ratio": regional_metrics["variance_ratio"],
        "matched_station_mean_annual_re": regional_metrics["re_climatology_zero"],
        "matched_station_mean_annual_ce": regional_metrics["ce_evaluation_mean"],
    }


def bootstrap_differences(frame: pd.DataFrame, observation: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + (0 if observation.startswith("qcf") else 1))
    station_ids = np.array(sorted(frame["station_id"].unique()))
    source_groups = {station_id: frame[frame["station_id"].eq(station_id)] for station_id in station_ids}
    distributions = {
        branch: {metric: [] for metric in ("annual_rmse", "annual_r", "decadal_rmse", "decadal_r")}
        for branch in BRANCHES
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(station_ids, size=len(station_ids), replace=True)
        pieces = []
        for occurrence, station_id in enumerate(sampled):
            piece = source_groups[station_id].copy()
            piece["bootstrap_station_id"] = f"{station_id}_{occurrence}"
            pieces.append(piece)
        replicate = pd.concat(pieces, ignore_index=True)
        current_column = BRANCHES["current_CESM"]
        current_valid = replicate[[observation, current_column]].notna().all(axis=1)
        current_annual = paired_metrics(
            replicate.loc[current_valid, observation].to_numpy(float),
            replicate.loc[current_valid, current_column].to_numpy(float),
        )
        current_decades = decadal_pairs(
            replicate.rename(columns={"station_id": "original_station_id", "bootstrap_station_id": "station_id"}),
            observation,
            current_column,
        )
        current_decadal = paired_metrics(
            current_decades["observed_degC"].to_numpy(float),
            current_decades["predicted_degC"].to_numpy(float),
        )
        for branch, column in BRANCHES.items():
            valid = replicate[[observation, column]].notna().all(axis=1)
            annual = paired_metrics(
                replicate.loc[valid, observation].to_numpy(float),
                replicate.loc[valid, column].to_numpy(float),
            )
            decades = decadal_pairs(
                replicate.rename(columns={"station_id": "original_station_id", "bootstrap_station_id": "station_id"}),
                observation,
                column,
            )
            decadal = paired_metrics(
                decades["observed_degC"].to_numpy(float), decades["predicted_degC"].to_numpy(float)
            )
            distributions[branch]["annual_rmse"].append(annual["rmse_degC"] - current_annual["rmse_degC"])
            distributions[branch]["annual_r"].append(annual["correlation_r"] - current_annual["correlation_r"])
            distributions[branch]["decadal_rmse"].append(decadal["rmse_degC"] - current_decadal["rmse_degC"])
            distributions[branch]["decadal_r"].append(decadal["correlation_r"] - current_decadal["correlation_r"])
    rows: list[dict[str, object]] = []
    for branch, metrics in distributions.items():
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=float)
            finite = array[np.isfinite(array)]
            lower, median, upper = np.quantile(finite, [0.025, 0.5, 0.975])
            lower_is_better = metric.endswith("rmse")
            improvement_probability = float(np.mean(finite < 0)) if lower_is_better else float(np.mean(finite > 0))
            rows.append({
                "observation": observation.removesuffix("_station_degC"),
                "branch": branch,
                "metric_delta_vs_current": metric,
                "replicate_count": int(len(finite)),
                "delta_median": float(median),
                "delta_ci95_lower": float(lower),
                "delta_ci95_upper": float(upper),
                "probability_better_than_current": improvement_probability,
            })
    return pd.DataFrame(rows)


def plot_results(aggregate: pd.DataFrame, station: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    qcf = aggregate[aggregate["observation"].eq("qcf")].set_index("branch").loc[list(BRANCHES)]
    colors = ["#355070", "#d97706", "#2a9d8f", "#7b2cbf"]
    labels = ["Current CESM", "Old 50:50", "Equal four", "Skill-independence"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    x = np.arange(len(BRANCHES))
    axes[0, 0].bar(x, qcf["annual_pooled_rmse_degC"], color=colors)
    axes[0, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 0].set_ylabel("RMSE (degC)")
    axes[0, 0].set_title("Annual station-pair error, GHCN-M qcf")

    axes[0, 1].bar(x - 0.18, qcf["annual_pooled_correlation_r"], 0.36, color=colors, alpha=0.65)
    axes[0, 1].bar(x + 0.18, qcf["decadal_correlation_r"], 0.36, color=colors)
    axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 1].set_ylabel("Pearson r")
    axes[0, 1].set_title("Annual (light) and decadal (solid) correlation")
    axes[0, 1].axhline(0, color="#222222", linewidth=0.8)

    axes[1, 0].bar(x - 0.18, qcf["median_station_re"], 0.36, color=colors, alpha=0.65)
    axes[1, 0].bar(x + 0.18, qcf["median_station_ce"], 0.36, color=colors)
    axes[1, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 0].set_ylabel("Skill score")
    axes[1, 0].set_title("Median station RE (light) and CE (solid)")
    axes[1, 0].axhline(0, color="#222222", linewidth=0.8)

    qcf_station = station[station["observation"].eq("qcf")].pivot(
        index="station_name", columns="branch", values="correlation_r"
    ).reindex(columns=list(BRANCHES))
    image = axes[1, 1].imshow(qcf_station.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[1, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 1].set_yticks(np.arange(len(qcf_station)), qcf_station.index)
    axes[1, 1].set_title("Station-level annual correlation")
    figure.colorbar(image, ax=axes[1, 1], label="Pearson r", shrink=0.85)

    figure.suptitle("Four temperature-background schemes: expanded instrumental evaluation", fontsize=15)
    figure.savefig(OUT / "four_scheme_instrumental_metrics.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    source_hash_before = sha256(SOURCE_CSV)
    frame = pd.read_csv(SOURCE_CSV)
    if not frame["year"].between(1951, 2000).all():
        raise RuntimeError("Input contains a year outside the declared 1951-2000 evaluation interval")

    station_tables: list[pd.DataFrame] = []
    decade_tables: list[pd.DataFrame] = []
    aggregate_rows: list[dict[str, object]] = []
    bootstrap_tables: list[pd.DataFrame] = []
    for observation_name, observation in OBSERVATIONS.items():
        for branch, column in BRANCHES.items():
            station = station_metrics(frame, observation, branch, column)
            decades = decadal_pairs(frame, observation, column)
            station_tables.append(station)
            decades.insert(0, "branch", branch)
            decades.insert(0, "observation", observation_name)
            decade_tables.append(decades)
            aggregate_rows.append(aggregate_metrics(frame, station, decades, observation, branch, column))
        bootstrap_tables.append(bootstrap_differences(frame, observation))

    station_all = pd.concat(station_tables, ignore_index=True)
    decades_all = pd.concat(decade_tables, ignore_index=True)
    aggregate = pd.DataFrame(aggregate_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    station_all.to_csv(OUT / "station_level_annual_metrics.csv", index=False)
    decades_all.to_csv(OUT / "station_decadal_pairs.csv", index=False)
    aggregate.to_csv(OUT / "instrumental_metrics_summary.csv", index=False)
    bootstrap.to_csv(OUT / "station_bootstrap_differences_vs_current.csv", index=False)
    plot_results(aggregate, station_all, bootstrap)

    source_hash_after = sha256(SOURCE_CSV)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Read-only source table changed during analysis")

    qcf = aggregate[aggregate["observation"].eq("qcf")].set_index("branch")
    summary = {
        "experiment": OUT.name,
        "status": "complete",
        "evaluation_interval": [1951, 2000],
        "observation_primary": "GHCN-M v4 TAVG qcf station anomalies",
        "observation_sensitivity": "GHCN-M v4 TAVG qcu station anomalies",
        "baseline": "station-and-calendar-month-specific 1961-1990 climatology inherited from source table",
        "station_count": int(frame["station_id"].nunique()),
        "branches": list(BRANCHES),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "station",
        "qcf_metrics": aggregate[aggregate["observation"].eq("qcf")].to_dict(orient="records"),
        "qcu_metrics": aggregate[aggregate["observation"].eq("qcu")].to_dict(orient="records"),
        "best_qcf_annual_rmse": str(qcf["annual_pooled_rmse_degC"].idxmin()),
        "best_qcf_annual_correlation": str(qcf["annual_pooled_correlation_r"].idxmax()),
        "best_qcf_decadal_rmse": str(qcf["decadal_rmse_degC"].idxmin()),
        "best_qcf_decadal_correlation": str(qcf["decadal_correlation_r"].idxmax()),
        "interpretation_boundary": (
            "Instrumental metrics evaluate the reconstructed candidate fields, not phase-locking skill of each "
            "forced-model background alone. Differences require uncertainty support and must not be interpreted "
            "as independent evidence when station sources overlap reconstruction calibration products."
        ),
        "locked_period_access": "none; input contains only 1951-2000",
        "source_table_sha256_before": source_hash_before,
        "source_table_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({
        "source": {"path": str(SOURCE_CSV), "sha256": source_hash_before},
        "source_experiment": str(SOURCE_DIR),
        "original_climate_or_station_files_opened": False,
    }, indent=2), encoding="utf-8")
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        "# Experiment contract\n\n"
        "- Read only the locked-safe 1951-2000 station-year pairing table from the completed four-scheme experiment.\n"
        "- Do not read 1921-1939, original reconstruction fields, or original station files.\n"
        "- Do not modify production climate, run EnKF, or run OGGM.\n"
        "- qcf is primary; qcu is a homogenization sensitivity.\n"
        "- Report annual and decadal correlation together with error, bias, variance, RE, CE, and trend metrics.\n"
        "- Use station bootstrap uncertainty before treating small scheme differences as meaningful.\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Final result",
        "",
        "Expanded GHCN-M instrumental metrics for the four fixed background schemes over 1951-2000.",
        "",
        "## Primary qcf summary",
        "",
        "| Branch | Annual RMSE | Annual r | Median station r | Decadal RMSE | Decadal r | Median RE | Median CE | Variance ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        row = qcf.loc[branch]
        report_lines.append(
            f"| {branch} | {row['annual_pooled_rmse_degC']:.3f} | {row['annual_pooled_correlation_r']:.3f} | "
            f"{row['median_station_correlation_r']:.3f} | {row['decadal_rmse_degC']:.3f} | "
            f"{row['decadal_correlation_r']:.3f} | {row['median_station_re']:.3f} | "
            f"{row['median_station_ce']:.3f} | {row['annual_pooled_variance_ratio']:.3f} |"
        )
    report_lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These metrics are valid instrumental consistency diagnostics for the final reconstructed candidate fields. "
        "They do not measure phase-locking skill of the forced-model backgrounds in isolation. A small metric "
        "difference is not a selection result unless its station-bootstrap interval supports separation.",
    ])
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
