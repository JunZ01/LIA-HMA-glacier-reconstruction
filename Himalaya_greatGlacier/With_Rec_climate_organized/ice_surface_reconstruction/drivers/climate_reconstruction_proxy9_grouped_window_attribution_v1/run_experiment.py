"""Run grouped proxy9 calibration-window attribution with frozen EnKF inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE = "archived_1940_2000"
FULL = "all_records_1920_1980"
GROUP_ORDER = (
    "high_impact_high_instability",
    "high_impact_lower_instability",
    "lower_impact_high_instability",
    "lower_impact_lower_instability",
    "duplicate_chronology_291_296",
)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-load-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v73(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            name: np.asarray(handle[name])
            for name in handle.keys()
            if not name.startswith("#")
        }


def anomaly(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    selected = (years >= period[0]) & (years <= period[1])
    return values - np.nanmean(values[..., selected], axis=-1, keepdims=True)


def rolling(values: np.ndarray, width: int = 15) -> np.ndarray:
    return pd.Series(values).rolling(width, center=True, min_periods=width).mean().to_numpy()


def period_metrics(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> dict[str, float]:
    selected = (years >= period[0]) & (years <= period[1])
    return {
        "mean_c": float(np.nanmean(values[selected])),
        "trend_c_per_century": float(100.0 * np.polyfit(years[selected], values[selected], 1)[0]),
        "std_c": float(np.nanstd(values[selected])),
    }


def build_groups(ranking: pd.DataFrame, quantile: float, duplicates: set[int]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    table = ranking.copy().sort_values("proxy9_record_1based").reset_index(drop=True)
    expected = np.arange(1, len(table) + 1)
    if not np.array_equal(table["proxy9_record_1based"].to_numpy(), expected):
        raise RuntimeError("ranking-record-order-or-coverage-invalid")
    table["impact_score"] = (
        table["archived_R_hma_mean_contribution_c"].abs()
        + table["archived_R_hma_trend_contribution_c_per_century"].abs()
    )
    correlation = table["minimum_hxb_correlation"].clip(-1.0, 1.0)
    table["instability_score"] = table["maximum_hxb_rms_difference"] * (1.0 - correlation) / 2.0
    nonduplicate = ~table["proxy9_record_1based"].isin(duplicates)
    impact_threshold = float(table.loc[nonduplicate, "impact_score"].quantile(quantile))
    instability_threshold = float(table.loc[nonduplicate, "instability_score"].quantile(quantile))
    high_impact = table["impact_score"] >= impact_threshold
    high_instability = table["instability_score"] >= instability_threshold
    table["group"] = ""
    table.loc[nonduplicate & high_impact & high_instability, "group"] = GROUP_ORDER[0]
    table.loc[nonduplicate & high_impact & ~high_instability, "group"] = GROUP_ORDER[1]
    table.loc[nonduplicate & ~high_impact & high_instability, "group"] = GROUP_ORDER[2]
    table.loc[nonduplicate & ~high_impact & ~high_instability, "group"] = GROUP_ORDER[3]
    table.loc[~nonduplicate, "group"] = GROUP_ORDER[4]
    if (table["group"] == "").any():
        raise RuntimeError("unassigned-records")
    masks = {name: table["group"].eq(name).to_numpy() for name in GROUP_ORDER}
    coverage = np.sum(np.stack(list(masks.values())), axis=0)
    if not np.all(coverage == 1):
        raise RuntimeError("group-partition-not-mutually-exclusive")
    thresholds = {
        "quantile": quantile,
        "impact_threshold": impact_threshold,
        "instability_threshold": instability_threshold,
    }
    return table, masks, thresholds


def plot_curves(years: np.ndarray, curves: dict[str, np.ndarray], output: Path, period: tuple[int, int]) -> None:
    selected = (years >= period[0]) & (years <= period[1])
    colors = {
        BASELINE: "#111111",
        "high_impact_high_instability": "#c43c39",
        "high_impact_lower_instability": "#e68632",
        "lower_impact_high_instability": "#5b63b7",
        "lower_impact_lower_instability": "#6f9d48",
        "duplicate_chronology_291_296": "#9b59a5",
        FULL: "#007f8b",
    }
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.6), sharex=True, constrained_layout=True)
    for name, values in curves.items():
        width = 2.4 if name in (BASELINE, FULL) else 1.45
        alpha = 1.0 if name in (BASELINE, FULL) else 0.88
        axes[0].plot(years[selected], values[selected], color=colors[name], lw=width, alpha=alpha, label=name)
        axes[1].plot(years[selected], rolling(values)[selected], color=colors[name], lw=width, alpha=alpha, label=name)
    for axis in axes:
        axis.axvspan(1901, 1950, color="#eeeeee", zorder=-2)
        axis.axvline(1950.5, color="#777777", lw=0.8, ls="--")
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA temperature: one proxy group moved to the 1920-1980 calibration operator")
    axes[1].set_title("Centered 15-year mean")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_attribution(summary: pd.DataFrame, output: Path) -> None:
    groups = summary[summary["branch"].isin(GROUP_ORDER)].copy()
    x = groups["early_rmse_improvement_c"].to_numpy()
    y = groups["late_rmse_change_c"].to_numpy()
    sizes = 35.0 + 3.0 * groups["record_count"].to_numpy()
    figure, axis = plt.subplots(figsize=(9.6, 6.8), constrained_layout=True)
    axis.scatter(x, y, s=sizes, c=["#c43c39", "#e68632", "#5b63b7", "#6f9d48", "#9b59a5"], edgecolor="white", linewidth=0.8)
    for _, row in groups.iterrows():
        axis.annotate(row["branch"], (row["early_rmse_improvement_c"], row["late_rmse_change_c"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    axis.axvline(0.0, color="#777777", lw=0.8)
    axis.axhline(0.0, color="#777777", lw=0.8)
    axis.grid(alpha=0.18)
    axis.set_xlabel("1901-1950 RMSE improvement (degC; positive is better)")
    axis.set_ylabel("1951-2000 RMSE change (degC; positive is worse)")
    axis.set_title("Early correction versus late-century cost by proxy group")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    archive = Path(cfg["archive_root"])
    source = cfg["sources"]
    output = root / cfg["result_root"]
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)
    paths = {
        "window_psm": root / source["window_psm"],
        "ranking": root / source["ranking"],
        "baseline_driver": root / source["baseline_driver"],
        "evaluation_driver": root / source["evaluation_driver"],
        "prior": root / source["prior"],
        "indices": archive / source["indices"],
        "proxy9": archive / source["proxy9"],
        "archived_regem": root / source["archived_regem"],
        "baseline_arrays": root / source["baseline_arrays"],
        "hma_mask": root / source["hma_mask"],
        "instrumental_netcdf": Path(source["instrumental_netcdf"]),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("controlled_enkf", paths["baseline_driver"])
    evaluation = load_module("network_evaluation", paths["evaluation_driver"])

    psm = load_v73(paths["window_psm"])
    fresh = np.asarray(psm["hxb"], dtype=float).transpose(2, 1, 0)
    repeat = np.asarray(psm["baseline_repeat_hxb"], dtype=float).T
    repeat_closure = float(np.nanmax(np.abs(fresh[:, 2] - repeat)))
    if repeat_closure > float(cfg["gates"]["window_repeat_hxb_max_abs"]):
        raise RuntimeError(f"window-repeat-closure-failed:{repeat_closure}")
    delta = fresh[:, 1] - fresh[:, 2]

    ranking = pd.read_csv(paths["ranking"])
    duplicates = set(int(value) for value in cfg["grouping"]["duplicate_records_1based"])
    group_table, masks, thresholds = build_groups(ranking, float(cfg["grouping"]["quantile"]), duplicates)
    group_table.to_csv(output / "proxy9_group_assignments.csv", index=False)
    group_counts = {name: int(mask.sum()) for name, mask in masks.items()}

    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    proxies = {BASELINE: {key: np.asarray(value).copy() for key, value in proxy.items()}}
    for name, mask in masks.items():
        item = {key: np.asarray(value).copy() for key, value in proxy.items()}
        item["hxb"] = proxy["hxb"].copy()
        item["hxb"][:, mask] += delta[:, mask]
        proxies[name] = item
    full_item = {key: np.asarray(value).copy() for key, value in proxy.items()}
    full_item["hxb"] = proxy["hxb"] + delta
    proxies[FULL] = full_item

    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"])
    mask2 = evaluation.source_grid(mask05[None].astype(float), source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    state_mask = mask2.T.reshape(-1)
    regional_weights = np.cos(np.deg2rad(state_lat)) * state_mask
    regional_weights /= regional_weights.sum()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)
    results = {
        name: base._run_branch(prior, samples, item, residuals, target_weights, regional_weights)
        for name, item in proxies.items()
    }
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(results[BASELINE]["analysis"]) - saved["proxy9_analysis"])))
    enkf_closure = max(
        float(result[key])
        for result in results.values()
        for key in (
            "state_increment_closure_max_abs_c",
            "target_increment_closure_max_abs_c",
            "regional_increment_closure_max_abs_c",
        )
    )
    if baseline_state_closure > float(cfg["gates"]["baseline_state_closure_max_abs_c"]) or enkf_closure > float(cfg["gates"]["enkf_closure_max_abs_c"]):
        raise RuntimeError(f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    reference = tuple(cfg["periods"]["reference"])
    regional = {name: anomaly(np.asarray(result["regional_analysis"]), years, reference) for name, result in results.items()}
    cubes = {
        name: evaluation.anomaly(
            evaluation.to_cube(np.asarray(result["analysis"]), len(grid_lat), len(grid_lon)),
            years,
            reference,
        )
        for name, result in results.items()
    }
    observations = {}
    observation_cubes = {}
    variables = {
        "Berkeley Earth": "tas_obs_bea_05deg",
        "CRU": "tas_obs_cru_tmp_05deg",
        "GISTEMP": "tas_obs_GIST_05deg",
        "HadCRUT5": "tas_obs_had5_05deg",
        "NOAA": "tas_obs_NOAA_05deg",
        "rec_ref": "tas_ensemble_mean_rec_ref",
    }
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        for label, variable in variables.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(
                evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon),
                years,
                reference,
            )
            observation_cubes[label] = field
            observations[label] = evaluation.hma_mean(field, grid_lat, mask2)

    pd.DataFrame({"year": years, **regional, **observations}).to_csv(output / "hma_regional_series.csv", index=False)
    plot_curves(years, regional, output / "figure_hma_grouped_window_curves_1850_2000.png", tuple(cfg["periods"]["plot"]))

    skill_rows = []
    for branch_name, field in cubes.items():
        for observation_name in ("Berkeley Earth", "CRU", "GISTEMP", "HadCRUT5", "NOAA"):
            for period_name in ("early", "late"):
                start, end = cfg["periods"][period_name]
                selected = (years >= start) & (years <= end)
                row = {"case": branch_name, "observation": observation_name, "period": f"{start}-{end}"}
                row.update(evaluation.metrics(field[selected], observation_cubes[observation_name][selected], years[selected], mask2))
                skill_rows.append(row)
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(output / "instrumental_skill_summary_2deg.csv", index=False)
    principal = (
        skill[~skill.observation.eq("NOAA")]
        .groupby(["case", "period"])[["median_lowpass_correlation_15y", "median_rmse", "median_abs_trend_difference"]]
        .mean()
        .reset_index()
    )
    principal.to_csv(output / "instrumental_skill_principal_mean.csv", index=False)

    period_rows = []
    for name, values in {**regional, **observations}.items():
        for period_name in ("early", "late"):
            start, end = cfg["periods"][period_name]
            period_rows.append({"series": name, "period": f"{start}-{end}", **period_metrics(values, years, (start, end))})
    periods = pd.DataFrame(period_rows)
    periods.to_csv(output / "hma_period_summary.csv", index=False)

    baseline_early = principal[(principal.case == BASELINE) & (principal.period == "1901-1950")].iloc[0]
    baseline_late = principal[(principal.case == BASELINE) & (principal.period == "1951-2000")].iloc[0]
    attribution_rows = []
    for branch in (*GROUP_ORDER, FULL):
        early = principal[(principal.case == branch) & (principal.period == "1901-1950")].iloc[0]
        late = principal[(principal.case == branch) & (principal.period == "1951-2000")].iloc[0]
        early_period = periods[(periods.series == branch) & (periods.period == "1901-1950")].iloc[0]
        late_period = periods[(periods.series == branch) & (periods.period == "1951-2000")].iloc[0]
        attribution_rows.append({
            "branch": branch,
            "record_count": 297 if branch == FULL else group_counts[branch],
            "early_mean_c": float(early_period.mean_c),
            "early_trend_c_per_century": float(early_period.trend_c_per_century),
            "early_lowpass_correlation_15y": float(early.median_lowpass_correlation_15y),
            "early_rmse_c": float(early.median_rmse),
            "early_rmse_improvement_c": float(baseline_early.median_rmse - early.median_rmse),
            "late_mean_c": float(late_period.mean_c),
            "late_trend_c_per_century": float(late_period.trend_c_per_century),
            "late_lowpass_correlation_15y": float(late.median_lowpass_correlation_15y),
            "late_rmse_c": float(late.median_rmse),
            "late_rmse_change_c": float(late.median_rmse - baseline_late.median_rmse),
            "late_correlation_change": float(late.median_lowpass_correlation_15y - baseline_late.median_lowpass_correlation_15y),
        })
    attribution = pd.DataFrame(attribution_rows)
    attribution.to_csv(output / "group_attribution_summary.csv", index=False)
    plot_attribution(attribution, output / "figure_early_benefit_late_cost.png")

    baseline_curve = regional[BASELINE]
    summed_group_delta = sum((regional[name] - baseline_curve for name in GROUP_ORDER), start=np.zeros_like(baseline_curve))
    full_delta = regional[FULL] - baseline_curve
    additivity_residual = full_delta - summed_group_delta
    additivity = {
        "annual_max_abs_c": float(np.max(np.abs(additivity_residual))),
        "annual_rmse_c": float(np.sqrt(np.mean(additivity_residual**2))),
        "early_mean_residual_c": float(np.mean(additivity_residual[(years >= 1901) & (years <= 1950)])),
        "late_mean_residual_c": float(np.mean(additivity_residual[(years >= 1951) & (years <= 2000)])),
    }
    pd.DataFrame({
        "year": years,
        "full_network_delta_c": full_delta,
        "sum_isolated_group_deltas_c": summed_group_delta,
        "nonadditivity_residual_c": additivity_residual,
    }).to_csv(output / "group_additivity_diagnostics.csv", index=False)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if pure else "fail",
        "window_repeat_hxb_max_abs": repeat_closure,
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "source_hashes_unchanged": pure,
        "group_thresholds": thresholds,
        "group_counts": group_counts,
        "group_attribution": attribution.to_dict(orient="records"),
        "group_additivity": additivity,
        "interpretation_boundary": "Grouped operator-only attribution. It does not select a production window, delete records, constrain signs, combine residual-error changes, or validate against rec_ref.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Grouped Calibration-Window Attribution v1\n\n"
        f"Status: **{summary['status']}**. See `group_attribution_summary.csv` and the two figures.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
