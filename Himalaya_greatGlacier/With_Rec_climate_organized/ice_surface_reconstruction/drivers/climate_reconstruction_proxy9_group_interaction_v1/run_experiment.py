"""Run the complete 2^5 proxy-group calibration-window interaction experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUPS = (
    ("G1", "high_impact_high_instability"),
    ("G2", "high_impact_lower_instability"),
    ("G3", "lower_impact_high_instability"),
    ("G4", "lower_impact_lower_instability"),
    ("G5", "duplicate_chronology_291_296"),
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


def subset_name(subset: frozenset[str]) -> str:
    return "baseline" if not subset else "+".join(sorted(subset))


def period_metrics(values: np.ndarray, years: np.ndarray, start: int, end: int) -> dict[str, float]:
    selected = (years >= start) & (years <= end)
    return {
        "mean_c": float(np.mean(values[selected])),
        "trend_c_per_century": float(100.0 * np.polyfit(years[selected], values[selected], 1)[0]),
        "std_c": float(np.std(values[selected])),
    }


def pareto_indices(frame: pd.DataFrame) -> np.ndarray:
    benefit = frame["early_rmse_improvement_c"].to_numpy()
    cost = frame["late_rmse_change_c"].to_numpy()
    corr_loss = -frame["late_correlation_change"].to_numpy()
    keep = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        dominates = (
            (benefit >= benefit[i])
            & (cost <= cost[i])
            & (corr_loss <= corr_loss[i])
            & ((benefit > benefit[i]) | (cost < cost[i]) | (corr_loss < corr_loss[i]))
        )
        if np.any(dominates):
            keep[i] = False
    return keep


def plot_curves(years: np.ndarray, curves: dict[frozenset[str], np.ndarray], summary: pd.DataFrame, output: Path) -> None:
    selected = (years >= 1850) & (years <= 2000)
    pareto = set(summary.loc[summary["pareto"], "subset"])
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.7), sharex=True, constrained_layout=True)
    for subset, values in curves.items():
        name = subset_name(subset)
        if not subset:
            color, width, alpha, zorder = "#111111", 2.5, 1.0, 5
        elif len(subset) == 5:
            color, width, alpha, zorder = "#b2182b", 2.5, 1.0, 5
        elif name in pareto:
            color, width, alpha, zorder = "#007f8b", 1.6, 0.9, 4
        else:
            color, width, alpha, zorder = "#a9a9a9", 0.75, 0.32, 1
        rolled = pd.Series(values).rolling(15, center=True, min_periods=15).mean().to_numpy()
        label = name if (not subset or len(subset) == 5 or name in pareto) else None
        axes[0].plot(years[selected], values[selected], color=color, lw=width, alpha=alpha, zorder=zorder, label=label)
        axes[1].plot(years[selected], rolled[selected], color=color, lw=width, alpha=alpha, zorder=zorder, label=label)
    for axis in axes:
        axis.axvspan(1901, 1950, color="#eeeeee", zorder=-2)
        axis.axvline(1950.5, color="#777777", lw=0.8, ls="--")
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("All 32 proxy-group calibration-window subsets")
    axes[1].set_title("Centered 15-year mean")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 7.2), constrained_layout=True)
    scatter = axis.scatter(
        summary["early_rmse_improvement_c"],
        summary["late_rmse_change_c"],
        c=summary["group_count"],
        cmap="viridis",
        s=55 + 15 * summary["group_count"],
        edgecolor=np.where(summary["pareto"], "#111111", "white"),
        linewidth=np.where(summary["pareto"], 1.4, 0.7),
    )
    for _, row in summary[summary["pareto"] | summary["subset"].isin(["baseline", "G1+G2+G3+G4+G5"])].iterrows():
        axis.annotate(row["subset"], (row["early_rmse_improvement_c"], row["late_rmse_change_c"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.axvline(0.0, color="#777777", lw=0.8)
    axis.axhline(0.0, color="#777777", lw=0.8)
    axis.grid(alpha=0.18)
    axis.set_xlabel("1901-1950 RMSE improvement (degC; positive is better)")
    axis.set_ylabel("1951-2000 RMSE change (degC; positive is worse)")
    axis.set_title("Early benefit and late cost across all 32 subsets")
    figure.colorbar(scatter, ax=axis, label="Number of changed groups")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_interactions(interactions: pd.DataFrame, output: Path) -> None:
    ranked = interactions[interactions["order"] >= 2].copy()
    ranked["magnitude"] = np.sqrt(
        ranked["early_mean_c"] ** 2
        + ranked["late_mean_c"] ** 2
        + (ranked["late_trend_c_per_century"] / 4.0) ** 2
    )
    ranked = ranked.nlargest(15, "magnitude").sort_values("magnitude")
    figure, axis = plt.subplots(figsize=(10.5, 7.4), constrained_layout=True)
    colors = np.where(ranked["late_trend_c_per_century"] < 0, "#b2182b", "#2166ac")
    axis.barh(ranked["term"], ranked["late_trend_c_per_century"], color=colors)
    axis.axvline(0.0, color="#555555", lw=0.8)
    axis.grid(axis="x", alpha=0.18)
    axis.set_xlabel("Interaction contribution to 1951-2000 trend (degC/century)")
    axis.set_title("Largest higher-order group interactions")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source = cfg["sources"]
    archive = Path(cfg["archive_root"])
    output = root / cfg["result_root"]
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)
    paths = {
        "grouped_result": root / source["grouped_result"],
        "window_psm": root / source["window_psm"],
        "baseline_driver": root / source["baseline_driver"],
        "evaluation_driver": root / source["evaluation_driver"],
        "helper_driver": root / source["helper_driver"],
        "prior": root / source["prior"],
        "indices": archive / source["indices"],
        "proxy9": archive / source["proxy9"],
        "archived_regem": root / source["archived_regem"],
        "baseline_arrays": root / source["baseline_arrays"],
        "hma_mask": root / source["hma_mask"],
        "instrumental_netcdf": Path(source["instrumental_netcdf"]),
    }
    hash_paths = {name: path for name, path in paths.items() if name != "grouped_result"}
    hash_paths["group_assignments"] = paths["grouped_result"] / "proxy9_group_assignments.csv"
    hash_paths["grouped_summary"] = paths["grouped_result"] / "summary.json"
    hashes_before = {name: sha256(path) for name, path in hash_paths.items()}

    helper = load_module("grouped_helper", paths["helper_driver"])
    base = load_module("controlled_enkf_interaction", paths["baseline_driver"])
    evaluation = load_module("network_evaluation_interaction", paths["evaluation_driver"])
    assignments = pd.read_csv(hash_paths["group_assignments"]).sort_values("proxy9_record_1based")
    if not np.array_equal(assignments["proxy9_record_1based"].to_numpy(), np.arange(1, 298)):
        raise RuntimeError("group-assignment-coverage-failed")
    masks = {short: assignments["group"].eq(long).to_numpy() for short, long in GROUPS}
    if not np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1):
        raise RuntimeError("group-assignment-partition-failed")

    psm = helper.load_v73(paths["window_psm"])
    fresh = np.asarray(psm["hxb"], dtype=float).transpose(2, 1, 0)
    repeat = np.asarray(psm["baseline_repeat_hxb"], dtype=float).T
    repeat_closure = float(np.nanmax(np.abs(fresh[:, 2] - repeat)))
    if repeat_closure > float(cfg["gates"]["window_repeat_hxb_max_abs"]):
        raise RuntimeError(f"window-repeat-closure-failed:{repeat_closure}")
    delta = fresh[:, 1] - fresh[:, 2]

    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
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

    subsets = [frozenset(combo) for count in range(6) for combo in itertools.combinations([short for short, _ in GROUPS], count)]
    results = {}
    regional = {}
    cubes = {}
    reference = tuple(cfg["periods"]["reference"])
    for subset in subsets:
        item = {key: np.asarray(value).copy() for key, value in proxy.items()}
        if subset:
            changed = np.any(np.stack([masks[name] for name in subset]), axis=0)
            item["hxb"] = proxy["hxb"].copy()
            item["hxb"][changed, :] += delta[changed, :]
        result = base._run_branch(prior, samples, item, residuals, target_weights, regional_weights)
        results[subset] = result
        regional[subset] = helper.anomaly(np.asarray(result["regional_analysis"]), years, reference)
        cubes[subset] = evaluation.anomaly(
            evaluation.to_cube(np.asarray(result["analysis"]), len(grid_lat), len(grid_lon)),
            years,
            reference,
        )

    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(results[frozenset()]["analysis"]) - saved["proxy9_analysis"])))
    enkf_closure = max(
        float(result[key])
        for result in results.values()
        for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c")
    )
    if baseline_state_closure > float(cfg["gates"]["baseline_state_closure_max_abs_c"]) or enkf_closure > float(cfg["gates"]["enkf_closure_max_abs_c"]):
        raise RuntimeError(f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    observation_cubes = {}
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        for label, variable in {
            "Berkeley Earth": "tas_obs_bea_05deg",
            "CRU": "tas_obs_cru_tmp_05deg",
            "GISTEMP": "tas_obs_GIST_05deg",
            "HadCRUT5": "tas_obs_had5_05deg",
            "NOAA": "tas_obs_NOAA_05deg",
        }.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            observation_cubes[label] = evaluation.anomaly(
                evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, reference
            )

    skill_rows = []
    for subset, field in cubes.items():
        for observation_name, observation_field in observation_cubes.items():
            for period_name in ("early", "late"):
                start, end = cfg["periods"][period_name]
                selected = (years >= start) & (years <= end)
                row = {"subset": subset_name(subset), "observation": observation_name, "period": f"{start}-{end}"}
                row.update(evaluation.metrics(field[selected], observation_field[selected], years[selected], mask2))
                skill_rows.append(row)
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(output / "instrumental_skill_all_subsets.csv", index=False)
    principal = (
        skill[~skill["observation"].eq("NOAA")]
        .groupby(["subset", "period"])[["median_lowpass_correlation_15y", "median_rmse", "median_abs_trend_difference"]]
        .mean()
        .reset_index()
    )
    baseline_early = principal[(principal["subset"] == "baseline") & (principal["period"] == "1901-1950")].iloc[0]
    baseline_late = principal[(principal["subset"] == "baseline") & (principal["period"] == "1951-2000")].iloc[0]
    rows = []
    for subset in subsets:
        name = subset_name(subset)
        early = principal[(principal["subset"] == name) & (principal["period"] == "1901-1950")].iloc[0]
        late = principal[(principal["subset"] == name) & (principal["period"] == "1951-2000")].iloc[0]
        early_curve = period_metrics(regional[subset], years, 1901, 1950)
        late_curve = period_metrics(regional[subset], years, 1951, 2000)
        rows.append({
            "subset": name,
            "group_count": len(subset),
            "early_mean_c": early_curve["mean_c"],
            "early_trend_c_per_century": early_curve["trend_c_per_century"],
            "early_lowpass_correlation_15y": float(early["median_lowpass_correlation_15y"]),
            "early_rmse_c": float(early["median_rmse"]),
            "early_rmse_improvement_c": float(baseline_early["median_rmse"] - early["median_rmse"]),
            "late_mean_c": late_curve["mean_c"],
            "late_trend_c_per_century": late_curve["trend_c_per_century"],
            "late_lowpass_correlation_15y": float(late["median_lowpass_correlation_15y"]),
            "late_rmse_c": float(late["median_rmse"]),
            "late_rmse_change_c": float(late["median_rmse"] - baseline_late["median_rmse"]),
            "late_correlation_change": float(late["median_lowpass_correlation_15y"] - baseline_late["median_lowpass_correlation_15y"]),
        })
    subset_summary = pd.DataFrame(rows)
    subset_summary["pareto"] = pareto_indices(subset_summary)
    subset_summary.to_csv(output / "subset_skill_summary.csv", index=False)

    effects = {frozenset(): regional[frozenset()].copy()}
    interaction_rows = []
    for order in range(1, 6):
        for subset in [item for item in subsets if len(item) == order]:
            effect = regional[subset].copy()
            for proper, proper_effect in effects.items():
                if proper < subset:
                    effect -= proper_effect
            effects[subset] = effect
            early = period_metrics(effect, years, 1901, 1950)
            late = period_metrics(effect, years, 1951, 2000)
            interaction_rows.append({
                "term": subset_name(subset),
                "order": order,
                "early_mean_c": early["mean_c"],
                "early_trend_c_per_century": early["trend_c_per_century"],
                "late_mean_c": late["mean_c"],
                "late_trend_c_per_century": late["trend_c_per_century"],
                "annual_rms_c": float(np.sqrt(np.mean(effect**2))),
                "annual_max_abs_c": float(np.max(np.abs(effect))),
            })
    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(output / "factorial_interaction_terms.csv", index=False)
    reconstructed_full = sum(effects.values(), start=np.zeros_like(years, dtype=float))
    full = frozenset(short for short, _ in GROUPS)
    factorial_closure = float(np.max(np.abs(reconstructed_full - regional[full])))
    if factorial_closure > float(cfg["gates"]["factorial_reconstruction_max_abs_c"]):
        raise RuntimeError(f"factorial-closure-failed:{factorial_closure}")

    pd.DataFrame({"year": years, **{subset_name(key): value for key, value in regional.items()}}).to_csv(
        output / "hma_regional_series_all_subsets.csv", index=False
    )
    plot_curves(years, regional, subset_summary, output / "figure_all_subset_curves_1850_2000.png")
    plot_tradeoff(subset_summary, output / "figure_subset_tradeoff.png")
    plot_interactions(interactions, output / "figure_interaction_terms.png")

    hashes_after = {name: sha256(path) for name, path in hash_paths.items()}
    pure = hashes_before == hashes_after
    top_interactions = (
        interactions[interactions["order"] >= 2]
        .assign(abs_late_trend=lambda frame: frame["late_trend_c_per_century"].abs())
        .nlargest(10, "abs_late_trend")
        .drop(columns="abs_late_trend")
    )
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if pure else "fail",
        "subset_count": len(subsets),
        "window_repeat_hxb_max_abs": repeat_closure,
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "factorial_reconstruction_max_abs_c": factorial_closure,
        "source_hashes_unchanged": pure,
        "pareto_subsets": subset_summary[subset_summary["pareto"]].to_dict(orient="records"),
        "largest_late_trend_interactions": top_interactions.to_dict(orient="records"),
        "interpretation_boundary": "Complete grouped operator factorial. No production calibration design is selected and rec_ref is not used for acceptance.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Calibration-Window Group Interaction v1\n\n"
        f"Status: **{summary['status']}**. All 32 subsets completed with exact factorial closure.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
