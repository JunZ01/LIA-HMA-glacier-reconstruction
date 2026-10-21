"""Evaluate the controlled proxy9 residual-missingness repair without OGGM."""

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


def anomaly(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    selected = (years >= period[0]) & (years <= period[1])
    return values - np.nanmean(values[..., selected], axis=-1, keepdims=True)


def rolling(values: np.ndarray, width: int = 15) -> np.ndarray:
    return pd.Series(values).rolling(width, center=True, min_periods=width).mean().to_numpy()


def trend(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(100.0 * np.polyfit(years[finite], values[finite], 1)[0])


def period_metrics(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> dict[str, float]:
    selected = (years >= period[0]) & (years <= period[1])
    return {
        "mean_c": float(np.nanmean(values[selected])),
        "trend_c_per_century": trend(years[selected], values[selected]),
        "std_c": float(np.nanstd(values[selected])),
    }


def hma_weights(base, mask2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    if mask2.shape != (len(grid_lat), len(grid_lon)):
        raise ValueError(f"hma-mask-shape:{mask2.shape}")
    state_mask = mask2.T.reshape(-1)
    weights = np.cos(np.deg2rad(state_lat)) * state_mask
    weights /= weights.sum()
    return grid_lat, grid_lon, weights


def plot_hma_curves(years: np.ndarray, curves: dict[str, np.ndarray], output: Path, plot_period: tuple[int, int]) -> None:
    selected = (years >= plot_period[0]) & (years <= plot_period[1])
    colors = {
        "archived_R": "#b2182b",
        "missingness_repaired_R": "#2166ac",
        "Berkeley Earth": "#333333",
        "CRU": "#5c5c5c",
        "GISTEMP": "#7a7a7a",
        "HadCRUT5": "#999999",
        "NOAA": "#b8b8b8",
        "rec_ref": "#2f8f46",
    }
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True, constrained_layout=True)
    for name, values in curves.items():
        width = 2.3 if name in {"archived_R", "missingness_repaired_R"} else 1.0
        alpha = 1.0 if name in {"archived_R", "missingness_repaired_R"} else 0.75
        axes[0].plot(years[selected], values[selected], color=colors[name], lw=width, alpha=alpha, label=name)
        axes[1].plot(years[selected], rolling(values)[selected], color=colors[name], lw=width, alpha=alpha, label=name)
    for axis in axes:
        axis.axvspan(1901, 1950, color="#eeeeee", zorder=-2)
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA annual temperature: residual missingness repair")
    axes[1].set_title("Centered 15-year mean")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_variance_diagnostic(frame: pd.DataFrame, output: Path) -> None:
    ratio = frame.corrected_error_variance / frame.archived_error_variance
    size = 18 + 90 * np.minimum(np.abs(frame.repaired_hma_trend_contribution_c_per_century), 1.0)
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    scatter = axis.scatter(frame.missing_calibration_count, ratio, c=frame.repaired_minus_archived_hma_trend_contribution_c_per_century, s=size, cmap="RdBu_r", vmin=-0.5, vmax=0.5, alpha=0.75, edgecolors="none")
    axis.axhline(1.0, color="#555555", lw=0.8)
    axis.set(xlabel="Missing calibration years (1940-2000)", ylabel="Corrected / archived EnKF error variance", title="Residual missingness changes proxy error weights")
    axis.grid(alpha=0.18)
    figure.colorbar(scatter, ax=axis, label="Change in 1901-1950 HMA trend contribution (degC/century)")
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
    paths = {
        "proxy9": archive / source["proxy9"],
        "prior": root / source["prior"],
        "indices": archive / source["indices"],
        "archived_regem": root / source["archived_regem"],
        "baseline_driver": root / source["baseline_driver"],
        "baseline_arrays": root / source["baseline_arrays"],
        "network_audit": root / source["network_audit"],
        "hma_mask": root / source["hma_mask"],
        "instrumental_netcdf": Path(source["instrumental_netcdf"]),
        "regem_input": output / "regem_input.mat",
        "corrected_regem": output / "regem_output.mat",
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("controlled_enkf", paths["baseline_driver"])
    jja = load_module("jja_evaluation", paths["baseline_driver"].parents[1] / "climate_reconstruction_jja_temperature_network_repair_v1" / "run_experiment.py")

    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    archived_residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    corrected_residuals = np.asarray(scipy.io.loadmat(paths["corrected_regem"])["R0v_filled"], dtype=float)
    prepared = scipy.io.loadmat(paths["regem_input"])
    repaired_input = np.asarray(prepared["R0v"], dtype=float)
    archived_input = np.asarray(prepared["R0v_archived"], dtype=float)
    availability = np.asarray(prepared["observation_available"], dtype=bool)
    observed = np.isfinite(repaired_input)
    observed_change = float(np.max(np.abs(corrected_residuals[observed] - repaired_input[observed])))

    mask05 = np.load(paths["hma_mask"])
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    grid_lat0, grid_lon0, _, _ = base._state_coordinates()
    mask2 = jja.source_grid(mask05[None].astype(float), source_lat, source_lon, grid_lat0, grid_lon0)[0] >= 0.5
    grid_lat, grid_lon, regional_weights = hma_weights(base, mask2)
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)
    branches = {
        "archived_R": base._run_branch(prior, samples, proxy, archived_residuals, target_weights, regional_weights),
        "missingness_repaired_R": base._run_branch(prior, samples, proxy, corrected_residuals, target_weights, regional_weights),
    }
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(branches["archived_R"]["analysis"]) - saved["proxy9_analysis"])))
    enkf_closure = max(float(branch[key]) for branch in branches.values() for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c"))

    reference_period = tuple(cfg["periods"]["reference"])
    regional = {name: anomaly(np.asarray(result["regional_analysis"]), years, reference_period) for name, result in branches.items()}
    cubes = {name: jja.anomaly(jja.to_cube(np.asarray(result["analysis"]), len(grid_lat), len(grid_lon)), years, reference_period) for name, result in branches.items()}
    observations: dict[str, np.ndarray] = {}
    observations_cube: dict[str, np.ndarray] = {}
    variable_names = {
        "Berkeley Earth": "tas_obs_bea_05deg",
        "CRU": "tas_obs_cru_tmp_05deg",
        "GISTEMP": "tas_obs_GIST_05deg",
        "HadCRUT5": "tas_obs_had5_05deg",
        "NOAA": "tas_obs_NOAA_05deg",
        "rec_ref": "tas_ensemble_mean_rec_ref",
    }
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        for label, variable in variable_names.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = jja.anomaly(jja.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, reference_period)
            observations_cube[label] = field
            observations[label] = jja.hma_mean(field, grid_lat, mask2)
    regional.update(observations)
    pd.DataFrame({"year": years, **regional}).to_csv(output / "hma_regional_series.csv", index=False)
    plot_hma_curves(years, regional, output / "figure_hma_temperature_curves_1850_2000.png", tuple(cfg["periods"]["plot"]))

    skill_rows = []
    for branch_name, field in cubes.items():
        for observation_name in ("Berkeley Earth", "CRU", "GISTEMP", "HadCRUT5", "NOAA"):
            for period_name in ("early", "late"):
                start, end = cfg["periods"][period_name]
                selected = (years >= start) & (years <= end)
                row = {"case": branch_name, "observation": observation_name, "period": f"{start}-{end}"}
                row.update(jja.metrics(field[selected], observations_cube[observation_name][selected], years[selected], mask2))
                skill_rows.append(row)
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(output / "instrumental_skill_summary_2deg.csv", index=False)

    audit = pd.read_csv(paths["network_audit"])
    preflight = pd.read_csv(output / "residual_missingness_preflight.csv")
    early = (years >= cfg["periods"]["early"][0]) & (years <= cfg["periods"]["early"][1])
    contribution_columns = {}
    for branch_name, result in branches.items():
        contributions = np.asarray(result["regional_contributions"])
        innovations = np.asarray(result["innovations"])
        weights = np.asarray(result["weights"])
        contribution_columns[f"{branch_name}_hma_mean_contribution_c"] = np.nanmean(contributions[:, early], axis=1)
        contribution_columns[f"{branch_name}_hma_trend_contribution_c_per_century"] = np.asarray([trend(years[early], row[early]) for row in contributions])
        contribution_columns[f"{branch_name}_innovation_mean"] = np.nanmean(innovations[:, early], axis=1)
        contribution_columns[f"{branch_name}_weight_rms"] = np.sqrt(np.nanmean(weights[:, early] ** 2, axis=1))
    record = audit.merge(preflight, on="proxy9_record_1based", how="validate")
    for name, values in contribution_columns.items():
        record[name] = values
    record["archived_error_variance"] = np.var(archived_residuals, axis=1, ddof=1)
    record["corrected_error_variance"] = np.var(corrected_residuals, axis=1, ddof=1)
    record["corrected_to_archived_error_variance_ratio"] = record.corrected_error_variance / record.archived_error_variance
    record["repaired_minus_archived_hma_mean_contribution_c"] = record.missingness_repaired_R_hma_mean_contribution_c - record.archived_R_hma_mean_contribution_c
    record["repaired_minus_archived_hma_trend_contribution_c_per_century"] = record.missingness_repaired_R_hma_trend_contribution_c_per_century - record.archived_R_hma_trend_contribution_c_per_century
    record.to_csv(output / "proxy9_record_calibration_error_contribution_diagnostics.csv", index=False)
    plot_variance_diagnostic(record, output / "figure_error_variance_and_hma_contribution_change.png")

    period_rows = []
    for name in ("archived_R", "missingness_repaired_R", "Berkeley Earth", "CRU", "GISTEMP", "HadCRUT5", "NOAA", "rec_ref"):
        for period_name in ("early", "late"):
            start, end = cfg["periods"][period_name]
            period_rows.append({"series": name, "period": f"{start}-{end}", **period_metrics(regional[name], years, (start, end))})
    period_table = pd.DataFrame(period_rows)
    period_table.to_csv(output / "hma_period_summary.csv", index=False)

    principal = skill[~skill.observation.eq("NOAA")]
    skill_summary = principal.groupby(["case", "period"])[["median_lowpass_correlation_15y", "median_rmse", "median_abs_trend_difference"]].mean().reset_index()
    skill_summary.to_csv(output / "instrumental_skill_principal_mean.csv", index=False)
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    source_pure = hashes_before == hashes_after
    gates = cfg["gates"]
    passed = bool(
        observed_change <= float(gates["observed_residual_max_abs_change"])
        and int(np.isnan(corrected_residuals).sum()) == int(gates["regem_output_nan_count"])
        and baseline_state_closure <= float(gates["baseline_state_closure_max_abs_c"])
        and enkf_closure <= float(gates["enkf_closure_max_abs_c"])
        and source_pure
    )
    early_archived = period_table[(period_table.series == "archived_R") & (period_table.period == "1901-1950")].iloc[0]
    early_repaired = period_table[(period_table.series == "missingness_repaired_R") & (period_table.period == "1901-1950")].iloc[0]
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if passed else "fail",
        "controlled_change": cfg["rules"]["only_changed_freedom"],
        "preflight": {
            "structural_zero_entries_replaced": int((~availability & np.isclose(archived_input, 0.0, atol=1e-15)).sum()),
            "affected_records": int((~availability).any(axis=1).sum()),
            "observed_residual_max_abs_change_after_regem": observed_change,
            "corrected_regem_nan_count": int(np.isnan(corrected_residuals).sum()),
        },
        "closure": {"baseline_state_max_abs_c": baseline_state_closure, "enkf_max_abs_c": enkf_closure},
        "early_hma": {
            "archived_mean_c": float(early_archived.mean_c),
            "repaired_mean_c": float(early_repaired.mean_c),
            "change_c": float(early_repaired.mean_c - early_archived.mean_c),
            "archived_trend_c_per_century": float(early_archived.trend_c_per_century),
            "repaired_trend_c_per_century": float(early_repaired.trend_c_per_century),
            "trend_change_c_per_century": float(early_repaired.trend_c_per_century - early_archived.trend_c_per_century),
        },
        "records": {
            "variance_ratio_below_0p8": int((record.corrected_to_archived_error_variance_ratio > 1.25).sum()),
            "largest_absolute_hma_trend_contribution_changes": record.reindex(record.repaired_minus_archived_hma_trend_contribution_c_per_century.abs().sort_values(ascending=False).index).head(12)[["proxy9_record_1based", "combined_name", "missing_calibration_count", "corrected_to_archived_error_variance_ratio", "repaired_minus_archived_hma_trend_contribution_c_per_century"]].to_dict(orient="records"),
        },
        "source_hashes_unchanged": source_pure,
        "interpretation_boundary": "This isolates residual missingness encoding. It does not test a new calibration window, seasonality, coefficient sign, proxy deletion, or a new prior.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Residual Missingness Repair v1\n\n"
        f"Status: **{summary['status']}**.\n\n"
        "This controlled experiment changes only the encoding of missing PSM residual years before rerunning the archived RegEM completion. "
        "See `summary.json`, `proxy9_record_calibration_error_contribution_diagnostics.csv`, and the HMA curve figure.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("hard-gate-failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
