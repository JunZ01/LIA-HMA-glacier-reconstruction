"""Run delta-anchored calibration-window EnKF branches and HMA validation."""

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


def load_v73(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name]) for name in handle.keys() if not name.startswith("#")}


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


def plot_curves(years: np.ndarray, curves: dict[str, np.ndarray], output: Path, period: tuple[int, int]) -> None:
    selected = (years >= period[0]) & (years <= period[1])
    colors = {
        "archived_1940_2000": "#111111",
        "delta_anchor_1901_1961": "#2166ac",
        "delta_anchor_1920_1980": "#b2182b",
        "Berkeley Earth": "#4d4d4d",
        "CRU": "#6c6c6c",
        "GISTEMP": "#898989",
        "HadCRUT5": "#a6a6a6",
        "NOAA": "#c4c4c4",
        "rec_ref": "#2f8f46",
    }
    figure, axes = plt.subplots(2, 1, figsize=(12.7, 8.3), sharex=True, constrained_layout=True)
    for name, values in curves.items():
        branch = name.startswith("archived") or name.startswith("delta_anchor")
        width = 2.2 if branch else 1.0
        alpha = 1.0 if branch else 0.72
        axes[0].plot(years[selected], values[selected], color=colors[name], lw=width, alpha=alpha, label=name)
        axes[1].plot(years[selected], rolling(values)[selected], color=colors[name], lw=width, alpha=alpha, label=name)
    for axis in axes:
        axis.axvspan(1901, 1950, color="#eeeeee", zorder=-2)
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA annual temperature under equal-length PSM calibration windows")
    axes[1].set_title("Centered 15-year mean")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=5, fontsize=7.8)
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
        "window_summary": root / source["window_summary"],
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

    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    proxies = {"archived_1940_2000": {key: np.asarray(value).copy() for key, value in proxy.items()}}
    for name, window_index in (("delta_anchor_1901_1961", 0), ("delta_anchor_1920_1980", 1)):
        item = {key: np.asarray(value).copy() for key, value in proxy.items()}
        item["hxb"] = proxy["hxb"] + (fresh[:, window_index] - fresh[:, 2])
        proxies[name] = item

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
    results = {name: base._run_branch(prior, samples, item, residuals, target_weights, regional_weights) for name, item in proxies.items()}
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(results["archived_1940_2000"]["analysis"]) - saved["proxy9_analysis"])))
    enkf_closure = max(float(result[key]) for result in results.values() for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c"))
    if baseline_state_closure > float(cfg["gates"]["baseline_state_closure_max_abs_c"]) or enkf_closure > float(cfg["gates"]["enkf_closure_max_abs_c"]):
        raise RuntimeError(f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    reference = tuple(cfg["periods"]["reference"])
    regional = {name: anomaly(np.asarray(result["regional_analysis"]), years, reference) for name, result in results.items()}
    cubes = {name: evaluation.anomaly(evaluation.to_cube(np.asarray(result["analysis"]), len(grid_lat), len(grid_lon)), years, reference) for name, result in results.items()}
    observations: dict[str, np.ndarray] = {}
    observation_cubes: dict[str, np.ndarray] = {}
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
            field = evaluation.anomaly(evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, reference)
            observation_cubes[label] = field
            observations[label] = evaluation.hma_mean(field, grid_lat, mask2)
    curves = {**regional, **observations}
    pd.DataFrame({"year": years, **curves}).to_csv(output / "hma_regional_series.csv", index=False)
    plot_curves(years, curves, output / "figure_hma_calibration_window_curves_1850_2000.png", tuple(cfg["periods"]["plot"]))

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
    principal = skill[~skill.observation.eq("NOAA")].groupby(["case", "period"])[["median_lowpass_correlation_15y", "median_rmse", "median_abs_trend_difference"]].mean().reset_index()
    principal.to_csv(output / "instrumental_skill_principal_mean.csv", index=False)

    period_rows = []
    for name, values in curves.items():
        for period_name in ("early", "late"):
            start, end = cfg["periods"][period_name]
            period_rows.append({"series": name, "period": f"{start}-{end}", **period_metrics(values, years, (start, end))})
    periods = pd.DataFrame(period_rows)
    periods.to_csv(output / "hma_period_summary.csv", index=False)

    contribution_rows = []
    for name, result in results.items():
        contribution = np.asarray(result["regional_contributions"])
        for start, end in ((1901, 1950), (1951, 2000)):
            selected = (years >= start) & (years <= end)
            contribution_rows.append({
                "branch": name,
                "period": f"{start}-{end}",
                "sum_proxy_mean_c": float(np.mean(contribution[:, selected].sum(axis=0))),
                "sum_proxy_trend_c_per_century": float(100.0 * np.polyfit(years[selected], contribution[:, selected].sum(axis=0), 1)[0]),
            })
    pd.DataFrame(contribution_rows).to_csv(output / "hma_increment_summary.csv", index=False)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if pure else "fail",
        "window_repeat_hxb_max_abs": repeat_closure,
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "source_hashes_unchanged": pure,
        "hma_period_summary": periods.to_dict(orient="records"),
        "principal_instrumental_skill": principal.to_dict(orient="records"),
        "interpretation_boundary": "Only the PSM calibration-window operator is changed. Archived RegEM error variances are frozen, so this does not combine the residual-missingness correction with the window test.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Calibration-Window EnKF v1\n\n"
        f"Status: **{summary['status']}**. This is a delta-anchored operator-only sensitivity with archived error variances frozen.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
