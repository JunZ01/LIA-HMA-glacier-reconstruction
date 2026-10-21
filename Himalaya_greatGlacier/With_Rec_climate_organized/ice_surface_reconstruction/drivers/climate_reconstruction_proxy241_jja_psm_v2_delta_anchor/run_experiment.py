"""Run the fixed-network proxy-241 JJA PSM delta-anchor experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("controlled_enkf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("baseline-driver-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.nanmax(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def trend(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    x = years[finite].astype(float)
    x -= x.mean()
    return float(100.0 * np.polyfit(x, values[finite], 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    archive = Path(cfg["archive_root"])
    src = cfg["sources"]
    paths = {
        "baseline_driver": root / src["baseline_driver"],
        "baseline_arrays": root / src["baseline_arrays"],
        "prior": root / src["prior"],
        "indices": archive / src["indices"],
        "proxy9": archive / src["proxy9"],
        "proxy9_regem": root / src["proxy9_regem"],
        "combined_proxy": Path(src["combined_proxy"]),
        "reference_netcdf": Path(src["reference_netcdf"]),
    }
    output = root / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    matlab_output = output / "proxy241_psm_delta_anchor.mat"
    if not matlab_output.exists():
        raise FileNotFoundError(f"missing-matlab-output:{matlab_output}")

    hashes_before = {name: sha256(path) for name, path in paths.items()}
    psm = scipy.io.loadmat(matlab_output, squeeze_me=True)
    gates = cfg["gates"]
    preflight = {
        "annual_data_max_abs": max_abs(psm["annual_data"], psm["baseline_data"]),
        "annual_b_max_abs": max_abs(psm["annual_b"], psm["baseline_b"]),
        "fresh_annual_hxb_max_abs": max_abs(psm["annual_hxb"], psm["baseline_hxb"]),
        "annual_mt_max_abs": max_abs(psm["annual_mt"], psm["baseline_mt"]),
        "fresh_annual_mp_max_abs": max_abs(psm["annual_mp"], psm["baseline_mp"]),
        "annual_residual_max_abs": max_abs(psm["annual_rvec"], psm["baseline_rvec"]),
        "anchored_annual_hxb_max_abs": max_abs(psm["anchored_annual_hxb"], psm["baseline_hxb"]),
        "common_rng_hxb_repeat_max_abs": max_abs(psm["annual_repeat_hxb"], psm["annual_hxb"]),
        "common_rng_mp_repeat_max_abs": max_abs(psm["annual_repeat_mp"], psm["annual_mp"]),
        "jja_hxb_delta_max_abs": float(np.nanmax(np.abs(psm["jja_hxb_delta"]))),
        "jja_hxb_delta_rms": float(np.sqrt(np.nanmean(np.square(psm["jja_hxb_delta"])))),
    }
    preflight["status"] = "pass" if (
        preflight["annual_data_max_abs"] <= gates["annual_data_max_abs"]
        and preflight["annual_b_max_abs"] <= gates["annual_b_max_abs"]
        and preflight["annual_mt_max_abs"] <= gates["annual_predictor_max_abs"]
        and preflight["annual_residual_max_abs"] <= gates["annual_residual_max_abs"]
        and preflight["anchored_annual_hxb_max_abs"] <= gates["anchored_annual_hxb_max_abs"]
        and preflight["common_rng_hxb_repeat_max_abs"] <= gates["common_rng_repeat_max_abs"]
        and preflight["common_rng_mp_repeat_max_abs"] <= gates["common_rng_repeat_max_abs"]
    ) else "fail"
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps(preflight, indent=2))
    if preflight["status"] != "pass":
        raise RuntimeError("delta-anchor-preflight-failed")

    base = load_module(paths["baseline_driver"])
    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        indices = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["proxy9_regem"])["R0v_filled"], dtype=float)
    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    target_weights = base._target_weights(
        grid_lat, grid_lon, float(cfg["target"]["latitude"]), float(cfg["target"]["longitude"])
    )
    regional_weights = np.cos(np.deg2rad(state_lat))
    regional_weights /= regional_weights.sum()

    baseline = base._run_branch(prior, indices, proxy, residuals, target_weights, regional_weights)
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_difference = float(np.max(np.abs(np.asarray(baseline["analysis"]) - saved["proxy9_analysis"])))
    if baseline_state_difference > gates["baseline_state_max_abs_c"]:
        raise RuntimeError(f"baseline-state-closure-failed:{baseline_state_difference}")

    focus = int(cfg["focus_record_1based"]) - 1
    jja_proxy = {key: np.asarray(value).copy() for key, value in proxy.items()}
    jja_proxy["hxb"][focus] = np.asarray(psm["anchored_jja_hxb"], dtype=float).reshape(-1)
    jja_proxy["coefficients"][focus] = np.asarray(psm["jja_b"], dtype=float).reshape(-1)
    jja = base._run_branch(prior, indices, jja_proxy, residuals, target_weights, regional_weights)
    maximum_closure = max(
        float(result[key])
        for result in (baseline, jja)
        for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c")
    )
    if maximum_closure > gates["enkf_closure_max_abs_c"]:
        raise RuntimeError(f"enkf-closure-failed:{maximum_closure}")

    ref_period = tuple(cfg["periods"]["reference"])
    early_period = tuple(cfg["periods"]["early_20c"])
    early = (years >= early_period[0]) & (years <= early_period[1])
    with h5py.File(paths["reference_netcdf"], "r") as handle:
        lat = np.asarray(handle["lat"])
        lon = np.asarray(handle["lon"])
        reference = np.asarray(
            handle[src["reference_variable"]][
                :, np.flatnonzero(np.isclose(lat, cfg["target"]["latitude"]))[0], np.flatnonzero(np.isclose(lon, cfg["target"]["longitude"]))[0]
            ],
            dtype=float,
        )

    cases = {"baseline_annual": baseline, "record241_jja_delta_anchored": jja}
    target_curves: dict[str, np.ndarray] = {}
    regional_curves: dict[str, np.ndarray] = {}
    metrics = []
    for name, result in cases.items():
        target = base._anomaly(np.asarray(result["target_analysis"]), years, ref_period)
        regional = base._anomaly(np.asarray(result["regional_analysis"]), years, ref_period)
        target_curves[name] = target
        regional_curves[name] = regional
        metric, _ = base._metrics(
            name,
            years,
            target,
            reference,
            early_period,
            int(cfg["phase"]["lowpass_window_years"]),
            tuple(cfg["phase"]["lag_years"]),
        )
        metrics.append({
            "case": name,
            "target_trend_c_per_century": metric["trend_c_per_century"],
            "rolling15_correlation": metric["rolling15_correlation"],
            "best_lag_years": metric["best_lag_years"],
            "phase_projection": base._phase_projection(target[early], reference[early], int(cfg["phase"]["lowpass_window_years"])),
            "hma_trend_c_per_century": trend(years[early], regional[early]),
            "record241_target_contribution_trend_c_per_century": trend(
                years[early], np.asarray(result["target_contributions"])[focus, early]
            ),
        })
    metrics_table = pd.DataFrame(metrics)
    baseline_row = metrics_table.iloc[0]
    for column in metrics_table.columns[1:]:
        if pd.api.types.is_numeric_dtype(metrics_table[column]):
            metrics_table[f"delta_{column}"] = metrics_table[column] - baseline_row[column]
    metrics_table.to_csv(output / "metrics_1901_1950.csv", index=False)
    pd.DataFrame({"year": years, **target_curves, "rec_ref": reference}).to_csv(output / "target_3584_curves.csv", index=False)
    pd.DataFrame({"year": years, **regional_curves}).to_csv(output / "hma_curves.csv", index=False)

    selected = (years >= cfg["periods"]["plot"][0]) & (years <= cfg["periods"]["plot"][1])
    center = early
    window = int(cfg["phase"]["lowpass_window_years"])
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True, constrained_layout=True)
    colors = {"baseline_annual": "#222222", "record241_jja_delta_anchored": "#c44e52"}
    for name in target_curves:
        target_smooth = pd.Series(target_curves[name] - target_curves[name][center].mean()).rolling(window, center=True, min_periods=window).mean()
        regional_smooth = pd.Series(regional_curves[name] - regional_curves[name][center].mean()).rolling(window, center=True, min_periods=window).mean()
        axes[0].plot(years[selected], target_smooth.to_numpy()[selected], color=colors[name], lw=2.3, label=name)
        axes[1].plot(years[selected], regional_smooth.to_numpy()[selected], color=colors[name], lw=2.3, label=name)
    ref_smooth = pd.Series(reference - reference[center].mean()).rolling(window, center=True, min_periods=window).mean()
    axes[0].plot(years[selected], ref_smooth.to_numpy()[selected], color="#4c72b0", lw=2.0, ls="--", label="rec_ref comparison")
    for axis, title in zip(axes, ("3584: proxy 241 JJA PSM, delta anchored", "Cosine-weighted HMA: proxy 241 JJA PSM, delta anchored")):
        axis.axvspan(1901, 1950, color="#dddddd", alpha=0.3)
        axis.axhline(0, color="#777777", lw=0.7)
        axis.grid(alpha=0.2)
        axis.set_title(title)
        axis.set_ylabel("Temperature anomaly (degC)")
        axis.legend(frameon=False)
    axes[1].set_xlabel("Year")
    fig.savefig(output / "proxy241_jja_delta_anchor_sensitivity.png", dpi=220)
    plt.close(fig)

    psm_summary = {
        "baseline": {key: float(np.asarray(psm[f"baseline_{key}"]).squeeze()) for key in ("pcr", "pcr_t", "pcr_p", "pval")},
        "annual_replay": {key: float(np.asarray(psm[f"annual_{key}"]).squeeze()) for key in ("pcr", "pcr_t", "pcr_p", "pval")},
        "jja": {key: float(np.asarray(psm[f"jja_{key}"]).squeeze()) for key in ("pcr", "pcr_t", "pcr_p", "pval")},
        "baseline_b": np.asarray(psm["baseline_b"], dtype=float).reshape(-1).tolist(),
        "jja_b": np.asarray(psm["jja_b"], dtype=float).reshape(-1).tolist(),
    }
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    status = "pass" if pure else "fail"
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": status,
        "source_hashes_unchanged": pure,
        "preflight": preflight,
        "baseline_state_max_abs_c": baseline_state_difference,
        "maximum_enkf_closure_c": maximum_closure,
        "psm": psm_summary,
        "metrics": metrics_table.to_dict(orient="records"),
        "operator_strategy": cfg["operator_strategy"],
        "interpretation_boundary": "This tests only the proxy-241 JJA-minus-annual operator perturbation, anchored to the archived annual HXb. It does not validate rec_ref, invalidate the chronology, or retune the network error model.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    (output / "input_manifest.json").write_text(json.dumps({"before": hashes_before, "after": hashes_after}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy 241 JJA PSM Delta-Anchor Sensitivity\n\n"
        f"Status: **{status}**. Source hashes unchanged: **{pure}**.\n\n"
        "## Annual replay preflight\n\n"
        f"```json\n{json.dumps(preflight, indent=2)}\n```\n\n"
        "## PSM diagnostics\n\n"
        f"```json\n{json.dumps(psm_summary, indent=2)}\n```\n\n"
        "## 1901-1950 reconstruction metrics\n\n"
        f"{metrics_table.to_markdown(index=False)}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if status != "pass":
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
