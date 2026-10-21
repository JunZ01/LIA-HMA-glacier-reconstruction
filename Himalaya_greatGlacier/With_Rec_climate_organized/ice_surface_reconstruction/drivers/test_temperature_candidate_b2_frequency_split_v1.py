from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
EVALUATION = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluation_v3"
STATIONS = EVALUATION / "station_year_candidate_b2_values.csv"
CURVES = EVALUATION / "candidate_b2_long_term_hma_curves_allowed.csv"
OUTPUT = LOOP / "temp_b2_frequency_split_v1"
WINDOWS = (21, 31, 51)
NETWORKS = ("candidate_proxy6", "candidate_proxy9", "candidate_joint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def paired(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(observed) & np.isfinite(predicted)
    obs = np.asarray(observed, dtype=float)[finite]
    pred = np.asarray(predicted, dtype=float)[finite]
    error = pred - obs
    correlation = float(pearsonr(obs, pred).statistic) if len(obs) >= 3 and np.std(pred) > 0 else np.nan
    return {
        "n": int(len(obs)),
        "rmse_degC": float(np.sqrt(np.mean(error**2))),
        "bias_degC": float(np.mean(error)),
        "correlation_r": correlation,
        "sign_agreement": float(np.mean(np.sign(obs) == np.sign(pred))),
    }


def trend(frame: pd.DataFrame, column: str) -> float:
    valid = frame[["year", column]].dropna()
    return float(linregress(valid["year"], valid[column]).slope * 100.0)


def decadal(frame: pd.DataFrame, observation: str, prediction: str) -> pd.DataFrame:
    rows = []
    for start in range(1940, 2000, 10):
        subset = frame[frame["year"].between(start, start + 9)]
        for station_id, group in subset.groupby("station_id"):
            valid = group[[observation, prediction]].dropna()
            if len(valid) >= 6:
                rows.append({"station_id": station_id, "observed": valid[observation].mean(), "predicted": valid[prediction].mean()})
    return pd.DataFrame(rows)


def highpass_by_station(frame: pd.DataFrame, branch: str, window: int) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("station_id"):
        ordered = group.sort_values("year")
        delta = ordered[branch] - ordered["balanced_background"]
        filtered = np.full(len(ordered), np.nan, dtype=float)
        for start, end in ((1500, 1920), (1940, 2000)):
            mask = ordered["year"].between(start, end).to_numpy()
            if not mask.any():
                continue
            values = delta.to_numpy()[mask]
            lowpass = pd.Series(values).rolling(window, center=True, min_periods=max(3, window // 2)).mean().to_numpy()
            filtered[mask] = values - lowpass
        output.loc[ordered.index] = ordered["balanced_background"].to_numpy() + filtered
    return output


def highpass_curve(curves: pd.DataFrame, branch: str, window: int) -> np.ndarray:
    delta = curves[branch] - curves["balanced_background"]
    result = np.full(len(curves), np.nan, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        mask = curves["year"].between(start, end).to_numpy()
        values = delta.to_numpy()[mask]
        lowpass = pd.Series(values).rolling(window, center=True, min_periods=max(3, window // 2)).mean().to_numpy()
        result[mask] = curves.loc[mask, "balanced_background"].to_numpy() + values - lowpass
    return result


def lowpass(values: np.ndarray, years: np.ndarray, window: int = 31) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        mask = (years >= start) & (years <= end)
        result[mask] = pd.Series(values[mask]).rolling(window, center=True, min_periods=max(3, window // 2)).mean().to_numpy()
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    stations = pd.read_csv(STATIONS)
    curves = pd.read_csv(CURVES)
    if stations["year"].between(1921, 1939).any() or curves["year"].between(1921, 1939).any():
        raise RuntimeError("sealed years present")

    candidates = []
    for window in WINDOWS:
        for branch in NETWORKS:
            name = f"{branch}_hp{window}"
            stations[name] = highpass_by_station(stations, branch, window)
            candidates.append((window, branch, name))

    metric_rows = []
    modern = stations[stations["year"].between(1940, 2000)]
    for observation in ("qcf", "qcu"):
        for window, branch, name in candidates:
            valid = modern[["year", "station_id", observation, name]].dropna()
            annual = paired(valid[observation].to_numpy(), valid[name].to_numpy())
            decades = decadal(modern, observation, name)
            decadal_metric = paired(decades["observed"].to_numpy(), decades["predicted"].to_numpy())
            metric_rows.append(
                {
                    "observation": observation,
                    "window_years": window,
                    "source_branch": branch,
                    **{f"annual_{key}": value for key, value in annual.items()},
                    **{f"decadal_{key}": value for key, value in decadal_metric.items()},
                    "observed_trend_degC_per_century": trend(valid, observation),
                    "predicted_trend_degC_per_century": trend(valid, name),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "frequency_split_instrumental_metrics.csv", index=False)

    years = curves["year"].to_numpy(int)
    reference = lowpass(curves["external_ref_mean_degC"].to_numpy(float), years)
    long_rows = []
    for window, branch, name in candidates:
        values = highpass_curve(curves, branch, window)
        curves[name] = values
        smooth_values = lowpass(values, years)
        for start, end in ((1500, 1849), (1500, 1879), (1850, 1920)):
            period = (years >= start) & (years <= end) & np.isfinite(reference) & np.isfinite(smooth_values)
            error = smooth_values[period] - reference[period]
            long_rows.append(
                {
                    "period": f"{start}-{end}",
                    "window_years": window,
                    "source_branch": branch,
                    "lowpass_rmse_vs_external_ref_degC": float(np.sqrt(np.mean(error**2))),
                    "lowpass_bias_vs_external_ref_degC": float(np.mean(error)),
                }
            )
    long_term = pd.DataFrame(long_rows)
    long_term.to_csv(OUTPUT / "frequency_split_long_term_metrics.csv", index=False)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True, constrained_layout=True)
    for start, end in ((1500, 1920), (1940, 2000)):
        period = curves["year"].between(start, end).to_numpy()
        top.plot(years[period], lowpass(curves["balanced_background"].to_numpy(float), years)[period], color="#777777", linewidth=1.5, label="Balanced background" if start == 1500 else None)
        top.plot(years[period], lowpass(curves["candidate_joint"].to_numpy(float), years)[period], color="#C33C2E", linewidth=2.0, label="Full joint posterior" if start == 1500 else None)
        top.plot(years[period], lowpass(curves["candidate_joint_hp31"].to_numpy(float), years)[period], color="#2878B5", linewidth=2.0, label="Joint high-pass increment (31y)" if start == 1500 else None)
        top.plot(years[period], reference[period], color="#7A4EAB", linewidth=1.6, label="External reference mean" if start == 1500 else None)
        top.plot(years[period], curves["instrument_mean_degC_smooth11"].to_numpy(float)[period], color="#E08B27", linewidth=1.5, label="Instrumental mean" if start == 1500 else None)
    top.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    top.set(ylabel="Temperature anomaly (degC)", title="Candidate B2 frequency-split mechanism sensitivity")
    top.legend(frameon=False, ncol=3, fontsize=8)
    top.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    qcf = metrics[(metrics["observation"] == "qcf") & (metrics["source_branch"] == "candidate_joint")]
    bottom.plot(qcf["window_years"], qcf["annual_correlation_r"], marker="o", label="Annual r")
    bottom.plot(qcf["window_years"], qcf["decadal_correlation_r"], marker="s", label="Decadal r")
    bottom.set(xlabel="High-pass cutoff window (years)", ylabel="Correlation", xticks=list(WINDOWS))
    bottom.grid(color="#DDDDDD", linewidth=0.6)
    bottom.legend(frameon=False, ncol=2)
    fig.savefig(OUTPUT / "candidate_b2_frequency_split_sensitivity.png", dpi=220)
    plt.close(fig)

    joint_qcf = metrics[(metrics["observation"] == "qcf") & (metrics["source_branch"] == "candidate_joint")].copy()
    robust_long_term = bool(
        (long_term[(long_term["period"] == "1500-1879") & (long_term["source_branch"] == "candidate_joint")]["lowpass_rmse_vs_external_ref_degC"] < 0.24).all()
    )
    modern_r_retained = bool((joint_qcf["annual_correlation_r"] >= 0.40).all())
    modern_trend_restored = bool((joint_qcf["predicted_trend_degC_per_century"] >= 0.75).all())
    summary = {
        "experiment": OUTPUT.name,
        "status": "complete",
        "windows_years": list(WINDOWS),
        "long_term_background_skill_robustly_retained": robust_long_term,
        "modern_annual_correlation_partly_retained": modern_r_retained,
        "modern_trend_restored": modern_trend_restored,
        "finding": "frequency splitting tests whether proxy covariance skill can be separated from the low-frequency anchor without tuning a blend weight",
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "enkf_rerun": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate B2 frequency-split sensitivity\n\n"
        f"Long-term background skill retained for all 21/31/51-year windows: `{robust_long_term}`.\n\n"
        f"Modern annual correlation at least 0.40 for all windows: `{modern_r_retained}`.\n\n"
        f"Modern trend at least 0.75 degC/century for all windows: `{modern_trend_restored}`.\n\n"
        "This is a mechanism sensitivity, not a promoted reconstruction or a tuned production correction.\n",
        encoding="utf-8",
    )
    manifest = {
        "inputs": {str(STATIONS): sha256(STATIONS), str(CURVES): sha256(CURVES)},
        "outputs": {str(path): sha256(path) for path in sorted(OUTPUT.iterdir()) if path.name != "source_manifest.json"},
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(joint_qcf.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
