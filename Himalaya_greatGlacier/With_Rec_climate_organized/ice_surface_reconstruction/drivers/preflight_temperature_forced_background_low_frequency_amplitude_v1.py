"""Test full forced-mean low-frequency amplitude with period-constant corrections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import linregress

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2"
)
EARLY = LOOP / (
    "temperature_forced_background_member_station_diagnostic_v1/"
    "station_year_background_members_allowed.csv"
)
MODERN = LOOP / (
    "temperature_forced_background_member_station_1940_1950_validation_v1/"
    "station_year_background_members_1940_1950.csv"
)
OUTPUT = LOOP / "temperature_forced_background_low_frequency_amplitude_preflight_v1"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
PERIODS = {"1880-1900": (1880, 1900), "1901-1920": (1901, 1920), "1940-1950": (1940, 1950)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slope(year: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 8:
        return np.nan
    return float(linregress(year[valid], values[valid]).slope * 100.0)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# Forced-background low-frequency amplitude preflight\n\n"
        "Use the equal three-model forced background only. Within each independent allowed "
        "validation segment, replace annual forced-minus-archived differences by their station-"
        "specific period mean, add that constant to the archived posterior, and compare with qcf. "
        "This removes model weather and preserves every original within-period trend exactly. "
        "Require both early segments to improve absolute mean bias and median station MAE, at least "
        "half of stations to improve, 1940-1950 median MAE regression no worse than 5%, absolute "
        "1940-1950 mean-bias regression no worse than 0.05 degC, and trend closure <=1e-10. "
        "No smoothing or bridge is designed here. No 1921-1939, EnKF, production, or OGGM access.\n",
        encoding="utf-8",
    )
    sources = {"early": EARLY, "modern": MODERN, "facts": FACTS, "driver": DRIVER, "contract": contract}
    before = {name: sha256(path) for name, path in sources.items()}
    values = pd.concat([pd.read_csv(EARLY), pd.read_csv(MODERN)], ignore_index=True)
    if values.duplicated(["station_id", "year"]).any():
        values = values.sort_values(["station_id", "year"]).drop_duplicates(["station_id", "year"], keep="last")
    if values["year"].between(1921, 1939).any():
        raise RuntimeError("Locked year present in derived input")

    station_rows = []
    annual_rows = []
    for period, (start, end) in PERIODS.items():
        segment = values[values["year"].between(start, end)].copy()
        for station_id, group in segment.groupby("station_id"):
            observed = group["qcf_station_degC"].to_numpy(float)
            original = group["project_six_member_degC"].to_numpy(float)
            delta = group["forced_mean_background_degC"].to_numpy(float) - group["archived_background_degC"].to_numpy(float)
            valid_delta = np.isfinite(delta)
            if not np.any(valid_delta):
                continue
            constant_delta = float(np.mean(delta[valid_delta]))
            candidate = original + constant_delta
            year = group["year"].to_numpy(float)
            valid = np.isfinite(observed) & np.isfinite(original) & np.isfinite(candidate)
            if int(valid.sum()) < 5:
                continue
            observed_trend = slope(year[valid], observed[valid])
            original_trend = slope(year[valid], original[valid])
            candidate_trend = slope(year[valid], candidate[valid])
            station_rows.append({
                "period": period,
                "station_id": station_id,
                "station_name": group["station_name"].iloc[0],
                "n_years": int(valid.sum()),
                "constant_forced_mean_background_correction_degC": constant_delta,
                "original_mean_bias_degC": float(np.mean(original[valid] - observed[valid])),
                "candidate_mean_bias_degC": float(np.mean(candidate[valid] - observed[valid])),
                "original_mae_degC": float(np.mean(np.abs(original[valid] - observed[valid]))),
                "candidate_mae_degC": float(np.mean(np.abs(candidate[valid] - observed[valid]))),
                "original_abs_trend_error": abs(original_trend - observed_trend),
                "candidate_abs_trend_error": abs(candidate_trend - observed_trend),
                "trend_closure_degC_per_century": abs(candidate_trend - original_trend),
            })
            for index in np.flatnonzero(valid):
                annual_rows.append({
                    "period": period, "station_id": station_id, "year": int(year[index]),
                    "qcf_station_degC": observed[index], "original_degC": original[index],
                    "candidate_degC": candidate[index], "constant_correction_degC": constant_delta,
                })
    station = pd.DataFrame(station_rows)
    aggregates = []
    for period, group in station.groupby("period"):
        original_bias = float(group["original_mean_bias_degC"].mean())
        candidate_bias = float(group["candidate_mean_bias_degC"].mean())
        original_mae = float(group["original_mae_degC"].median())
        candidate_mae = float(group["candidate_mae_degC"].median())
        aggregates.append({
            "period": period,
            "station_count": len(group),
            "original_equal_station_mean_bias_degC": original_bias,
            "candidate_equal_station_mean_bias_degC": candidate_bias,
            "absolute_bias_improvement_fraction": (abs(original_bias) - abs(candidate_bias)) / abs(original_bias),
            "original_median_station_mae_degC": original_mae,
            "candidate_median_station_mae_degC": candidate_mae,
            "median_station_mae_improvement_fraction": (original_mae - candidate_mae) / original_mae,
            "stations_improving_mae_fraction": float(np.mean(group["candidate_mae_degC"] < group["original_mae_degC"])),
            "maximum_trend_closure_degC_per_century": float(group["trend_closure_degC_per_century"].max()),
            "mean_constant_correction_degC": float(group["constant_forced_mean_background_correction_degC"].mean()),
        })
    aggregate = pd.DataFrame(aggregates).set_index("period")
    gates = {
        "early_absolute_bias_improves": bool(np.all(aggregate.loc[["1880-1900", "1901-1920"], "absolute_bias_improvement_fraction"] > 0)),
        "early_median_station_mae_improves": bool(np.all(aggregate.loc[["1880-1900", "1901-1920"], "median_station_mae_improvement_fraction"] > 0)),
        "early_majority_of_stations_improve": bool(np.all(aggregate.loc[["1880-1900", "1901-1920"], "stations_improving_mae_fraction"] >= 0.5)),
        "modern_median_station_mae_regression_no_worse_than_5pct": bool(aggregate.loc["1940-1950", "median_station_mae_improvement_fraction"] >= -0.05),
        "modern_absolute_mean_bias_regression_no_worse_than_0p05_degC": bool(
            abs(aggregate.loc["1940-1950", "candidate_equal_station_mean_bias_degC"])
            - abs(aggregate.loc["1940-1950", "original_equal_station_mean_bias_degC"]) <= 0.05
        ),
        "trend_closure_at_most_1e_10": bool(aggregate["maximum_trend_closure_degC_per_century"].max() <= 1e-10),
    }
    passes = bool(all(gates.values()))
    decision = (
        "authorize_full_forced_mean_low_frequency_spatial_candidate_design"
        if passes else "reject_full_forced_mean_low_frequency_amplitude_before_spatial_candidate"
    )
    station.to_csv(OUTPUT / "station_period_metrics.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(OUTPUT / "station_year_counterfactual.csv", index=False)
    aggregate.reset_index().to_csv(OUTPUT / "aggregate_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    x = np.arange(len(aggregate))
    axes[0].bar(x - 0.18, aggregate["original_equal_station_mean_bias_degC"], 0.36, color="#D55E00", label="Original")
    axes[0].bar(x + 0.18, aggregate["candidate_equal_station_mean_bias_degC"], 0.36, color="#0072B2", label="Full forced low-frequency")
    axes[0].set_xticks(x, aggregate.index, rotation=25, ha="right")
    axes[0].set_ylabel("Equal-station mean bias (degC)")
    axes[0].legend(frameon=False)
    axes[1].bar(x, aggregate["median_station_mae_improvement_fraction"] * 100, color="#009E73")
    axes[1].axhline(0, color="#555555", lw=0.8)
    axes[1].set_xticks(x, aggregate.index, rotation=25, ha="right")
    axes[1].set_ylabel("Median station MAE improvement (%)")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "full_forced_low_frequency_amplitude_preflight.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "status": "complete", "experiment": "temperature_forced_background_low_frequency_amplitude_preflight_v1",
        "decision": decision, "passes_all_frozen_gates": passes,
        "aggregate": aggregate.reset_index().set_index("period").to_dict(orient="index"),
        "gates": gates, "locked_1921_1939_numeric_values_accessed": 0,
        "all_declared_sources_unchanged": unchanged, "production_authorized": False,
        "enkf_run": False, "oggm_run": False,
        "interpretation_boundary": "Passing authorizes only construction of a smooth full-forced-mean low-frequency spatial diagnostic candidate.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = ["# Full Forced-Mean Low-Frequency Amplitude Preflight", "", f"Decision: `{decision}`.", ""]
    for period, row in aggregate.iterrows():
        lines.append(
            f"- {period}: bias `{row.original_equal_station_mean_bias_degC:+.3f}` to "
            f"`{row.candidate_equal_station_mean_bias_degC:+.3f} degC`; median MAE change "
            f"`{row.median_station_mae_improvement_fraction:+.1%}`; correction "
            f"`{row.mean_constant_correction_degC:+.3f} degC`."
        )
    lines.extend(["", f"- All frozen gates pass: `{str(passes).lower()}`.", "- Trend closure is algebraic; no model annual weather is inserted.", "- No 1921-1939 value, EnKF, production file, or OGGM run was used."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
