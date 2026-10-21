from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress, pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import evaluate_temperature_candidate_b2_stage_c_v1 as evaluation
import temperature_candidate_b2_stage_c_core_v1 as core


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
STATIC_PACKAGE = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/"
    "candidate_b2_stage_c_static_input_v2.npz"
)
LONG_CURVES = evaluation.LONG_CURVES
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_no_proxy_background_evaluation_v1"


def smooth_segmented(values: np.ndarray, years: np.ndarray, size: int = 31) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        mask = (years >= start) & (years <= end)
        output[mask] = uniform_filter1d(values[mask], size=size, mode="reflect")
    return output


def curve_metrics(years: np.ndarray, estimate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(estimate) & np.isfinite(target)
    x = years[valid].astype(float)
    estimate = estimate[valid]
    target = target[valid]
    if len(x) < 10:
        return {name: np.nan for name in ("n", "rmse_degC", "mae_degC", "bias_degC", "correlation_r", "estimate_trend_degC_per_century", "target_trend_degC_per_century", "absolute_trend_error_degC_per_century")}
    error = estimate - target
    correlation = float(pearsonr(estimate, target).statistic) if np.std(estimate) > 0 and np.std(target) > 0 else np.nan
    estimate_trend = float(linregress(x, estimate).slope * 100.0)
    target_trend = float(linregress(x, target).slope * 100.0)
    return {
        "n": int(len(x)),
        "rmse_degC": float(np.sqrt(np.mean(error**2))),
        "mae_degC": float(np.mean(np.abs(error))),
        "bias_degC": float(np.mean(error)),
        "correlation_r": correlation,
        "estimate_trend_degC_per_century": estimate_trend,
        "target_trend_degC_per_century": target_trend,
        "absolute_trend_error_degC_per_century": abs(estimate_trend - target_trend),
    }


def station_metrics(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    regional = []
    modern = panel[panel["year"].between(1940, 2000)]
    for observation in ("qcf", "qcu"):
        for branch in ("balanced_background", "archived_project"):
            valid = modern[["station_id", "year", "region_group", observation, branch]].dropna()
            annual = evaluation.paired_metrics(valid[observation].to_numpy(), valid[branch].to_numpy())
            decades = evaluation.decadal_pairs(modern, observation, branch)
            decadal = evaluation.paired_metrics(
                decades.get("observed", pd.Series(dtype=float)).to_numpy(),
                decades.get("predicted", pd.Series(dtype=float)).to_numpy(),
            )
            rows.append(
                {
                    "observation": observation,
                    "branch": branch,
                    **{f"annual_{name}": value for name, value in annual.items()},
                    **{f"decadal_{name}": value for name, value in decadal.items()},
                    "observed_trend_degC_per_century": evaluation.trend(valid, observation),
                    "predicted_trend_degC_per_century": evaluation.trend(valid, branch),
                    "absolute_trend_error_degC_per_century": abs(
                        evaluation.trend(valid, observation) - evaluation.trend(valid, branch)
                    ),
                }
            )
            for region, group in valid.groupby("region_group"):
                regional.append(
                    {
                        "observation": observation,
                        "branch": branch,
                        "region_group": region,
                        **evaluation.paired_metrics(
                            group[observation].to_numpy(), group[branch].to_numpy()
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(regional)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    if core.sha256(STATIC_PACKAGE) != core.EXPECTED_STATIC_PACKAGE_SHA256:
        raise RuntimeError("static-package-hash")
    with np.load(STATIC_PACKAGE, allow_pickle=False) as package:
        years = np.asarray(package["years"], dtype=np.int32)
        background_field = evaluation.own_baseline(
            np.asarray(package["balanced_state_forced_mean"], dtype=float), years
        )
        weights = np.asarray(package["hma_weights"], dtype=float)
        valid = weights > 0
        background_hma = np.average(
            background_field[:, valid], axis=1, weights=weights[valid]
        )
        latitude = np.asarray(package["retained_state_latitude"], dtype=float)
        longitude = np.asarray(package["retained_state_longitude"], dtype=float)
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("sealed-year-present")

    long = pd.read_csv(LONG_CURVES)
    curves = pd.DataFrame(
        {"year": years, "balanced_background_degC": background_hma}
    ).merge(long, on="year", how="left", validate="one_to_one")
    curves["balanced_background_smooth31"] = smooth_segmented(
        curves["balanced_background_degC"].to_numpy(float), years
    )
    curves["original_project_smooth31_rebuilt"] = smooth_segmented(
        curves["original_project_degC"].to_numpy(float), years
    )
    curves["external_ref_smooth31_rebuilt"] = smooth_segmented(
        curves["external_ref_mean_degC"].to_numpy(float), years
    )
    curves["instrument_smooth11_rebuilt"] = smooth_segmented(
        curves["instrument_mean_degC"].to_numpy(float), years, size=11
    )

    metric_rows = []
    comparisons = (
        (1500, 1849, "external_reference", "external_ref_smooth31_rebuilt", 31),
        (1500, 1879, "external_reference", "external_ref_smooth31_rebuilt", 31),
        (1850, 1920, "external_reference", "external_ref_smooth31_rebuilt", 31),
        (1940, 2000, "instrumental_mean", "instrument_smooth11_rebuilt", 11),
    )
    for start, end, target_name, target_column, scale in comparisons:
        mask = curves["year"].between(start, end).to_numpy()
        for branch, column in (
            ("balanced_background", "balanced_background_smooth31"),
            ("archived_project", "original_project_smooth31_rebuilt"),
        ):
            metric_rows.append(
                {
                    "period": f"{start}-{end}",
                    "target": target_name,
                    "comparison_scale_years": scale,
                    "branch": branch,
                    **curve_metrics(
                        years[mask],
                        curves.loc[mask, column].to_numpy(float),
                        curves.loc[mask, target_column].to_numpy(float),
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)

    panel = evaluation.load_station_panel()
    panel, mapping = evaluation.add_candidate_values(
        panel,
        years,
        latitude,
        longitude,
        {"balanced_background": background_field},
    )
    station, regional = station_metrics(panel)

    qcf = station[station["observation"].eq("qcf")].set_index("branch")
    background_station = qcf.loc["balanced_background"]
    archived_station = qcf.loc["archived_project"]
    diagnostic = {
        "background_beats_archived_qcf_annual_rmse": bool(
            background_station["annual_rmse_degC"] < archived_station["annual_rmse_degC"]
        ),
        "background_beats_archived_qcf_annual_correlation": bool(
            background_station["annual_correlation_r"] > archived_station["annual_correlation_r"]
        ),
        "background_beats_archived_qcf_decadal_rmse": bool(
            background_station["decadal_rmse_degC"] < archived_station["decadal_rmse_degC"]
        ),
        "background_beats_archived_qcf_decadal_correlation": bool(
            background_station["decadal_correlation_r"] > archived_station["decadal_correlation_r"]
        ),
    }

    curves.to_csv(OUT / "balanced_background_long_term_hma_curves_allowed.csv", index=False)
    metrics.to_csv(OUT / "balanced_background_long_term_metrics.csv", index=False)
    panel.to_csv(OUT / "balanced_background_station_year_values.csv", index=False)
    mapping.to_csv(OUT / "station_to_background_state_mapping.csv", index=False)
    station.to_csv(OUT / "balanced_background_station_metrics.csv", index=False)
    regional.to_csv(OUT / "balanced_background_west_central_east_metrics.csv", index=False)

    fig, axis = plt.subplots(figsize=(13.2, 6.3), constrained_layout=True)
    for start, end in ((1500, 1920), (1940, 2000)):
        segment = curves[curves["year"].between(start, end)]
        axis.plot(segment["year"], segment["balanced_background_smooth31"], color="#C43C2B", linewidth=2.1, label="Balanced four-model background" if start == 1500 else None)
        axis.plot(segment["year"], segment["original_project_smooth31_rebuilt"], color="#111111", linewidth=1.2, label="Archived reconstruction" if start == 1500 else None)
        axis.plot(segment["year"], segment["external_ref_smooth31_rebuilt"], color="#6A51A3", linewidth=1.1, label="External reconstruction mean" if start == 1500 else None)
        axis.plot(segment["year"], segment["instrument_smooth11_rebuilt"], color="#E08214", linewidth=1.4, label="Instrumental mean" if start == 1500 else None)
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.set(xlabel="Year", ylabel="Temperature anomaly (degC)", title="Candidate B2 no-proxy balanced background")
    axis.legend(frameon=False, ncol=2)
    fig.savefig(OUT / "balanced_background_long_term_hma_curves_allowed.png", dpi=200, facecolor="white")
    plt.close(fig)

    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "diagnostic": diagnostic,
        "qcf_balanced_background_metrics": background_station.to_dict(),
        "qcf_archived_project_metrics": archived_station.to_dict(),
        "interpretation_boundary": (
            "background-only diagnosis; no real proxy value or EnKF update is used"
        ),
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "static_package": STATIC_PACKAGE,
        "allowed_long_curves": LONG_CURVES,
        "early_stations": evaluation.EARLY_STATIONS,
        "modern_stations": evaluation.MODERN_STATIONS,
        "driver": DRIVER,
        "evaluation_helpers": Path(evaluation.__file__).resolve(),
    }
    manifest = {
        name: {"path": str(path), "sha256": core.sha256(path)}
        for name, path in sources.items()
    }
    manifest["real_proxy_observations_read"] = False
    manifest["prohibited_source_accessed"] = False
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
