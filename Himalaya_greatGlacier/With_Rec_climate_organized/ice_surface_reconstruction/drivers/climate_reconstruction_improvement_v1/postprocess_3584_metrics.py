from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def _trend(years: np.ndarray, values: np.ndarray) -> float:
    centered = years.astype(float) - float(np.mean(years))
    return float(100.0 * np.polyfit(centered, values, 1)[0])


def _target_series(
    field: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    cube = np.empty((len(grid_lat), len(grid_lon), field.shape[1]), dtype=float)
    for year_index in range(field.shape[1]):
        cube[:, :, year_index] = field[:, year_index].reshape(
            (len(grid_lat), len(grid_lon)), order="F"
        )
    return np.asarray(
        RegularGridInterpolator((grid_lat, grid_lon), cube)([[33.75, 75.75]])[0],
        dtype=float,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_localization_pilot_v1"
    )
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()
    output = args.output_dir.resolve()

    prepared = np.load(output / "inputs/localization_pilot_inputs.npz")
    instrumental = np.load(output / "inputs/instrumental_targets.npz")
    reconstructions = np.load(output / "reconstruction_means.npz")
    years = prepared["years"]
    reference = (years >= 1961) & (years <= 1990)
    analysis = (years >= 1901) & (years <= 2000)
    analysis_years = years[analysis]
    early = (analysis_years >= 1901) & (analysis_years <= 1930)
    late = (analysis_years >= 1971) & (analysis_years <= 2000)

    rows = []
    branch_names = [name for name in reconstructions.files if name != "years"]
    for branch in branch_names:
        target = _target_series(
            reconstructions[branch], prepared["grid_lat"], prepared["grid_lon"]
        )
        target = target - np.mean(target[reference])
        rec = target[analysis]
        for product_index, product in enumerate(instrumental["product_names"]):
            obs = instrumental["target_3584"][product_index]
            obs = obs - np.mean(obs[reference])
            obs = obs[analysis]
            rec_roll = _rolling(rec, 30)
            obs_roll = _rolling(obs, 30)
            valid_roll = np.isfinite(rec_roll) & np.isfinite(obs_roll)
            rec_change = float(np.mean(rec[late]) - np.mean(rec[early]))
            obs_change = float(np.mean(obs[late]) - np.mean(obs[early]))
            rec_trend = _trend(analysis_years, rec)
            obs_trend = _trend(analysis_years, obs)
            rows.append(
                {
                    "branch": branch,
                    "product": str(product),
                    "annual_rmse_c": float(np.sqrt(np.mean((rec - obs) ** 2))),
                    "annual_correlation": float(np.corrcoef(rec, obs)[0, 1]),
                    "rolling30_rmse_c": float(
                        np.sqrt(np.mean((rec_roll[valid_roll] - obs_roll[valid_roll]) ** 2))
                    ),
                    "rolling30_correlation": float(
                        np.corrcoef(rec_roll[valid_roll], obs_roll[valid_roll])[0, 1]
                    ),
                    "reconstruction_trend_c_per_century": rec_trend,
                    "observation_trend_c_per_century": obs_trend,
                    "trend_absolute_error_c_per_century": abs(rec_trend - obs_trend),
                    "reconstruction_early_late_change_c": rec_change,
                    "observation_early_late_change_c": obs_change,
                    "early_late_absolute_error_c": abs(rec_change - obs_change),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "target_3584_metrics.csv", index=False)
    summary = (
        frame.groupby("branch")
        .agg(
            median_rolling30_correlation=("rolling30_correlation", "median"),
            median_rolling30_rmse_c=("rolling30_rmse_c", "median"),
            median_trend_absolute_error_c_per_century=(
                "trend_absolute_error_c_per_century",
                "median",
            ),
            median_early_late_absolute_error_c=("early_late_absolute_error_c", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output / "target_3584_summary.csv", index=False)
    payload = summary.to_dict(orient="records")
    (output / "target_3584_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
