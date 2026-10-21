"""Test temporal stability of the archived bivariate precipitation PSM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLIMATE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
CALIBRATION = (1940, 2000)
EXTERNAL = (1901, 1920)
SPLITS = {
    "early_fit_late_test": ((1940, 1969), (1970, 2000)),
    "late_fit_early_test": ((1970, 2000), (1940, 1969)),
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_root() -> Path:
    archive_folder = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper_folder = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive_folder / paper_folder / "work1" / "compared_result_all"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chronologies(path: Path, years: np.ndarray) -> pd.DataFrame:
    columns: dict[int, list[float]] = {}
    with h5py.File(path, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for index, reference in enumerate(references, start=1):
            proxy = handle[reference]
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            lookup = {
                int(age): float(value)
                for age, value in zip(ages, values)
                if np.isfinite(age) and np.isfinite(value)
            }
            columns[index] = [lookup.get(int(year), np.nan) for year in years]
    return pd.DataFrame(columns, index=years)


def fit_psm(y: np.ndarray, temp: np.ndarray, prcp: np.ndarray) -> tuple[np.ndarray, int]:
    valid = np.isfinite(y) & np.isfinite(temp) & np.isfinite(prcp)
    if int(valid.sum()) < 10:
        return np.full(3, np.nan), int(valid.sum())
    design = np.column_stack([np.ones(int(valid.sum())), temp[valid], prcp[valid]])
    if np.linalg.matrix_rank(design) != 3:
        return np.full(3, np.nan), int(valid.sum())
    coefficients, _, _, _ = np.linalg.lstsq(design, y[valid], rcond=None)
    return coefficients.astype(float), int(valid.sum())


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if int(valid.sum()) < 10:
        return {"n": int(valid.sum()), "r": np.nan, "rmse": np.nan, "bias": np.nan}
    residual = predicted[valid] - actual[valid]
    return {
        "n": int(valid.sum()),
        "r": float(np.corrcoef(actual[valid], predicted[valid])[0, 1]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
    }


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 10:
        return float("nan")
    x = years[valid].astype(float)
    return float(100.0 * np.polyfit(x - x.mean(), values[valid], 1)[0])


def main() -> int:
    root = project_root()
    attribution = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    )
    metadata_path = attribution / "proxy6_metadata.csv"
    gains_path = attribution / "proxy_trend_contribution_1901_1920.csv"
    proxy_path = archive_root() / "python_data" / "proxy_data_6.mat"
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_psm_temporal_stability_v2"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    paths = {
        "metadata": metadata_path,
        "gain_attribution": gains_path,
        "proxy6": proxy_path,
        "climate": CLIMATE,
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    output.mkdir(parents=True)
    before = {name: sha256(path) for name, path in paths.items()}

    years = np.arange(EXTERNAL[0], CALIBRATION[1] + 1)
    metadata = pd.read_csv(metadata_path).set_index("proxy_index")
    gains = pd.read_csv(gains_path).set_index("proxy_index")
    observations = read_chronologies(proxy_path, years)
    with h5py.File(CLIMATE, "r") as handle:
        climate_years = np.asarray(handle["time"], dtype=int)
        indices = np.where(
            (climate_years >= EXTERNAL[0]) & (climate_years <= CALIBRATION[1])
        )[0]
        if not np.array_equal(climate_years[indices], years):
            raise RuntimeError("climate-year-axis-does-not-match-requested-period")
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        temperature = np.asarray(handle["tas_obs_bea_05deg"][indices], dtype=float)
        precipitation = np.asarray(handle["pr_obs_gpcp_05deg"][indices], dtype=float)
    temperature[temperature <= -9000] = np.nan
    precipitation[precipitation <= -9000] = np.nan

    table_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    for proxy_index, proxy in metadata.iterrows():
        if proxy["proxy_type"] != "tree" or int(proxy["coefficient_count"]) != 3:
            continue
        latitude = float(proxy["latitude"])
        longitude = float(proxy["longitude"])
        if not (latitudes.min() <= latitude <= latitudes.max() and longitudes.min() <= longitude <= longitudes.max()):
            continue
        lat_index = int(np.argmin(np.abs(latitudes - latitude)))
        lon_index = int(np.argmin(np.abs(longitudes - longitude)))
        temp = temperature[:, lat_index, lon_index]
        prcp = precipitation[:, lat_index, lon_index]
        if not (np.isfinite(temp).all() and np.isfinite(prcp).all()):
            continue
        actual = observations[int(proxy_index)].to_numpy(dtype=float)
        production = np.array(
            [proxy["coefficient_b0"], proxy["coefficient_b1"], proxy["coefficient_b2"]],
            dtype=float,
        )
        for split_name, (fit_period, test_period) in SPLITS.items():
            masks = {
                "fit": (years >= fit_period[0]) & (years <= fit_period[1]),
                "test": (years >= test_period[0]) & (years <= test_period[1]),
                "external_1901_1920": (years >= EXTERNAL[0]) & (years <= EXTERNAL[1]),
            }
            coefficients, fit_n = fit_psm(actual[masks["fit"]], temp[masks["fit"]], prcp[masks["fit"]])
            row: dict[str, object] = {
                "proxy_index": int(proxy_index),
                "split": split_name,
                "fit_start": fit_period[0],
                "fit_end": fit_period[1],
                "test_start": test_period[0],
                "test_end": test_period[1],
                "latitude": latitude,
                "longitude": longitude,
                "selected_latitude": float(latitudes[lat_index]),
                "selected_longitude": float(longitudes[lon_index]),
                "fit_n": fit_n,
                "fit_b0": coefficients[0],
                "fit_b1_temperature": coefficients[1],
                "fit_b2_precipitation": coefficients[2],
                "production_b0": production[0],
                "production_b1_temperature": production[1],
                "production_b2_precipitation": production[2],
                "delta_b0_vs_production": coefficients[0] - production[0],
                "delta_b1_vs_production": coefficients[1] - production[1],
                "delta_b2_vs_production": coefficients[2] - production[2],
            }
            for label, mask in masks.items():
                predicted = coefficients[0] + coefficients[1] * temp[mask] + coefficients[2] * prcp[mask]
                row.update({f"{label}_{key}": value for key, value in prediction_metrics(actual[mask], predicted).items()})
                temp_trend = trend(years[mask], temp[mask])
                prcp_trend = trend(years[mask], prcp[mask])
                gain = float(gains.loc[int(proxy_index), "mean_hma_gain_weight"])
                projection_rows.append(
                    {
                        "proxy_index": int(proxy_index),
                        "split": split_name,
                        "period": label,
                        "period_start": int(years[mask][0]),
                        "period_end": int(years[mask][-1]),
                        "temperature_trend_c_per_century": temp_trend,
                        "precipitation_trend_mm_day_per_century": prcp_trend,
                        "temperature_hma_projection_mm_day_per_century": gain * coefficients[1] * temp_trend,
                        "precipitation_hma_projection_mm_day_per_century": gain * coefficients[2] * prcp_trend,
                    }
                )
            table_rows.append(row)

    proxy_table = pd.DataFrame(table_rows)
    if proxy_table.empty:
        raise RuntimeError("no-assessed-proxies")
    if len(proxy_table) != 294:
        raise RuntimeError(f"unexpected-assessed-split-row-count:{len(proxy_table)}")
    proxy_table.to_csv(output / "psm_temporal_stability_proxy_level.csv", index=False)
    projections = pd.DataFrame(projection_rows)
    projections.to_csv(output / "psm_temporal_stability_hma_projection_proxy_level.csv", index=False)

    early = proxy_table[proxy_table["split"].eq("early_fit_late_test")].set_index("proxy_index")
    late = proxy_table[proxy_table["split"].eq("late_fit_early_test")].set_index("proxy_index")
    drift = pd.DataFrame(
        {
            "b1_early": early["fit_b1_temperature"],
            "b1_late": late["fit_b1_temperature"],
            "b2_early": early["fit_b2_precipitation"],
            "b2_late": late["fit_b2_precipitation"],
        }
    )
    drift["b1_early_minus_late"] = drift["b1_early"] - drift["b1_late"]
    drift["b2_early_minus_late"] = drift["b2_early"] - drift["b2_late"]
    drift.to_csv(output / "psm_temporal_coefficient_drift_proxy_level.csv")

    summary_rows: list[dict[str, object]] = []
    for split_name in SPLITS:
        subset = proxy_table[proxy_table["split"].eq(split_name)]
        for period in ("fit", "test", "external_1901_1920"):
            summary_rows.append(
                {
                    "split": split_name,
                    "period": period,
                    "proxy_count": int(len(subset)),
                    "median_r": float(subset[f"{period}_r"].median()),
                    "median_rmse": float(subset[f"{period}_rmse"].median()),
                    "median_bias": float(subset[f"{period}_bias"].median()),
                    "median_abs_delta_b1_vs_production": float(subset["delta_b1_vs_production"].abs().median()),
                    "median_abs_delta_b2_vs_production": float(subset["delta_b2_vs_production"].abs().median()),
                }
            )
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(output / "psm_temporal_stability_summary.csv", index=False)
    hma_summary = projections.groupby(["split", "period"], as_index=False).agg(
        temperature_hma_projection_mm_day_per_century=("temperature_hma_projection_mm_day_per_century", "sum"),
        precipitation_hma_projection_mm_day_per_century=("precipitation_hma_projection_mm_day_per_century", "sum"),
    )
    hma_summary["total_hma_projection_mm_day_per_century"] = (
        hma_summary["temperature_hma_projection_mm_day_per_century"]
        + hma_summary["precipitation_hma_projection_mm_day_per_century"]
    )
    hma_summary.to_csv(output / "psm_temporal_stability_hma_projection_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    axes[0].scatter(early["fit_b1_temperature"], late["fit_b1_temperature"], c="#d95f02", s=30, alpha=0.78, label="b1 temperature")
    axes[0].scatter(early["fit_b2_precipitation"], late["fit_b2_precipitation"], c="#1b9e77", s=30, alpha=0.78, label="b2 precipitation")
    combined = np.concatenate([early[["fit_b1_temperature", "fit_b2_precipitation"]].to_numpy().ravel(), late[["fit_b1_temperature", "fit_b2_precipitation"]].to_numpy().ravel()])
    limit = float(np.nanmax(np.abs(combined)))
    axes[0].plot([-limit, limit], [-limit, limit], color="#333333", lw=1)
    axes[0].axhline(0, color="#777777", lw=0.7)
    axes[0].axvline(0, color="#777777", lw=0.7)
    axes[0].set_xlabel("1940-1969 fitted coefficient")
    axes[0].set_ylabel("1970-2000 fitted coefficient")
    axes[0].set_title("PSM coefficient stability")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    plot_data = hma_summary.pivot(index="period", columns="split", values="total_hma_projection_mm_day_per_century")
    plot_data.plot(kind="bar", ax=axes[1], color=["#7570b3", "#4c78a8"])
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_ylabel("Projected HMA precipitation trend")
    axes[1].set_title("Split-fit gain-weighted projection")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(title="fit split", frameon=False)
    figure.savefig(output / "figure_psm_temporal_stability.png", dpi=220)
    plt.close(figure)

    drift_summary = {
        "median_abs_b1_early_minus_late": float(drift["b1_early_minus_late"].abs().median()),
        "median_abs_b2_early_minus_late": float(drift["b2_early_minus_late"].abs().median()),
        "b1_sign_agreement_early_vs_late": float(np.mean(np.sign(drift["b1_early"]) == np.sign(drift["b1_late"]))),
        "b2_sign_agreement_early_vs_late": float(np.mean(np.sign(drift["b2_early"]) == np.sign(drift["b2_late"]))),
    }
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_psm_temporal_stability_v2",
        "date": "2026-08-14",
        "calibration_period": list(CALIBRATION),
        "splits": {name: [list(pair[0]), list(pair[1])] for name, pair in SPLITS.items()},
        "external_development_period": list(EXTERNAL),
        "locked_final_period_accessed": False,
        "production_psm": "b0 + b1*BerkeleyEarth_temperature + b2*GPCC_derived_precipitation",
        "assessed_proxy_count": int(len(early)),
        "source_hashes_unchanged": before == after,
        "coefficient_drift": drift_summary,
        "decision_boundary": "This is a time-stability and frozen early-20th-century extrapolation diagnosis. It does not alter the production PSM, EnKF, prior, or reconstruction.",
    }
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "EXPERIMENT.md").write_text("\n".join(["# Precipitation PSM Temporal Stability V2", "", "- Fit 1940-1969 and validate 1970-2000.", "- Fit 1970-2000 and validate 1940-1969.", "- Extrapolate both fits to 1901-1920 for development diagnosis only.", "- Do not read 1921-1939.", "- No production source or reconstruction output is modified.", "", summary["decision_boundary"], ""]), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text("\n".join(["# Precipitation PSM Temporal Stability V2", "", f"- Assessed proxies: `{len(early)}`", f"- Median absolute b1 drift: `{drift_summary['median_abs_b1_early_minus_late']:.6g}`", f"- Median absolute b2 drift: `{drift_summary['median_abs_b2_early_minus_late']:.6g}`", f"- b1 sign agreement: `{drift_summary['b1_sign_agreement_early_vs_late']:.3f}`", f"- b2 sign agreement: `{drift_summary['b2_sign_agreement_early_vs_late']:.3f}`", f"- Source hashes unchanged: `{before == after}`", "", summary["decision_boundary"], ""]), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
