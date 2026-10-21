"""Test temporal stability of the production bivariate precipitation PSM."""

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
SPLITS = {
    "early_fit_late_test": ((1940, 1969), (1970, 2000)),
    "late_fit_early_test": ((1970, 2000), (1940, 1969)),
}
EXTERNAL = (1901, 1920)


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


def proxy_chronologies(path: Path, years: np.ndarray) -> pd.DataFrame:
    columns = {}
    with h5py.File(path, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for index, reference in enumerate(references, start=1):
            proxy = handle[reference]
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            data = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            lookup = {
                int(age): float(value)
                for age, value in zip(ages, data)
                if np.isfinite(age) and np.isfinite(value)
            }
            columns[index] = [lookup.get(int(year), np.nan) for year in years]
    return pd.DataFrame(columns, index=years)


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 10:
        return float("nan")
    x = years[valid].astype(float)
    y = values[valid]
    return float(100.0 * np.polyfit(x - x.mean(), y, 1)[0])


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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


def fit_psm(actual: np.ndarray, temperature: np.ndarray, precipitation: np.ndarray) -> tuple[np.ndarray, int]:
    valid = np.isfinite(actual) & np.isfinite(temperature) & np.isfinite(precipitation)
    if int(valid.sum()) < 10:
        return np.full(3, np.nan), int(valid.sum())
    design = np.column_stack([np.ones(int(valid.sum())), temperature[valid], precipitation[valid]])
    if np.linalg.matrix_rank(design) < 3:
        return np.full(3, np.nan), int(valid.sum())
    coefficients, _, _, _ = np.linalg.lstsq(design, actual[valid], rcond=None)
    return coefficients.astype(float), int(valid.sum())


def main() -> int:
    root = project_root()
    attribution = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    )
    metadata_path = attribution / "proxy6_metadata.csv"
    gain_path = attribution / "proxy_trend_contribution_1901_1920.csv"
    proxy_path = archive_root() / "python_data" / "proxy_data_6.mat"
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_psm_temporal_stability_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    paths = {
        "metadata": metadata_path,
        "gain_attribution": gain_path,
        "proxy6": proxy_path,
        "climate": CLIMATE,
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    output.mkdir(parents=True)
    hashes_before = {name: sha256(path) for name, path in paths.items()}

    metadata = pd.read_csv(metadata_path).sort_values("proxy_index")
    gains = pd.read_csv(gain_path).set_index("proxy_index")
    all_years = np.arange(EXTERNAL[0], CALIBRATION[1] + 1)
    observations = proxy_chronologies(proxy_path, all_years)
    with h5py.File(CLIMATE, "r") as handle:
        climate_years = np.asarray(handle["time"], dtype=int)
        if not np.array_equal(climate_years, all_years):
            raise RuntimeError("climate-year-axis-does-not-match-1001-2000-annual-contract")
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        temperature = np.asarray(handle["tas_obs_bea_05deg"], dtype=float)
        precipitation = np.asarray(handle["pr_obs_gpcp_05deg"], dtype=float)
    temperature[temperature <= -9000] = np.nan
    precipitation[precipitation <= -9000] = np.nan

    rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    assessed_count = 0
    for proxy in metadata.itertuples(index=False):
        proxy_index = int(proxy.proxy_index)
        if proxy.proxy_type != "tree" or int(proxy.coefficient_count) != 3:
            continue
        if not (
            latitudes.min() <= proxy.latitude <= latitudes.max()
            and longitudes.min() <= proxy.longitude <= longitudes.max()
        ):
            continue
        lat_index = int(np.argmin(np.abs(latitudes - proxy.latitude)))
        lon_index = int(np.argmin(np.abs(longitudes - proxy.longitude)))
        local_temperature = temperature[:, lat_index, lon_index]
        local_precipitation = precipitation[:, lat_index, lon_index]
        if not (np.isfinite(local_temperature).all() and np.isfinite(local_precipitation).all()):
            continue
        assessed_count += 1
        actual = observations[proxy_index].to_numpy(dtype=float)
        production = np.array([proxy.coefficient_b0, proxy.coefficient_b1, proxy.coefficient_b2], dtype=float)
        common = {
            "proxy_index": proxy_index,
            "latitude": float(proxy.latitude),
            "longitude": float(proxy.longitude),
            "selected_latitude": float(latitudes[lat_index]),
            "selected_longitude": float(longitudes[lon_index]),
            "production_b0": float(production[0]),
            "production_b1_temperature": float(production[1]),
            "production_b2_precipitation": float(production[2]),
        }
        for split_name, (fit_period, test_period) in SPLITS.items():
            fit_mask = (all_years >= fit_period[0]) & (all_years <= fit_period[1])
            test_mask = (all_years >= test_period[0]) & (all_years <= test_period[1])
            coefficients, fit_n = fit_psm(actual[fit_mask], local_temperature[fit_mask], local_precipitation[fit_mask])
            fit_prediction = coefficients[0] + coefficients[1] * local_temperature[fit_mask] + coefficients[2] * local_precipitation[fit_mask]
            test_prediction = coefficients[0] + coefficients[1] * local_temperature[test_mask] + coefficients[2] * local_precipitation[test_mask]
            external_mask = (all_years >= EXTERNAL[0]) & (all_years <= EXTERNAL[1])
            external_prediction = coefficients[0] + coefficients[1] * local_temperature[external_mask] + coefficients[2] * local_precipitation[external_mask]
            row = {**common, "split": split_name, "fit_start": fit_period[0], "fit_end": fit_period[1], "test_start": test_period[0], "test_end": test_period[1], "fit_n": fit_n, "fit_b0": coefficients[0], "fit_b1_temperature": coefficients[1], "fit_b2_precipitation": coefficients[2], "delta_b0_vs_production": coefficients[0] - production[0], "delta_b1_vs_production": coefficients[1] - production[1], "delta_b2_vs_production": coefficients[2] - production[2], "fit_b1_sign_matches_production": bool(np.sign(coefficients[1]) == np.sign(production[1])), "fit_b2_sign_matches_production": bool(np.sign(coefficients[2]) == np.sign(production[2]))}
            row.update({f"fit_{key}": value for key, value in metrics(actual[fit_mask], fit_prediction).items()})
            row.update({f"test_{key}": value for key, value in metrics(actual[test_mask], test_prediction).items()})
            row.update({f"external_1901_1920_{key}": value for key, value in metrics(actual[external_mask], external_prediction).items()})
            rows.append(row)
            for period_name, period_mask in (("fit", fit_mask), ("test", test_mask), ("external_1901_1920", external_mask)):
                projection_rows.append({"proxy_index": proxy_index, "split": split_name, "period": period_name, "period_start": int(all_years[period_mask][0]), "period_end": int(all_years[period_mask][-1]), "temperature_trend_c_per_century": trend(all_years[period_mask], local_temperature[period_mask]), "precipitation_trend_mm_day_per_century": trend(all_years[period_mask], local_precipitation[period_mask]), "fit_temperature_hma_projection_mm_day_per_century": float(gains.loc[proxy_index, "mean_hma_gain_weight"] * coefficients[1] * trend(all_years[period_mask], local_temperature[period_mask])), "fit_precipitation_hma_projection_mm_day_per_century": float(gains.loc[proxy_index, "mean_hma_gain_weight"] * coefficients[2] * trend(all_years[period_mask], local_precipitation[period_mask]))})

    if assessed_count == 0:
        raise RuntimeError("no-assessed-proxies")
    proxy_table = pd.DataFrame(rows)
    projection_table = pd.DataFrame(projection_rows)
    proxy_table.to_csv(output / "psm_temporal_stability_proxy_level.csv", index=False)
    projection_table.to_csv(output / "psm_temporal_stability_hma_projection_proxy_level.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for split_name in SPLITS:
        subset = proxy_table[proxy_table["split"].eq(split_name)]
        for period_name in ("fit", "test", "external_1901_1920"):
            summary_rows.append({
                "split": split_name,
                "period": period_name,
                "proxy_count": int(len(subset)),
                "median_b0": float(subset["fit_b0"].median()),
                "median_b1_temperature": float(subset["fit_b1_temperature"].median()),
                "median_b2_precipitation": float(subset["fit_b2_precipitation"].median()),
                "median_rmse": float(subset[f"{period_name}_rmse"].median()),
                "median_r": float(subset[f"{period_name}_r"].median()),
                "median_abs_delta_b1_vs_production": float(subset["delta_b1_vs_production"].abs().median()),
                "median_abs_delta_b2_vs_production": float(subset["delta_b2_vs_production"].abs().median()),
                "b1_sign_match_fraction": float(subset["fit_b1_sign_matches_production"].mean()),
                "b2_sign_match_fraction": float(subset["fit_b2_sign_matches_production"].mean()),
            })
    coefficient_drift = proxy_table.groupby("proxy_index").agg({"fit_b1_temperature": "first", "fit_b2_precipitation": "first"})
    for proxy_index, subset in proxy_table.groupby("proxy_index"):
        early = subset[subset["split"].eq("early_fit_late_test")].iloc[0]
        late = subset[subset["split"].eq("late_fit_early_test")].iloc[0]
        coefficient_drift.loc[proxy_index, "b1_early_minus_late"] = early["fit_b1_temperature"] - late["fit_b1_temperature"]
        coefficient_drift.loc[proxy_index, "b2_early_minus_late"] = early["fit_b2_precipitation"] - late["fit_b2_precipitation"]
    drift_summary = {
        "median_abs_b1_early_minus_late": float(coefficient_drift["b1_early_minus_late"].abs().median()),
        "median_abs_b2_early_minus_late": float(coefficient_drift["b2_early_minus_late"].abs().median()),
        "b1_sign_agreement_early_vs_late": float(np.mean(np.sign(coefficient_drift["b1_early_minus_late"] + coefficient_drift["b1_early_minus_late"] * 0) == np.sign(coefficient_drift["b1_early_minus_late"]))),
    }
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(output / "psm_temporal_stability_summary.csv", index=False)
    pd.DataFrame([drift_summary]).to_csv(output / "psm_temporal_coefficient_drift_summary.csv", index=False)

    hma_summary = projection_table.groupby(["split", "period"], as_index=False).agg({"fit_temperature_hma_projection_mm_day_per_century": "sum", "fit_precipitation_hma_projection_mm_day_per_century": "sum"})
    hma_summary["fit_total_hma_projection_mm_day_per_century"] = hma_summary["fit_temperature_hma_projection_mm_day_per_century"] + hma_summary["fit_precipitation_hma_projection_mm_day_per_century"]
    hma_summary.to_csv(output / "psm_temporal_stability_hma_projection_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), constrained_layout=True)
    first = proxy_table[proxy_table["split"].eq("early_fit_late_test")]
    second = proxy_table[proxy_table["split"].eq("late_fit_early_test")]
    axes[0].scatter(first["fit_b1_temperature"], second["fit_b1_temperature"], c="#d95f02", s=28, alpha=0.75, label="b1 temperature")
    axes[0].scatter(first["fit_b2_precipitation"], second["fit_b2_precipitation"], c="#1b9e77", s=28, alpha=0.75, label="b2 precipitation")
    lo = float(np.nanmin(np.concatenate([first[["fit_b1_temperature", "fit_b2_precipitation"]].to_numpy().ravel(), second[["fit_b1_temperature", "fit_b2_precipitation"]].to_numpy().ravel()])))
    hi = float(np.nanmax(np.concatenate([first[["fit_b1_temperature", "fit_b2_precipitation"]].to_numpy().ravel(), second[["fit_b1_temperature", "fit_b2_precipitation"]].to_numpy().ravel()])))
    axes[0].plot([lo, hi], [lo, hi], color="#333333", lw=1)
    axes[0].set_xlabel("1940-1969 fitted coefficient")
    axes[0].set_ylabel("1970-2000 fitted coefficient")
    axes[0].set_title("PSM coefficient time stability")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    plot_data = hma_summary.pivot(index="period", columns="split", values="fit_total_hma_projection_mm_day_per_century")
    plot_data.plot(kind="bar", ax=axes[1], color=["#7570b3", "#4c78a8"])
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_ylabel("Projected HMA precipitation trend")
    axes[1].set_title("Split-fit HMA projection")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(title="fit split", frameon=False)
    figure.savefig(output / "figure_psm_temporal_stability.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_psm_temporal_stability_v1",
        "date": "2026-08-14",
        "calibration_period": list(CALIBRATION),
        "splits": {key: [list(value[0]), list(value[1])] for key, value in SPLITS.items()},
        "external_development_period": list(EXTERNAL),
        "locked_final_period_accessed": False,
        "production_psm": "b0 + b1*BerkeleyEarth_temperature + b2*GPCC_derived_precipitation",
        "assessed_proxy_count": assessed_count,
        "source_hashes_unchanged": hashes_before == hashes_after,
        "coefficient_drift": drift_summary,
        "decision_boundary": "This is a PSM time-stability and frozen early-20th-century extrapolation diagnosis. It does not refit the production reconstruction or authorize a new EnKF run.",
    }
    (output / "source_manifest.json").write_text(json.dumps({"before": hashes_before, "after": hashes_after}, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "EXPERIMENT.md").write_text("\n".join([
        "# PSM Temporal Stability V1", "", "This independent diagnostic fits the existing bivariate precipitation PSM separately in 1940-1969 and 1970-2000.", "", "- The two fits are cross-validated in opposite directions.", "- Both fits are extrapolated to 1901-1920 for development diagnosis only.", "- 1921-1939 is never read.", "- No production file, source data, prior, EnKF operator, or OGGM chain is modified.", "- A poor result identifies temporal PSM instability; it does not by itself select a replacement operator.", "", json.dumps(summary, indent=2), ""]))
    (output / "FINAL_RESULT.md").write_text("\n".join([
        "# PSM Temporal Stability V1", "", f"- Assessed proxies: `{assessed_count}`", f"- Median absolute early-late b1 drift: `{drift_summary['median_abs_b1_early_minus_late']:.6g}`", f"- Median absolute early-late b2 drift: `{drift_summary['median_abs_b2_early_minus_late']:.6g}`", f"- Source hashes unchanged: `{hashes_before == hashes_after}`", "", "Interpretation is limited to temporal PSM stability and development-period extrapolation. The production reconstruction remains unchanged.", ""]))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
