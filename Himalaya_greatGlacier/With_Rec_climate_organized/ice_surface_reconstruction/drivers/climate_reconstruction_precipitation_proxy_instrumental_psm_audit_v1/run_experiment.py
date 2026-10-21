"""Audit proxy trends against local instrumental bivariate PSM predictors."""

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
YEARS = (1901, 1920)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 10:
        return float("nan")
    x = years[valid].astype(float)
    y = values[valid]
    return float(100.0 * np.polyfit(x - x.mean(), y, 1)[0])


def correlation(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    root = project_root()
    source = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/"
        "proxy_trend_contribution_1901_1920.csv"
    )
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_proxy_instrumental_psm_audit_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    if not source.exists() or not CLIMATE.exists():
        raise FileNotFoundError("missing-input")
    output.mkdir(parents=True)
    hashes_before = {"trend_attribution": sha256(source), "climate": sha256(CLIMATE)}

    proxies = pd.read_csv(source).sort_values("proxy_index")
    with h5py.File(CLIMATE, "r") as handle:
        years_all = np.asarray(handle["time"], dtype=int)
        indices = np.where((years_all >= YEARS[0]) & (years_all <= YEARS[1]))[0]
        years = years_all[indices]
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        temperature = np.asarray(handle["tas_obs_bea_05deg"][indices], dtype=float)
        precipitation = np.asarray(handle["pr_obs_gpcp_05deg"][indices], dtype=float)
    temperature[temperature <= -9000] = np.nan
    precipitation[precipitation <= -9000] = np.nan

    rows = []
    for proxy in proxies.itertuples(index=False):
        row = {
            "proxy_index": int(proxy.proxy_index),
            "latitude": float(proxy.latitude),
            "longitude": float(proxy.longitude),
            "proxy_type": proxy.proxy_type,
            "psm_method": proxy.psm_method,
            "actual_proxy_trend_per_century": float(
                proxy.innovation_trend_proxy_units_per_century
            ),
            "hma_gain_weight": float(proxy.mean_hma_gain_weight),
            "actual_hma_contribution_trend_mm_day_per_century": float(
                proxy.contribution_trend_mm_day_per_century
            ),
        }
        if proxy.proxy_type != "tree" or not np.isfinite(proxy.coefficient_b2):
            row["assessment_status"] = "non_tree_or_non_bivariate"
            rows.append(row)
            continue
        if not (
            latitudes.min() <= proxy.latitude <= latitudes.max()
            and longitudes.min() <= proxy.longitude <= longitudes.max()
        ):
            row["assessment_status"] = "outside_combined_grid"
            rows.append(row)
            continue
        lat_index = int(np.argmin(np.abs(latitudes - proxy.latitude)))
        lon_index = int(np.argmin(np.abs(longitudes - proxy.longitude)))
        local_temperature = temperature[:, lat_index, lon_index]
        local_precipitation = precipitation[:, lat_index, lon_index]
        if not (
            np.isfinite(local_temperature).all()
            and np.isfinite(local_precipitation).all()
        ):
            row["assessment_status"] = "incomplete_instrumental_predictors"
            rows.append(row)
            continue
        temperature_trend = trend(years, local_temperature)
        precipitation_trend = trend(years, local_precipitation)
        temperature_component = float(proxy.coefficient_b1 * temperature_trend)
        precipitation_component = float(proxy.coefficient_b2 * precipitation_trend)
        predicted_proxy_trend = temperature_component + precipitation_component
        row.update(
            {
                "assessment_status": "assessed",
                "selected_latitude": float(latitudes[lat_index]),
                "selected_longitude": float(longitudes[lon_index]),
                "temperature_trend_c_per_century": temperature_trend,
                "precipitation_trend_mm_day_per_century": precipitation_trend,
                "temperature_proxy_component_per_century": temperature_component,
                "precipitation_proxy_component_per_century": precipitation_component,
                "instrumental_psm_predicted_proxy_trend_per_century": predicted_proxy_trend,
                "proxy_minus_instrumental_psm_trend_per_century": float(
                    proxy.innovation_trend_proxy_units_per_century
                    - predicted_proxy_trend
                ),
                "predicted_hma_temperature_contribution_mm_day_per_century": float(
                    proxy.mean_hma_gain_weight * temperature_component
                ),
                "predicted_hma_precipitation_contribution_mm_day_per_century": float(
                    proxy.mean_hma_gain_weight * precipitation_component
                ),
                "predicted_hma_total_contribution_mm_day_per_century": float(
                    proxy.mean_hma_gain_weight * predicted_proxy_trend
                ),
            }
        )
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_csv(output / "proxy_instrumental_psm_audit_1901_1920.csv", index=False)
    assessed = audit[audit["assessment_status"].eq("assessed")].copy()
    if assessed.empty:
        raise RuntimeError("no-assessed-proxies")

    actual = assessed["actual_proxy_trend_per_century"]
    predicted = assessed["instrumental_psm_predicted_proxy_trend_per_century"]
    temperature_component = assessed["temperature_proxy_component_per_century"]
    precipitation_component = assessed["precipitation_proxy_component_per_century"]
    residual = actual - predicted
    sign_agreement = float(np.mean(np.sign(actual) == np.sign(predicted)))
    total_actual_hma = float(
        assessed["actual_hma_contribution_trend_mm_day_per_century"].sum()
    )
    predicted_hma_temperature = float(
        assessed["predicted_hma_temperature_contribution_mm_day_per_century"].sum()
    )
    predicted_hma_precipitation = float(
        assessed["predicted_hma_precipitation_contribution_mm_day_per_century"].sum()
    )
    predicted_hma_total = predicted_hma_temperature + predicted_hma_precipitation

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    limit = float(
        np.nanmax(np.abs(np.concatenate([actual.to_numpy(), predicted.to_numpy()])))
    )
    scatter = axes[0].scatter(
        predicted,
        actual,
        c=assessed["hma_gain_weight"],
        cmap="PiYG",
        s=34,
        alpha=0.82,
    )
    axes[0].plot([-limit, limit], [-limit, limit], color="#333333", lw=1.0)
    axes[0].axhline(0, color="#777777", lw=0.7)
    axes[0].axvline(0, color="#777777", lw=0.7)
    axes[0].set_xlabel("Instrumental PSM-predicted proxy trend")
    axes[0].set_ylabel("Actual proxy trend")
    axes[0].set_title("1901-1920 proxy trend consistency")
    axes[0].grid(alpha=0.2)
    figure.colorbar(scatter, ax=axes[0], label="HMA Kalman gain weight")
    component_totals = [
        predicted_hma_temperature,
        predicted_hma_precipitation,
        predicted_hma_total,
        total_actual_hma,
    ]
    axes[1].bar(
        ["Temperature\nterm", "Precipitation\nterm", "Predicted\ntotal", "Actual proxy\nprojection"],
        component_totals,
        color=["#d95f02", "#1b9e77", "#7570b3", "#4c78a8"],
    )
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_ylabel("HMA trend contribution (mm/day/century)")
    axes[1].set_title("Instrumental PSM component projection")
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output / "figure_proxy_instrumental_psm_audit.png", dpi=220)
    plt.close(figure)

    hashes_after = {"trend_attribution": sha256(source), "climate": sha256(CLIMATE)}
    status_counts = audit["assessment_status"].value_counts().to_dict()
    result = {
        "experiment_id": "climate_reconstruction_precipitation_proxy_instrumental_psm_audit_v1",
        "date": "2026-08-13",
        "years": [1901, 1920],
        "locked_final_period_accessed": False,
        "production_psm": "b0 + b1*BerkeleyEarth_temperature + b2*GPCC_derived_precipitation",
        "source_hashes_unchanged": hashes_before == hashes_after,
        "assessment_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "assessed_proxy_count": int(len(assessed)),
        "actual_vs_instrumental_predicted_trend_correlation": correlation(actual, predicted),
        "actual_vs_temperature_component_correlation": correlation(actual, temperature_component),
        "actual_vs_precipitation_component_correlation": correlation(actual, precipitation_component),
        "actual_vs_predicted_trend_rmse_proxy_units_per_century": float(
            np.sqrt(np.mean(residual**2))
        ),
        "actual_vs_predicted_sign_agreement_fraction": sign_agreement,
        "assessed_actual_hma_trend_contribution_mm_day_per_century": total_actual_hma,
        "instrumental_temperature_term_hma_trend_contribution_mm_day_per_century": predicted_hma_temperature,
        "instrumental_precipitation_term_hma_trend_contribution_mm_day_per_century": predicted_hma_precipitation,
        "instrumental_total_psm_hma_trend_contribution_mm_day_per_century": predicted_hma_total,
        "decision_boundary": (
            "Nearest combined-grid instrumental predictors are a diagnostic approximation to the "
            "production spatial extraction. This audit does not refit or validate a PSM."
        ),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Precipitation Proxy Instrumental PSM Audit V1",
                "",
                f"- Assessed proxies: `{len(assessed)}`",
                f"- Actual vs predicted trend correlation: `{result['actual_vs_instrumental_predicted_trend_correlation']:.6g}`",
                f"- Sign agreement: `{sign_agreement:.3f}`",
                f"- Instrumental temperature-term HMA contribution: `{predicted_hma_temperature:.6g} mm/day/century`",
                f"- Instrumental precipitation-term HMA contribution: `{predicted_hma_precipitation:.6g} mm/day/century`",
                f"- Actual assessed-proxy HMA contribution: `{total_actual_hma:.6g} mm/day/century`",
                f"- Source hashes unchanged: `{hashes_before == hashes_after}`",
                "",
                result["decision_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
