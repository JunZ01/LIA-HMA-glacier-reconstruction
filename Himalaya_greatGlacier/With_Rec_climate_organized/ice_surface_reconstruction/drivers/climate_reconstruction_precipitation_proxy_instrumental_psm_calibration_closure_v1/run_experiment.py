"""Check nearest-grid reproduction of archived 1940-2000 PSM calibration."""

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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 10:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    root = project_root()
    attribution = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    )
    metadata_path = attribution / "proxy6_metadata.csv"
    proxy_path = archive_root() / "python_data" / "proxy_data_6.mat"
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_proxy_instrumental_psm_calibration_closure_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    paths = {
        "metadata": metadata_path,
        "proxy6": proxy_path,
        "climate": CLIMATE,
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    output.mkdir(parents=True)
    hashes_before = {name: sha256(path) for name, path in paths.items()}

    metadata = pd.read_csv(metadata_path).sort_values("proxy_index")
    years = np.arange(CALIBRATION[0], CALIBRATION[1] + 1)
    observations = proxy_chronologies(proxy_path, years)
    with h5py.File(CLIMATE, "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int)
        indices = np.where((all_years >= CALIBRATION[0]) & (all_years <= CALIBRATION[1]))[0]
        if not np.array_equal(all_years[indices], years):
            raise RuntimeError("calibration-year-mismatch")
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        temperature = np.asarray(handle["tas_obs_bea_05deg"][indices], dtype=float)
        precipitation = np.asarray(handle["pr_obs_gpcp_05deg"][indices], dtype=float)
    temperature[temperature <= -9000] = np.nan
    precipitation[precipitation <= -9000] = np.nan

    rows = []
    for proxy in metadata.itertuples(index=False):
        row = {
            "proxy_index": int(proxy.proxy_index),
            "latitude": float(proxy.latitude),
            "longitude": float(proxy.longitude),
            "proxy_type": proxy.proxy_type,
            "psm_method": proxy.psm_method,
            "archived_psm_correlation": float(proxy.psm_correlation),
        }
        if proxy.proxy_type != "tree" or int(proxy.coefficient_count) != 3:
            row["closure_status"] = "non_tree_or_non_bivariate"
            rows.append(row)
            continue
        if not (
            latitudes.min() <= proxy.latitude <= latitudes.max()
            and longitudes.min() <= proxy.longitude <= longitudes.max()
        ):
            row["closure_status"] = "outside_combined_grid"
            rows.append(row)
            continue
        lat_index = int(np.argmin(np.abs(latitudes - proxy.latitude)))
        lon_index = int(np.argmin(np.abs(longitudes - proxy.longitude)))
        local_temperature = temperature[:, lat_index, lon_index]
        local_precipitation = precipitation[:, lat_index, lon_index]
        actual = observations[int(proxy.proxy_index)].to_numpy(dtype=float)
        predicted = (
            float(proxy.coefficient_b0)
            + float(proxy.coefficient_b1) * local_temperature
            + float(proxy.coefficient_b2) * local_precipitation
        )
        valid = np.isfinite(actual) & np.isfinite(predicted)
        if int(valid.sum()) < 10:
            row["closure_status"] = "insufficient_complete_years"
            rows.append(row)
            continue
        reproduced_correlation = correlation(actual, predicted)
        row.update(
            {
                "closure_status": "assessed",
                "selected_latitude": float(latitudes[lat_index]),
                "selected_longitude": float(longitudes[lon_index]),
                "complete_year_count": int(valid.sum()),
                "reproduced_psm_correlation": reproduced_correlation,
                "correlation_difference": reproduced_correlation
                - float(proxy.psm_correlation),
                "prediction_rmse_proxy_units": float(
                    np.sqrt(np.mean((predicted[valid] - actual[valid]) ** 2))
                ),
                "prediction_bias_proxy_units": float(
                    np.mean(predicted[valid] - actual[valid])
                ),
            }
        )
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_csv(output / "psm_calibration_closure_1940_2000.csv", index=False)
    assessed = audit[audit["closure_status"].eq("assessed")].copy()
    if assessed.empty:
        raise RuntimeError("no-assessed-proxies")
    correlation_of_correlations = correlation(
        assessed["archived_psm_correlation"].to_numpy(dtype=float),
        assessed["reproduced_psm_correlation"].to_numpy(dtype=float),
    )
    median_abs_difference = float(assessed["correlation_difference"].abs().median())
    close_fraction_005 = float(
        np.mean(assessed["correlation_difference"].abs() <= 0.05)
    )
    close_fraction_010 = float(
        np.mean(assessed["correlation_difference"].abs() <= 0.10)
    )

    figure, axis = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    axis.scatter(
        assessed["archived_psm_correlation"],
        assessed["reproduced_psm_correlation"],
        c=assessed["correlation_difference"],
        cmap="RdBu_r",
        s=34,
        alpha=0.82,
    )
    limits = [-0.2, 1.0]
    axis.plot(limits, limits, color="#333333", lw=1.0)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Archived PSM correlation")
    axis.set_ylabel("Nearest-grid reproduced correlation")
    axis.set_title("1940-2000 calibration-period PSM closure")
    axis.grid(alpha=0.2)
    figure.savefig(output / "figure_psm_calibration_closure.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    if median_abs_difference <= 0.05 and close_fraction_010 >= 0.8:
        closure_decision = "nearest_grid_approximation_is_adequate_for_trend_diagnosis"
    else:
        closure_decision = "nearest_grid_approximation_does_not_reproduce_production_psm"
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_proxy_instrumental_psm_calibration_closure_v1",
        "date": "2026-08-13",
        "period": [1940, 2000],
        "period_role": "calibration_consistency_only",
        "locked_final_period_accessed": False,
        "source_hashes_unchanged": hashes_before == hashes_after,
        "status_counts": {
            str(key): int(value)
            for key, value in audit["closure_status"].value_counts().to_dict().items()
        },
        "assessed_proxy_count": int(len(assessed)),
        "archived_vs_reproduced_correlation_correlation": correlation_of_correlations,
        "median_absolute_correlation_difference": median_abs_difference,
        "correlation_difference_within_0p05_fraction": close_fraction_005,
        "correlation_difference_within_0p10_fraction": close_fraction_010,
        "closure_decision": closure_decision,
        "interpretation_boundary": (
            "This checks the combined nearest-grid approximation against archived "
            "calibration behavior. It is not independent proxy or climate validation."
        ),
    }
    (output / "source_manifest.json").write_text(
        json.dumps({"before": hashes_before, "after": hashes_after}, indent=2),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Precipitation Proxy PSM Calibration Closure V1",
                "",
                f"- Assessed proxies: `{len(assessed)}`",
                f"- Median absolute correlation difference: `{median_abs_difference:.6g}`",
                f"- Fraction within 0.05: `{close_fraction_005:.3f}`",
                f"- Fraction within 0.10: `{close_fraction_010:.3f}`",
                f"- Decision: `{closure_decision}`",
                f"- Source hashes unchanged: `{hashes_before == hashes_after}`",
                "",
                summary["interpretation_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
