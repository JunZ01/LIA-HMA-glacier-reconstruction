"""Audit CRU and GPCC-derived precipitation disagreement in the disclosed window."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERIOD = (1901, 1920)
FILL_VALUE_LIMIT = -9000.0


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


def weighted_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[None, :, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights, axis=(1, 2))
    denominator = np.sum(valid * weights, axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def spatial_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    x = left[valid]
    y = right[valid]
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def main() -> int:
    root = project_root()
    baseline_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1"
    result_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_cru_gpcc_uncertainty_1901_1920_v1"
    netcdf = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    inputs = {
        "baseline_summary": baseline_dir / "summary.json",
        "locked_contract": baseline_dir / "LOCKED_FINAL_PERIOD.json",
        "netcdf": netcdf,
    }
    if missing := [str(path) for path in inputs.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if result_dir.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{result_dir}")
    summary_contract = json.loads(inputs["baseline_summary"].read_text(encoding="utf-8"))
    locked_contract = json.loads(inputs["locked_contract"].read_text(encoding="utf-8"))
    if summary_contract["locked_final_period_evaluated"] or locked_contract["status"] != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")
    before = {name: sha256(path) for name, path in inputs.items() if name != "netcdf"}

    with h5py.File(netcdf, "r") as handle:
        years_all = np.asarray(handle["time"], dtype=int)
        indices = np.flatnonzero((years_all >= PERIOD[0]) & (years_all <= PERIOD[1]))
        years = np.arange(PERIOD[0], PERIOD[1] + 1)
        if not np.array_equal(years_all[indices], years):
            raise RuntimeError("time-window-contract-failed")
        latitudes = np.asarray(handle["lat"], dtype=float)
        required = ["pr_obs_cru_pre_05deg", "pr_obs_gpcp_05deg", "pr_ensemble_mean_rec_result"]
        if missing := [name for name in required if name not in handle]:
            raise KeyError(f"missing-variables:{missing}")
        cru = np.asarray(handle[required[0]][indices, :, :], dtype=float)
        gpcc = np.asarray(handle[required[1]][indices, :, :], dtype=float)
        rec_result = np.asarray(handle[required[2]][indices, :, :], dtype=float)
    for values in (cru, gpcc, rec_result):
        values[values <= FILL_VALUE_LIMIT] = np.nan

    annual_rows = []
    grid_shape = cru.shape[1:]
    for offset, year in enumerate(years):
        c = cru[offset]
        g = gpcc[offset]
        r = rec_result[offset]
        jointly_valid = np.isfinite(c) & np.isfinite(g)
        disagreement = c - g
        annual_rows.append(
            {
                "year": int(year),
                "cru_finite_grid_fraction": float(np.isfinite(c).mean()),
                "gpcc_finite_grid_fraction": float(np.isfinite(g).mean()),
                "joint_finite_grid_fraction": float(jointly_valid.mean()),
                "cru_gpcc_spatial_correlation": spatial_correlation(c, g),
                "cru_gpcc_spatial_rmse_mm_day": float(np.sqrt(np.nanmean(np.square(disagreement[jointly_valid])))),
                "cru_minus_gpcc_hma_mean_mm_day": float(weighted_mean(disagreement[None, :, :], latitudes)[0]),
                "cru_hma_mean_mm_day": float(weighted_mean(c[None, :, :], latitudes)[0]),
                "gpcc_hma_mean_mm_day": float(weighted_mean(g[None, :, :], latitudes)[0]),
                "product_midpoint_hma_mean_mm_day": float(weighted_mean(((c + g) / 2.0)[None, :, :], latitudes)[0]),
                "product_half_spread_hma_mean_mm_day": float(weighted_mean((np.abs(disagreement) / 2.0)[None, :, :], latitudes)[0]),
                "rec_result_hma_mean_mm_day": float(weighted_mean(r[None, :, :], latitudes)[0]),
            }
        )
    annual = pd.DataFrame(annual_rows)
    annual["rec_result_distance_to_cru_mm_day"] = annual["rec_result_hma_mean_mm_day"] - annual["cru_hma_mean_mm_day"]
    annual["rec_result_distance_to_gpcc_mm_day"] = annual["rec_result_hma_mean_mm_day"] - annual["gpcc_hma_mean_mm_day"]
    annual["rec_result_outside_hma_product_bounds"] = (annual["rec_result_hma_mean_mm_day"] < annual[["cru_hma_mean_mm_day", "gpcc_hma_mean_mm_day"]].min(axis=1)) | (annual["rec_result_hma_mean_mm_day"] > annual[["cru_hma_mean_mm_day", "gpcc_hma_mean_mm_day"]].max(axis=1))

    difference_trend = np.full(grid_shape, np.nan)
    for lat_index in range(grid_shape[0]):
        for lon_index in range(grid_shape[1]):
            difference_trend[lat_index, lon_index] = trend(years, cru[:, lat_index, lon_index] - gpcc[:, lat_index, lon_index])
    finite_trends = difference_trend[np.isfinite(difference_trend)]
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_cru_gpcc_uncertainty_1901_1920_v1",
        "period": list(PERIOD),
        "locked_final_period_accessed": False,
        "netcdf_access": "Only the 20 annual indices for 1901-1920 were read.",
        "coverage_definition": "Finite-value grid fraction in the supplied gridded products; not station density or a direct measure of real-world accuracy.",
        "gpcc_label": "Stored pr_obs_gpcp_05deg is GPCC-derived, per the baseline contract.",
        "annual_product_disagreement": {
            "median_spatial_correlation": float(annual["cru_gpcc_spatial_correlation"].median()),
            "median_spatial_rmse_mm_day": float(annual["cru_gpcc_spatial_rmse_mm_day"].median()),
            "median_hma_half_spread_mm_day": float(annual["product_half_spread_hma_mean_mm_day"].median()),
            "cru_minus_gpcc_hma_trend_mm_day_per_century": trend(years, annual["cru_minus_gpcc_hma_mean_mm_day"].to_numpy()),
        },
        "product_trends_mm_day_per_century": {
            "cru": trend(years, annual["cru_hma_mean_mm_day"].to_numpy()),
            "gpcc": trend(years, annual["gpcc_hma_mean_mm_day"].to_numpy()),
            "midpoint": trend(years, annual["product_midpoint_hma_mean_mm_day"].to_numpy()),
            "rec_result": trend(years, annual["rec_result_hma_mean_mm_day"].to_numpy()),
        },
        "rec_result_relative_to_product_bounds": {
            "years_outside_cru_gpcc_hma_bounds": int(annual["rec_result_outside_hma_product_bounds"].sum()),
            "total_years": int(len(annual)),
        },
        "gridcell_cru_minus_gpcc_trend_mm_day_per_century": {
            "median": float(np.nanmedian(finite_trends)),
            "p10": float(np.nanpercentile(finite_trends, 10)),
            "p90": float(np.nanpercentile(finite_trends, 90)),
        },
        "interpretation_limit": "CRU-GPCC disagreement is a product-disagreement envelope, not a calibrated confidence interval and not proof that either product is correct.",
    }

    result_dir.mkdir(parents=True)
    annual.to_csv(result_dir / "cru_gpcc_annual_uncertainty_1901_1920.csv", index=False)
    pd.DataFrame({"latitude_index": np.repeat(np.arange(grid_shape[0]), grid_shape[1]), "longitude_index": np.tile(np.arange(grid_shape[1]), grid_shape[0]), "cru_minus_gpcc_trend_mm_day_per_century": difference_trend.reshape(-1)}).to_csv(result_dir / "cru_minus_gpcc_gridcell_trend_1901_1920.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), constrained_layout=True)
    axes[0].plot(years, annual["cru_hma_mean_mm_day"], color="#6A3D9A", linewidth=1.7, label="CRU")
    axes[0].plot(years, annual["gpcc_hma_mean_mm_day"], color="#7A7A7A", linewidth=1.7, label="GPCC-derived")
    axes[0].fill_between(years, annual[["cru_hma_mean_mm_day", "gpcc_hma_mean_mm_day"]].min(axis=1), annual[["cru_hma_mean_mm_day", "gpcc_hma_mean_mm_day"]].max(axis=1), color="#BDBDBD", alpha=0.35, label="product-disagreement envelope")
    axes[0].plot(years, annual["rec_result_hma_mean_mm_day"], color="#D55E00", linewidth=2, label="rec_result")
    axes[0].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[0].set_ylabel("HMA anomaly (mm day$^{-1}$)")
    axes[0].set_title("HMA precipitation products and reconstruction")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(years, annual["cru_gpcc_spatial_correlation"], color="#0072B2", linewidth=1.7, label="spatial correlation")
    axes[1].plot(years, annual["cru_gpcc_spatial_rmse_mm_day"], color="#D55E00", linewidth=1.7, label="spatial RMSE (mm day$^{-1}$)")
    axes[1].plot(years, annual["product_half_spread_hma_mean_mm_day"], color="#009E73", linewidth=1.7, label="HMA half-spread (mm day$^{-1}$)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Product disagreement metric")
    axes[1].set_title("CRU-GPCC product disagreement, not observational truth error")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(result_dir / "cru_gpcc_uncertainty_1901_1920.png", dpi=180)
    plt.close(figure)

    report = [
        "# CRU-GPCC Product Disagreement Audit",
        "",
        "Period: 1901-1920 only. The locked 1921-1939 final test period was not accessed.",
        "",
        f"- Median annual CRU-GPCC spatial correlation: `{summary['annual_product_disagreement']['median_spatial_correlation']:.3f}`.",
        f"- Median annual spatial RMSE: `{summary['annual_product_disagreement']['median_spatial_rmse_mm_day']:.4f}` mm/day.",
        f"- rec_result is outside the annual HMA CRU-GPCC bounds in `{summary['rec_result_relative_to_product_bounds']['years_outside_cru_gpcc_hma_bounds']}/{len(annual)}` years.",
        "",
        "This is a product-disagreement audit, not a claim that CRU or GPCC is the true early-century precipitation field.",
    ]
    (result_dir / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: sha256(path) for name, path in inputs.items() if name != "netcdf"}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (result_dir / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "netcdf_sha256_from_baseline_contract": summary_contract["source_sha256_before"], "analysis_driver_sha256": sha256(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
