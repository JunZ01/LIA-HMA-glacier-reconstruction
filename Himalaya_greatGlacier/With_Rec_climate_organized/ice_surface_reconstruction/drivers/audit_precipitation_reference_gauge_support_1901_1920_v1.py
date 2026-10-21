"""Audit early CRU/GPCC precipitation reliability using station and gauge support."""

from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import netcdf_file


EXPERIMENT_ID = "climate_reconstruction_precipitation_reference_gauge_support_1901_1920_v1"
TARGET_YEARS = np.arange(1901, 1921, dtype=int)
REFERENCE_YEARS = np.arange(1961, 1991, dtype=int)
LOCKED_PERIOD = (1921, 1939)
HMA_CLIMATE = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
CRU_PRECIPITATION = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\input-recon\cruts.2004151855.v4.04\pre\cru_ts4.04.1901.2019.pre.dat.nc")
GPCC_PRECIPITATION = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc")
FILL_LIMIT = -9000.0


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path, include_hash: bool = False) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if include_hash:
        result["sha256"] = digest(path)
    return result


def assert_unlocked(years: np.ndarray, label: str) -> None:
    if np.any((years >= LOCKED_PERIOD[0]) & (years <= LOCKED_PERIOD[1])):
        raise RuntimeError(f"locked-period-requested:{label}")


def exact_indices(source: np.ndarray, target: np.ndarray, label: str) -> np.ndarray:
    result = []
    for value in target:
        matches = np.flatnonzero(np.isclose(source, value, atol=1e-8, rtol=0))
        if len(matches) != 1:
            raise RuntimeError(f"coordinate-not-unique:{label}:{value}:{len(matches)}")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=int)


def clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values[(values <= FILL_LIMIT) | (values >= 1e20)] = np.nan
    return values


def read_raw_periods(
    path: Path,
    variable_names: list[str],
    source_start_year: int,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    periods = {"target": TARGET_YEARS, "reference": REFERENCE_YEARS}
    output: dict[str, dict[str, np.ndarray]] = {name: {} for name in periods}
    with netcdf_file(path, "r", mmap=True) as handle:
        source_lat = np.array(handle.variables["lat"][:], dtype=float, copy=True)
        source_lon = np.array(handle.variables["lon"][:], dtype=float, copy=True)
        lat_indices = exact_indices(source_lat, target_lat, f"{path.name}:lat")
        lon_indices = exact_indices(source_lon, target_lon, f"{path.name}:lon")
        lat_start, lat_stop = int(lat_indices.min()), int(lat_indices.max()) + 1
        lon_start, lon_stop = int(lon_indices.min()), int(lon_indices.max()) + 1
        lat_offsets = lat_indices - lat_start
        lon_offsets = lon_indices - lon_start
        for period_name, years in periods.items():
            assert_unlocked(years, f"{path.name}:{period_name}")
            start = (int(years[0]) - source_start_year) * 12
            stop = (int(years[-1]) - source_start_year + 1) * 12
            expected = len(years) * 12
            for variable_name in variable_names:
                block = np.array(
                    handle.variables[variable_name][start:stop, lat_start:lat_stop, lon_start:lon_stop],
                    dtype=float,
                    copy=True,
                )
                values = clean(block[:, lat_offsets, :][:, :, lon_offsets])
                if values.shape != (expected, len(target_lat), len(target_lon)):
                    raise RuntimeError(f"raw-shape-contract-failed:{path.name}:{variable_name}:{values.shape}")
                output[period_name][variable_name] = values
    return output


def annual_candidates(monthly: np.ndarray, years: np.ndarray) -> dict[str, np.ndarray]:
    shaped = monthly.reshape(len(years), 12, monthly.shape[1], monthly.shape[2])
    days = np.asarray(
        [[calendar.monthrange(int(year), month)[1] for month in range(1, 13)] for year in years],
        dtype=float,
    )
    return {
        "legacy_mean_monthly_total_div30": np.nanmean(shaped, axis=1) / 30.0,
        "mean_monthly_total_div30p4375": np.nanmean(shaped, axis=1) / (365.25 / 12.0),
        "physical_annual_total_div_days": np.nansum(shaped, axis=1) / days.sum(axis=1)[:, None, None],
        "mean_monthly_daily_rate": np.nanmean(shaped / days[:, :, None, None], axis=1),
    }


def anomaly(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return target - np.nanmean(reference, axis=0, keepdims=True)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def closure_metrics(candidate: np.ndarray, stored: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(candidate) & np.isfinite(stored)
    return {
        "finite_pair_count": int(valid.sum()),
        "correlation": correlation(candidate, stored),
        "rmse_mm_day": float(np.sqrt(np.mean(np.square(candidate[valid] - stored[valid])))),
        "maximum_absolute_error_mm_day": float(np.max(np.abs(candidate[valid] - stored[valid]))),
    }


def weighted_mean(field: np.ndarray, latitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    values = np.where(mask[None, :, :], field, np.nan)
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def trend(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(TARGET_YEARS[valid], values[valid], 1)[0] * 100.0)


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "hma_climate": HMA_CLIMATE,
        "cru_precipitation": CRU_PRECIPITATION,
        "gpcc_precipitation": GPCC_PRECIPITATION,
        "baseline_summary": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/summary.json",
        "locked_contract": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    contract = json.loads(sources["locked_contract"].read_text(encoding="utf-8"))
    baseline = json.loads(sources["baseline_summary"].read_text(encoding="utf-8"))
    if contract["status"] != "locked_not_evaluated" or baseline["locked_final_period_evaluated"]:
        raise RuntimeError("locked-period-contract-failed")
    assert_unlocked(TARGET_YEARS, "stored-target")
    assert_unlocked(REFERENCE_YEARS, "stored-reference")
    before = {
        name: source_state(path, include_hash=name in {"baseline_summary", "locked_contract", "facts"})
        for name, path in sources.items()
    }

    with h5py.File(HMA_CLIMATE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in TARGET_YEARS])
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        stored_cru = clean(np.asarray(handle["pr_obs_cru_pre_05deg"][indices, :, :], dtype=float))
        stored_gpcc = clean(np.asarray(handle["pr_obs_gpcp_05deg"][indices, :, :], dtype=float))
        rec_result = clean(np.asarray(handle["pr_ensemble_mean_rec_result"][indices, :, :], dtype=float))

    cru_raw = read_raw_periods(CRU_PRECIPITATION, ["pre", "stn"], 1901, latitudes, longitudes)
    gpcc_raw = read_raw_periods(
        GPCC_PRECIPITATION,
        ["precip", "numgauge", "infilled_numgauges", "interpolation_error", "interpolation_error_infilled"],
        1891,
        latitudes,
        longitudes,
    )

    closure_rows = []
    selected_candidates: dict[str, str] = {}
    for product, target_monthly, reference_monthly, stored in (
        ("cru", cru_raw["target"]["pre"], cru_raw["reference"]["pre"], stored_cru),
        ("gpcc", gpcc_raw["target"]["precip"], gpcc_raw["reference"]["precip"], stored_gpcc),
    ):
        target_candidates = annual_candidates(target_monthly, TARGET_YEARS)
        reference_candidates = annual_candidates(reference_monthly, REFERENCE_YEARS)
        product_rows = []
        for label in target_candidates:
            candidate = anomaly(target_candidates[label], reference_candidates[label])
            row = {"product": product, "candidate": label, **closure_metrics(candidate, stored)}
            closure_rows.append(row)
            product_rows.append(row)
        selected = sorted(product_rows, key=lambda row: (-float(row["correlation"]), float(row["rmse_mm_day"])))[0]
        selected_candidates[product] = str(selected["candidate"])
    closure = pd.DataFrame(closure_rows)

    cru_station = cru_raw["target"]["stn"]
    gpcc_gauge = gpcc_raw["target"]["numgauge"]
    gpcc_infilled = gpcc_raw["target"]["infilled_numgauges"]
    gpcc_error = gpcc_raw["target"]["interpolation_error"]
    gpcc_error_infilled = gpcc_raw["target"]["interpolation_error_infilled"]
    cru_station_fraction = np.mean(np.nan_to_num(cru_station, nan=0.0) > 0, axis=0)
    gpcc_gauge_fraction = np.mean(np.nan_to_num(gpcc_gauge, nan=0.0) > 0, axis=0)
    gpcc_infilled_fraction = np.mean(np.nan_to_num(gpcc_infilled, nan=0.0) > 0, axis=0)
    joint_complete = (
        np.all(np.isfinite(stored_cru), axis=0)
        & np.all(np.isfinite(stored_gpcc), axis=0)
        & np.all(np.isfinite(rec_result), axis=0)
    )
    masks = {
        "all_joint_complete": joint_complete,
        "dual_high": joint_complete & (cru_station_fraction >= 0.5) & (gpcc_gauge_fraction >= 0.5),
        "dual_any": joint_complete & (cru_station_fraction > 0) & (gpcc_gauge_fraction > 0),
        "single_any": joint_complete & ((cru_station_fraction > 0) ^ (gpcc_gauge_fraction > 0)),
        "neither_any": joint_complete & (cru_station_fraction == 0) & (gpcc_gauge_fraction == 0),
        "cru_high": joint_complete & (cru_station_fraction >= 0.5),
        "gpcc_high": joint_complete & (gpcc_gauge_fraction >= 0.5),
    }
    products = {"cru": stored_cru, "gpcc": stored_gpcc, "rec_result": rec_result}
    area_weights = np.cos(np.deg2rad(latitudes))[:, None]
    joint_area = float(np.sum(area_weights * joint_complete))
    annual = pd.DataFrame({"year": TARGET_YEARS})
    score_rows = []
    for mask_name, mask in masks.items():
        curves = {name: weighted_mean(values, latitudes, mask) for name, values in products.items()}
        for product_name, values in curves.items():
            annual[f"{mask_name}_{product_name}_mm_day"] = values
        cru_trend = trend(curves["cru"])
        gpcc_trend = trend(curves["gpcc"])
        rec_trend = trend(curves["rec_result"])
        lower = np.minimum(curves["cru"], curves["gpcc"])
        upper = np.maximum(curves["cru"], curves["gpcc"])
        score_rows.append({
            "support_mask": mask_name,
            "gridcell_count": int(mask.sum()),
            "area_fraction_of_joint_complete": float(np.sum(area_weights * mask) / joint_area) if joint_area > 0 else float("nan"),
            "cru_trend_mm_day_per_century": cru_trend,
            "gpcc_trend_mm_day_per_century": gpcc_trend,
            "rec_result_trend_mm_day_per_century": rec_trend,
            "rec_result_minus_more_negative_product_trend": rec_trend - min(cru_trend, gpcc_trend),
            "rec_result_outside_annual_product_bounds_years": int(np.sum((curves["rec_result"] < lower) | (curves["rec_result"] > upper))),
            "cru_gpcc_annual_correlation": correlation(curves["cru"], curves["gpcc"]),
            "cru_gpcc_annual_rmse_mm_day": float(np.sqrt(np.nanmean(np.square(curves["cru"] - curves["gpcc"])))),
        })
    scores = pd.DataFrame(score_rows)

    category = np.full(joint_complete.shape, "outside_joint_complete", dtype=object)
    category[masks["neither_any"]] = "neither_any"
    category[masks["single_any"]] = "single_any"
    category[masks["dual_any"]] = "dual_any"
    category[masks["dual_high"]] = "dual_high"
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    grid = pd.DataFrame({
        "latitude": latitude_grid.reshape(-1),
        "longitude": longitude_grid.reshape(-1),
        "joint_complete": joint_complete.reshape(-1),
        "support_category": category.reshape(-1),
        "cru_station_month_fraction_1901_1920": cru_station_fraction.reshape(-1),
        "cru_mean_station_count_1901_1920": np.nanmean(cru_station, axis=0).reshape(-1),
        "gpcc_direct_gauge_month_fraction_1901_1920": gpcc_gauge_fraction.reshape(-1),
        "gpcc_mean_direct_gauge_count_1901_1920": np.nanmean(gpcc_gauge, axis=0).reshape(-1),
        "gpcc_infilled_gauge_month_fraction_1901_1920": gpcc_infilled_fraction.reshape(-1),
        "gpcc_mean_infilled_gauge_count_1901_1920": np.nanmean(gpcc_infilled, axis=0).reshape(-1),
        "gpcc_mean_interpolation_error_mm_month_1901_1920": np.nanmean(gpcc_error, axis=0).reshape(-1),
        "gpcc_mean_infilled_interpolation_error_mm_month_1901_1920": np.nanmean(gpcc_error_infilled, axis=0).reshape(-1),
    })

    strict_count = int(masks["dual_high"].sum())
    primary_mask = "dual_high" if strict_count >= 20 else "dual_any"
    primary = scores.set_index("support_mask").loc[primary_mask]
    rec_trend = float(primary["rec_result_trend_mm_day_per_century"])
    product_lower = min(float(primary["cru_trend_mm_day_per_century"]), float(primary["gpcc_trend_mm_day_per_century"]))
    persists = bool(np.isfinite(rec_trend) and np.isfinite(product_lower) and rec_trend < product_lower)
    classification = (
        "reconstruction_negative_trend_persists_on_supported_cells"
        if persists
        else "reference_support_changes_trend_diagnosis"
    )

    output.mkdir(parents=True)
    closure.to_csv(output / "raw_monthly_to_stored_annual_closure.csv", index=False)
    scores.to_csv(output / "support_stratum_scores.csv", index=False)
    annual.to_csv(output / "support_stratum_hma_curves_1901_1920.csv", index=False)
    grid.to_csv(output / "hma_grid_reference_support_1901_1920.csv", index=False)

    colors = {"cru": "#6A3D9A", "gpcc": "#777777", "rec_result": "#D55E00"}
    labels = {"cru": "CRU", "gpcc": "GPCC", "rec_result": "rec_result"}
    panels = ["all_joint_complete", "dual_high", "dual_any", "neither_any"]
    titles = {
        "all_joint_complete": "All jointly complete cells",
        "dual_high": "Direct support in at least 50% of months",
        "dual_any": "Both products have some direct support",
        "neither_any": "Neither product has direct support",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True, constrained_layout=True)
    for axis, mask_name in zip(axes.reshape(-1), panels):
        count = int(masks[mask_name].sum())
        if count == 0:
            axis.text(0.5, 0.5, "No cells", transform=axis.transAxes, ha="center", va="center")
        else:
            for product_name in products:
                axis.plot(TARGET_YEARS, annual[f"{mask_name}_{product_name}_mm_day"], color=colors[product_name], linewidth=2.0 if product_name == "rec_result" else 1.6, label=labels[product_name])
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.set_title(f"{titles[mask_name]} (n={count})")
        axis.set_ylabel("HMA precipitation anomaly (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.savefig(output / "support_stratified_hma_precipitation_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "target_period": TARGET_YEARS.tolist(),
        "reference_period": [1961, 1990],
        "locked_1921_1939_accessed": False,
        "raw_monthly_to_stored_selected_candidates": selected_candidates,
        "raw_monthly_to_stored_closure": closure.to_dict(orient="records"),
        "support_definitions": {
            "dual_high": "Both CRU stn and GPCC numgauge are positive in at least 50% of 1901-1920 months.",
            "dual_any": "Both CRU stn and GPCC numgauge are positive in at least one 1901-1920 month.",
            "single_any": "Exactly one product has any direct station/gauge support.",
            "neither_any": "Neither product has any direct station/gauge support.",
        },
        "support_scores": scores.to_dict(orient="records"),
        "primary_supported_mask": primary_mask,
        "classification": classification,
        "interpretation_boundary": "Station and gauge counts quantify direct input support, not truth or a calibrated uncertainty interval. The audit does not authorize replacing CRU, GPCC, or the reconstruction.",
    }
    after = {
        name: source_state(path, include_hash=name in {"baseline_summary", "locked_contract", "facts"})
        for name, path in sources.items()
    }
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "driver_sha256": digest(Path(__file__))}, indent=2),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-reference-support-audit")
    (output / "FINAL_RESULT.md").write_text(
        "# Early Precipitation Reference Support Audit v1\n\n"
        f"- Jointly complete cells: `{int(masks['all_joint_complete'].sum())}`.\n"
        f"- Dual-high cells: `{strict_count}`.\n"
        f"- Dual-any cells: `{int(masks['dual_any'].sum())}`.\n"
        f"- Neither-any cells: `{int(masks['neither_any'].sum())}`.\n"
        f"- Primary supported mask: `{primary_mask}`.\n"
        f"- CRU/GPCC/rec_result trends on the primary mask: `{primary['cru_trend_mm_day_per_century']:+.3f}` / `{primary['gpcc_trend_mm_day_per_century']:+.3f}` / `{primary['rec_result_trend_mm_day_per_century']:+.3f} mm/day/century`.\n\n"
        f"Classification: `{classification}`. Station and gauge support is not observational truth or a production correction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
