"""Audit GPCC v2022 support and annualization at frozen proxy9 sites."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


START_YEAR = 1940
END_YEAR = 2000


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def file_state(path: Path, include_hash: bool = True) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty-csv:{path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_03_gpcc_calibration_target_audit"
    )
    proxy_path = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/"
        "proxy9_allowed_1940_2000_official_baseline_v2.mat"
    )
    gpcc_path = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc")
    facts_path = root / "LIA_PROJECT_FACTS.md"
    driver_path = Path(__file__).resolve()

    for path in (proxy_path, gpcc_path, facts_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)

    before = {
        "proxy9": file_state(proxy_path),
        "gpcc_v2022": file_state(gpcc_path, include_hash=False),
        "facts": file_state(facts_path),
        "driver": file_state(driver_path),
    }

    proxy = scipy.io.loadmat(
        proxy_path, variable_names=["locations", "model_type", "archived_b"]
    )
    locations = np.asarray(proxy["locations"], dtype=float)
    model_type = np.asarray(proxy["model_type"], dtype=int).ravel()
    archived_b = np.asarray(proxy["archived_b"], dtype=float)
    if locations.shape != (297, 2) or model_type.shape != (297,):
        raise RuntimeError("unexpected-proxy9-shape")
    if np.count_nonzero(model_type == 2) != 290:
        raise RuntimeError("unexpected-precipitation-sensitive-proxy-count")

    with xr.open_dataset(gpcc_path, decode_times=True) as dataset:
        selected = dataset[[
            "precip", "numgauge", "infilled_numgauges", "interpolation_error"
        ]].sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
        if selected.sizes["time"] != (END_YEAR - START_YEAR + 1) * 12:
            raise RuntimeError("unexpected-month-count")
        source_lat = np.asarray(selected["lat"], dtype=float)
        source_lon = np.asarray(selected["lon"], dtype=float)
        lat_index = np.asarray(
            [int(np.argmin(np.abs(source_lat - value))) for value in locations[:, 0]], dtype=int
        )
        lon_index = np.asarray(
            [int(np.argmin(np.abs(source_lon - value))) for value in locations[:, 1]], dtype=int
        )
        site_lat = source_lat[lat_index]
        site_lon = source_lon[lon_index]
        years = np.asarray(selected["time"].dt.year, dtype=int)

        arrays: dict[str, np.ndarray] = {}
        for name in ("precip", "numgauge", "infilled_numgauges", "interpolation_error"):
            raw = np.asarray(selected[name].values, dtype=float)
            arrays[name] = raw[:, lat_index, lon_index]

    unique_cells, inverse = np.unique(
        np.column_stack([site_lat, site_lon]), axis=0, return_inverse=True
    )
    calendar_days = np.asarray(
        [366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
         for year in range(START_YEAR, END_YEAR + 1)],
        dtype=float,
    )

    annual_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    year_values: dict[int, dict[str, np.ndarray]] = {}
    precip = arrays["precip"]
    gauge = arrays["numgauge"]
    infilled = arrays["infilled_numgauges"]
    error = arrays["interpolation_error"]
    annual_legacy = np.full((END_YEAR - START_YEAR + 1, locations.shape[0]), np.nan)
    annual_physical = np.full_like(annual_legacy, np.nan)
    annual_complete = np.zeros_like(annual_legacy, dtype=bool)

    for year_index, year in enumerate(range(START_YEAR, END_YEAR + 1)):
        month = years == year
        p = precip[month]
        g = gauge[month]
        f = infilled[month]
        e = error[month]
        finite_count = np.sum(np.isfinite(p), axis=0)
        complete = finite_count == 12
        legacy = np.nansum(p, axis=0) / 12.0 / 30.0
        physical = np.full(locations.shape[0], np.nan)
        physical[complete] = np.sum(p[:, complete], axis=0) / calendar_days[year_index]
        annual_legacy[year_index] = legacy
        annual_physical[year_index] = physical
        annual_complete[year_index] = complete
        year_values[year] = {
            "direct_fraction": np.mean(np.isfinite(g) & (g > 0), axis=0),
            "infilled_fraction": np.mean(np.isfinite(f) & (f > 0), axis=0),
            "complete": complete,
        }
        for record in range(locations.shape[0]):
            annual_rows.append({
                "record_index_1based": record + 1,
                "year": year,
                "model_type": int(model_type[record]),
                "precipitation_sensitive": bool(model_type[record] == 2),
                "grid_lat": float(site_lat[record]),
                "grid_lon": float(site_lon[record]),
                "finite_precip_months": int(finite_count[record]),
                "direct_gauge_months": int(np.sum(np.isfinite(g[:, record]) & (g[:, record] > 0))),
                "infilled_gauge_months": int(np.sum(np.isfinite(f[:, record]) & (f[:, record] > 0))),
                "legacy_mm_day": float(legacy[record]),
                "physical_complete_mm_day": float(physical[record]),
                "legacy_minus_physical_mm_day": float(legacy[record] - physical[record]),
            })

    for record in range(locations.shape[0]):
        p = precip[:, record]
        g = gauge[:, record]
        f = infilled[:, record]
        e = error[:, record]
        complete_years = annual_complete[:, record]
        direct_month = np.isfinite(g) & (g > 0)
        infilled_month = np.isfinite(f) & (f > 0)
        physical = annual_physical[:, record]
        legacy = annual_legacy[:, record]
        valid_difference = np.isfinite(physical) & np.isfinite(legacy)
        record_rows.append({
            "record_index_1based": record + 1,
            "model_type": int(model_type[record]),
            "precipitation_sensitive": bool(model_type[record] == 2),
            "proxy_lat": float(locations[record, 0]),
            "proxy_lon": float(locations[record, 1]),
            "grid_lat": float(site_lat[record]),
            "grid_lon": float(site_lon[record]),
            "unique_grid_cell_index_1based": int(inverse[record] + 1),
            "archived_b2": float(archived_b[record, 2]),
            "finite_precip_month_fraction": float(np.mean(np.isfinite(p))),
            "complete_year_fraction": float(np.mean(complete_years)),
            "direct_gauge_month_fraction": float(np.mean(direct_month)),
            "no_direct_gauge_month_fraction": float(np.mean(~direct_month)),
            "infilled_gauge_month_fraction": float(np.mean(infilled_month)),
            "median_numgauge": float(np.nanmedian(g)),
            "median_infilled_numgauges": float(np.nanmedian(f)),
            "median_interpolation_error_mm_month": float(np.nanmedian(e)),
            "median_legacy_minus_physical_mm_day": (
                float(np.median(legacy[valid_difference] - physical[valid_difference]))
                if np.any(valid_difference) else float("nan")
            ),
        })

    sensitive = model_type == 2
    unique_sensitive_cells = np.unique(inverse[sensitive]).size
    all_complete = annual_complete[:, sensitive]
    direct_fraction = np.mean(np.isfinite(gauge[:, sensitive]) & (gauge[:, sensitive] > 0), axis=0)
    infilled_fraction = np.mean(np.isfinite(infilled[:, sensitive]) & (infilled[:, sensitive] > 0), axis=0)
    difference = annual_legacy[:, sensitive] - annual_physical[:, sensitive]
    valid_difference = np.isfinite(difference)
    site_error = np.nanmedian(error[:, sensitive], axis=0)
    site_gauges = np.nanmedian(gauge[:, sensitive], axis=0)

    direct_by_year = np.asarray([
        np.median(year_values[year]["direct_fraction"][sensitive])
        for year in range(START_YEAR, END_YEAR + 1)
    ])
    complete_by_year = np.asarray([
        np.mean(year_values[year]["complete"][sensitive])
        for year in range(START_YEAR, END_YEAR + 1)
    ])

    summary = {
        "status": "complete",
        "scope": {
            "years": [START_YEAR, END_YEAR],
            "locked_1921_1939_accessed": False,
            "proxy_records": int(locations.shape[0]),
            "precipitation_sensitive_records": int(sensitive.sum()),
            "unique_gpcc_cells_all_records": int(unique_cells.shape[0]),
            "unique_gpcc_cells_precipitation_sensitive": int(unique_sensitive_cells),
        },
        "source_identity": {
            "label_in_loader": "gpcp",
            "actual_product": "GPCC Full Data Monthly Product Version 2022",
            "precip_units": "mm/month",
            "version_is_embedded_but_source_was_not_frozen_with_historical_psm": True,
        },
        "precipitation_sensitive_support": {
            "finite_precip_month_fraction": float(np.mean(np.isfinite(precip[:, sensitive]))),
            "complete_site_year_fraction": float(np.mean(all_complete)),
            "records_complete_all_61_years": int(np.sum(np.all(all_complete, axis=0))),
            "direct_gauge_month_fraction": float(np.mean(np.isfinite(gauge[:, sensitive]) & (gauge[:, sensitive] > 0))),
            "records_with_direct_gauge_in_at_least_half_months": int(np.sum(direct_fraction >= 0.5)),
            "records_with_no_direct_gauge_in_any_month": int(np.sum(direct_fraction == 0)),
            "infilled_gauge_month_fraction": float(np.mean(np.isfinite(infilled[:, sensitive]) & (infilled[:, sensitive] > 0))),
            "record_direct_gauge_fraction": finite_summary(direct_fraction),
            "record_infilled_gauge_fraction": finite_summary(infilled_fraction),
            "record_median_numgauge": finite_summary(site_gauges),
            "record_median_interpolation_error_mm_month": finite_summary(site_error),
        },
        "annualization": {
            "legacy_rule": "nansum(monthly_mm)/12/30",
            "physical_rule": "sum(12 complete monthly totals)/calendar_days_in_year",
            "comparable_site_year_count": int(np.sum(valid_difference)),
            "legacy_minus_physical_mm_day": finite_summary(difference[valid_difference]),
            "maximum_absolute_difference_mm_day": float(np.max(np.abs(difference[valid_difference]))),
            "partial_year_site_year_count": int(np.sum(~all_complete)),
            "partial_years_are_artificially_low_under_legacy_rule": bool(np.any(~all_complete)),
        },
        "decision": "freeze_and_rebuild_gpcc_calibration_target_before_any_new_psm_fit",
        "reason": (
            "The current file identity and preprocessing do not exactly reproduce the archived target; "
            "support diagnostics determine whether station-poor records require exclusion or uncertainty weighting."
        ),
    }

    write_csv(output / "proxy_record_support.csv", record_rows)
    write_csv(output / "proxy_record_year_annualization.csv", annual_rows)

    plot_years = np.arange(START_YEAR, END_YEAR + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axes[0, 0].plot(plot_years, direct_by_year, color="#005f73", linewidth=2)
    axes[0, 0].set_title("Median monthly direct-gauge support")
    axes[0, 0].set_ylabel("Fraction of months")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(plot_years, complete_by_year, color="#0a9396", linewidth=2)
    axes[0, 1].set_title("Complete GPCC precipitation years")
    axes[0, 1].set_ylabel("Fraction of proxy records")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].hist(direct_fraction, bins=np.linspace(0, 1, 21), color="#ee9b00", edgecolor="white")
    axes[1, 0].set_title("Direct-gauge support across precipitation PSMs")
    axes[1, 0].set_xlabel("Fraction of 1940-2000 months with gauge")
    axes[1, 0].set_ylabel("Proxy records")

    finite_delta = difference[valid_difference]
    axes[1, 1].hist(finite_delta, bins=35, color="#bb3e03", edgecolor="white")
    axes[1, 1].set_title("Legacy minus physical annualization")
    axes[1, 1].set_xlabel("mm/day")
    axes[1, 1].set_ylabel("Proxy-years")
    fig.suptitle("GPCC v2022 calibration-target audit, 1940-2000", fontsize=15)
    fig.savefig(output / "gpcc_calibration_target_audit.png", dpi=180)
    plt.close(fig)

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_result = (
        "# GPCC Calibration-Target Audit\n\n"
        f"- Scope: {START_YEAR}-{END_YEAR}; locked 1921-1939 was not accessed.\n"
        f"- Precipitation-sensitive proxy records: `{sensitive.sum()}` on `{unique_sensitive_cells}` unique GPCC cells.\n"
        f"- Complete proxy-year fraction: `{summary['precipitation_sensitive_support']['complete_site_year_fraction']:.6f}`.\n"
        f"- Direct-gauge month fraction: `{summary['precipitation_sensitive_support']['direct_gauge_month_fraction']:.6f}`.\n"
        f"- Records with direct gauges in at least half of months: "
        f"`{summary['precipitation_sensitive_support']['records_with_direct_gauge_in_at_least_half_months']}/{sensitive.sum()}`.\n"
        f"- Records with no direct gauge in any month: "
        f"`{summary['precipitation_sensitive_support']['records_with_no_direct_gauge_in_any_month']}/{sensitive.sum()}`.\n"
        f"- Median legacy-minus-physical annualization: "
        f"`{summary['annualization']['legacy_minus_physical_mm_day']['median']:.9f} mm/day`; "
        f"maximum absolute difference `{summary['annualization']['maximum_absolute_difference_mm_day']:.9f} mm/day`.\n"
        f"- Partial proxy-years: `{summary['annualization']['partial_year_site_year_count']}`.\n\n"
        "Decision: do not fit another PSM against the mutable loader output. Freeze a versioned, "
        "completeness-aware GPCC calibration target and use gauge support as an explicit validation/uncertainty variable.\n"
    )
    (output / "FINAL_RESULT.md").write_text(final_result, encoding="utf-8")

    after = {
        "proxy9": file_state(proxy_path),
        "gpcc_v2022": file_state(gpcc_path, include_hash=False),
        "facts": file_state(facts_path),
        "driver": file_state(driver_path),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_before": before,
        "sources_after": after,
        "source_states_unchanged": before == after,
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["source_states_unchanged"]:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
