"""Read-only GPCC v2022 support audit at frozen proxy9 calibration sites."""

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


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path, digest: bool = True) -> dict[str, object]:
    stat = path.stat()
    item: dict[str, object] = {
        "path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)
    }
    if digest:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
        item["sha256"] = value.hexdigest()
    return item


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = root()
    output = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_03_gpcc_calibration_target_audit_v2"
    )
    proxy_path = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/"
        "proxy9_allowed_1940_2000_official_baseline_v2.mat"
    )
    gpcc_path = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc")
    facts_path = project / "LIA_PROJECT_FACTS.md"
    driver_path = Path(__file__).resolve()
    for path in (proxy_path, gpcc_path, facts_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)

    before = {
        "proxy9": state(proxy_path),
        "gpcc_v2022": state(gpcc_path, digest=False),
        "facts": state(facts_path),
        "driver": state(driver_path),
    }
    proxy = scipy.io.loadmat(
        proxy_path, variable_names=["locations", "model_type", "archived_b"]
    )
    locations = np.asarray(proxy["locations"], dtype=float)
    model_type = np.asarray(proxy["model_type"], dtype=int).ravel()
    b2 = np.asarray(proxy["archived_b"], dtype=float)[:, 2]
    sensitive = model_type == 2
    if locations.shape != (297, 2) or sensitive.sum() != 290:
        raise RuntimeError("unexpected-proxy9-contract")

    with xr.open_dataset(gpcc_path, decode_times=True) as dataset:
        selected = dataset[[
            "precip", "numgauge", "infilled_numgauges", "interpolation_error"
        ]].sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
        if selected.sizes["time"] != 732:
            raise RuntimeError("unexpected-month-count")
        source_lat = np.asarray(selected.lat, dtype=float)
        source_lon = np.asarray(selected.lon, dtype=float)
        lat_index = np.asarray([np.argmin(np.abs(source_lat - x)) for x in locations[:, 0]])
        lon_index = np.asarray([np.argmin(np.abs(source_lon - x)) for x in locations[:, 1]])
        site_lat = source_lat[lat_index]
        site_lon = source_lon[lon_index]
        ilat = xr.DataArray(lat_index, dims="site")
        ilon = xr.DataArray(lon_index, dims="site")
        arrays = {
            name: np.asarray(selected[name].isel(lat=ilat, lon=ilon).values, dtype=float)
            for name in ("precip", "numgauge", "infilled_numgauges", "interpolation_error")
        }
        month_year = np.asarray(selected.time.dt.year, dtype=int)

    cells, inverse = np.unique(np.column_stack([site_lat, site_lon]), axis=0, return_inverse=True)
    precip = arrays["precip"]
    gauge = arrays["numgauge"]
    infilled = arrays["infilled_numgauges"]
    error = arrays["interpolation_error"]
    n_years = END_YEAR - START_YEAR + 1
    legacy = np.full((n_years, len(model_type)), np.nan)
    physical = np.full_like(legacy, np.nan)
    complete = np.zeros_like(legacy, dtype=bool)
    direct_by_year = np.full(n_years, np.nan)

    annual_rows: list[dict[str, object]] = []
    for yi, year in enumerate(range(START_YEAR, END_YEAR + 1)):
        pick = month_year == year
        p = precip[pick]
        g = gauge[pick]
        f = infilled[pick]
        finite_months = np.sum(np.isfinite(p), axis=0)
        complete[yi] = finite_months == 12
        legacy[yi] = np.nansum(p, axis=0) / 12.0 / 30.0
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        physical[yi, complete[yi]] = np.sum(p[:, complete[yi]], axis=0) / (366 if leap else 365)
        direct_by_year[yi] = np.median(np.mean(np.isfinite(g[:, sensitive]) & (g[:, sensitive] > 0), axis=0))
        for record in range(len(model_type)):
            annual_rows.append({
                "record_index_1based": record + 1,
                "year": year,
                "precipitation_sensitive": bool(sensitive[record]),
                "finite_precip_months": int(finite_months[record]),
                "direct_gauge_months": int(np.sum(np.isfinite(g[:, record]) & (g[:, record] > 0))),
                "infilled_gauge_months": int(np.sum(np.isfinite(f[:, record]) & (f[:, record] > 0))),
                "legacy_mm_day": float(legacy[yi, record]),
                "physical_complete_mm_day": float(physical[yi, record]),
                "legacy_minus_physical_mm_day": float(legacy[yi, record] - physical[yi, record]),
            })

    direct_fraction = np.mean(np.isfinite(gauge) & (gauge > 0), axis=0)
    infilled_fraction = np.mean(np.isfinite(infilled) & (infilled > 0), axis=0)
    record_rows: list[dict[str, object]] = []
    for record in range(len(model_type)):
        delta = legacy[:, record] - physical[:, record]
        record_rows.append({
            "record_index_1based": record + 1,
            "model_type": int(model_type[record]),
            "precipitation_sensitive": bool(sensitive[record]),
            "proxy_lat": float(locations[record, 0]),
            "proxy_lon": float(locations[record, 1]),
            "grid_lat": float(site_lat[record]),
            "grid_lon": float(site_lon[record]),
            "unique_grid_cell_index_1based": int(inverse[record] + 1),
            "archived_b2": float(b2[record]),
            "finite_precip_month_fraction": float(np.mean(np.isfinite(precip[:, record]))),
            "complete_year_fraction": float(np.mean(complete[:, record])),
            "direct_gauge_month_fraction": float(direct_fraction[record]),
            "infilled_gauge_month_fraction": float(infilled_fraction[record]),
            "median_numgauge": float(np.nanmedian(gauge[:, record])),
            "median_infilled_numgauges": float(np.nanmedian(infilled[:, record])),
            "median_interpolation_error_mm_month": float(np.nanmedian(error[:, record])),
            "median_legacy_minus_physical_mm_day": float(np.nanmedian(delta)),
        })

    all_complete = complete[:, sensitive]
    delta = legacy[:, sensitive] - physical[:, sensitive]
    finite_delta = delta[np.isfinite(delta)]
    summary = {
        "status": "complete",
        "scope": {
            "years": [START_YEAR, END_YEAR],
            "locked_1921_1939_accessed": False,
            "proxy_records": int(len(model_type)),
            "precipitation_sensitive_records": int(sensitive.sum()),
            "unique_gpcc_cells_all": int(len(cells)),
            "unique_gpcc_cells_precipitation_sensitive": int(np.unique(inverse[sensitive]).size),
        },
        "source_identity": {
            "loader_label": "gpcp",
            "actual_product": "GPCC Full Data Monthly Product Version 2022",
            "precipitation_units": "mm/month",
            "historical_psm_source_snapshot_frozen": False,
            "current_source_exactly_reproduces_stored_annual_field": False,
        },
        "precipitation_sensitive_support": {
            "finite_precip_month_fraction": float(np.mean(np.isfinite(precip[:, sensitive]))),
            "complete_site_year_fraction": float(np.mean(all_complete)),
            "records_complete_all_61_years": int(np.sum(np.all(all_complete, axis=0))),
            "direct_gauge_month_fraction": float(np.mean(np.isfinite(gauge[:, sensitive]) & (gauge[:, sensitive] > 0))),
            "records_direct_gauge_at_least_half_months": int(np.sum(direct_fraction[sensitive] >= 0.5)),
            "records_no_direct_gauge_any_month": int(np.sum(direct_fraction[sensitive] == 0)),
            "infilled_gauge_month_fraction": float(np.mean(np.isfinite(infilled[:, sensitive]) & (infilled[:, sensitive] > 0))),
            "record_direct_gauge_fraction": stats(direct_fraction[sensitive]),
            "record_infilled_gauge_fraction": stats(infilled_fraction[sensitive]),
            "record_median_numgauge": stats(np.nanmedian(gauge[:, sensitive], axis=0)),
            "record_median_interpolation_error_mm_month": stats(np.nanmedian(error[:, sensitive], axis=0)),
        },
        "annualization": {
            "legacy_rule": "nansum(monthly_mm)/12/30",
            "physical_rule": "sum(complete_12_month_totals)/calendar_days",
            "partial_site_year_count": int(np.sum(~all_complete)),
            "comparable_site_year_count": int(finite_delta.size),
            "legacy_minus_physical_mm_day": stats(finite_delta),
            "maximum_absolute_difference_mm_day": float(np.max(np.abs(finite_delta))),
        },
        "decision": "freeze_versioned_completeness_aware_gpcc_target_before_new_psm_fit",
    }

    write_csv(output / "proxy_record_support.csv", record_rows)
    write_csv(output / "proxy_record_year_annualization.csv", annual_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    years = np.arange(START_YEAR, END_YEAR + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    axes[0].plot(years, direct_by_year, color="#005f73", linewidth=2)
    axes[0].set(title="Direct-gauge support by year", ylabel="Median monthly fraction", ylim=(-0.02, 1.02))
    axes[0].grid(alpha=0.25)
    axes[1].hist(direct_fraction[sensitive], bins=np.linspace(0, 1, 21), color="#ee9b00", edgecolor="white")
    axes[1].set(title="Support across precipitation PSMs", xlabel="Months with direct gauge", ylabel="Proxy records")
    axes[2].hist(finite_delta, bins=35, color="#bb3e03", edgecolor="white")
    axes[2].set(title="Legacy minus physical annualization", xlabel="mm/day", ylabel="Proxy-years")
    fig.suptitle("GPCC v2022 calibration-target audit, 1940-2000", fontsize=14)
    fig.savefig(output / "gpcc_calibration_target_audit.png", dpi=180)
    plt.close(fig)

    support = summary["precipitation_sensitive_support"]
    annual = summary["annualization"]
    report = (
        "# GPCC Calibration-Target Audit\n\n"
        f"- Scope: 1940-2000; locked 1921-1939 accessed: `false`.\n"
        f"- Precipitation-sensitive records: `{sensitive.sum()}` on "
        f"`{summary['scope']['unique_gpcc_cells_precipitation_sensitive']}` unique GPCC cells.\n"
        f"- Complete site-year fraction: `{support['complete_site_year_fraction']:.6f}`; "
        f"partial site-years: `{annual['partial_site_year_count']}`.\n"
        f"- Direct-gauge month fraction: `{support['direct_gauge_month_fraction']:.6f}`.\n"
        f"- Records with direct gauges in at least half the months: "
        f"`{support['records_direct_gauge_at_least_half_months']}/{sensitive.sum()}`; "
        f"records with none: `{support['records_no_direct_gauge_any_month']}/{sensitive.sum()}`.\n"
        f"- Median legacy-minus-physical conversion: "
        f"`{annual['legacy_minus_physical_mm_day']['median']:.9f} mm/day`; maximum absolute difference "
        f"`{annual['maximum_absolute_difference_mm_day']:.9f} mm/day`.\n\n"
        "Decision: the arithmetic conversion is quantified, but another PSM must not be fit against "
        "the mutable loader target. First freeze a versioned, completeness-aware GPCC target and carry "
        "gauge support into validation or observation-error weighting.\n"
    )
    (output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    after = {
        "proxy9": state(proxy_path),
        "gpcc_v2022": state(gpcc_path, digest=False),
        "facts": state(facts_path),
        "driver": state(driver_path),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_before": before,
        "sources_after": after,
        "source_states_unchanged": before == after,
        "draft_v1_executed": False,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
