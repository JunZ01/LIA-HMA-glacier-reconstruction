from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_northern_pages_monthly_summer_temperature_v1"
EVENT_YEARS = np.arange(1905, 1921, dtype=int)
BASELINE = (1905, 1914)
EVENT = (1915, 1918)
SEASONS = {
    "annual": tuple(range(1, 13)),
    "JJA": (6, 7, 8),
    "AMJJAS": (4, 5, 6, 7, 8, 9),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def period_change(values: np.ndarray) -> np.ndarray:
    baseline = (EVENT_YEARS >= BASELINE[0]) & (EVENT_YEARS <= BASELINE[1])
    event = (EVENT_YEARS >= EVENT[0]) & (EVENT_YEARS <= EVENT[1])
    return np.nanmean(values[event], axis=0) - np.nanmean(values[baseline], axis=0)


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    drivers = Path(__file__).resolve().parent
    rolling = load_module(drivers / "run_clean_gpcc_psm_rolling_pseudo_event_v1.py", "rolling_monthly_reader")
    base = load_module(drivers / "audit_positive_b2_event_seasonal_psm_v1.py", "positive_b2_monthly_sources")
    annual_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_northern_pages_annual_temperature_products_v2"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "grid_mapping": annual_root / "north_proxy_to_unique_temperature_grid_mapping.csv",
        "annual_product_changes": annual_root / "regional_temperature_product_event_changes.csv",
        "berkeley_monthly": Path(base.BERKELEY_TEMPERATURE),
        "cru_monthly": Path(base.CRU_TEMPERATURE),
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    mapping = pd.read_csv(sources["grid_mapping"])
    points = mapping.drop_duplicates(["region", "grid_cell_id"]).sort_values(["region", "grid_cell_id"]).reset_index(drop=True)
    if len(points) != 15:
        raise RuntimeError(f"unique-grid-cell-count-contract-failed:{len(points)}")
    latitudes = points["grid_latitude"].to_numpy(dtype=float)
    longitudes = points["grid_longitude"].to_numpy(dtype=float)

    cru = rolling.read_netcdf3_points(base.CRU_TEMPERATURE, ["tmp", "stn"], 1901, latitudes, longitudes)
    berkeley = rolling.read_berkeley_points(base.BERKELEY_TEMPERATURE, latitudes, longitudes)
    labels = rolling.monthly_labels(1905, 1920)

    curve_rows = []
    point_change_rows = []
    station_rows = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for season, months in SEASONS.items():
        arrays[("Berkeley Earth", season)] = rolling.aggregate_temperature(berkeley["event"], labels, EVENT_YEARS, months)
        arrays[("CRU TS", season)] = rolling.aggregate_temperature(cru["event"]["tmp"], labels, EVENT_YEARS, months)
        station_annual = rolling.aggregate_temperature(cru["event"]["stn"], labels, EVENT_YEARS, months)
        for point_index, point in points.iterrows():
            station_values = station_annual[:, point_index]
            station_rows.append({
                "region": point["region"],
                "grid_cell_id": point["grid_cell_id"],
                "season": season,
                "finite_year_count_1905_1918": int(np.isfinite(station_values[:14]).sum()),
                "years_with_positive_station_count_1905_1918": int(np.sum(station_values[:14] > 0)),
                "mean_cru_station_count_1905_1918": float(np.nanmean(station_values[:14])),
            })

    for (product, season), values in arrays.items():
        changes = period_change(values)
        for point_index, point in points.iterrows():
            for year_index, year in enumerate(EVENT_YEARS):
                curve_rows.append({
                    "region": point["region"],
                    "grid_cell_id": point["grid_cell_id"],
                    "product": product,
                    "season": season,
                    "year": int(year),
                    "temperature_anomaly_c": float(values[year_index, point_index]) if np.isfinite(values[year_index, point_index]) else np.nan,
                })
            point_change_rows.append({
                "region": point["region"],
                "grid_cell_id": point["grid_cell_id"],
                "product": product,
                "season": season,
                "baseline_finite_year_count": int(np.isfinite(values[:10, point_index]).sum()),
                "event_finite_year_count": int(np.isfinite(values[10:14, point_index]).sum()),
                "event_change_c": float(changes[point_index]) if np.isfinite(changes[point_index]) else np.nan,
            })

    curves = pd.DataFrame(curve_rows)
    point_changes = pd.DataFrame(point_change_rows)
    station = pd.DataFrame(station_rows)
    regional_rows = []
    for (region, product, season), group in point_changes.groupby(["region", "product", "season"]):
        complete = group[(group["baseline_finite_year_count"] == 10) & (group["event_finite_year_count"] == 4)]
        regional_rows.append({
            "region": region,
            "product": product,
            "season": season,
            "unique_grid_cell_count": int(len(group)),
            "complete_grid_cell_count": int(len(complete)),
            "complete_grid_cell_fraction": float(len(complete) / len(group)),
            "unique_cell_mean_event_change_c": float(complete["event_change_c"].mean()) if len(complete) else np.nan,
            "unique_cell_min_event_change_c": float(complete["event_change_c"].min()) if len(complete) else np.nan,
            "unique_cell_max_event_change_c": float(complete["event_change_c"].max()) if len(complete) else np.nan,
            "negative_cell_fraction": float((complete["event_change_c"] < 0).mean()) if len(complete) else np.nan,
        })
    regional = pd.DataFrame(regional_rows)

    annual_reference = pd.read_csv(sources["annual_product_changes"])
    annual_reference = annual_reference[annual_reference["product"].isin(["Berkeley Earth", "CRU TS"])][
        ["region", "product", "unique_cell_mean_event_change_c"]
    ].rename(columns={"unique_cell_mean_event_change_c": "compiled_annual_event_change_c"})
    annual_monthly = regional[regional["season"] == "annual"][
        ["region", "product", "unique_cell_mean_event_change_c"]
    ].rename(columns={"unique_cell_mean_event_change_c": "monthly_aggregated_annual_event_change_c"})
    annual_closure = annual_monthly.merge(annual_reference, on=["region", "product"], validate="one_to_one")
    annual_closure["difference_c"] = annual_closure["monthly_aggregated_annual_event_change_c"] - annual_closure["compiled_annual_event_change_c"]

    output.mkdir(parents=True)
    curves.to_csv(output / "monthly_product_point_annualized_curves.csv", index=False)
    point_changes.to_csv(output / "monthly_product_point_event_changes.csv", index=False)
    regional.to_csv(output / "monthly_product_regional_event_changes.csv", index=False)
    station.to_csv(output / "cru_station_support_by_cell_and_season.csv", index=False)
    annual_closure.to_csv(output / "monthly_vs_compiled_annual_closure.csv", index=False)

    regions = ["north_west_ge35_lt80E", "north_east_ge35_90to101E"]
    colors = {"Berkeley Earth": "#0072B2", "CRU TS": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True, constrained_layout=True)
    for row_index, region in enumerate(regions):
        for column_index, season in enumerate(SEASONS):
            axis = axes[row_index, column_index]
            for product in colors:
                selected = curves[(curves["region"] == region) & (curves["product"] == product) & (curves["season"] == season)]
                mean_curve = selected.groupby("year")["temperature_anomaly_c"].mean().reindex(EVENT_YEARS)
                baseline = float(mean_curve.loc[BASELINE[0]:BASELINE[1]].mean())
                axis.plot(EVENT_YEARS, mean_curve.to_numpy(dtype=float) - baseline, color=colors[product], linewidth=2.0, label=product)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.axvspan(EVENT[0], EVENT[1], color="0.7", alpha=0.18)
            axis.set_title(f"{region.replace('_', ' ')} | {season}", fontsize=10)
            if column_index == 0:
                axis.set_ylabel("Anomaly from 1905-1914 mean (degC)")
            if row_index == 1:
                axis.set_xlabel("Year")
            axis.set_xlim(1905, 1920)
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    fig.savefig(output / "northern_monthly_temperature_season_curves_1905_1920.png", dpi=220)
    plt.close(fig)

    summer = regional[regional["season"].isin(["JJA", "AMJJAS"])].copy()
    complete_summer = summer[summer["complete_grid_cell_fraction"] == 1.0]
    all_summer_positive = bool((complete_summer["unique_cell_mean_event_change_c"] > 0).all() and len(complete_summer) == 8)
    summary = {
        "read_periods": [[1905, 1920], [1941, 2000]],
        "locked_1921_1939_accessed": False,
        "event_definition": "mean(1915-1918) minus mean(1905-1914)",
        "aggregation": "15 unique 0.5-degree cells inherited from the annual audit; equal-cell regional means",
        "products": ["Berkeley Earth monthly", "CRU TS monthly"],
        "seasons": list(SEASONS),
        "all_eight_region_product_summer_combinations_complete": bool(len(complete_summer) == 8),
        "all_complete_jja_and_amjjas_region_product_changes_positive": all_summer_positive,
        "regional_event_changes": regional.to_dict(orient="records"),
        "maximum_absolute_monthly_vs_compiled_annual_difference_c": float(annual_closure["difference_c"].abs().max()),
        "classification": (
            "monthly_berkeley_and_cru_support_summer_warming_in_both_northern_blocks"
            if all_summer_positive
            else "monthly_summer_temperature_sign_differs_by_product_or_region"
        ),
        "interpretation": (
            "A summer warming sign opposite to the proxy decline rules out annual averaging as the explanation, "
            "but it still cannot distinguish sparse early observations from chronology or PSM nonstationarity."
        ),
    }

    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-monthly-comparison")
    (output / "FINAL_RESULT.md").write_text(
        "# Northern PAGES Monthly Summer-temperature Comparison v1\n\n"
        f"All eight JJA/AMJJAS region-product combinations are complete: {bool(len(complete_summer) == 8)}; "
        f"all are positive: {all_summer_positive}.\n\n"
        f"Classification: `{summary['classification']}`. This removes annual averaging as the explanation for the sign conflict, "
        "but does not choose between sparse early instrumental coverage and proxy/PSM nonstationarity.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
