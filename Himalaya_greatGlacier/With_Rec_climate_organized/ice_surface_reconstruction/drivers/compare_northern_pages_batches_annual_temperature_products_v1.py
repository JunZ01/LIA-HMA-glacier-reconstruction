from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_northern_pages_annual_temperature_products_v1"
CLIMATE_FILE = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
BASELINE_YEARS = np.arange(1905, 1915, dtype=int)
EVENT_YEARS = np.arange(1915, 1919, dtype=int)
DISPLAY_YEARS = np.arange(1905, 1921, dtype=int)
PRODUCTS = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU TS": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
    "rec_ref": "tas_ensemble_mean_rec_ref",
    "rec_result": "tas_ensemble_mean_rec_result",
}
OBSERVATIONS = tuple(list(PRODUCTS)[:5])


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


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values.astype(float) - float(target))))


def clean_values(dataset: h5py.Dataset, time_indices: np.ndarray, lat_index: int, lon_index: int) -> np.ndarray:
    values = np.asarray(dataset[time_indices, lat_index, lon_index], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", [-9999.])).reshape(-1)[0])
    values[np.isclose(values, fill) | (values <= -9000) | (values >= 1e20)] = np.nan
    return values


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    study_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_pages2k_fixed_network_study_batch_v1"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "study_metadata": study_root / "fixed_network_study_metadata.csv",
        "climate_file": CLIMATE_FILE,
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    records = pd.read_csv(sources["study_metadata"])
    pages = records[records["source_family"].eq("PAGES2k2017")].copy()
    pages["region"] = "outside_tested_north_blocks"
    pages.loc[pages["latitude"].ge(35.0) & pages["longitude"].lt(80.0), "region"] = "north_west_ge35_lt80E"
    pages.loc[
        pages["latitude"].ge(35.0) & pages["longitude"].ge(90.0) & pages["longitude"].lt(101.0),
        "region",
    ] = "north_east_ge35_90to101E"
    tested = pages[pages["region"].ne("outside_tested_north_blocks")].copy()
    if len(tested) != 29:
        raise RuntimeError(f"north-record-count-contract-failed:{len(tested)}")

    cell_rows = []
    annual_rows = []
    change_rows = []
    with h5py.File(CLIMATE_FILE, "r") as handle:
        years = np.asarray(handle["time"][:], dtype=int)
        latitudes = np.asarray(handle["lat"][:], dtype=float)
        longitudes = np.asarray(handle["lon"][:], dtype=float)
        if not all(variable in handle for variable in PRODUCTS.values()):
            raise RuntimeError("missing-required-temperature-variable")
        display_indices = np.asarray([int(np.where(years == year)[0][0]) for year in DISPLAY_YEARS], dtype=int)
        for row in tested.itertuples(index=False):
            lat_index = nearest_index(latitudes, row.latitude)
            lon_index = nearest_index(longitudes, row.longitude)
            cell_rows.append({
                "proxy_index": int(row.proxy_index),
                "site_id": row.site_id,
                "data_author": row.data_author,
                "region": row.region,
                "proxy_latitude": float(row.latitude),
                "proxy_longitude": float(row.longitude),
                "grid_latitude": float(latitudes[lat_index]),
                "grid_longitude": float(longitudes[lon_index]),
                "lat_index_0based": lat_index,
                "lon_index_0based": lon_index,
                "latitude_offset_degrees": float(latitudes[lat_index] - row.latitude),
                "longitude_offset_degrees": float(longitudes[lon_index] - row.longitude),
                "grid_cell_id": f"{float(latitudes[lat_index]):.2f},{float(longitudes[lon_index]):.2f}",
            })
        cells = pd.DataFrame(cell_rows)
        unique_cells = cells.drop_duplicates(["region", "grid_cell_id"]).copy()
        for region, region_cells in unique_cells.groupby("region"):
            product_cell_curves: dict[str, list[np.ndarray]] = {name: [] for name in PRODUCTS}
            for cell in region_cells.itertuples(index=False):
                for product, variable in PRODUCTS.items():
                    values = clean_values(handle[variable], display_indices, int(cell.lat_index_0based), int(cell.lon_index_0based))
                    product_cell_curves[product].append(values)
                    baseline = values[np.isin(DISPLAY_YEARS, BASELINE_YEARS)]
                    event = values[np.isin(DISPLAY_YEARS, EVENT_YEARS)]
                    complete = bool(np.isfinite(baseline).all() and np.isfinite(event).all())
                    change_rows.append({
                        "region": region,
                        "grid_cell_id": cell.grid_cell_id,
                        "grid_latitude": float(cell.grid_latitude),
                        "grid_longitude": float(cell.grid_longitude),
                        "product": product,
                        "baseline_finite_year_count": int(np.isfinite(baseline).sum()),
                        "event_finite_year_count": int(np.isfinite(event).sum()),
                        "complete_14_year_event": complete,
                        "event_change_c": float(np.mean(event) - np.mean(baseline)) if complete else np.nan,
                    })
            for product, curves in product_cell_curves.items():
                matrix = np.vstack(curves)
                for year_index, year in enumerate(DISPLAY_YEARS):
                    finite = np.isfinite(matrix[:, year_index])
                    annual_rows.append({
                        "region": region,
                        "product": product,
                        "year": int(year),
                        "finite_grid_cell_count": int(finite.sum()),
                        "total_grid_cell_count": int(matrix.shape[0]),
                        "unique_cell_mean_temperature_anomaly_c": float(np.mean(matrix[finite, year_index])) if finite.any() else np.nan,
                    })

    cells = pd.DataFrame(cell_rows)
    annual = pd.DataFrame(annual_rows)
    cell_changes = pd.DataFrame(change_rows)
    group_rows = []
    for (region, product), group in cell_changes.groupby(["region", "product"]):
        complete = group[group["complete_14_year_event"]]
        group_rows.append({
            "region": region,
            "product": product,
            "unique_grid_cell_count": int(len(group)),
            "complete_grid_cell_count": int(len(complete)),
            "complete_grid_cell_fraction": float(len(complete) / len(group)),
            "unique_cell_mean_event_change_c": float(complete["event_change_c"].mean()) if len(complete) else np.nan,
            "unique_cell_min_event_change_c": float(complete["event_change_c"].min()) if len(complete) else np.nan,
            "unique_cell_max_event_change_c": float(complete["event_change_c"].max()) if len(complete) else np.nan,
            "negative_cell_fraction": float((complete["event_change_c"] < 0).mean()) if len(complete) else np.nan,
        })
    group_changes = pd.DataFrame(group_rows)

    consensus_rows = []
    for region in sorted(group_changes["region"].unique()):
        obs = group_changes[(group_changes["region"] == region) & group_changes["product"].isin(OBSERVATIONS)]
        consensus_rows.append({
            "region": region,
            "observation_product_count": int(len(obs)),
            "complete_observation_product_count": int(obs["unique_cell_mean_event_change_c"].notna().sum()),
            "negative_observation_product_count": int((obs["unique_cell_mean_event_change_c"] < 0).sum()),
            "observation_product_mean_event_change_c": float(obs["unique_cell_mean_event_change_c"].mean()),
            "observation_product_min_event_change_c": float(obs["unique_cell_mean_event_change_c"].min()),
            "observation_product_max_event_change_c": float(obs["unique_cell_mean_event_change_c"].max()),
            "observation_product_std_event_change_c": float(obs["unique_cell_mean_event_change_c"].std(ddof=0)),
        })
    consensus = pd.DataFrame(consensus_rows)

    output.mkdir(parents=True)
    cells.to_csv(output / "north_proxy_to_unique_temperature_grid_mapping.csv", index=False)
    annual.to_csv(output / "regional_annual_temperature_curves_1905_1920.csv", index=False)
    cell_changes.to_csv(output / "grid_cell_temperature_event_changes.csv", index=False)
    group_changes.to_csv(output / "regional_temperature_product_event_changes.csv", index=False)
    consensus.to_csv(output / "regional_observation_product_consensus.csv", index=False)

    colors = {
        "Berkeley Earth": "#0072B2", "CRU TS": "#D55E00", "GISTEMP": "#009E73",
        "HadCRUT5": "#CC79A7", "NOAA": "#56B4E9", "rec_ref": "#1B1B1B", "rec_result": "#E69F00",
    }
    regions = ["north_west_ge35_lt80E", "north_east_ge35_90to101E"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), constrained_layout=True)
    for row_index, region in enumerate(regions):
        axis_curve = axes[row_index, 0]
        region_annual = annual[annual["region"] == region]
        for product in PRODUCTS:
            curve = region_annual[region_annual["product"] == product].sort_values("year")
            values = curve["unique_cell_mean_temperature_anomaly_c"].to_numpy(dtype=float)
            baseline = float(np.mean(values[np.isin(curve["year"].to_numpy(dtype=int), BASELINE_YEARS)]))
            axis_curve.plot(
                curve["year"], values - baseline, color=colors[product],
                linewidth=2.2 if product in {"rec_ref", "rec_result"} else 1.15,
                alpha=1.0 if product in {"rec_ref", "rec_result"} else 0.78,
                label=product,
            )
        axis_curve.axhline(0, color="black", linewidth=0.7)
        axis_curve.axvspan(1915, 1918, color="0.7", alpha=0.18)
        axis_curve.set_xlim(1905, 1920)
        axis_curve.set_ylabel("Annual anomaly from 1905-1914 mean (degC)")
        axis_curve.set_title(region.replace("_", " "))
        if row_index == 1:
            axis_curve.set_xlabel("Year")
        if row_index == 0:
            axis_curve.legend(ncol=3, fontsize=8, frameon=False)

        axis_change = axes[row_index, 1]
        region_change = group_changes[group_changes["region"] == region].set_index("product").loc[list(PRODUCTS)]
        positions = np.arange(len(region_change))
        axis_change.scatter(region_change["unique_cell_mean_event_change_c"], positions, c=[colors[name] for name in region_change.index], s=62)
        for position, item in enumerate(region_change.itertuples()):
            axis_change.plot([item.unique_cell_min_event_change_c, item.unique_cell_max_event_change_c], [position, position], color="0.45", linewidth=1.1, zorder=0)
        axis_change.axvline(0, color="black", linewidth=0.8)
        axis_change.set_yticks(positions, region_change.index, fontsize=8)
        axis_change.set_xlabel("1915-1918 minus 1905-1914 (degC)")
        axis_change.set_title("Unique-cell mean and cell range")
    fig.savefig(output / "northern_temperature_products_1905_1920.png", dpi=220)
    plt.close(fig)

    west = consensus[consensus["region"] == regions[0]].iloc[0]
    east = consensus[consensus["region"] == regions[1]].iloc[0]
    summary = {
        "event_definition": "mean(1915-1918) minus mean(1905-1914)",
        "aggregation": "nearest 0.5-degree grid cell for each proxy; duplicate cells removed; unique cells equally weighted",
        "annual_only_limitation": "This compiled product comparison is annual. The PAGES metadata mostly labels the proxies as Summer, so this is a broad forcing audit rather than a seasonality-valid PSM test.",
        "north_record_count": int(len(tested)),
        "north_unique_proxy_site_count": int(tested["site_id"].nunique()),
        "north_unique_grid_cell_count": int(cells["grid_cell_id"].nunique()),
        "west_observation_consensus": west.to_dict(),
        "east_observation_consensus": east.to_dict(),
        "classification": (
            "five_product_annual_temperature_consensus_supports_cooling_in_both_northern_blocks"
            if int(west["negative_observation_product_count"]) >= 4 and int(east["negative_observation_product_count"]) >= 4
            else "annual_instrumental_products_do_not_show_strong_cross_product_cooling_consensus_in_both_blocks"
        ),
        "interpretation": "Annual product agreement or disagreement does not adjudicate a summer-only proxy response or early station representativeness.",
    }

    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-comparison")
    (output / "FINAL_RESULT.md").write_text(
        "# Northern PAGES Annual Temperature-product Comparison v1\n\n"
        f"West observation-product mean event change: {float(west['observation_product_mean_event_change_c']):+.3f} degC "
        f"({int(west['negative_observation_product_count'])}/5 products negative). "
        f"East: {float(east['observation_product_mean_event_change_c']):+.3f} degC "
        f"({int(east['negative_observation_product_count'])}/5 negative).\n\n"
        f"Classification: `{summary['classification']}`. This comparison is annual and therefore cannot replace a summer-season PSM test.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
