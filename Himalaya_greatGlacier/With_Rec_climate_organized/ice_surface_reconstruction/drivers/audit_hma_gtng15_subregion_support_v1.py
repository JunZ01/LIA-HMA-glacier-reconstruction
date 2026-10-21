"""Audit GTN-G HMA subregion support for climate proxies and Core-191 systems."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from shapely.geometry import Point


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1"
GTNG = OUTPUT / "raw/hma_gtng_202307_subregions.gpkg"
REFERENCE = RESULTS / "climate_reconstruction_improvement_loop_v2/round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
CORE191 = RESULTS / "v3_flowline_geometry_audit_core191_v3/v3_flowline_system_geometry.csv"
SYSTEM_MANIFEST = PROJECT / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/system_manifest_work/system_manifest_245_v0.csv"
TEMPERATURE_PROXY = RESULTS / "climate_reconstruction_improvement_loop_v2/round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv"
MOISTURE_PROXY = RESULTS / "climate_reconstruction_improvement_loop_v2/round_12_expanded_moisture_nested_cross_product_v1/network_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_points(frame: pd.DataFrame, polygons: gpd.GeoDataFrame, lon: str, lat: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    region_rows = list(polygons[["o1region", "o2region", "full_name", "long_code", "geometry"]].itertuples(index=False))
    for source in frame.itertuples(index=False):
        point = Point(float(getattr(source, lon)), float(getattr(source, lat)))
        matches = [region for region in region_rows if region.geometry.covers(point)]
        record = source._asdict()
        record["match_count"] = len(matches)
        if len(matches) == 1:
            match = matches[0]
            record.update({
                "o1region": match.o1region,
                "o2region": match.o2region,
                "full_name": match.full_name,
                "long_code": match.long_code,
                "assignment_status": "unique",
            })
        else:
            record.update({
                "o1region": None,
                "o2region": None,
                "full_name": None,
                "long_code": None,
                "assignment_status": "unmatched" if len(matches) == 0 else "ambiguous",
            })
        rows.append(record)
    return pd.DataFrame(rows)


def unique_sites(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    coordinates = frame[["longitude", "latitude"]].round(4).drop_duplicates()
    return int(len(coordinates))


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    sources = {
        "gtng": GTNG,
        "reference": REFERENCE,
        "hma_mask": HMA_MASK,
        "core191": CORE191,
        "system_manifest": SYSTEM_MANIFEST,
        "temperature_proxy": TEMPERATURE_PROXY,
        "moisture_proxy": MOISTURE_PROXY,
    }
    before = {name: sha256(path) for name, path in sources.items()}

    polygons = gpd.read_file(GTNG, layer="gtng_202307_hma").to_crs(4326).sort_values("o2region").reset_index(drop=True)
    if len(polygons) != 15 or polygons["o2region"].duplicated().any():
        raise RuntimeError("GTN-G region contract failed")

    with netcdf_file(REFERENCE, "r", mmap=False) as dataset:
        latitude = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        longitude = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    mask = np.load(HMA_MASK).astype(bool)
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    grid = pd.DataFrame({
        "grid_row": np.where(mask)[0],
        "grid_col": np.where(mask)[1],
        "latitude": lat_grid[mask],
        "longitude": lon_grid[mask],
        "area_weight_coslat": np.cos(np.deg2rad(lat_grid[mask])),
    })
    grid_assignment = assign_points(grid, polygons, "longitude", "latitude")
    grid_assignment.to_csv(OUTPUT / "hma_grid_gtng15_assignment.csv", index=False)

    core_ids = set(pd.read_csv(CORE191, usecols=["lia_unique_id"])["lia_unique_id"].astype(int))
    system = pd.read_csv(SYSTEM_MANIFEST)
    system = system[system["lia_unique_id"].astype(int).isin(core_ids)].copy()
    if len(system) != 191 or system["lia_unique_id"].nunique() != 191:
        raise RuntimeError("Core-191 manifest join failed")
    system_assignment = assign_points(system, polygons, "centroid_lon", "centroid_lat")
    system_assignment.to_csv(OUTPUT / "core191_gtng15_assignment.csv", index=False)

    temperature = pd.read_csv(TEMPERATURE_PROXY)
    moisture = pd.read_csv(MOISTURE_PROXY)
    temperature_assignment = assign_points(temperature, polygons, "longitude", "latitude")
    moisture_assignment = assign_points(moisture, polygons, "longitude", "latitude")
    temperature_assignment.to_csv(OUTPUT / "temperature_proxy_gtng15_assignment.csv", index=False)
    moisture_assignment.to_csv(OUTPUT / "moisture_proxy_gtng15_assignment.csv", index=False)

    equal_area = polygons.to_crs(6933)
    inventory_rows: list[dict[str, object]] = []
    total_grid_weight = float(grid_assignment["area_weight_coslat"].sum())
    for index, region in polygons.iterrows():
        code = region["o2region"]
        region_grid = grid_assignment[grid_assignment["o2region"].eq(code)]
        region_temp = temperature_assignment[temperature_assignment["o2region"].eq(code)]
        region_moisture = moisture_assignment[moisture_assignment["o2region"].eq(code)]
        region_system = system_assignment[system_assignment["o2region"].eq(code)]
        inventory_rows.append({
            "o1region": region["o1region"],
            "o2region": code,
            "full_name": region["full_name"],
            "long_code": region["long_code"],
            "polygon_area_km2": float(equal_area.geometry.iloc[index].area / 1e6),
            "hma_grid_cell_count": int(len(region_grid)),
            "hma_grid_area_weight_fraction": float(region_grid["area_weight_coslat"].sum() / total_grid_weight),
            "temperature_proxy_identity_count": int(len(region_temp)),
            "temperature_proxy_unique_site_count": unique_sites(region_temp),
            "moisture_proxy_identity_count": int(len(region_moisture)),
            "moisture_proxy_unique_site_count": unique_sites(region_moisture),
            "core191_system_count": int(len(region_system)),
            "core191_lia_area_km2": float(region_system["lia_area_km2"].sum()),
        })
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(OUTPUT / "gtng15_region_support_inventory.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(polygons)))
    polygons.plot(ax=axes[0], color=colors, edgecolor="#333333", linewidth=0.65)
    unique_grid = grid_assignment[grid_assignment["assignment_status"].eq("unique")]
    unmatched_grid = grid_assignment[grid_assignment["assignment_status"].eq("unmatched")]
    axes[0].scatter(unique_grid["longitude"], unique_grid["latitude"], s=2, color="#777777", alpha=0.22, label="Assigned HMA grid")
    if not unmatched_grid.empty:
        axes[0].scatter(unmatched_grid["longitude"], unmatched_grid["latitude"], s=5, color="#000000", alpha=0.55, label="Unmatched HMA grid")
    unique_system = system_assignment[system_assignment["assignment_status"].eq("unique")]
    axes[0].scatter(unique_system["centroid_lon"], unique_system["centroid_lat"], s=12, color="#111111", marker="o", label="Core-191")
    unique_temp = temperature_assignment[temperature_assignment["assignment_status"].eq("unique")]
    axes[0].scatter(unique_temp["longitude"], unique_temp["latitude"], s=22, color="#D55E00", marker="^", label="Temperature proxy")
    unique_moisture = moisture_assignment[moisture_assignment["assignment_status"].eq("unique")]
    axes[0].scatter(unique_moisture["longitude"], unique_moisture["latitude"], s=28, color="#0072B2", marker="s", label="Moisture proxy")
    for region in polygons.itertuples(index=False):
        point = region.geometry.representative_point()
        axes[0].text(point.x, point.y, region.o2region, fontsize=8, ha="center", va="center")
    axes[0].set(title="GTN-G 2023 HMA subregions and current evidence", xlabel="Longitude", ylabel="Latitude")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    plot_inventory = inventory.sort_values("o2region", ascending=False)
    y = np.arange(len(plot_inventory))
    axes[1].barh(y - 0.22, plot_inventory["temperature_proxy_unique_site_count"], height=0.4, color="#D55E00", label="Temperature proxy sites")
    axes[1].barh(y + 0.22, plot_inventory["core191_system_count"], height=0.4, color="#333333", label="Core-191 systems")
    axes[1].set_yticks(y, [f"{row.o2region}  {row.full_name}" for row in plot_inventory.itertuples(index=False)])
    axes[1].set(xlabel="Count", title="Regional evidence support")
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.6)
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    figure_path = OUTPUT / "gtng15_region_support_inventory.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "gtng_region_count": int(len(polygons)),
        "gtng_source_sha256": before["gtng"],
        "hma_grid_cell_count": int(len(grid_assignment)),
        "hma_grid_unique_assignment_count": int(grid_assignment["assignment_status"].eq("unique").sum()),
        "hma_grid_unmatched_count": int(grid_assignment["assignment_status"].eq("unmatched").sum()),
        "hma_grid_ambiguous_count": int(grid_assignment["assignment_status"].eq("ambiguous").sum()),
        "temperature_proxy_identity_count": int(len(temperature_assignment)),
        "temperature_proxy_unique_assignment_count": int(temperature_assignment["assignment_status"].eq("unique").sum()),
        "temperature_proxy_unmatched_count": int(temperature_assignment["assignment_status"].eq("unmatched").sum()),
        "moisture_proxy_identity_count": int(len(moisture_assignment)),
        "moisture_proxy_unique_assignment_count": int(moisture_assignment["assignment_status"].eq("unique").sum()),
        "moisture_proxy_unmatched_count": int(moisture_assignment["assignment_status"].eq("unmatched").sum()),
        "core191_system_count": int(len(system_assignment)),
        "core191_unique_assignment_count": int(system_assignment["assignment_status"].eq("unique").sum()),
        "core191_unmatched_count": int(system_assignment["assignment_status"].eq("unmatched").sum()),
        "source_url": "https://github.com/Hydrological-Observation-Group/HMA-GIS-Data",
        "boundary_citation": "GTN-G Glacier Regions 2023-07, doi:10.5904/gtng-glacreg-2023-07",
        "interpretation_limit": "This is a support and spatial-coverage audit. It does not authorize region-specific PSM fitting.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = {name: sha256(path) for name, path in sources.items()}
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "sources_unchanged": before == after}, indent=2),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(inventory.to_string(index=False))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
