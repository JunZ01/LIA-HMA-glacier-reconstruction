"""Screen DoD2k v2.0 metadata for new HMA hydroclimate evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_URL = "https://raw.githubusercontent.com/lluecke/dod2k/main/data/dod2k_v2.0/dod2k_v2.0_compact_metadata.csv"
DOWNLOAD_URL = "https://cdn.jsdelivr.net/gh/lluecke/dod2k@main/data/dod2k_v2.0/dod2k_v2.0_compact_metadata.csv"
HYDROCLIMATE = {"moisture", "temperature+moisture"}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def haversine_to_points(lat: np.ndarray, lon: np.ndarray, point_lat: np.ndarray, point_lon: np.ndarray) -> np.ndarray:
    result = np.full(lat.shape, np.nan)
    point_lat_rad = np.deg2rad(point_lat)
    point_lon_rad = np.deg2rad(point_lon)
    for start in range(0, len(lat), 256):
        stop = min(start + 256, len(lat))
        lat_rad = np.deg2rad(lat[start:stop])[:, None]
        lon_rad = np.deg2rad(lon[start:stop])[:, None]
        delta_lat = point_lat_rad[None] - lat_rad
        delta_lon = point_lon_rad[None] - lon_rad
        value = np.sin(delta_lat / 2) ** 2 + np.cos(lat_rad) * np.cos(point_lat_rad)[None] * np.sin(delta_lon / 2) ** 2
        result[start:stop] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1))).min(axis=1)
    return result


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_34_sisalv3_hma_hydroclimate_source_audit_v1"
    paths = {
        "dod2k_metadata": output / "dod2k_v2.0_compact_metadata.csv",
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "reference_grid": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "existing_full_inventory": loop / "round_11_full_pages_lmr_hma_source_inventory_v2/full_pages_lmr_hma_candidate_inventory.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    metadata = pd.read_csv(paths["dod2k_metadata"])
    if metadata.shape != (4781, 20) or metadata.datasetId.duplicated().any():
        raise RuntimeError("dod2k-metadata-contract-failed")
    mask = np.load(paths["hma_mask"]).astype(bool)
    with netcdf_file(paths["reference_grid"], "r", mmap=False) as dataset:
        latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    if mask.shape != (len(latitudes), len(longitudes)):
        raise RuntimeError("hma-grid-contract-failed")
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    hma_lat = lat_grid[mask]
    hma_lon = lon_grid[mask]

    existing = pd.read_csv(paths["existing_full_inventory"])
    existing = existing[np.isfinite(existing.latitude) & np.isfinite(existing.longitude)].copy()
    metadata_lat = pd.to_numeric(metadata.geo_meanLat, errors="coerce").to_numpy(float)
    metadata_lon = pd.to_numeric(metadata.geo_meanLon, errors="coerce").to_numpy(float)
    finite_coordinates = np.isfinite(metadata_lat) & np.isfinite(metadata_lon)
    distance_hma = np.full(len(metadata), np.nan)
    distance_existing = np.full(len(metadata), np.nan)
    distance_hma[finite_coordinates] = haversine_to_points(metadata_lat[finite_coordinates], metadata_lon[finite_coordinates], hma_lat, hma_lon)
    distance_existing[finite_coordinates] = haversine_to_points(
        metadata_lat[finite_coordinates], metadata_lon[finite_coordinates],
        existing.latitude.to_numpy(float), existing.longitude.to_numpy(float),
    )
    metadata["distance_to_hma_mask_km"] = distance_hma
    metadata["distance_to_existing_pages_lmr_site_km"] = distance_existing

    variable = metadata.interpretation_variable.fillna("").str.strip().str.lower()
    eligible = finite_coordinates & variable.isin(HYDROCLIMATE) & (distance_hma <= 500)
    candidates = metadata.loc[eligible].copy()
    candidates["candidate_site_status"] = np.where(
        candidates.distance_to_existing_pages_lmr_site_km <= 10,
        "existing_site_requires_identity_review",
        "new_site_candidate",
    )
    candidates["non_tree_archive"] = ~candidates.archiveType.fillna("").str.lower().eq("wood")
    existing_identity_text = " ".join(existing.formal_identity.fillna("").map(normalize).tolist())
    candidates["dataset_name_token_seen_in_existing_identity"] = [
        bool(name and name in existing_identity_text) for name in candidates.dataSetName.map(normalize)
    ]
    candidates["priority_for_allowed_period_acquisition"] = np.select(
        [
            candidates.non_tree_archive & candidates.candidate_site_status.eq("new_site_candidate"),
            candidates.non_tree_archive,
            candidates.candidate_site_status.eq("new_site_candidate"),
        ],
        ["1_non_tree_new_site", "2_non_tree_existing_site", "3_tree_new_site"],
        default="4_tree_existing_site",
    )
    candidates = candidates.sort_values([
        "priority_for_allowed_period_acquisition", "distance_to_hma_mask_km", "originalDatabase", "datasetId"
    ])
    export_columns = [
        "datasetId", "archiveType", "dataSetName", "geo_meanElev", "geo_meanLat", "geo_meanLon", "geo_siteName",
        "distance_to_hma_mask_km", "distance_to_existing_pages_lmr_site_km", "candidate_site_status",
        "dataset_name_token_seen_in_existing_identity", "interpretation_direction", "interpretation_seasonality",
        "interpretation_variable", "interpretation_variableDetail", "paleoData_proxy", "paleoData_variableName",
        "originalDatabase", "originalDataURL", "priority_for_allowed_period_acquisition",
    ]
    candidates[export_columns].to_csv(output / "dod2k_hma_hydroclimate_metadata_candidates.csv", index=False)

    source_counts = candidates.groupby(["originalDatabase", "candidate_site_status"]).size().rename("record_count").reset_index()
    archive_counts = candidates.groupby(["archiveType", "candidate_site_status"]).size().rename("record_count").reset_index()
    source_counts.to_csv(output / "candidate_counts_by_source.csv", index=False)
    archive_counts.to_csv(output / "candidate_counts_by_archive.csv", index=False)
    new_site = candidates[candidates.candidate_site_status == "new_site_candidate"]
    priority = candidates[candidates.priority_for_allowed_period_acquisition == "1_non_tree_new_site"]
    summary = {
        "status": "complete",
        "metadata_only_audit": True,
        "locked_1921_1939_numeric_values_accessed": False,
        "dod2k_record_count": int(len(metadata)),
        "hma_hydroclimate_within_500km_record_count": int(len(candidates)),
        "new_site_candidate_count": int(len(new_site)),
        "existing_site_identity_review_count": int(len(candidates) - len(new_site)),
        "non_tree_new_site_priority_count": int(len(priority)),
        "candidate_source_counts": {str(key): int(value) for key, value in candidates.originalDatabase.value_counts().items()},
        "candidate_archive_counts": {str(key): int(value) for key, value in candidates.archiveType.value_counts().items()},
        "metadata_source_sha256": before["dod2k_metadata"]["sha256"],
        "source_urls": {"canonical": SOURCE_URL, "download_transport": DOWNLOAD_URL},
        "numeric_coverage_not_yet_assessed": True,
        "decision": "acquire_only_candidate_allowed_period_year_value_columns_and_verify_identity_chronology" if len(candidates) else "no_dod2k_hma_hydroclimate_expansion",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    colors = {"new_site_candidate": "#0072b2", "existing_site_requires_identity_review": "#d55e00"}
    for status, group in candidates.groupby("candidate_site_status"):
        axes[0].scatter(group.geo_meanLon, group.geo_meanLat, s=28, alpha=0.75, label=status.replace("_", " "), color=colors[status])
    axes[0].scatter(hma_lon, hma_lat, s=2, alpha=0.15, color="#333333", label="HMA mask")
    axes[0].set(xlabel="Longitude", ylabel="Latitude", title="DoD2k hydroclimate candidates")
    axes[0].legend(frameon=False, fontsize=8)
    source_plot = source_counts.pivot(index="originalDatabase", columns="candidate_site_status", values="record_count").fillna(0)
    source_plot.plot.bar(ax=axes[1], color=[colors.get(column, "#777777") for column in source_plot.columns])
    axes[1].set(xlabel="", ylabel="Record count", title="Candidates by source and site status")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("DoD2k v2.0 HMA hydroclimate metadata audit", fontsize=14)
    figure.savefig(output / "dod2k_hma_hydroclimate_metadata_audit.png", dpi=180)
    plt.close(figure)

    lines = [
        "# DoD2k HMA Hydroclimate Metadata Audit",
        "",
        f"- DoD2k records: `{len(metadata)}`.",
        f"- Hydroclimate records within 500 km of HMA: `{len(candidates)}`.",
        f"- New-site candidates: `{len(new_site)}`; existing-site identity review: `{len(candidates) - len(new_site)}`.",
        f"- Priority non-tree new-site candidates: `{len(priority)}`.",
        "- No year or proxy-value file was read; calibration and LIA coverage remain unassessed.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    acquisition = {
        "canonical_metadata_url": SOURCE_URL,
        "download_transport_url": DOWNLOAD_URL,
        "downloaded_size_bytes": before["dod2k_metadata"]["size_bytes"],
        "downloaded_sha256": before["dod2k_metadata"]["sha256"],
        "noaa_sisal_direct_attempt": "connection_timeout",
        "cyverse_direct_attempt": "redirected_to_access_page",
        "github_raw_attempt": "connection_reset",
        "jsdelivr_attempt": "success",
    }
    (output / "acquisition_manifest.json").write_text(json.dumps(acquisition, indent=2), encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    pure = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": pure}, indent=2), encoding="utf-8")
    if not pure:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
