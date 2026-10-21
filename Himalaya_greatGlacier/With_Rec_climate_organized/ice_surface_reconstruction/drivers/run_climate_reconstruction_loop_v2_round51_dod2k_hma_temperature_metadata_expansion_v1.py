"""Screen DoD2k metadata for additional HMA temperature evidence."""

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


TEMPERATURE_CLASSES = {"temperature", "temperature+moisture"}


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
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def haversine_to_points(
    lat: np.ndarray,
    lon: np.ndarray,
    point_lat: np.ndarray,
    point_lon: np.ndarray,
) -> np.ndarray:
    result = np.full(lat.shape, np.nan)
    point_lat_rad = np.deg2rad(point_lat)
    point_lon_rad = np.deg2rad(point_lon)
    for start in range(0, len(lat), 256):
        stop = min(start + 256, len(lat))
        lat_rad = np.deg2rad(lat[start:stop])[:, None]
        lon_rad = np.deg2rad(lon[start:stop])[:, None]
        delta_lat = point_lat_rad[None] - lat_rad
        delta_lon = point_lon_rad[None] - lon_rad
        value = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat_rad)
            * np.cos(point_lat_rad)[None]
            * np.sin(delta_lon / 2) ** 2
        )
        result[start:stop] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1))).min(axis=1)
    return result


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_51_dod2k_hma_temperature_metadata_expansion_v1"
    round34 = loop / "round_34_sisalv3_hma_hydroclimate_source_audit_v1"
    paths = {
        "dod2k_metadata": round34 / "dod2k_v2.0_compact_metadata.csv",
        "round34_candidates": round34 / "dod2k_hma_hydroclimate_metadata_candidates.csv",
        "hma_mask": project
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "reference_grid": loop
        / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "existing_full_inventory": loop
        / "round_11_full_pages_lmr_hma_source_inventory_v2/full_pages_lmr_hma_candidate_inventory.csv",
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
    prior = pd.read_csv(paths["round34_candidates"])
    prior_ids = set(prior.datasetId.astype(str))
    existing = pd.read_csv(paths["existing_full_inventory"])
    existing = existing[np.isfinite(existing.latitude) & np.isfinite(existing.longitude)].copy()

    mask = np.load(paths["hma_mask"]).astype(bool)
    with netcdf_file(paths["reference_grid"], "r", mmap=False) as dataset:
        latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    if mask.shape != (len(latitudes), len(longitudes)):
        raise RuntimeError("hma-grid-contract-failed")
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    hma_lat = lat_grid[mask]
    hma_lon = lon_grid[mask]

    metadata_lat = pd.to_numeric(metadata.geo_meanLat, errors="coerce").to_numpy(float)
    metadata_lon = pd.to_numeric(metadata.geo_meanLon, errors="coerce").to_numpy(float)
    finite = np.isfinite(metadata_lat) & np.isfinite(metadata_lon)
    distance_hma = np.full(len(metadata), np.nan)
    distance_existing = np.full(len(metadata), np.nan)
    distance_hma[finite] = haversine_to_points(
        metadata_lat[finite], metadata_lon[finite], hma_lat, hma_lon
    )
    distance_existing[finite] = haversine_to_points(
        metadata_lat[finite],
        metadata_lon[finite],
        existing.latitude.to_numpy(float),
        existing.longitude.to_numpy(float),
    )
    metadata["distance_to_hma_mask_km"] = distance_hma
    metadata["distance_to_existing_pages_lmr_site_km"] = distance_existing

    variable = metadata.interpretation_variable.fillna("").str.strip().str.lower()
    eligible = finite & variable.isin(TEMPERATURE_CLASSES) & (distance_hma <= 500)
    candidates = metadata.loc[eligible].copy()
    candidates["already_routed_by_round34"] = candidates.datasetId.astype(str).isin(prior_ids)
    candidates["candidate_site_status"] = np.where(
        candidates.distance_to_existing_pages_lmr_site_km <= 10,
        "existing_site_requires_identity_review",
        "new_site_candidate",
    )
    candidates["non_wood_archive"] = ~candidates.archiveType.fillna("").str.lower().eq("wood")
    existing_identity_text = " ".join(existing.formal_identity.fillna("").map(normalize).tolist())
    candidates["dataset_name_token_seen_in_existing_identity"] = [
        bool(name and name in existing_identity_text)
        for name in candidates.dataSetName.map(normalize)
    ]
    candidates["new_temperature_evidence_priority"] = np.select(
        [
            ~candidates.already_routed_by_round34
            & candidates.non_wood_archive
            & candidates.candidate_site_status.eq("new_site_candidate"),
            ~candidates.already_routed_by_round34 & candidates.non_wood_archive,
            ~candidates.already_routed_by_round34
            & candidates.candidate_site_status.eq("new_site_candidate"),
            ~candidates.already_routed_by_round34,
        ],
        [
            "1_non_wood_new_site",
            "2_non_wood_existing_site",
            "3_wood_new_site",
            "4_wood_existing_site",
        ],
        default="5_already_round34",
    )
    candidates = candidates.sort_values(
        [
            "new_temperature_evidence_priority",
            "distance_to_hma_mask_km",
            "originalDatabase",
            "datasetId",
        ]
    )
    export_columns = [
        "datasetId",
        "archiveType",
        "dataSetName",
        "geo_meanElev",
        "geo_meanLat",
        "geo_meanLon",
        "geo_siteName",
        "distance_to_hma_mask_km",
        "distance_to_existing_pages_lmr_site_km",
        "candidate_site_status",
        "already_routed_by_round34",
        "dataset_name_token_seen_in_existing_identity",
        "interpretation_direction",
        "interpretation_seasonality",
        "interpretation_variable",
        "interpretation_variableDetail",
        "paleoData_proxy",
        "paleoData_variableName",
        "originalDatabase",
        "originalDataURL",
        "new_temperature_evidence_priority",
    ]
    candidates[export_columns].to_csv(output / "dod2k_hma_temperature_metadata_candidates.csv", index=False)

    new = candidates[~candidates.already_routed_by_round34].copy()
    priority = new[
        new.new_temperature_evidence_priority == "1_non_wood_new_site"
    ].copy()
    source_counts = (
        new.groupby(["originalDatabase", "candidate_site_status"])
        .size()
        .rename("record_count")
        .reset_index()
    )
    archive_counts = (
        new.groupby(["archiveType", "candidate_site_status"])
        .size()
        .rename("record_count")
        .reset_index()
    )
    source_counts.to_csv(output / "new_candidate_counts_by_source.csv", index=False)
    archive_counts.to_csv(output / "new_candidate_counts_by_archive.csv", index=False)
    summary = {
        "status": "complete",
        "metadata_only_audit": True,
        "locked_1921_1939_numeric_values_accessed": False,
        "dod2k_record_count": int(len(metadata)),
        "hma_temperature_within_500km_record_count": int(len(candidates)),
        "already_routed_by_round34_count": int(candidates.already_routed_by_round34.sum()),
        "new_temperature_candidate_count": int(len(new)),
        "new_site_candidate_count": int(
            np.sum(new.candidate_site_status == "new_site_candidate")
        ),
        "new_non_wood_candidate_count": int(new.non_wood_archive.sum()),
        "priority_non_wood_new_site_count": int(len(priority)),
        "new_candidate_source_counts": {
            str(key): int(value) for key, value in new.originalDatabase.value_counts().items()
        },
        "new_candidate_archive_counts": {
            str(key): int(value) for key, value in new.archiveType.value_counts().items()
        },
        "numeric_coverage_not_yet_assessed": True,
        "decision": (
            "advance_new_temperature_candidates_to_allowed_period_coverage_audit"
            if len(new)
            else "no_new_dod2k_temperature_evidence"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    colors = {
        "new_site_candidate": "#0072b2",
        "existing_site_requires_identity_review": "#d55e00",
    }
    for status, group in new.groupby("candidate_site_status"):
        axes[0].scatter(
            group.geo_meanLon,
            group.geo_meanLat,
            s=30,
            alpha=0.78,
            label=status.replace("_", " "),
            color=colors[status],
        )
    axes[0].scatter(hma_lon, hma_lat, s=2, alpha=0.13, color="#333333", label="HMA mask")
    axes[0].set(xlabel="Longitude", ylabel="Latitude", title="New DoD2k temperature candidates")
    axes[0].legend(frameon=False, fontsize=8)
    if len(archive_counts):
        plot = archive_counts.pivot(
            index="archiveType", columns="candidate_site_status", values="record_count"
        ).fillna(0)
        plot.plot.bar(
            ax=axes[1],
            color=[colors.get(column, "#777777") for column in plot.columns],
        )
    axes[1].set(xlabel="", ylabel="Record count", title="New candidates by archive")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(output / "dod2k_hma_temperature_metadata_expansion.png", dpi=180)
    plt.close(figure)

    lines = [
        "# DoD2k HMA Temperature Metadata Expansion",
        "",
        f"- Temperature-classified records within 500 km: `{len(candidates)}`.",
        f"- Already routed by Round 34: `{int(candidates.already_routed_by_round34.sum())}`.",
        f"- New temperature candidates: `{len(new)}`.",
        f"- New sites: `{int(np.sum(new.candidate_site_status == 'new_site_candidate'))}`; new non-wood records: `{int(new.non_wood_archive.sum())}`.",
        f"- Priority non-wood new sites: `{len(priority)}`.",
        "- No year or proxy-value file was read; coverage and calibration remain unassessed.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
