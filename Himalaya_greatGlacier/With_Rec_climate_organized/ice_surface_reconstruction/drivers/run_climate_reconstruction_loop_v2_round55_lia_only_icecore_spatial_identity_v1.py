"""Test whether LIA-only DoD2k ice-core records add independent spatial sites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def haversine(lat: float, lon: float, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)
    lat2 = np.deg2rad(latitudes)
    lon2 = np.deg2rad(longitudes)
    value = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_55_lia_only_icecore_spatial_identity_v1"
    paths = {
        "coverage": loop
        / "round_52_dod2k_temperature_allowed_period_coverage_v1/dod2k_hma_temperature_coverage.csv",
        "inventory": loop
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

    coverage = pd.read_csv(paths["coverage"])
    selected = coverage[
        coverage.lia_evidence_eligible.astype(bool)
        & ~coverage.modern_calibration_capable.astype(bool)
    ].copy()
    if len(selected) != 7 or set(selected.archiveType) != {"GlacierIce"}:
        raise RuntimeError("lia-only-record-contract-failed")
    inventory = pd.read_csv(paths["inventory"])
    inventory = inventory[np.isfinite(inventory.latitude) & np.isfinite(inventory.longitude)].copy()

    rows: list[dict[str, object]] = []
    for item in selected.itertuples(index=False):
        distances = haversine(
            float(item.geo_meanLat),
            float(item.geo_meanLon),
            inventory.latitude.to_numpy(float),
            inventory.longitude.to_numpy(float),
        )
        position = int(np.argmin(distances))
        nearest = inventory.iloc[position]
        distance = float(distances[position])
        rows.append(
            {
                "datasetId": item.datasetId,
                "dataSetName": item.dataSetName,
                "site_name": item.geo_siteName,
                "latitude": float(item.geo_meanLat),
                "longitude": float(item.geo_meanLon),
                "proxy": item.paleoData_proxy,
                "weak_round30_window_support_count": int(
                    item.weak_round30_window_support_count
                ),
                "modern_finite_count": int(item.modern_1940_2000_finite_count),
                "nearest_round11_distance_km": distance,
                "nearest_round11_identity": nearest.formal_identity,
                "nearest_round11_archive": nearest.archive_type,
                "adds_independent_spatial_site": bool(distance > 10.0),
            }
        )
    result = pd.DataFrame(rows).sort_values(["site_name", "datasetId"])
    result.to_csv(output / "lia_only_icecore_spatial_identity.csv", index=False)
    independent = result[result.adds_independent_spatial_site]
    site_clusters = result[["site_name", "latitude", "longitude"]].drop_duplicates()
    summary = {
        "status": "complete",
        "metadata_only_audit": True,
        "locked_1921_1939_numeric_values_accessed": False,
        "lia_only_record_count": int(len(result)),
        "distinct_candidate_site_count": int(len(site_clusters)),
        "existing_site_record_count": int(np.sum(~result.adds_independent_spatial_site)),
        "independent_new_site_count": int(len(independent)),
        "independent_new_site_ids": independent.datasetId.tolist(),
        "decision": (
            "advance_independent_lia_only_sites_to_direction_validation"
            if len(independent)
            else "close_lia_only_temperature_records_as_no_new_spatial_support"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.scatter(
        inventory.longitude,
        inventory.latitude,
        s=9,
        alpha=0.25,
        color="#777777",
        label="Round 11 sites",
    )
    axis.scatter(
        result.longitude,
        result.latitude,
        s=65,
        color="#d55e00",
        label="LIA-only ice-core records",
    )
    for row in site_clusters.itertuples(index=False):
        axis.annotate(row.site_name, (row.longitude, row.latitude), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="LIA-only ice-core sites versus current inventory",
    )
    axis.legend(frameon=False)
    figure.savefig(output / "lia_only_icecore_spatial_identity.png", dpi=180)
    plt.close(figure)

    lines = [
        "# LIA-Only Ice-Core Spatial Identity",
        "",
        f"- Records/sites: `{len(result)}/{len(site_clusters)}`.",
        f"- Existing-site records: `{summary['existing_site_record_count']}`.",
        f"- Independent new sites: `{summary['independent_new_site_count']}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged},
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
