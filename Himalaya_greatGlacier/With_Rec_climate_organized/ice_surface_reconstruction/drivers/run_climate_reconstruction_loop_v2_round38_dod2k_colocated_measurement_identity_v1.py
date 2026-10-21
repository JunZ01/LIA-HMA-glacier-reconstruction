"""Resolve co-located DoD2k records by archive and measurement identity."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root() -> Path:
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


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def archive_group(value: object) -> str:
    token = normalize(value)
    if token in {"wood", "tree", "treering"}:
        return "wood"
    if "speleothem" in token or "cave" in token:
        return "speleothem"
    if "glacier" in token or "ice" in token:
        return "glacier_ice"
    return token


def measurement_group(value: object) -> str:
    token = normalize(value)
    if "d18o" in token or "delta18o" in token:
        return "d18o"
    if token == "trw" or "ringwidth" in token or "trsgi" in token or "breit" in token:
        return "ring_width"
    if "dust" in token:
        return "dust"
    return token


def haversine(lat: float, lon: float, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.deg2rad(lat), np.deg2rad(lon)
    lat2, lon2 = np.deg2rad(latitudes), np.deg2rad(longitudes)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def main() -> int:
    project = root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_38_dod2k_colocated_measurement_identity_v1"
    paths = {
        "round36": loop / "round_36_dod2k_hma_identity_chronology_admissibility_v1/dod2k_identity_chronology_admissibility.csv",
        "round11": loop / "round_11_full_pages_lmr_hma_source_inventory_v2/full_pages_lmr_hma_candidate_inventory.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
        "facts": project / "LIA_PROJECT_FACTS.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    round36 = pd.read_csv(paths["round36"])
    queue = round36[round36.route == "source_identity_confirmation_required"].copy()
    if len(queue) != 7:
        raise RuntimeError("round36-identity-queue-contract-failed")
    inventory = pd.read_csv(paths["round11"])
    inventory = inventory[np.isfinite(inventory.latitude) & np.isfinite(inventory.longitude)].copy()
    inventory["archive_group"] = inventory.archive_type.map(archive_group)
    inventory["measurement_group"] = inventory.measurement.map(measurement_group)

    rows: list[dict[str, object]] = []
    nearby_rows: list[dict[str, object]] = []
    for item in queue.itertuples(index=False):
        distances = haversine(float(item.latitude), float(item.longitude), inventory.latitude.to_numpy(float), inventory.longitude.to_numpy(float))
        nearby = inventory.loc[distances <= 10].copy()
        nearby["distance_km"] = distances[distances <= 10]
        candidate_archive = archive_group(item.archiveType)
        candidate_measurement = measurement_group(item.paleoData_proxy)
        same_archive = nearby.archive_group == candidate_archive
        same_measurement = nearby.measurement_group == candidate_measurement
        exact_measurement = nearby[same_archive & same_measurement]
        different_measurement = nearby[same_archive & ~same_measurement]
        for record in nearby.itertuples(index=False):
            nearby_rows.append({
                "datasetId": item.datasetId,
                "round11_formal_identity": record.formal_identity,
                "distance_km": float(record.distance_km),
                "round11_archive": record.archive_type,
                "round11_measurement": record.measurement,
                "same_archive_group": record.archive_group == candidate_archive,
                "same_measurement_group": record.measurement_group == candidate_measurement,
            })

        name = normalize(item.dataSetName)
        similarities = [SequenceMatcher(None, name, normalize(identity)).ratio() if name else 0.0 for identity in nearby.formal_identity]
        maximum_name_similarity = max(similarities, default=0.0)
        annual_gates = bool(
            candidate_archive == "wood"
            and candidate_measurement == "d18o"
            and str(item.interpretation_direction).lower() in {"positive", "negative"}
            and bool(str(item.interpretation_seasonality).strip())
            and bool(item.annual_modern_chronology_gate)
            and bool(item.annual_lia_chronology_gate)
        )
        source_linked_dataset = bool(item.datasetId == "pages2k_2216" or maximum_name_similarity >= 0.80)
        if len(exact_measurement):
            route = "exclude_likely_existing_measurement"
        elif annual_gates and len(different_measurement):
            route = "colocated_distinct_annual_measurement_candidate"
        elif source_linked_dataset and len(different_measurement):
            route = "source_linked_distinct_low_frequency_only"
        else:
            route = "identity_unresolved"
        rows.append({
            "datasetId": item.datasetId,
            "dataSetName": item.dataSetName,
            "geo_siteName": item.geo_siteName,
            "latitude": item.latitude,
            "longitude": item.longitude,
            "archiveType": item.archiveType,
            "paleoData_proxy": item.paleoData_proxy,
            "interpretation_variable": item.interpretation_variable,
            "interpretation_direction": item.interpretation_direction,
            "interpretation_seasonality": item.interpretation_seasonality,
            "nearby_round11_record_count": int(len(nearby)),
            "same_archive_same_measurement_count": int(len(exact_measurement)),
            "same_archive_different_measurement_count": int(len(different_measurement)),
            "maximum_dataset_name_formal_identity_similarity": maximum_name_similarity,
            "annual_metadata_chronology_gates": annual_gates,
            "route": route,
            "originalDataURL": item.originalDataURL,
        })

    result = pd.DataFrame(rows).sort_values(["route", "datasetId"])
    nearby = pd.DataFrame(nearby_rows).sort_values(["datasetId", "distance_km"])
    result.to_csv(output / "colocated_measurement_identity.csv", index=False)
    nearby.to_csv(output / "nearby_round11_measurement_matches.csv", index=False)
    annual = result[result.route == "colocated_distinct_annual_measurement_candidate"]
    low_frequency = result[result.route == "source_linked_distinct_low_frequency_only"]
    annual.to_csv(output / "colocated_distinct_annual_candidates.csv", index=False)
    low_frequency.to_csv(output / "source_linked_low_frequency_candidates.csv", index=False)

    route_counts = {str(key): int(value) for key, value in result.route.value_counts().items()}
    summary = {
        "status": "complete",
        "metadata_only": True,
        "locked_1921_1939_numeric_values_accessed": False,
        "input_identity_queue_count": int(len(result)),
        "route_counts": route_counts,
        "colocated_distinct_annual_candidate_count": int(len(annual)),
        "colocated_distinct_annual_candidate_ids": annual.datasetId.tolist(),
        "likely_existing_measurement_count": int((result.route == "exclude_likely_existing_measurement").sum()),
        "source_linked_low_frequency_count": int(len(low_frequency)),
        "decision": "advance_colocated_distinct_annual_candidates_to_nested_validation_with_site_cluster_sensitivity" if len(annual) else "close_colocated_annual_evidence",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    order = ["colocated_distinct_annual_measurement_candidate", "exclude_likely_existing_measurement", "source_linked_distinct_low_frequency_only", "identity_unresolved"]
    counts = [route_counts.get(value, 0) for value in order]
    labels = ["Distinct annual", "Likely existing", "Source-linked low-frequency", "Unresolved"]
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.bar(labels, counts, color=["#0072b2", "#d55e00", "#009e73", "#e69f00"])
    axis.set(ylabel="Record count", title="Co-located archive and measurement identity")
    axis.tick_params(axis="x", rotation=15)
    for index, count in enumerate(counts):
        axis.text(index, count + 0.05, str(count), ha="center")
    figure.savefig(output / "colocated_measurement_identity.png", dpi=180)
    plt.close(figure)

    lines = [
        "# DoD2k Co-Located Measurement Identity",
        "",
        f"- Identity queue: `{len(result)}`.",
        f"- Co-located distinct annual measurements: `{len(annual)}`: `{', '.join(annual.datasetId)}`.",
        f"- Likely existing same measurements: `{summary['likely_existing_measurement_count']}`.",
        f"- Source-linked distinct low-frequency records: `{len(low_frequency)}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
