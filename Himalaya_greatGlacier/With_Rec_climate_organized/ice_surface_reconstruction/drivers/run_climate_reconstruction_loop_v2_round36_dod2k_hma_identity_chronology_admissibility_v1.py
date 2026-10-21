"""Audit identity, chronology, and admissibility of Round 35 DoD2k records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCKED_START = 1921
LOCKED_END = 1939


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def file_state(path: Path) -> dict[str, object]:
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


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<missing>"} else text


def haversine(lat: float, lon: float, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)
    lat2 = np.deg2rad(latitudes)
    lon2 = np.deg2rad(longitudes)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def chronology_metrics(years: np.ndarray) -> dict[str, object]:
    years = np.asarray(years, dtype=float)
    unique = np.unique(years)
    gaps = np.diff(unique)
    if len(years) < 2:
        order = "single"
    elif np.all(np.diff(years) > 0):
        order = "strictly_increasing"
    elif np.all(np.diff(years) < 0):
        order = "strictly_decreasing"
    else:
        order = "non_monotonic_or_duplicate"
    return {
        "count": int(len(years)),
        "unique_count": int(len(unique)),
        "duplicate_year_count": int(len(years) - len(unique)),
        "source_order": order,
        "median_gap_years": float(np.median(gaps)) if len(gaps) else np.nan,
        "maximum_gap_years": float(np.max(gaps)) if len(gaps) else np.nan,
        "integer_year_fraction": float(np.mean(np.isclose(years, np.round(years), atol=1e-8))) if len(years) else np.nan,
    }


def affine_match(
    candidate_years: np.ndarray,
    candidate_values: np.ndarray,
    reference_years: np.ndarray,
    reference_values: np.ndarray,
) -> tuple[int, float, float]:
    rounded = np.round(candidate_years)
    integer = np.isclose(candidate_years, rounded, atol=1e-8)
    lookup: dict[int, float] = {}
    duplicated: set[int] = set()
    for year, value in zip(rounded[integer].astype(int), candidate_values[integer]):
        if year in lookup:
            duplicated.add(int(year))
        lookup[int(year)] = float(value)
    for year in duplicated:
        lookup.pop(year, None)
    candidate: list[float] = []
    reference: list[float] = []
    for year, value in zip(reference_years.astype(int), reference_values):
        if year in lookup and np.isfinite(value) and not (LOCKED_START <= year <= LOCKED_END):
            candidate.append(lookup[year])
            reference.append(float(value))
    count = len(candidate)
    if count < 3:
        return count, np.nan, np.nan
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    coefficient = np.linalg.lstsq(np.column_stack([b, np.ones(count)]), a, rcond=None)[0]
    residual = float(np.max(np.abs(a - np.column_stack([b, np.ones(count)]) @ coefficient)))
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    return count, correlation, residual


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_36_dod2k_hma_identity_chronology_admissibility_v1"
    paths = {
        "round34_metadata": loop / "round_34_sisalv3_hma_hydroclimate_source_audit_v1/dod2k_hma_hydroclimate_metadata_candidates.csv",
        "round35_coverage": loop / "round_35_dod2k_hma_hydroclimate_coverage_audit_v1/dod2k_hma_hydroclimate_coverage.csv",
        "round35_values": loop / "round_35_dod2k_hma_hydroclimate_coverage_audit_v1/dod2k_hma_hydroclimate_allowed_values.csv",
        "round11_inventory": loop / "round_11_full_pages_lmr_hma_source_inventory_v2/full_pages_lmr_hma_candidate_inventory.csv",
        "round12_manifest": loop / "round_12_expanded_moisture_nested_cross_product_v1/network_manifest.csv",
        "round12_network": loop / "round_12_expanded_moisture_nested_cross_product_v1/expanded_moisture_network_v1.mat",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: file_state(path) for name, path in paths.items()}

    metadata = pd.read_csv(paths["round34_metadata"])
    coverage_all = pd.read_csv(paths["round35_coverage"])
    coverage = coverage_all[coverage_all.both_lia_and_modern_capable].copy()
    if len(coverage) != 25 or coverage.datasetId.duplicated().any():
        raise RuntimeError("round35-both-capable-contract-failed")
    coverage = coverage.merge(
        metadata[[
            "datasetId", "distance_to_hma_mask_km", "distance_to_existing_pages_lmr_site_km",
            "dataset_name_token_seen_in_existing_identity", "paleoData_variableName",
            "priority_for_allowed_period_acquisition",
        ]],
        on="datasetId",
        how="left",
        validate="one_to_one",
    )
    values = pd.read_csv(paths["round35_values"])
    values = values[values.datasetId.isin(set(coverage.datasetId))].copy()
    if values.empty or values.datasetId.nunique() != 25:
        raise RuntimeError("round35-allowed-values-contract-failed")
    if ((values.year_ce >= LOCKED_START) & (values.year_ce <= LOCKED_END)).any():
        raise RuntimeError("locked-period-value-present")

    inventory = pd.read_csv(paths["round11_inventory"])
    inventory = inventory[np.isfinite(inventory.latitude) & np.isfinite(inventory.longitude)].copy()
    manifest = pd.read_csv(paths["round12_manifest"])
    network = loadmat(paths["round12_network"], squeeze_me=True)
    network_years = np.asarray(network["years"], dtype=int).reshape(-1)
    observations = np.asarray(network["observations"], dtype=float)
    identity_ids = [str(item).strip() for item in np.asarray(network["identity_ids"]).reshape(-1)]
    if observations.shape != (61, 22) or not np.array_equal(network_years, np.arange(1940, 2001)):
        raise RuntimeError("round12-modern-network-contract-failed")
    if len(manifest) != 22 or identity_ids != manifest.formal_identity.astype(str).tolist():
        raise RuntimeError("round12-identity-contract-failed")

    inventory_tokens = inventory.formal_identity.fillna("").map(normalize).tolist()
    result_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    for item in coverage.itertuples(index=False):
        series = values[values.datasetId == item.datasetId]
        years = series.year_ce.to_numpy(float)
        proxy_values = series.paleo_value.to_numpy(float)
        all_metrics = chronology_metrics(years)
        lia_mask = (years >= 1400) & (years <= 1899)
        modern_mask = (years >= 1940) & (years <= 2000)
        lia_metrics = chronology_metrics(years[lia_mask])
        modern_metrics = chronology_metrics(years[modern_mask])

        distances = haversine(
            float(item.geo_meanLat), float(item.geo_meanLon),
            inventory.latitude.to_numpy(float), inventory.longitude.to_numpy(float),
        )
        nearest_position = int(np.argmin(distances))
        nearest = inventory.iloc[nearest_position]
        dataset_token = normalize(item.dataSetName)
        token_match_positions = [index for index, token in enumerate(inventory_tokens) if dataset_token and dataset_token in token]

        best_match: dict[str, object] | None = None
        for column, identity in enumerate(identity_ids):
            count, correlation, residual = affine_match(years, proxy_values, network_years, observations[:, column])
            matched = bool(count >= 20 and np.isfinite(correlation) and abs(correlation) >= 0.999999 and residual <= 1e-8)
            row = {
                "datasetId": item.datasetId,
                "round12_identity": identity,
                "overlap_count": count,
                "correlation": correlation,
                "maximum_abs_affine_residual": residual,
                "passes_affine_identity": matched,
            }
            match_rows.append(row)
            if matched and (best_match is None or count > int(best_match["overlap_count"])):
                best_match = row

        if best_match is not None:
            identity_status = "current_round12_sequence_duplicate"
        elif token_match_positions:
            identity_status = "existing_round11_formal_token_match"
        elif float(distances[nearest_position]) <= 10:
            identity_status = "same_site_distinct_or_identity_unresolved"
        else:
            identity_status = "independent_new_site"

        direction = clean(item.interpretation_direction).lower()
        seasonality = clean(item.interpretation_seasonality)
        variable = clean(item.interpretation_variable).lower()
        archive = clean(item.archiveType).lower()
        proxy = clean(item.paleoData_proxy).lower()
        annual_measurement = archive == "wood" and proxy in {"d18o", "ring width", "trw"}
        documented_direction = direction in {"positive", "negative"}
        annual_modern = bool(
            modern_metrics["unique_count"] >= 50
            and modern_metrics["duplicate_year_count"] == 0
            and modern_metrics["integer_year_fraction"] == 1.0
            and modern_metrics["median_gap_years"] <= 1.0
            and modern_metrics["maximum_gap_years"] <= 1.0
        )
        annual_lia = bool(
            int(item.lia_occupied_50year_window_count) >= 3
            and lia_metrics["duplicate_year_count"] == 0
            and lia_metrics["median_gap_years"] <= 1.0
        )
        annual_psm = bool(
            identity_status == "independent_new_site"
            and variable == "moisture"
            and documented_direction
            and bool(seasonality)
            and annual_measurement
            and annual_modern
            and annual_lia
        )
        low_frequency = bool(
            identity_status == "independent_new_site"
            and bool(item.low_frequency_validation_eligible)
            and all_metrics["duplicate_year_count"] == 0
            and not annual_psm
        )
        if identity_status == "current_round12_sequence_duplicate":
            route = "exclude_current_network_duplicate"
        elif identity_status == "existing_round11_formal_token_match":
            route = "exclude_existing_project_identity"
        elif identity_status == "same_site_distinct_or_identity_unresolved":
            route = "source_identity_confirmation_required"
        elif annual_psm:
            route = "annual_moisture_psm_candidate"
        elif low_frequency:
            route = "independent_low_frequency_validation"
        else:
            route = "not_admissible"

        result_rows.append({
            "datasetId": item.datasetId,
            "dataSetName": item.dataSetName,
            "geo_siteName": item.geo_siteName,
            "latitude": float(item.geo_meanLat),
            "longitude": float(item.geo_meanLon),
            "originalDatabase": item.originalDatabase,
            "archiveType": item.archiveType,
            "paleoData_proxy": item.paleoData_proxy,
            "interpretation_variable": item.interpretation_variable,
            "interpretation_direction": clean(item.interpretation_direction),
            "interpretation_seasonality": clean(item.interpretation_seasonality),
            "round34_site_status": item.candidate_site_status,
            "identity_status": identity_status,
            "route": route,
            "nearest_round11_distance_km": float(distances[nearest_position]),
            "nearest_round11_identity": nearest.formal_identity,
            "nearest_round11_measurement": nearest.measurement,
            "round11_formal_token_match_count": len(token_match_positions),
            "round11_formal_token_matches": " || ".join(inventory.iloc[token_match_positions].formal_identity.astype(str)) if token_match_positions else "",
            "round12_affine_match_identity": best_match["round12_identity"] if best_match else "",
            "round12_affine_match_overlap": int(best_match["overlap_count"]) if best_match else 0,
            "round12_affine_match_correlation": float(best_match["correlation"]) if best_match else np.nan,
            "round12_affine_match_max_residual": float(best_match["maximum_abs_affine_residual"]) if best_match else np.nan,
            "all_duplicate_year_count": all_metrics["duplicate_year_count"],
            "source_year_order": all_metrics["source_order"],
            "lia_unique_count": lia_metrics["unique_count"],
            "lia_median_gap_years": lia_metrics["median_gap_years"],
            "lia_maximum_gap_years": lia_metrics["maximum_gap_years"],
            "lia_occupied_50year_window_count": int(item.lia_occupied_50year_window_count),
            "modern_unique_count": modern_metrics["unique_count"],
            "modern_median_gap_years": modern_metrics["median_gap_years"],
            "modern_maximum_gap_years": modern_metrics["maximum_gap_years"],
            "modern_integer_year_fraction": modern_metrics["integer_year_fraction"],
            "annual_measurement_gate": annual_measurement,
            "documented_direction_gate": documented_direction,
            "documented_seasonality_gate": bool(seasonality),
            "annual_modern_chronology_gate": annual_modern,
            "annual_lia_chronology_gate": annual_lia,
            "annual_moisture_psm_admissible": annual_psm,
            "independent_low_frequency_admissible": low_frequency,
            "originalDataURL": item.originalDataURL,
        })

    result = pd.DataFrame(result_rows).sort_values(["route", "originalDatabase", "datasetId"])
    matches = pd.DataFrame(match_rows)
    result.to_csv(output / "dod2k_identity_chronology_admissibility.csv", index=False)
    matches[matches.passes_affine_identity].to_csv(output / "current_network_sequence_matches.csv", index=False)
    result[result.annual_moisture_psm_admissible].to_csv(output / "annual_moisture_psm_candidates.csv", index=False)
    result[result.independent_low_frequency_admissible].to_csv(output / "independent_low_frequency_candidates.csv", index=False)
    result[result.route == "source_identity_confirmation_required"].to_csv(output / "source_identity_confirmation_queue.csv", index=False)

    route_counts = {str(key): int(value) for key, value in result.route.value_counts().items()}
    identity_counts = {str(key): int(value) for key, value in result.identity_status.value_counts().items()}
    summary = {
        "status": "complete",
        "locked_1921_1939_values_present_or_accessed": False,
        "input_both_capable_count": int(len(result)),
        "route_counts": route_counts,
        "identity_status_counts": identity_counts,
        "current_round12_sequence_duplicate_count": int((result.identity_status == "current_round12_sequence_duplicate").sum()),
        "existing_round11_formal_token_match_count": int((result.identity_status == "existing_round11_formal_token_match").sum()),
        "source_identity_confirmation_required_count": int((result.route == "source_identity_confirmation_required").sum()),
        "annual_moisture_psm_candidate_count": int(result.annual_moisture_psm_admissible.sum()),
        "annual_moisture_psm_candidate_ids": result.loc[result.annual_moisture_psm_admissible, "datasetId"].tolist(),
        "independent_low_frequency_candidate_count": int(result.independent_low_frequency_admissible.sum()),
        "independent_low_frequency_candidate_ids": result.loc[result.independent_low_frequency_admissible, "datasetId"].tolist(),
        "decision": "advance_independent_annual_candidates_to_nested_cross_product_validation" if result.annual_moisture_psm_admissible.any() else ("advance_independent_records_as_low_frequency_validation_only" if result.independent_low_frequency_admissible.any() else "close_dod2k_expansion_pending_source_identity_resolution"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    route_order = [
        "annual_moisture_psm_candidate", "independent_low_frequency_validation",
        "source_identity_confirmation_required", "exclude_existing_project_identity",
        "exclude_current_network_duplicate", "not_admissible",
    ]
    counts = [route_counts.get(route, 0) for route in route_order]
    labels = ["Annual PSM", "Low-frequency", "Identity review", "Existing identity", "Network duplicate", "Not admissible"]
    axes[0].barh(labels, counts, color=["#0072b2", "#009e73", "#e69f00", "#7f7f7f", "#d55e00", "#cc79a7"])
    axes[0].set(xlabel="Record count", title="Frozen admissibility routes")
    for index, count in enumerate(counts):
        axes[0].text(count + 0.1, index, str(count), va="center")

    independent = result[result.identity_status == "independent_new_site"].sort_values("latitude")
    for row_index, row in enumerate(independent.itertuples(index=False)):
        series = values[values.datasetId == row.datasetId]
        allowed = series[(series.year_ce >= 1400) & (series.year_ce <= 2000)]
        axes[1].scatter(allowed.year_ce, np.full(len(allowed), row_index), s=6, alpha=0.55)
    axes[1].axvspan(1900, 1920, color="#dddddd", alpha=0.5)
    axes[1].set(
        xlabel="Allowed year CE", ylabel="Independent record",
        yticks=np.arange(len(independent)), yticklabels=independent.datasetId,
        title="Independent-candidate temporal support",
    )
    figure.suptitle("DoD2k HMA identity and chronology audit", fontsize=14)
    figure.savefig(output / "dod2k_identity_chronology_admissibility.png", dpi=180)
    plt.close(figure)

    result_lines = [
        "# DoD2k HMA Identity and Chronology Audit",
        "",
        f"- Input both-capable records: `{len(result)}`.",
        f"- Annual moisture PSM candidates: `{summary['annual_moisture_psm_candidate_count']}`: `{', '.join(summary['annual_moisture_psm_candidate_ids'])}`.",
        f"- Independent low-frequency candidates: `{summary['independent_low_frequency_candidate_count']}`: `{', '.join(summary['independent_low_frequency_candidate_ids'])}`.",
        f"- Current-network sequence duplicates: `{summary['current_round12_sequence_duplicate_count']}`; existing formal identities: `{summary['existing_round11_formal_token_match_count']}`.",
        f"- Source identity confirmation required: `{summary['source_identity_confirmation_required_count']}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    after = {name: file_state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(json.dumps({
        "sources_before": before,
        "sources_after": after,
        "source_states_unchanged": unchanged,
    }, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
