"""Resolve identity and chronology of both-capable DoD2k temperature records."""

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
MODERN_YEARS = np.arange(1940, 2001)


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


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def cleaned(value: object) -> str:
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
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def chronology(years: np.ndarray) -> dict[str, object]:
    years = np.asarray(years, dtype=float)
    unique = np.unique(years)
    gaps = np.diff(unique)
    return {
        "count": int(len(years)),
        "unique_count": int(len(unique)),
        "duplicate_count": int(len(years) - len(unique)),
        "median_gap": float(np.median(gaps)) if len(gaps) else np.nan,
        "maximum_gap": float(np.max(gaps)) if len(gaps) else np.nan,
        "integer_fraction": (
            float(np.mean(np.isclose(years, np.round(years), atol=1e-8)))
            if len(years)
            else np.nan
        ),
    }


def affine_match(
    candidate_years: np.ndarray,
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
) -> tuple[int, float, float]:
    rounded = np.round(candidate_years)
    integer = np.isclose(candidate_years, rounded, atol=1e-8)
    lookup: dict[int, float] = {}
    duplicates: set[int] = set()
    for year, value in zip(rounded[integer].astype(int), candidate_values[integer]):
        if int(year) in lookup:
            duplicates.add(int(year))
        lookup[int(year)] = float(value)
    for year in duplicates:
        lookup.pop(year, None)
    candidate: list[float] = []
    reference: list[float] = []
    for index, year in enumerate(MODERN_YEARS):
        if year in lookup and np.isfinite(reference_values[index]):
            candidate.append(lookup[year])
            reference.append(float(reference_values[index]))
    count = len(candidate)
    if count < 3:
        return count, np.nan, np.nan
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(reference, dtype=float)
    design = np.column_stack([right, np.ones(count)])
    coefficient = np.linalg.lstsq(design, left, rcond=None)[0]
    residual = float(np.max(np.abs(left - design @ coefficient)))
    correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if np.std(left) > 0 and np.std(right) > 0
        else np.nan
    )
    return count, correlation, residual


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_53_dod2k_temperature_identity_chronology_v1"
    round52 = loop / "round_52_dod2k_temperature_allowed_period_coverage_v1"
    paths = {
        "round52_coverage": round52 / "dod2k_hma_temperature_coverage.csv",
        "round52_values": round52 / "dod2k_hma_temperature_allowed_values.csv",
        "round11_inventory": loop
        / "round_11_full_pages_lmr_hma_source_inventory_v2/full_pages_lmr_hma_candidate_inventory.csv",
        "proxy_inventory": loop
        / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "proxy_matrix": project
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    coverage_all = pd.read_csv(paths["round52_coverage"])
    coverage = coverage_all[coverage_all.both_lia_and_modern_capable.astype(bool)].copy()
    if len(coverage) != 155 or coverage.datasetId.duplicated().any():
        raise RuntimeError("round52-both-capable-contract-failed")
    coverage["datasetId"] = coverage.datasetId.astype(str)
    values = pd.read_csv(paths["round52_values"])
    values["datasetId"] = values.datasetId.astype(str)
    values = values[values.datasetId.isin(set(coverage.datasetId))].copy()
    if values.datasetId.nunique() != 155:
        raise RuntimeError("round52-values-contract-failed")
    if ((values.year_ce >= LOCKED_START) & (values.year_ce <= LOCKED_END)).any():
        raise RuntimeError("locked-period-value-present")

    inventory = pd.read_csv(paths["round11_inventory"])
    inventory = inventory[np.isfinite(inventory.latitude) & np.isfinite(inventory.longitude)].copy()
    inventory_tokens = inventory.formal_identity.fillna("").map(normalize).tolist()
    proxy_inventory = pd.read_csv(paths["proxy_inventory"])
    proxy = loadmat(paths["proxy_matrix"], variable_names=["observations"])
    observations = np.asarray(proxy["observations"], dtype=float)
    if observations.shape != (61, 297):
        raise RuntimeError(f"proxy-observation-shape-failed:{observations.shape}")
    identity_by_column: list[str] = []
    for index in range(1, 298):
        rows = proxy_inventory[proxy_inventory.record_index_1based == index]
        if rows.empty:
            raise RuntimeError(f"proxy-inventory-index-missing:{index}")
        identity_by_column.append(str(rows.iloc[0].identity_key))

    match_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    for item in coverage.itertuples(index=False):
        series = values[values.datasetId == item.datasetId]
        years = series.year_ce.to_numpy(float)
        proxy_values = series.paleo_value.to_numpy(float)
        lia_years = years[(years >= 1400) & (years <= 1899)]
        modern_years = years[(years >= 1940) & (years <= 2000)]
        lia_metrics = chronology(lia_years)
        modern_metrics = chronology(modern_years)

        distances = haversine(
            float(item.geo_meanLat),
            float(item.geo_meanLon),
            inventory.latitude.to_numpy(float),
            inventory.longitude.to_numpy(float),
        )
        nearest_position = int(np.argmin(distances))
        nearest = inventory.iloc[nearest_position]
        dataset_token = normalize(item.dataSetName)
        token_positions = [
            index
            for index, token in enumerate(inventory_tokens)
            if dataset_token and dataset_token in token
        ]

        best_match: dict[str, object] | None = None
        for column, identity in enumerate(identity_by_column):
            count, corr, residual = affine_match(
                years, proxy_values, observations[:, column]
            )
            passes = bool(
                count >= 20
                and np.isfinite(corr)
                and abs(corr) >= 0.999999
                and residual <= 1e-8
            )
            if passes:
                match = {
                    "datasetId": item.datasetId,
                    "proxy_identity": identity,
                    "overlap_count": count,
                    "correlation": corr,
                    "maximum_abs_affine_residual": residual,
                    "passes_affine_identity": True,
                }
                match_rows.append(match)
                if best_match is None or count > int(best_match["overlap_count"]):
                    best_match = match

        if best_match is not None:
            identity_status = "current_proxy_sequence_duplicate"
        elif token_positions:
            identity_status = "existing_round11_formal_token_match"
        elif float(distances[nearest_position]) <= 10:
            identity_status = "same_site_distinct_or_identity_unresolved"
        else:
            identity_status = "independent_new_site"

        direction = cleaned(item.interpretation_direction).lower()
        seasonality = cleaned(item.interpretation_seasonality)
        variable = cleaned(item.interpretation_variable).lower()
        documented = direction in {"positive", "negative"} and bool(seasonality)
        annual_modern = bool(
            modern_metrics["unique_count"] >= 50
            and modern_metrics["duplicate_count"] == 0
            and modern_metrics["integer_fraction"] == 1.0
            and modern_metrics["median_gap"] <= 1.0
            and modern_metrics["maximum_gap"] <= 1.0
        )
        annual_lia = bool(
            lia_metrics["duplicate_count"] == 0
            and lia_metrics["median_gap"] <= 1.0
            and int(item.lia_occupied_50year_window_count) >= 3
        )
        annual_candidate = bool(
            identity_status == "independent_new_site"
            and variable in {"temperature", "temperature+moisture"}
            and documented
            and annual_modern
            and annual_lia
        )
        low_frequency_candidate = bool(
            identity_status == "independent_new_site"
            and variable in {"temperature", "temperature+moisture"}
            and documented
            and bool(item.low_frequency_validation_eligible)
            and not annual_candidate
        )
        if identity_status == "current_proxy_sequence_duplicate":
            route = "exclude_current_proxy_sequence_duplicate"
        elif identity_status == "existing_round11_formal_token_match":
            route = "exclude_existing_project_identity"
        elif identity_status == "same_site_distinct_or_identity_unresolved":
            route = "source_identity_confirmation_required"
        elif annual_candidate:
            route = "annual_temperature_psm_candidate"
        elif low_frequency_candidate:
            route = "independent_low_frequency_temperature_validation"
        else:
            route = "not_admissible"

        result_rows.append(
            {
                "datasetId": item.datasetId,
                "dataSetName": item.dataSetName,
                "geo_siteName": item.geo_siteName,
                "latitude": float(item.geo_meanLat),
                "longitude": float(item.geo_meanLon),
                "archiveType": item.archiveType,
                "paleoData_proxy": item.paleoData_proxy,
                "paleoData_variableName": item.paleoData_variableName,
                "originalDatabase": item.originalDatabase,
                "interpretation_direction": cleaned(item.interpretation_direction),
                "interpretation_seasonality": seasonality,
                "identity_status": identity_status,
                "route": route,
                "nearest_round11_distance_km": float(distances[nearest_position]),
                "nearest_round11_identity": nearest.formal_identity,
                "round11_formal_token_match_count": len(token_positions),
                "round11_formal_token_matches": (
                    " || ".join(inventory.iloc[token_positions].formal_identity.astype(str))
                    if token_positions
                    else ""
                ),
                "proxy_affine_match_identity": (
                    best_match["proxy_identity"] if best_match else ""
                ),
                "proxy_affine_match_overlap": (
                    int(best_match["overlap_count"]) if best_match else 0
                ),
                "proxy_affine_match_correlation": (
                    float(best_match["correlation"]) if best_match else np.nan
                ),
                "proxy_affine_match_max_residual": (
                    float(best_match["maximum_abs_affine_residual"])
                    if best_match
                    else np.nan
                ),
                "lia_unique_count": lia_metrics["unique_count"],
                "lia_median_gap_years": lia_metrics["median_gap"],
                "lia_maximum_gap_years": lia_metrics["maximum_gap"],
                "modern_unique_count": modern_metrics["unique_count"],
                "modern_median_gap_years": modern_metrics["median_gap"],
                "modern_maximum_gap_years": modern_metrics["maximum_gap"],
                "documented_direction_and_seasonality_gate": documented,
                "annual_modern_chronology_gate": annual_modern,
                "annual_lia_chronology_gate": annual_lia,
                "annual_temperature_psm_admissible": annual_candidate,
                "independent_low_frequency_admissible": low_frequency_candidate,
                "weak_round30_window_support_count": int(
                    item.weak_round30_window_support_count
                ),
                "originalDataURL": item.originalDataURL,
            }
        )

    result = pd.DataFrame(result_rows).sort_values(["route", "datasetId"])
    matches = pd.DataFrame(match_rows)
    result.to_csv(output / "dod2k_temperature_identity_chronology.csv", index=False)
    matches.to_csv(output / "current_proxy_sequence_matches.csv", index=False)
    result[result.annual_temperature_psm_admissible].to_csv(
        output / "annual_temperature_psm_candidates.csv", index=False
    )
    result[result.independent_low_frequency_admissible].to_csv(
        output / "independent_low_frequency_temperature_candidates.csv", index=False
    )
    result[result.route == "source_identity_confirmation_required"].to_csv(
        output / "source_identity_confirmation_queue.csv", index=False
    )

    annual_ids = result.loc[result.annual_temperature_psm_admissible, "datasetId"].tolist()
    low_ids = result.loc[result.independent_low_frequency_admissible, "datasetId"].tolist()
    summary = {
        "status": "complete",
        "locked_1921_1939_values_present_or_accessed": False,
        "input_both_capable_count": int(len(result)),
        "route_counts": {
            str(key): int(value) for key, value in result.route.value_counts().items()
        },
        "identity_status_counts": {
            str(key): int(value)
            for key, value in result.identity_status.value_counts().items()
        },
        "current_proxy_sequence_duplicate_count": int(
            np.sum(result.identity_status == "current_proxy_sequence_duplicate")
        ),
        "existing_round11_formal_token_match_count": int(
            np.sum(result.identity_status == "existing_round11_formal_token_match")
        ),
        "source_identity_confirmation_required_count": int(
            np.sum(result.route == "source_identity_confirmation_required")
        ),
        "annual_temperature_psm_candidate_count": len(annual_ids),
        "annual_temperature_psm_candidate_ids": annual_ids,
        "independent_low_frequency_candidate_count": len(low_ids),
        "independent_low_frequency_candidate_ids": low_ids,
        "decision": (
            "advance_independent_annual_temperature_candidates_to_nested_validation"
            if annual_ids
            else (
                "advance_independent_temperature_records_as_low_frequency_validation"
                if low_ids
                else "close_dod2k_temperature_expansion_pending_new_evidence"
            )
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    route_counts = result.route.value_counts()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    axes[0].barh(route_counts.index, route_counts.values, color="#0072b2")
    axes[0].set(xlabel="Record count", title="Temperature identity routes")
    independent = result[result.identity_status == "independent_new_site"].copy()
    for row_index, row in enumerate(independent.itertuples(index=False)):
        record = values[values.datasetId == row.datasetId]
        selected = record[(record.year_ce >= 1400) & (record.year_ce <= 2000)]
        axes[1].scatter(selected.year_ce, np.full(len(selected), row_index), s=7, alpha=0.6)
    axes[1].set(
        xlabel="Allowed year CE",
        ylabel="Independent record",
        yticks=np.arange(len(independent)),
        yticklabels=independent.datasetId,
        title="Independent temperature support",
    )
    figure.savefig(output / "dod2k_temperature_identity_chronology.png", dpi=180)
    plt.close(figure)

    lines = [
        "# DoD2k Temperature Identity and Chronology",
        "",
        f"- Input both-capable records: `{len(result)}`.",
        f"- Current-proxy sequence duplicates: `{summary['current_proxy_sequence_duplicate_count']}`; existing formal identities: `{summary['existing_round11_formal_token_match_count']}`.",
        f"- Identity review required: `{summary['source_identity_confirmation_required_count']}`.",
        f"- Annual temperature candidates: `{annual_ids}`.",
        f"- Independent low-frequency candidates: `{low_ids}`.",
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
