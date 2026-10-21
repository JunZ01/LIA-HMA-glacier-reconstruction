from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from docx import Document


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_205_setp_winter_temperature_data_recovery_v1"
)
DRIVER = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
    / "run_round205_setp_winter_temperature_data_recovery_v1.py"
)
LEDGER = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_109_cross_network_identity_closure_v1"
    / "proxy_evidence_master_ledger_identity_closed.csv"
)
SUPPLEMENT = OUT / "382_2019_4695_MOESM1_ESM.docx"
SERIES = OUT / "Changdu_winter_Tmin.csv"

EXPECTED_SUPPLEMENT_SHA256 = (
    "5b474997ee6fc9e32238cf314e1d8e09a8b8d25e176baf9e43e288ca9acde8a8"
)
EXPECTED_SERIES_SHA256 = (
    "9957b1717ac11f0ecafac739963ae77c7e96a47c65c9068b372ef17cfe4924d6"
)
EXPECTED_HEADERS = [
    "Sites",
    "Species",
    "Lon.",
    "Lat.",
    "Elev (m, a.s.l.)",
    "First Year",
    "Last Year",
    "EPS > 0.85/Number of cores",
    "MS",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def normalized_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]+\d*", value.lower())
        if len(token) >= 2
    }


def read_station_metadata_only() -> list[dict[str, object]]:
    if sha256(SUPPLEMENT) != EXPECTED_SUPPLEMENT_SHA256:
        raise RuntimeError("Official Springer supplement hash changed")

    document = Document(SUPPLEMENT)
    if not document.tables:
        raise RuntimeError("Springer supplement has no tables")

    # Contract boundary: Table S1 is the first table. Do not traverse later
    # tables or paragraphs because this round is metadata-only.
    table = document.tables[0]
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    if headers != EXPECTED_HEADERS:
        raise RuntimeError(f"Unexpected Table S1 headers: {headers!r}")
    if len(table.rows) != 13:
        raise RuntimeError(f"Expected 12 Table S1 sites, found {len(table.rows) - 1}")

    stations = []
    for row in table.rows[1:]:
        values = [cell.text.strip() for cell in row.cells]
        eps_year, core_count = values[7].split("/", maxsplit=1)
        stations.append(
            {
                "site": values[0],
                "species": values[1],
                "longitude": float(values[2]),
                "latitude": float(values[3]),
                "elevation_m": int(values[4]),
                "first_year": int(values[5]),
                "last_year": int(values[6]),
                "eps_gt_085_year": int(eps_year),
                "core_count_at_eps_threshold": int(core_count),
                "mean_sensitivity": float(values[8]),
            }
        )
    return stations


def read_unique_tree_ledger() -> list[dict[str, str]]:
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        lat = as_float(row.get("latitude", ""))
        lon = as_float(row.get("longitude", ""))
        archive = " ".join(
            [row.get("archive_type", ""), row.get("canonical_archive_class", "")]
        ).lower()
        group = row.get("evidence_group_id", "")
        if lat is not None and lon is not None and "tree" in archive and group:
            by_group[group].append(row)

    unique = []
    for group, group_rows in sorted(by_group.items()):
        preferred = next(
            (row for row in group_rows if row.get("network") == "proxy6"),
            group_rows[0],
        )
        unique.append(preferred)
    return unique


def identity_text(row: dict[str, str]) -> str:
    fields = [
        "record_id",
        "site_name",
        "public_identity_id",
        "metadata_identities",
        "formal_metadata_identity",
        "evidence_group_id",
        "publication_cite_key",
    ]
    return " ".join(row.get(field, "") for field in fields)


def classify_candidate(
    station: dict[str, object], row: dict[str, str], distance_km: float
) -> str:
    start = as_float(row.get("first_year", ""))
    end = as_float(row.get("last_year", ""))
    start_delta = abs(start - int(station["first_year"])) if start is not None else math.inf
    end_delta = abs(end - int(station["last_year"])) if end is not None else math.inf
    site_tokens = normalized_tokens(str(station["site"]))
    ledger_tokens = normalized_tokens(identity_text(row))
    direct_code_match = bool(site_tokens & ledger_tokens)

    if direct_code_match and distance_km <= 10 and start_delta <= 20 and end_delta <= 20:
        return "metadata_exact_candidate_requires_source_confirmation"
    if distance_km <= 35 and start_delta <= 35 and end_delta <= 20:
        return "probable_or_related_chronology_candidate"
    if distance_km <= 75:
        return "nearby_site_candidate_not_identity_closed"
    return "no_close_spatial_candidate"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(SERIES) != EXPECTED_SERIES_SHA256:
        raise RuntimeError("Official CASEarth CSV hash changed")

    stations = read_station_metadata_only()
    ledger = read_unique_tree_ledger()

    station_columns = list(stations[0])
    with (OUT / "springer_table_s1_station_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=station_columns)
        writer.writeheader()
        writer.writerows(stations)

    candidate_rows: list[dict[str, object]] = []
    nearest_classifications: list[str] = []
    for station in stations:
        candidates = []
        for row in ledger:
            lat = as_float(row.get("latitude", ""))
            lon = as_float(row.get("longitude", ""))
            if lat is None or lon is None:
                continue
            distance = haversine_km(
                float(station["latitude"]),
                float(station["longitude"]),
                lat,
                lon,
            )
            candidates.append((distance, row))
        candidates.sort(key=lambda item: item[0])

        nearest_classification = ""
        for rank, (distance, row) in enumerate(candidates[:3], start=1):
            classification = classify_candidate(station, row, distance)
            if rank == 1:
                nearest_classification = classification
                nearest_classifications.append(classification)
            candidate_rows.append(
                {
                    "site": station["site"],
                    "candidate_rank": rank,
                    "distance_km": round(distance, 3),
                    "classification": classification,
                    "evidence_group_id": row.get("evidence_group_id", ""),
                    "record_id": row.get("record_id", ""),
                    "ledger_latitude": row.get("latitude", ""),
                    "ledger_longitude": row.get("longitude", ""),
                    "ledger_first_year": row.get("first_year", ""),
                    "ledger_last_year": row.get("last_year", ""),
                    "public_identity_id": row.get("public_identity_id", ""),
                    "metadata_identities": row.get("metadata_identities", ""),
                    "publication_cite_key": row.get("publication_cite_key", ""),
                    "nearest_site_classification": nearest_classification,
                }
            )

    candidate_columns = list(candidate_rows[0])
    with (OUT / "springer_station_nearest_ledger_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_columns)
        writer.writeheader()
        writer.writerows(candidate_rows)

    classification_counts = {
        value: nearest_classifications.count(value)
        for value in sorted(set(nearest_classifications))
    }
    summary = {
        "round": 205,
        "article_doi": "10.1007/s00382-019-04695-3",
        "dataset_doi": "10.11888/Geogra.tpdc.270139",
        "dataset_id": "5feae826819aec33049b7ca7",
        "official_csv": {
            "filename": SERIES.name,
            "bytes": SERIES.stat().st_size,
            "sha256": sha256(SERIES),
            "opened_or_parsed_in_round_205": False,
        },
        "springer_supplement": {
            "filename": SUPPLEMENT.name,
            "bytes": SUPPLEMENT.stat().st_size,
            "sha256": sha256(SUPPLEMENT),
            "tables_traversed": 1,
            "later_tables_or_paragraphs_traversed": False,
        },
        "study": {
            "target": "previous-November to current-February minimum temperature",
            "region": [94.5, 97.5, 29.5, 31.5],
            "reconstruction_period": [1340, 2007],
            "calibration_verification_period": [1960, 2007],
            "calibration_target": "CRU TS 4.01 regional Tmin",
            "site_count": len(stations),
            "nested_reconstruction_count": 8,
        },
        "identity_audit": {
            "current_ledger_unique_tree_evidence_groups": len(ledger),
            "nearest_candidate_classification_counts": classification_counts,
            "exact_identity_closed_count": 0,
            "reason": (
                "Spatial and chronology metadata can flag candidates but cannot prove "
                "record identity without source chronology identifiers or numeric "
                "sequence comparison. No site is promoted to exact overlap here."
            ),
        },
        "decision": "admit_for_locked_safe_shape_comparison_not_absolute_validation",
        "scientific_boundary": (
            "The series is winter Tmin calibrated to CRU TS 4.01, so it is not an "
            "independent absolute-temperature truth for annual HMA mean temperature. "
            "It can test low-frequency phase and warm-bias direction in southeastern "
            "Tibet after source-overlap sensitivity is reported."
        ),
        "next_round": (
            "Parse the official CSV with year-first filtering; skip 1921-1939 rows "
            "before converting the temperature token, then compare allowed segments "
            "with the frozen project reconstruction."
        ),
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    manifest_paths = [
        DRIVER,
        LEDGER,
        SUPPLEMENT,
        SERIES,
        OUT / "springer_article.html",
        OUT / "dataset_detail.json",
        OUT / "dataset_metadata.json",
        OUT / "file_list.json",
        OUT / "all_files.json",
    ]
    manifest = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": (
                "official_series_held_unopened"
                if path == SERIES
                else "read_only_input_or_reproducibility_source"
            ),
        }
        for path in manifest_paths
        if path.exists()
    ]
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    result = f"""# Round 205 final result

## Decision

`admit_for_locked_safe_shape_comparison_not_absolute_validation`

The official CASEarth annual reconstruction CSV was recovered and preserved
unopened in this round. Its SHA256 is
`{EXPECTED_SERIES_SHA256}`.

## New evidence

- Huang et al. (2019) reconstruct previous-November to current-February minimum
  temperature for southeastern Tibet from 1340 to 2007.
- Springer Table S1 identifies 12 high-elevation tree-ring sites. Only the first
  metadata table was traversed; later supplement content was not read.
- The study uses a different season and a newly sampled 12-site network, but its
  temperature scale is calibrated to CRU TS 4.01 for 1960-2007.

## Identity boundary

The nearest-candidate audit produced: `{classification_counts}`. These are
candidate relationships, not confirmed duplicates. Exact identity remains zero
because coordinates and chronology bounds alone cannot prove that two archived
series are the same measurement.

## Scientific boundary

This dataset can test low-frequency phase and the direction of the southeastern
Tibetan winter-temperature bias. It cannot independently validate the absolute
annual HMA temperature level because it is both seasonal and CRU-calibrated.

## Next action

Round 206 will parse each CSV row's year token first. Rows for 1921-1939 will be
discarded before the temperature token is converted. Only 1340-1920 and
1940-2007 may enter plots or diagnostics. No interpolation across the locked gap
is permitted.

No production reconstruction, EnKF update, climate correction, OGGM run,
locked-period annual value access, or Jionglacuo access occurred.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
