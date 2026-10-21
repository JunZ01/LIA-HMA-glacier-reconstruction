from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_204_setp_ppr_temperature_field_metadata_v1"
)
LEDGER = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_109_cross_network_identity_closure_v1"
    / "proxy_evidence_master_ledger_identity_closed.csv"
)


STATIONS = [
    ("LiJ", 100.22, 26.87, 0.50, 3, 1509, 2004, 0.46, 21.2, 0.13, 0.66),
    ("XingGL", 99.70, 27.83, 0.35, 1, 1516, 2004, 0.52, 27.1, 0.20, 0.45),
    ("DaoC", 100.30, 29.05, 0.50, 4, 1542, 2004, 0.43, 18.1, 0.09, 0.57),
    ("LaS", 91.13, 29.67, 0.40, 4, 1542, 1998, 0.46, 20.8, 0.01, 0.91),
    ("LinZ", 94.33, 29.67, 0.65, 4, 1516, 2004, 0.49, 23.9, 0.17, 0.52),
    ("XinL", 100.32, 30.93, 0.75, 4, 1730, 2004, 0.44, 19.0, 0.07, 0.72),
    ("DaoF", 101.12, 30.98, 0.70, 3, 1542, 2004, 0.43, 18.1, 0.08, 0.62),
    ("XiaoJ", 102.35, 31.00, 0.60, 2, 1520, 2002, 0.49, 23.9, 0.17, 0.64),
    ("BanG", 90.02, 31.38, 0.50, 14, 1553, 2001, 0.54, 29.5, 0.13, 0.85),
    ("DingQ", 95.60, 31.42, 0.70, 11, 1553, 2002, 0.52, 27.4, 0.15, 0.60),
    ("NaQ", 92.07, 31.48, 0.60, 9, 1553, 1998, 0.60, 36.1, 0.17, 0.76),
    ("GanZ", 100.00, 31.62, 0.70, 10, 1542, 2002, 0.73, 53.5, 0.43, 0.54),
    ("DeG", 98.58, 31.80, 0.70, 5, 1542, 2004, 0.61, 37.4, 0.27, 0.64),
    ("MaEK", 102.23, 31.90, 0.65, 8, 1730, 2001, 0.61, 37.5, 0.21, 0.64),
    ("NangQ", 96.48, 32.20, 0.80, 10, 1553, 2002, 0.58, 33.9, 0.23, 0.70),
    ("SeD", 100.33, 32.28, 0.70, 11, 1520, 2002, 0.70, 49.3, 0.38, 0.50),
    ("SongP", 103.57, 32.65, 0.70, 8, 1730, 2002, 0.60, 35.6, 0.25, 0.54),
    ("HongY", 102.55, 32.80, 0.75, 8, 1730, 2001, 0.48, 22.7, 0.09, 0.54),
    ("ZaD", 95.30, 32.90, 0.70, 15, 1480, 2001, 0.59, 34.4, 0.15, 0.77),
    ("ShiQ", 98.10, 32.98, 0.80, 12, 1520, 2002, 0.69, 47.1, 0.31, 0.57),
    ("YuS", 97.02, 33.02, 0.80, 9, 1553, 2002, 0.60, 35.9, 0.25, 0.64),
    ("JiuZ", 101.48, 33.43, 0.75, 12, 1520, 2002, 0.70, 49.1, 0.36, 0.48),
    ("RuoEG", 102.97, 33.58, 0.80, 10, 1730, 2001, 0.57, 28.7, 0.16, 0.54),
    ("DaR", 99.65, 33.75, 0.80, 9, 1480, 2002, 0.50, 25.0, 0.14, 0.56),
    ("QingSH", 97.13, 33.80, 0.80, 12, 1520, 2002, 0.52, 27.1, 0.10, 0.55),
    ("QuML", 95.78, 34.13, 0.80, 14, 1553, 2002, 0.63, 39.8, 0.25, 0.69),
    ("HeN", 101.60, 34.67, 0.80, 8, 1520, 2001, 0.69, 48.0, 0.39, 0.50),
    ("MaD", 98.22, 34.92, 0.80, 13, 1553, 2001, 0.56, 31.3, 0.18, 0.55),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as stream:
        ledger = list(csv.DictReader(stream))

    domain_tree = []
    for row in ledger:
        lat = as_float(row.get("latitude", ""))
        lon = as_float(row.get("longitude", ""))
        archive = " ".join(
            [row.get("archive_type", ""), row.get("canonical_archive_class", "")]
        ).lower()
        if (
            lat is not None
            and lon is not None
            and 25.0 <= lat <= 35.0
            and 90.0 <= lon <= 104.0
            and "tree" in archive
        ):
            domain_tree.append(row)

    station_columns = [
        "station",
        "longitude",
        "latitude",
        "search_spatial_correlation",
        "retained_chronology_count",
        "instrument_start",
        "instrument_end",
        "reconstruction_start",
        "reconstruction_end",
        "R",
        "R2_percent",
        "RE",
        "RMSE_degC",
    ]
    with (OUT / "station_reconstruction_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(station_columns)
        for station in STATIONS:
            writer.writerow(station[:5] + (1961, 2004) + station[5:])

    selected_columns = [
        "network",
        "network_record_1based",
        "record_id",
        "latitude",
        "longitude",
        "archive_type",
        "canonical_archive_class",
        "source_family",
        "data_name",
        "site_name",
        "public_identity_id",
        "evidence_group_id",
        "publication_cite_key",
        "publication_year",
    ]
    with (OUT / "current_project_setp_tree_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=selected_columns)
        writer.writeheader()
        for row in domain_tree:
            writer.writerow({key: row.get(key, "") for key in selected_columns})

    evidence_groups = sorted(
        {row.get("evidence_group_id", "") for row in domain_tree if row.get("evidence_group_id")}
    )
    source_counts = Counter(row.get("source_family", "") or "<missing>" for row in domain_tree)
    starts = [station[5] for station in STATIONS]
    summary = {
        "round": 204,
        "doi": "10.2113/2022/9648384",
        "study_network": {
            "chronology_count": 53,
            "itrdb_chronology_count": 37,
            "itrdb_fraction": 37 / 53,
            "width_count": 51,
            "mxd_count": 1,
            "carbon_isotope_count": 1,
        },
        "field": {
            "target": "May-June mean temperature",
            "station_count": len(STATIONS),
            "instrumental_target": "Chinese Homogeneous Temperature Data Set",
            "instrumental_period": [1961, 2004],
            "common_reconstruction_period": [1730, 1998],
            "longest_single_period": [1480, 2002],
            "stations_starting_by_1500": sum(year <= 1500 for year in starts),
            "stations_starting_by_1520": sum(year <= 1520 for year in starts),
            "stations_starting_at_1730": sum(year == 1730 for year in starts),
        },
        "public_data": {
            "paper_statement": "Dataset is available upon request to the corresponding author or first author.",
            "crossref_numeric_attachment": False,
            "doaj_numeric_attachment": False,
            "openalex_numeric_attachment": False,
            "publisher_automation_result": "Cloudflare challenge HTML, not article PDF or supplement",
            "lzu_repository_automation_result": "HTTP 503 response",
        },
        "project_overlap_context": {
            "setp_domain_tree_records": len(domain_tree),
            "setp_domain_unique_evidence_groups": len(evidence_groups),
            "source_family_counts": dict(sorted(source_counts.items())),
            "source_level_overlap": True,
            "exact_record_overlap_resolved": False,
            "reason": (
                "The paper states that 37 ITRDB chronologies were already used in the "
                "PAGES2k Asia reconstruction; the current project contains PAGES2k/LMR "
                "tree records in the same domain, but the paper's 53-record identity table "
                "and numeric field are not publicly exposed."
            ),
        },
        "admission_gates": {
            "geographic_relevance": "pass",
            "full_1500_1899_field_coverage": "fail_only_two_stations_reach_1500",
            "public_numeric_data": "fail_author_request_only",
            "record_level_identity_closure": "fail_missing_53_record_table",
            "independent_proxy_network": "fail_confirmed_source_level_overlap",
        },
        "decision": "comparison_only_pending_numeric_data",
        "permitted_next_use": (
            "If authors provide the 28-station reconstruction and 53-record identity table, "
            "run locked-safe validation for 1480-1920 and 1940-2002, with the 1921-1939 "
            "values skipped before decoding. Use it as source-overlap-aware comparison, not "
            "as independent absolute truth."
        ),
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    manifest_paths = [
        LEDGER,
        OUT / "crossref.json",
        OUT / "openalex.json",
        OUT / "doaj.json",
        OUT / "publisher_page.html",
        OUT / "publisher_pdf_endpoint_challenge.html",
        OUT / "lzu_480451.html",
        OUT / "lzu_481739.html",
    ]
    manifest = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": "read_only_input" if path == LEDGER else "downloaded_metadata_or_failed_endpoint_snapshot",
        }
        for path in manifest_paths
        if path.exists()
    ]
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    result = f"""# Round 204 final result

## Decision

`comparison_only_pending_numeric_data`

Li et al. (2022) is relevant to eastern HMA, but it is not an immediately
usable independent validator for the full 1500-1899 warm-bias question.

## What passed

- The target is May-June temperature over 25-35 N, 90-104 E.
- The reconstruction contains 28 station targets.
- Two station reconstructions (`ZaD` and `DaR`) reach 1480.
- Calibration uses the Chinese Homogeneous Temperature Data Set for 1961-2004;
  CRU TS 3.20 is described for spatial-correlation analysis rather than as the
  station reconstruction target.

## What failed

- Only 2 of 28 stations reach 1500; the common field begins in 1730. This is
  not a complete 1500-1899 regional field.
- The article says the numerical dataset is available only on request. Crossref,
  DOAJ, and OpenAlex expose no numeric attachment. Automated publisher access
  returned a Cloudflare challenge, and the LZU repository returned HTTP 503.
- The paper uses 53 chronologies, including 37 ITRDB records explicitly linked
  by the authors to the existing PAGES2k Asia reconstruction. Exact overlap
  cannot be closed without the 53-record identity table, so independence must
  not be claimed.
- The current project ledger already contains {len(domain_tree)} tree-record
  representations ({len(evidence_groups)} evidence groups) in the same SETP
  domain, making record-level identity closure mandatory before validation.

## Next admissible action

Request from the authors: (1) the 28-station annual reconstructed field,
(2) the 53 chronology identifiers/coordinates and reliable start years, and
(3) the station-to-chronology predictor membership. If delivered, parse years
with 1921-1939 skipped before value conversion and evaluate 1480-1920 plus
1940-2002. The result remains source-overlap-aware comparison evidence, not
independent absolute truth.

No production reconstruction, EnKF update, climate correction, OGGM run,
locked-period annual value access, or Jionglacuo access occurred.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
