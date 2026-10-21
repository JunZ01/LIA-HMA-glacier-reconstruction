from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from audit_external_proxy_identity_closure_v1 import OUTPUT, RAW, sha256


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
COMBINED_METADATA = OUTPUT / "combined_proxy_metadata_only.csv"
MASTER_LEDGER = PROJECT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_improvement_loop_v2"
    r"\round_109_cross_network_identity_closure_v1"
    r"\proxy_evidence_master_ledger_identity_closed.csv"
)


SITES = [
    {"candidate": "T_HENGDUAN_2024", "site": "DLL", "lat": 27 + 47 / 60, "lon": 98 + 28 / 60, "species": "Larix speciosa", "elevation_m": 3232.42},
    {"candidate": "T_HENGDUAN_2024", "site": "FGH", "lat": 27 + 11 / 60, "lon": 98 + 43 / 60, "species": "Larix speciosa", "elevation_m": 2992},
    {"candidate": "T_HENGDUAN_2024", "site": "KQH", "lat": 28 + 4 / 60, "lon": 98 + 45 / 60, "species": "Larix speciosa", "elevation_m": 3531},
    {"candidate": "T_HENGDUAN_2024", "site": "XYS", "lat": 26 + 26 / 60, "lon": 98 + 46 / 60, "species": "Larix speciosa", "elevation_m": 3199},
    {"candidate": "T_HENGDUAN_2024", "site": "YPL", "lat": 27 + 10 / 60, "lon": 98 + 43 / 60, "species": "Larix speciosa", "elevation_m": 2992},
    {"candidate": "T_HAIZI_2024", "site": "HZS", "lat": 29.41, "lon": 100.13, "species": "Juniperus saltuaria", "elevation_m": 4525},
    {"candidate": "P_SALWEEN_BRB_2020", "site": "BRB1", "lat": 31.76, "lon": 93.61, "species": "Juniperus tibetica", "elevation_m": 4159},
    {"candidate": "P_SALWEEN_BRB_2020", "site": "BRB2", "lat": 31.57, "lon": 93.60, "species": "Juniperus tibetica", "elevation_m": 4400},
    {"candidate": "P_SALWEEN_BRB_2020", "site": "BRB3", "lat": 31.54, "lon": 93.39, "species": "Juniperus tibetica", "elevation_m": 4136},
    {"candidate": "P_BRAHMAPUTRA_JLL_2020", "site": "JLL", "lat": 30.60, "lon": 93.66, "species": "Larix griffithiana", "elevation_m": 4090},
]


DECISIONS = [
    {
        "candidate": "T_HENGDUAN_2024",
        "identity_class": "new_archive_material",
        "identity_confidence": "high",
        "sequence_fingerprint_required": False,
        "immediate_role": "temperature_psm_fitting_candidate",
        "ingestion_decision": "release_regional_chronology_for_small_psm_experiment",
        "reason": "Five Larix speciosa sites are distinct in species and coordinates from the nearest PAGES2k records; the workbook contains a regional chronology.",
        "source_url": "https://doi.org/10.17632/889n6ccx4z.1",
    },
    {
        "candidate": "T_HAIZI_2024",
        "identity_class": "new_archive_material",
        "identity_confidence": "high",
        "sequence_fingerprint_required": False,
        "immediate_role": "temperature_psm_fitting_candidate",
        "ingestion_decision": "release_std_chronology_for_small_psm_experiment",
        "reason": "HZS uses Juniperus saltuaria sampled in 2019/2020 at 4500-4600 m; nearby PAGES2k records are older, different-site material ending in 2006/2007.",
        "source_url": "https://doi.org/10.1038/s41598-024-79096-6",
    },
    {
        "candidate": "P_SALWEEN_BRB_2020",
        "identity_class": "new_archive_material",
        "identity_confidence": "high",
        "sequence_fingerprint_required": False,
        "immediate_role": "external_hydroclimate_consistency_benchmark",
        "ingestion_decision": "hold_until_raw_brb_chronology_or_operator_uncertainty_is_acquired",
        "reason": "BRB1-3 have no close record in the current 311-record pool, but the acquired workbook contains a station-calibrated precipitation reconstruction rather than the raw regional chronology.",
        "source_url": "https://doi.org/10.1016/j.ecolind.2020.106189",
    },
    {
        "candidate": "P_BRAHMAPUTRA_JLL_2020",
        "identity_class": "new_archive_material",
        "identity_confidence": "high",
        "sequence_fingerprint_required": False,
        "immediate_role": "external_hydroclimate_consistency_benchmark",
        "ingestion_decision": "hold_until_raw_jll_chronology_or_operator_uncertainty_is_acquired",
        "reason": "JLL has no close record in the current 311-record pool, but the acquired workbook contains a station-calibrated precipitation reconstruction rather than the raw chronology.",
        "source_url": "https://doi.org/10.1016/j.ecolind.2020.106189",
    },
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def main() -> None:
    proxy_rows = read_csv(COMBINED_METADATA)
    ledger_rows = read_csv(MASTER_LEDGER)
    identity = {}
    for row in ledger_rows:
        if row["network"] != "proxy9" or not row["archive_column_1based"]:
            continue
        identity[int(float(row["archive_column_1based"]))] = row

    nearest_rows = []
    for site in SITES:
        ranked = []
        for record in proxy_rows:
            dist = distance_km(site["lat"], site["lon"], float(record["latitude"]), float(record["longitude"]))
            ranked.append((dist, record))
        dist, nearest = min(ranked, key=lambda item: item[0])
        record_id = int(nearest["record_1based"])
        known = identity.get(record_id, {})
        nearest_rows.append(
            {
                **site,
                "nearest_combined_record_1based": record_id,
                "nearest_distance_km": round(dist, 3),
                "nearest_latitude": nearest["latitude"],
                "nearest_longitude": nearest["longitude"],
                "nearest_public_identity": known.get("formal_metadata_identity", ""),
                "nearest_public_site_name": known.get("site_name", ""),
                "nearest_measurement": known.get("measurement", nearest.get("archive", "")),
                "nearest_first_year": known.get("first_year", ""),
                "nearest_last_year": known.get("last_year", ""),
                "nearest_source_family": nearest["data_name"],
            }
        )

    write_csv(OUTPUT / "candidate_site_nearest_existing_records.csv", nearest_rows)
    write_csv(OUTPUT / "candidate_identity_decisions.csv", DECISIONS)

    originals = [
        RAW / "hengduan_upload_information.xlsx",
        RAW / "haizi_temperature_supplement.xlsx",
        RAW / "salween_brahmaputra_precipitation.xlsx",
        RAW / "hengduan_DLL",
        RAW / "hengduan_FGH",
        RAW / "hengduan_KQH",
        RAW / "hengduan_XYS",
        RAW / "hengduan_YPL",
        MASTER_LEDGER,
    ]
    manifest = {
        "sources": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in originals
        ],
        "source_files_modified": False,
        "locked_interval": [1921, 1939],
        "locked_interval_values_retained_compared_or_output": False,
        "sequence_fingerprint_gate": "waived_because_material_identity_is_decisive",
    }
    (OUTPUT / "final_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "new_temperature_fitting_chronologies": 2,
        "new_precipitation_archive_materials": 2,
        "precipitation_records_immediately_released_for_enkf": 0,
        "precipitation_hold_reason": "downloaded files are calibrated climate reconstructions, not raw BRB/JLL chronologies",
        "formal_blind_validation_records_added": 0,
        "next_experiment": "small two-record temperature PSM and low-frequency contribution test",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = """# External Proxy Identity Closure 2026-08-18 V1

## Result

The identity audit closes two genuinely new temperature fitting chronologies: the five-site western Hengduan `Larix speciosa` regional chronology and the Haizi Mountain `Juniperus saltuaria` STD chronology. Their material identity is distinct from the current PAGES2k network by site/species/collection provenance, so a sequence fingerprint is not needed to establish non-duplication.

The Salween BRB and Brahmaputra JLL materials are also new relative to the current 311-record pool. However, the acquired workbook contains station-calibrated precipitation reconstructions, not the raw BRB/JLL chronologies. They are released only as external hydroclimate consistency benchmarks; direct EnKF ingestion is held until the raw chronologies or a defensible observation-error/operator model is acquired.

## Locked interval

No 1921-1939 candidate or current-proxy values were retained, compared, summarized, or written. This closure used workbook structure, selected metadata/earliest rows, current proxy metadata variables, publication metadata, and site/material provenance.

## Next experiment

Run one small temperature-only experiment with the Hengduan regional chronology and Haizi STD chronology. Test whether each adds low-frequency and eastern/southeastern HMA skill under blocked calibration using only admissible periods. Do not modify the production proxy files and do not promote either record unless it improves the frozen cross-product gates.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
