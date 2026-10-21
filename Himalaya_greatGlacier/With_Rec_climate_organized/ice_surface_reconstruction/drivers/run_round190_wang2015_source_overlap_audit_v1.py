from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WANG_SITES = [
    ("Angren", 29.250, 87.183, 1612, 1998, False),
    ("Chuanxi", 30.000, 101.000, 1650, 1994, False),
    ("Gongshan", 27.683, 98.417, 1585, 2005, True),
    ("Litang", 30.233, 100.267, 1694, 2007, True),
    ("Nangqian", 32.200, 96.483, 1624, 2002, False),
    ("Linzhi", 29.367, 94.267, 1500, 1998, False),
    ("Qamdo", 31.167, 96.950, 1000, 2010, False),
    ("Qumalai", 33.800, 96.200, 1395, 2010, False),
    ("Shaluli", 31.133, 99.783, 1446, 2008, False),
    ("Shangrila", 27.617, 99.800, 1571, 2007, True),
    ("Wulan", 37.033, 98.650, 1000, 2004, True),
    ("Zhongdian", 28.367, 99.767, 1475, 2003, False),
]

HENGDUAN_SITES = [
    ("DLL", 27 + 47 / 60, 98 + 28 / 60),
    ("FGH", 27 + 11 / 60, 98 + 43 / 60),
    ("KQH", 28 + 4 / 60, 98 + 45 / 60),
    ("XYS", 26 + 26 / 60, 98 + 46 / 60),
    ("YPL", 27 + 10 / 60, 98 + 43 / 60),
]

HAIZI = ("Haizi", 29.41, 100.13)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_190_wang2015_source_overlap_audit_v1"
    output.mkdir(parents=True, exist_ok=False)

    round53 = loop / "round_53_dod2k_temperature_identity_chronology_v1/dod2k_temperature_identity_chronology.csv"
    round111 = loop / "round_111_validation_evidence_independence_ledger_v1/temperature_evidence_independence_ledger.csv"
    round149 = loop / "round_149_noaa_temperature_candidate_identity_v1/summary.json"
    round180 = loop / "round_180_hengduan_chronology_modern_oof_v2/summary.json"
    round183 = loop / "round_183_haizi_903y_chronology_modern_oof_v2/summary.json"
    sources = [round53, round111, round149, round180, round183]
    hashes_before = {str(path): sha256(path) for path in sources}

    dod = [row for row in read_csv(round53) if row.get("originalDatabase", "").startswith("PAGES")]
    ledger = read_csv(round111)
    overlap_ids = {
        row["site_name"]: row
        for row in ledger
        if row.get("status") == "current_proxy_sequence_or_site_overlap"
    }

    site_rows: list[dict[str, Any]] = []
    for name, lat, lon, first, last, pages_overlap in WANG_SITES:
        nearest = min(
            dod,
            key=lambda row: distance_km(lat, lon, float(row["latitude"]), float(row["longitude"])),
        )
        nearest_distance = distance_km(lat, lon, float(nearest["latitude"]), float(nearest["longitude"]))
        matching_ledger = sorted(
            key for key in overlap_ids if key.lower() in {name.lower(), nearest.get("geo_siteName", "").lower()}
        )
        site_rows.append(
            {
                "wang_site": name,
                "latitude": lat,
                "longitude": lon,
                "first_year": first,
                "last_year": last,
                "wang_declared_pages_asia2k_overlap": pages_overlap,
                "nearest_pages2kv2_dataset": nearest.get("datasetId", ""),
                "nearest_pages2kv2_name": nearest.get("dataSetName", ""),
                "nearest_pages2kv2_site": nearest.get("geo_siteName", ""),
                "nearest_pages2kv2_distance_km": nearest_distance,
                "nearest_pages2kv2_identity_status": nearest.get("identity_status", ""),
                "round111_overlap_names": ";".join(matching_ledger),
                "project_independence_class": (
                    "confirmed_source_overlap"
                    if pages_overlap
                    else "no_publication_declared_overlap_but_spatial_metadata_only"
                ),
            }
        )
    write_csv(output / "wang12_project_overlap.csv", site_rows)

    candidate_rows: list[dict[str, Any]] = []
    for candidate_name, candidate_lat, candidate_lon in HENGDUAN_SITES + [HAIZI]:
        nearest_wang = min(
            WANG_SITES,
            key=lambda row: distance_km(candidate_lat, candidate_lon, row[1], row[2]),
        )
        nearest_distance = distance_km(candidate_lat, candidate_lon, nearest_wang[1], nearest_wang[2])
        family = "Hengduan" if candidate_name != "Haizi" else "Haizi"
        if family == "Hengduan" and nearest_distance <= 15:
            independence = "unresolved_near_site_requires_raw_identity"
        elif nearest_distance > 50:
            independence = "site_independent_no_known_source_identity"
        else:
            independence = "spatially_near_requires_source_review"
        candidate_rows.append(
            {
                "candidate_family": family,
                "candidate_site": candidate_name,
                "latitude": candidate_lat,
                "longitude": candidate_lon,
                "nearest_wang_site": nearest_wang[0],
                "nearest_distance_km": nearest_distance,
                "independence_class": independence,
            }
        )
    write_csv(output / "wang12_candidate_overlap.csv", candidate_rows)

    published_overlap_count = sum(int(row[5]) for row in WANG_SITES)
    hengduan_min = min(row["nearest_distance_km"] for row in candidate_rows if row["candidate_family"] == "Hengduan")
    haizi_distance = next(row["nearest_distance_km"] for row in candidate_rows if row["candidate_family"] == "Haizi")
    hashes_after = {str(path): sha256(path) for path in sources}
    if hashes_before != hashes_after:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "wang_site_count": len(WANG_SITES),
        "wang_publication_declared_pages_overlap_count": published_overlap_count,
        "wang_fully_independent_of_project_network": False,
        "wang_role": "partially_source_overlapping_regional_trajectory_benchmark_only",
        "hengduan_nearest_wang_distance_km": hengduan_min,
        "wang_vs_hengduan_independence": "unresolved_pending_chin041_raw_sample_identity",
        "haizi_nearest_wang_distance_km": haizi_distance,
        "wang_vs_haizi_independence": "site_independent_no_known_source_identity",
        "annual_values_decoded": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
        "decision": "recover_chin041_raw_and_compare_sample_ids_before_dual_operator_test",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"sources": hashes_after}, indent=2), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 190 Wang 2015 Source-Overlap Audit\n\n"
        f"- Wang sites: `{len(WANG_SITES)}`; publication-declared PAGES Asia2k overlaps: `{published_overlap_count}`.\n"
        "- Wang 2015 is not fully independent of the current project network.\n"
        f"- Nearest Hengduan--Wang site distance: `{hengduan_min:.2f} km`; raw identity remains unresolved.\n"
        f"- Nearest Haizi--Wang site distance: `{haizi_distance:.2f} km`; no known source identity is shared.\n"
        "- No annual proxy, reconstruction, climate, or locked-period value was decoded.\n\n"
        "Decision: recover official CHIN041 raw measurements and compare sample IDs before testing a dual chronology operator.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

