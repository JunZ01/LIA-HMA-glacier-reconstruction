from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


QUERY = (
    "https://www.ncei.noaa.gov/access/paleo-search/study/search.json?"
    "dataTypeId=18&reconstructionsOnly=N&minLat=25&maxLat=40&"
    "minLon=80&maxLon=105&earliestYear=1500&latestYear=2000&"
    "timeFormat=CE&timeMethod=entireOver&headersOnly=true"
)
CODE = re.compile(r"ITRDB\s+([A-Z]{2,5}\d{3})", re.IGNORECASE)


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


def number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_171_noaa_hma_temperature_raw_inventory_v1"
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = loop / (
        "round_110_unresolved_proxy_metadata_recovery_v4/"
        "proxy_evidence_master_ledger_metadata_recovered.csv"
    )
    catalog_path = root / (
        "validation_data/external_sources/"
        "hma_temperature_duan_mxd_metadata_recovery_20260819_v1/"
        "archive_metadata_unmodified/ITRDBmetadata12January2022.txt"
    )
    local_hashes_before = {
        "ledger": sha256(ledger_path),
        "itrdb_catalog": sha256(catalog_path),
    }

    request = urllib.request.Request(QUERY, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
    api_path = output / "noaa_headers_only_response_unmodified.json"
    api_path.write_bytes(payload)
    api = json.loads(payload.decode("utf-8"))
    studies = api.get("study", [])

    catalog: dict[str, dict[str, object]] = {}
    with catalog_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 9 or not fields[0].isdigit():
                continue
            catalog[fields[1].upper()] = {
                "catalog_study_id": fields[0],
                "code": fields[1].upper(),
                "catalog_name": fields[2],
                "latitude": number(fields[3]),
                "longitude": number(fields[4]),
                "elevation_m": number(fields[5]),
                "species_code": fields[6],
                "first_year": number(fields[7]),
                "last_year": number(fields[8]),
            }

    with ledger_path.open("r", encoding="utf-8", newline="") as stream:
        ledger = list(csv.DictReader(stream))
    unique_ledger = [row for row in ledger if row["network"] == "proxy6"]

    special = {
        "CHIN074": ("closed_qamdo2_modern_oof", "Round 167"),
        "CHIN075": ("not_temperature_candidate", "Maxiong scPDSI source"),
        "NPL063": ("not_temperature_candidate", "legacy Rara precipitation/TRW source"),
        "CHIN037": ("not_independent_temperature_candidate", "ambiguous Fan/Duan-relative source"),
        "CHIN039": ("not_independent_temperature_candidate", "ambiguous Fan/Duan-relative source"),
        "CHIN040": ("not_independent_temperature_candidate", "ambiguous Fan/Duan-relative source"),
    }
    inventory: list[dict[str, object]] = []
    for study in studies:
        name = str(study.get("studyName", ""))
        match = CODE.search(name)
        code = match.group(1).upper() if match else ""
        site = catalog.get(code, {})
        lat = site.get("latitude")
        lon = site.get("longitude")
        first = site.get("first_year")
        last = site.get("last_year")
        matches: list[str] = []
        if all(value is not None for value in (lat, lon, first, last)):
            for row in unique_ledger:
                row_lat = number(row.get("latitude", ""))
                row_lon = number(row.get("longitude", ""))
                row_first = number(row.get("first_year", ""))
                row_last = number(row.get("last_year", ""))
                if None in (row_lat, row_lon, row_first, row_last):
                    continue
                if (
                    abs(float(lat) - row_lat) <= 0.16
                    and abs(float(lon) - row_lon) <= 0.16
                    and abs(float(first) - row_first) <= 20
                    and abs(float(last) - row_last) <= 20
                ):
                    matches.append(row["record_id"])
        if code in special:
            disposition, reason = special[code]
        elif not code:
            disposition, reason = "catalogue_aggregate_not_site_candidate", "no ITRDB site code"
        elif matches:
            disposition, reason = "existing_proxy_network_source", "coordinate and chronology-bound metadata match"
        else:
            disposition, reason = "metadata_review_required", "no automatic ledger match"
        inventory.append(
            {
                "noaa_study_id": study.get("NOAAStudyId", ""),
                "study_name": name,
                "itrdb_code": code,
                "contribution_date": study.get("contributionDate", ""),
                "latitude": lat,
                "longitude": lon,
                "species_code": site.get("species_code", ""),
                "first_year": first,
                "last_year": last,
                "ledger_match_count": len(matches),
                "ledger_matches": ";".join(matches),
                "disposition": disposition,
                "reason": reason,
            }
        )

    review = [row for row in inventory if row["disposition"] == "metadata_review_required"]
    advancing = [
        row
        for row in inventory
        if row["disposition"] == "advance_new_temperature_candidate"
    ]
    counts: dict[str, int] = {}
    for row in inventory:
        key = str(row["disposition"])
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "query": QUERY,
        "api_final_url": final_url,
        "api_content_type": content_type,
        "api_response_sha256": sha256(api_path),
        "study_header_count": len(studies),
        "disposition_counts": counts,
        "metadata_review_required_codes": [row["itrdb_code"] for row in review],
        "advancing_candidate_codes": [row["itrdb_code"] for row in advancing],
        "annual_proxy_or_climate_values_downloaded_or_parsed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "local_source_hashes_before": local_hashes_before,
        "local_source_hashes_after": {
            "ledger": sha256(ledger_path),
            "itrdb_catalog": sha256(catalog_path),
        },
        "decision": (
            "review_unmatched_metadata_before_closure"
            if review
            else "close_current_noaa_catalogue_as_source_of_new_temperature_candidate"
        ),
    }
    if summary["local_source_hashes_before"] != summary["local_source_hashes_after"]:
        raise RuntimeError("local-source-hash-changed")
    fieldnames = list(inventory[0]) if inventory else []
    with (output / "candidate_inventory.csv").open(
        "w", encoding="ascii", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 171 NOAA HMA Raw-Temperature Inventory\n\n"
        f"- NOAA study headers: `{len(studies)}`.\n"
        f"- Existing-network sources: `{counts.get('existing_proxy_network_source', 0)}`.\n"
        f"- Metadata review required: `{len(review)}`.\n"
        f"- New candidates advancing now: `{len(advancing)}`.\n"
        "- Annual values downloaded or parsed: `false`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
