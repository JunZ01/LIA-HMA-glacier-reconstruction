from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_TYPES = {
    7: "ice_cores",
    13: "paleolimnology",
    17: "speleothems",
    19: "other_collections",
}
BASE_QUERY = (
    "https://www.ncei.noaa.gov/access/paleo-search/study/search.json?"
    "dataTypeId={data_type_id}&minLat=25&maxLat=40&minLon=80&maxLon=105&"
    "earliestYear=1500&latestYear=2000&timeFormat=CE&timeMethod=entireOver&"
    "headersOnly=true"
)
DETAIL_QUERY = (
    "https://www.ncei.noaa.gov/access/paleo-search/study/search.json?xmlId={xml_id}"
)
USER_AGENT = "Mozilla/5.0 HMA-climate-metadata-audit/1.0"
TEMPERATURE_TERMS = (
    "temperature",
    "delta 18o",
    "delta18o",
    "oxygen isotope",
    "isotope ratio>delta 18o",
)


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


def fetch(url: str) -> tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type", "")


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bounds(site: dict[str, Any]) -> tuple[float, float, float, float] | None:
    properties = site.get("geo", {}).get("properties", {}) or {}
    south = number(properties.get("southernmostLatitude"))
    north = number(properties.get("northernmostLatitude"))
    west = number(properties.get("westernmostLongitude"))
    east = number(properties.get("easternmostLongitude"))
    if None not in (south, north, west, east):
        return float(south), float(north), float(west), float(east)
    coordinates = site.get("geo", {}).get("geometry", {}).get("coordinates", [])
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        lat = number(coordinates[0])
        lon = number(coordinates[1])
        if lat is not None and lon is not None:
            return lat, lat, lon, lon
    return None


def table_years(table: dict[str, Any]) -> tuple[float | None, float | None]:
    return number(table.get("earliestYearCE")), number(table.get("mostRecentYearCE"))


def table_variables(table: dict[str, Any]) -> tuple[list[str], list[str]]:
    descriptions: list[str] = []
    urls: list[str] = []
    for data_file in table.get("dataFile", []) or []:
        url = str(data_file.get("fileUrl", "")).strip()
        if url:
            urls.append(url)
        for variable in data_file.get("variables", []) or []:
            fields = [
                variable.get("cvWhat"),
                variable.get("cvMaterial"),
                variable.get("cvSeasonality"),
                variable.get("cvDetail"),
                variable.get("cvMethod"),
                variable.get("cvAdditionalInfo"),
                variable.get("cvShortName"),
            ]
            text = " | ".join(str(value) for value in fields if value not in (None, ""))
            if text:
                descriptions.append(text)
    return sorted(set(descriptions)), sorted(set(urls))


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_175_noaa_hma_non_tree_temperature_inventory_v2"
    output.mkdir(parents=True, exist_ok=False)
    raw_headers = output / "headers_unmodified"
    raw_details = output / "study_metadata_unmodified"
    raw_headers.mkdir()
    raw_details.mkdir()

    ledger_path = loop / (
        "round_110_unresolved_proxy_metadata_recovery_v4/"
        "proxy_evidence_master_ledger_metadata_recovered.csv"
    )
    ledger_hash_before = sha256(ledger_path)
    with ledger_path.open("r", encoding="utf-8", newline="") as stream:
        ledger = list(csv.DictReader(stream))

    header_records: dict[str, dict[str, Any]] = {}
    request_manifest: list[dict[str, Any]] = []
    for data_type_id, data_type_name in DATA_TYPES.items():
        url = BASE_QUERY.format(data_type_id=data_type_id)
        payload, final_url, content_type = fetch(url)
        path = raw_headers / f"data_type_{data_type_id}_{data_type_name}.json"
        path.write_bytes(payload)
        response = json.loads(payload.decode("utf-8"))
        studies = response.get("study", []) or []
        request_manifest.append(
            {
                "data_type_id": data_type_id,
                "data_type_name": data_type_name,
                "query": url,
                "final_url": final_url,
                "content_type": content_type,
                "study_count": len(studies),
                "response_sha256": sha256(path),
            }
        )
        for study in studies:
            xml_id = str(study.get("xmlId", "")).strip()
            if xml_id:
                header_records[xml_id] = study

    details: list[dict[str, Any]] = []
    for index, xml_id in enumerate(sorted(header_records, key=lambda value: int(value))):
        payload, final_url, content_type = fetch(DETAIL_QUERY.format(xml_id=xml_id))
        path = raw_details / f"xml_id_{xml_id}.json"
        path.write_bytes(payload)
        response = json.loads(payload.decode("utf-8"))
        studies = response.get("study", []) or []
        if len(studies) != 1:
            raise RuntimeError(f"unexpected-study-count:{xml_id}:{len(studies)}")
        study = studies[0]
        study["_metadata_final_url"] = final_url
        study["_metadata_content_type"] = content_type
        study["_metadata_sha256"] = sha256(path)
        details.append(study)
        if index + 1 < len(header_records):
            time.sleep(0.08)

    rows: list[dict[str, Any]] = []
    for study in details:
        study_text = " ".join(
            str(study.get(key, ""))
            for key in ("studyName", "studyNotes", "scienceKeywords", "dataType")
        ).lower()
        for site in study.get("site", []) or []:
            site_bounds = bounds(site)
            if site_bounds is None:
                continue
            south, north, west, east = site_bounds
            center_lat = (south + north) / 2.0
            center_lon = (west + east) / 2.0
            spatial_extent = max(north - south, east - west)
            exact_hma_site = (
                25 <= center_lat <= 40
                and 80 <= center_lon <= 105
                and spatial_extent <= 5
            )
            for table in site.get("paleoData", []) or []:
                earliest, latest = table_years(table)
                variable_descriptions, data_urls = table_variables(table)
                variable_text = " ".join(variable_descriptions).lower()
                temperature_metadata = any(
                    term in f"{study_text} {variable_text}" for term in TEMPERATURE_TERMS
                )
                covers_target = (
                    earliest is not None
                    and latest is not None
                    and earliest <= 1500
                    and latest >= 2000
                )
                ledger_matches: list[str] = []
                if exact_hma_site:
                    for proxy in ledger:
                        proxy_lat = number(proxy.get("latitude"))
                        proxy_lon = number(proxy.get("longitude"))
                        proxy_first = number(proxy.get("first_year"))
                        proxy_last = number(proxy.get("last_year"))
                        if None in (proxy_lat, proxy_lon, proxy_first, proxy_last):
                            continue
                        if (
                            abs(center_lat - float(proxy_lat)) <= 0.16
                            and abs(center_lon - float(proxy_lon)) <= 0.16
                            and earliest is not None
                            and latest is not None
                            and abs(earliest - float(proxy_first)) <= 20
                            and abs(latest - float(proxy_last)) <= 20
                        ):
                            ledger_matches.append(str(proxy.get("record_id", "")))
                if not exact_hma_site:
                    disposition = "exclude_nonlocal_or_broad_site"
                elif not covers_target:
                    disposition = "exclude_target_period_not_covered"
                elif not temperature_metadata:
                    disposition = "exclude_no_temperature_sensitive_metadata"
                elif ledger_matches:
                    disposition = "existing_proxy_network_source"
                else:
                    disposition = "metadata_candidate_needs_raw_identity_frequency_audit"
                rows.append(
                    {
                        "noaa_study_id": study.get("NOAAStudyId", ""),
                        "xml_id": study.get("xmlId", ""),
                        "study_name": study.get("studyName", ""),
                        "data_type": study.get("dataType", ""),
                        "site_name": site.get("siteName", ""),
                        "center_latitude": center_lat,
                        "center_longitude": center_lon,
                        "region": "central" if center_lon < 90 else "east",
                        "spatial_extent_degrees": spatial_extent,
                        "table_name": table.get("dataTableName", ""),
                        "earliest_year_ce": earliest,
                        "most_recent_year_ce": latest,
                        "temperature_sensitive_metadata": temperature_metadata,
                        "variable_descriptions": ";".join(variable_descriptions),
                        "data_file_urls": ";".join(data_urls),
                        "ledger_matches": ";".join(sorted(set(ledger_matches))),
                        "disposition": disposition,
                    }
                )

    rows.sort(
        key=lambda row: (
            row["disposition"],
            str(row["noaa_study_id"]),
            str(row["site_name"]),
            str(row["table_name"]),
        )
    )
    with (output / "candidate_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    candidates = [
        row
        for row in rows
        if row["disposition"]
        == "metadata_candidate_needs_raw_identity_frequency_audit"
    ]
    disposition_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["disposition"])
        disposition_counts[key] = disposition_counts.get(key, 0) + 1
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "request_manifest": request_manifest,
        "unique_study_metadata_count": len(details),
        "site_table_row_count": len(rows),
        "disposition_counts": disposition_counts,
        "candidate_count": len(candidates),
        "candidate_study_ids": sorted(
            {str(row["noaa_study_id"]) for row in candidates}
        ),
        "candidate_table_keys": [
            f"{row['noaa_study_id']}::{row['site_name']}::{row['table_name']}"
            for row in candidates
        ],
        "annual_proxy_or_climate_values_downloaded_or_parsed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "ledger_sha256_before": ledger_hash_before,
        "ledger_sha256_after": sha256(ledger_path),
        "decision": (
            "audit_candidate_raw_files_identity_and_temporal_resolution"
            if candidates
            else "close_current_noaa_non_tree_inventory"
        ),
    }
    if summary["ledger_sha256_before"] != summary["ledger_sha256_after"]:
        raise RuntimeError("ledger-source-hash-changed")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 175 NOAA HMA Non-Tree Temperature Inventory\n\n"
        f"- Unique NOAA study metadata records: `{len(details)}`.\n"
        f"- Site-table rows assessed: `{len(rows)}`.\n"
        f"- New metadata candidates: `{len(candidates)}`.\n"
        "- Annual values downloaded or parsed: `false`.\n"
        "- Locked-period values accessed: `false`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
