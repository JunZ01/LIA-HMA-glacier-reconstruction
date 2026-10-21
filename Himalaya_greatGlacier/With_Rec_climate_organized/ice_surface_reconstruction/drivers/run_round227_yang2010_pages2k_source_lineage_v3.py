from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V1 = LOOP / "round_227_yang2010_pages2k_source_lineage_v1"
V2_FAILED = LOOP / "round_227_yang2010_pages2k_source_lineage_v2"
R226 = LOOP / "round_226_yang2010_setp_622y_metadata_recovery_v1"
OUT = LOOP / "round_227_yang2010_pages2k_source_lineage_v3"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
CONTRACT = DRIVER.with_name("round227_yang2010_pages2k_source_lineage_v3_contract.md")
WORKBOOK = V1 / "sdata201788-s3.xlsx"
R226_SUMMARY = R226 / "summary.json"
FAU_ITEM = R226 / "official_sources_unmodified" / "openfau_item.json"
CATALOG_URLS = {
    "Asi_222": "https://catalog.data.gov/dataset/noaa-wds-paleoclimatology-brauning-mainling-juti-itrdb-chin044",
    "Asi_223": "https://catalog.data.gov/dataset/noaa-wds-paleoclimatology-brauning-nyinchi-juti-itrdb-chin045",
}
CATALOG_TERMS = {
    "Asi_222": ("CHIN044", "Mainling", "JUTI", "Juniperus tibetica", "Tibetan juniper"),
    "Asi_223": ("CHIN045", "Nyinchi", "JUTI", "Juniperus tibetica", "Tibetan juniper"),
}


class TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 LIA metadata audit"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def visible_text(payload: bytes) -> str:
    parser = TextCollector()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return " ".join(parser.parts)


def prefixed(row: dict[str, object], prefix: str) -> object:
    values = [value for key, value in row.items() if key.lower().startswith(prefix.lower())]
    if len(values) != 1:
        raise RuntimeError(f"Expected one metadata field beginning {prefix!r}: {list(row)}")
    return values[0]


def target_rows() -> list[dict[str, object]]:
    workbook = load_workbook(WORKBOOK, read_only=True, data_only=False)
    try:
        rows = workbook["Table S1"].iter_rows(values_only=True)
        headers = [str(value) for value in next(rows)]
        selected = [dict(zip(headers, values)) for values in rows if values[0] in CATALOG_URLS]
    finally:
        workbook.close()
    if {row["PAGES 2k ID"] for row in selected} != set(CATALOG_URLS):
        raise RuntimeError(f"Target rows are not unique: {selected}")
    return sorted(selected, key=lambda row: str(row["PAGES 2k ID"]))


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 12742.0176 * math.asin(math.sqrt(a))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    required = [FACTS, DRIVER, CONTRACT, WORKBOOK, R226_SUMMARY, FAU_ITEM]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("Missing metadata-only input: " + "; ".join(missing))
    facts_before = sha256(FACTS)
    OUT.mkdir(parents=True)
    official = OUT / "official_metadata_unmodified"
    official.mkdir()

    catalog_sources: dict[str, dict[str, object]] = {}
    for pages_id, url in CATALOG_URLS.items():
        payload = download(url)
        path = official / f"data_gov_{pages_id}.html"
        path.write_bytes(payload)
        text = visible_text(payload)
        absent = [term for term in CATALOG_TERMS[pages_id] if term.lower() not in text.lower()]
        if absent:
            raise RuntimeError(f"Catalog identity/species preflight failed for {pages_id}: {absent}")
        catalog_sources[pages_id] = {
            "url": url,
            "path": str(path),
            "sha256": sha256(path),
            "verified_identity_species_terms": list(CATALOG_TERMS[pages_id]),
        }

    rows = target_rows()
    candidate = json.loads(R226_SUMMARY.read_text(encoding="utf-8"))["candidate"]
    fau = json.loads(FAU_ITEM.read_text(encoding="utf-8"))
    authors = [entry["value"] for entry in fau["metadata"]["dc.contributor.author"]]
    comparisons: list[dict[str, object]] = []
    for row in rows:
        pages_id = str(row["PAGES 2k ID"])
        latitude = float(prefixed(row, "lat ("))
        longitude = float(prefixed(row, "long ("))
        comparisons.append(
            {
                "pages2k_id": pages_id,
                "itrdb_site": row["site name"],
                "latitude": latitude,
                "longitude": longitude,
                "distance_from_yang_site_km": round(
                    distance_km(
                        float(candidate["site_latitude"]),
                        float(candidate["site_longitude"]),
                        latitude,
                        longitude,
                    ),
                    3,
                ),
                "coverage": f"{row['min year']}-{row['max year']}",
                "pages2k_reference_doi": row["reference 1 doi"],
                "pages2k_species": "Juniperus tibetica",
                "yang_species": candidate["species"],
                "same_publication_doi": str(row["reference 1 doi"]).lower() == str(candidate["doi"]).lower(),
                "same_exact_coordinate": latitude == float(candidate["site_latitude"])
                and longitude == float(candidate["site_longitude"]),
                "same_species_label": False,
                "same_record_end_year": int(row["max year"]) == 1998,
                "original_data_url": row["original data URL"],
                "wdc_link": row["link to data at WDC-paleo"],
            }
        )

    evidence = {
        "yang_authors": authors,
        "shared_regional_investigator": any("br" in author.lower() and "uning" in author.lower() for author in authors),
        "pages2k_archive_investigator_label": "Brauning",
        "all_publication_dois_differ": all(not item["same_publication_doi"] for item in comparisons),
        "all_exact_coordinates_differ": all(not item["same_exact_coordinate"] for item in comparisons),
        "all_species_labels_differ": all(not item["same_species_label"] for item in comparisons),
        "all_record_end_years_differ": all(not item["same_record_end_year"] for item in comparisons),
        "decision": "independent_tree_core_collections_supported_same_regional_investigator",
        "scope": "tree_core_independence_only_not_regional_climate_independence",
    }
    with (OUT / "corrected_target_metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "lineage_comparison.json").write_text(
        json.dumps({"candidate": candidate, "targets": comparisons, "evidence": evidence}, indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )

    facts_after = sha256(FACTS)
    if facts_after != facts_before:
        raise RuntimeError("Canonical facts changed during read-only audit")
    summary = {
        "round": 227,
        "version": 3,
        "supersedes": {
            "v1_identifier_matching": str(V1 / "summary.json"),
            "v2_status": "preflight_failed_missing_coverage_on_migrated_chin045_catalog_page",
            "v2_path_retained": str(V2_FAILED),
        },
        "candidate": candidate,
        "targets": comparisons,
        "evidence": evidence,
        "catalog_sources": catalog_sources,
        "annual_values_read": False,
        "locked_period_values_read": False,
        "jionglacuo_accessed": False,
        "production_files_modified": False,
        "facts_hash_unchanged": facts_after == facts_before,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(CONTRACT, OUT / "ROUND_CONTRACT.md")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "facts": {"path": str(FACTS), "sha256": facts_after},
                "pages2k_workbook": {"path": str(WORKBOOK), "sha256": sha256(WORKBOOK)},
                "round226_summary": {"path": str(R226_SUMMARY), "sha256": sha256(R226_SUMMARY)},
                "fau_item": {"path": str(FAU_ITEM), "sha256": sha256(FAU_ITEM)},
                "catalog_sources": catalog_sources,
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 227 v3 source-lineage closure\n\n"
        "- `Asi_222` = `CHIN044 / Mainling`; `Asi_223` = `CHIN045 / Nyinchi`.\n"
        "- The two PAGES2k rows and Yang 2010 have different DOI, exact coordinate, "
        "species label, and record end year.\n"
        "- Br\u00e4uning is shared, so these are independent tree-core collections from the "
        "same regional investigator network, not independent regional climate forcing.\n"
        f"- Decision: `{evidence['decision']}`.\n"
        "- No annual, locked-period, or Jionglacuo value was read.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
