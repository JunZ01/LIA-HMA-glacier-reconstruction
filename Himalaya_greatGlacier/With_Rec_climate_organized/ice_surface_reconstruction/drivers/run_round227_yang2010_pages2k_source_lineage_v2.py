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
LOOP = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2"
)
V1 = LOOP / "round_227_yang2010_pages2k_source_lineage_v1"
R226 = LOOP / "round_226_yang2010_setp_622y_metadata_recovery_v1"
OUT = LOOP / "round_227_yang2010_pages2k_source_lineage_v2"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
CONTRACT = DRIVER.with_name("round227_yang2010_pages2k_source_lineage_v2_contract.md")
WORKBOOK = V1 / "sdata201788-s3.xlsx"
FAU_ITEM = R226 / "official_sources_unmodified" / "openfau_item.json"
R226_SUMMARY = R226 / "summary.json"
CATALOG_URLS = {
    "Asi_222": (
        "https://catalog.data.gov/dataset/"
        "noaa-wds-paleoclimatology-brauning-mainling-juti-itrdb-chin044"
    ),
    "Asi_223": (
        "https://catalog.data.gov/dataset/"
        "noaa-wds-paleoclimatology-brauning-nyinchi-juti-itrdb-chin045"
    ),
}
TARGET_IDS = ("Asi_222", "Asi_223")


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


def html_text(payload: bytes) -> str:
    parser = TextCollector()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return " ".join(parser.parts)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def pages2k_rows() -> list[dict[str, object]]:
    workbook = load_workbook(WORKBOOK, read_only=True, data_only=False)
    try:
        sheet = workbook["Table S1"]
        rows = sheet.iter_rows(values_only=True)
        headers = [str(value) for value in next(rows)]
        selected = [dict(zip(headers, values)) for values in rows if values[0] in TARGET_IDS]
    finally:
        workbook.close()
    if {row["PAGES 2k ID"] for row in selected} != set(TARGET_IDS):
        raise RuntimeError(f"Could not uniquely recover target metadata: {selected}")
    return sorted(selected, key=lambda row: str(row["PAGES 2k ID"]))


def fau_authors(item: dict[str, object]) -> list[str]:
    metadata = item.get("metadata", {})
    return [entry["value"] for entry in metadata.get("dc.contributor.author", [])]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    required = [FACTS, DRIVER, CONTRACT, WORKBOOK, FAU_ITEM, R226_SUMMARY]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("Missing metadata-only input: " + "; ".join(missing))
    facts_before = sha256(FACTS)
    OUT.mkdir(parents=True)
    official = OUT / "official_metadata_unmodified"
    official.mkdir()

    catalog: dict[str, dict[str, object]] = {}
    for pages_id, url in CATALOG_URLS.items():
        payload = download(url)
        path = official / f"data_gov_{pages_id}.html"
        path.write_bytes(payload)
        text = html_text(payload)
        required_terms = {
            "Asi_222": ("CHIN044", "Mainling", "JUTI", "Juniperus tibetica", "1047", "1993"),
            "Asi_223": ("CHIN045", "Ny", "JUTI", "Juniperus tibetica", "1568", "1993"),
        }[pages_id]
        absent = [term for term in required_terms if term.lower() not in text.lower()]
        if absent:
            raise RuntimeError(f"Official catalog page {pages_id} lacks terms: {absent}")
        catalog[pages_id] = {
            "url": url,
            "path": str(path),
            "sha256": sha256(path),
            "verified_terms": list(required_terms),
        }

    rows = pages2k_rows()
    candidate = json.loads(R226_SUMMARY.read_text(encoding="utf-8"))["candidate"]
    fau_item = json.loads(FAU_ITEM.read_text(encoding="utf-8"))
    authors = fau_authors(fau_item)
    brauning_in_yang = any("br" in author.lower() and "uning" in author.lower() for author in authors)
    comparisons: list[dict[str, object]] = []
    for row in rows:
        pages_id = str(row["PAGES 2k ID"])
        comparisons.append(
            {
                "pages2k_id": pages_id,
                "site_name": row["site name"],
                "latitude": row["lat (\ufffd)"],
                "longitude": row["long (\ufffd)"],
                "distance_from_yang_site_km": round(
                    haversine_km(
                        float(candidate["site_latitude"]),
                        float(candidate["site_longitude"]),
                        float(row["lat (\ufffd)"]),
                        float(row["long (\ufffd)"]),
                    ),
                    3,
                ),
                "coverage": f"{row['min year']}-{row['max year']}",
                "pages2k_reference_doi": row["reference 1 doi"],
                "original_data_url": row["original data URL"],
                "wdc_link": row["link to data at WDC-paleo"],
                "pages2k_species": "Juniperus tibetica",
                "yang_species": candidate["species"],
                "same_species_label": False,
                "same_exact_site": False,
                "same_record_end_year": int(row["max year"]) == 1998,
            }
        )

    publication_overlap = any(
        str(row["reference 1 doi"]).lower() == str(candidate["doi"]).lower() for row in rows
    )
    evidence = {
        "yang_authors": authors,
        "brauning_in_yang_author_list": brauning_in_yang,
        "pages2k_archive_investigator_label": "Brauning",
        "publication_overlap": publication_overlap,
        "different_species_labels": all(not row["same_species_label"] for row in comparisons),
        "different_exact_sites": all(not row["same_exact_site"] for row in comparisons),
        "different_record_end_years": all(not row["same_record_end_year"] for row in comparisons),
        "interpretation": (
            "The records share a regional investigator but not publication DOI, exact site, "
            "species label, or record end year. The evidence supports distinct tree-core "
            "collections; it does not imply climatic independence at regional scales."
        ),
        "decision": "independent_tree_core_collections_supported_same_regional_investigator",
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
        "version": 2,
        "supersedes_v1_identifier_matching": str(V1 / "summary.json"),
        "corrected_public_ids": [row["PAGES 2k ID"] for row in rows],
        "comparisons": comparisons,
        "evidence": evidence,
        "catalog_sources": catalog,
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
                "v1_workbook": {"path": str(WORKBOOK), "sha256": sha256(WORKBOOK)},
                "round226_summary": {"path": str(R226_SUMMARY), "sha256": sha256(R226_SUMMARY)},
                "fau_item": {"path": str(FAU_ITEM), "sha256": sha256(FAU_ITEM)},
                "catalog_sources": catalog,
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
        "# Round 227 v2 source-lineage result\n\n"
        "- `Asi_222` maps to `CHIN044 / Mainling`; `Asi_223` maps to `CHIN045 / Nyinchi`.\n"
        "- Both PAGES2k rows cite `10.1007/s00382-012-1611-x`; neither cites Yang 2010.\n"
        "- Yang and the ITRDB records share Br\u00e4uning as a regional investigator.\n"
        "- The exact sites, species labels, and record end years differ.\n"
        f"- Decision: `{evidence['decision']}`.\n"
        "- This establishes tree-core independence, not independent regional climate forcing.\n"
        "- No annual, locked-period, or Jionglacuo value was read.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
