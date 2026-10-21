from __future__ import annotations

import csv
import hashlib
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_226_yang2010_setp_622y_metadata_recovery_v1"
DRIVER = Path(__file__).resolve()
LEDGER = (
    LOOP
    / "round_110_unresolved_proxy_metadata_recovery_v4"
    / "proxy_evidence_master_ledger_metadata_recovered.csv"
)
ITEM_ID = "45d87814-8667-46e0-aafc-55b47244701e"
ITEM_API = f"https://open.fau.de/server/api/core/items/{ITEM_ID}"
BUNDLES_API = f"{ITEM_API}/bundles?size=100"
DOI = "10.1177/0959683609350388"
TITLE = "A 622-year regional temperature history of southeast Tibet derived from tree rings"
SITE_LAT = 29 + 22 / 60
SITE_LON = 94 + 16 / 60
PDF_MAGIC = b"%PDF-"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 LIA climate metadata audit",
            "Accept": "application/json,application/pdf;q=0.9,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def request_json(url: str) -> dict[str, object]:
    return json.loads(request_bytes(url).decode("utf-8"))


def as_float(value: object) -> float | None:
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


def nearest_tree_groups() -> list[dict[str, object]]:
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates: list[tuple[float, dict[str, str]]] = []
    seen: set[str] = set()
    for row in rows:
        archive = " ".join(
            [row.get("archive_type", ""), row.get("canonical_archive_class", "")]
        ).lower()
        lat = as_float(row.get("latitude"))
        lon = as_float(row.get("longitude"))
        group = row.get("evidence_group_id", "")
        if "tree" not in archive or lat is None or lon is None or not group or group in seen:
            continue
        seen.add(group)
        candidates.append((haversine_km(SITE_LAT, SITE_LON, lat, lon), row))
    candidates.sort(key=lambda item: item[0])
    output: list[dict[str, object]] = []
    for rank, (distance, row) in enumerate(candidates[:15], start=1):
        output.append(
            {
                "rank": rank,
                "distance_km": round(distance, 3),
                "evidence_group_id": row.get("evidence_group_id", ""),
                "record_id": row.get("record_id", ""),
                "latitude": row.get("latitude", ""),
                "longitude": row.get("longitude", ""),
                "first_year": row.get("first_year", ""),
                "last_year": row.get("last_year", ""),
                "public_identity_id": row.get("public_identity_id", ""),
                "metadata_identities": row.get("metadata_identities", ""),
                "publication_cite_key": row.get("publication_cite_key", ""),
                "publication_author": row.get("publication_author", ""),
                "publication_year": row.get("publication_year", ""),
            }
        )
    return output


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    OUT.mkdir(parents=True)
    official = OUT / "official_sources_unmodified"
    official.mkdir()

    item = request_json(ITEM_API)
    bundles = request_json(BUNDLES_API)
    write_json(official / "openfau_item.json", item)
    write_json(official / "openfau_bundles.json", bundles)

    bundle_rows = bundles.get("_embedded", {}).get("bundles", [])
    bitstreams: list[dict[str, object]] = []
    for bundle in bundle_rows:
        url = bundle.get("_links", {}).get("bitstreams", {}).get("href")
        if not url:
            continue
        inventory = request_json(f"{url}?size=100" if "?" not in url else url)
        write_json(official / f"openfau_{str(bundle['name']).lower()}_bitstreams.json", inventory)
        for entry in inventory.get("_embedded", {}).get("bitstreams", []):
            bitstreams.append(
                {
                    "bundle": bundle.get("name"),
                    "name": entry.get("name"),
                    "size_bytes": entry.get("sizeBytes"),
                    "uuid": entry.get("uuid"),
                    "content_url": entry.get("_links", {}).get("content", {}).get("href"),
                    "checksum_algorithm": entry.get("checkSum", {}).get("checkSumAlgorithm"),
                    "checksum": entry.get("checkSum", {}).get("value"),
                }
            )

    originals = [row for row in bitstreams if row["bundle"] == "ORIGINAL"]
    if len(originals) != 1 or originals[0]["name"] != "2972_yang_622_year_4223.pdf":
        raise RuntimeError(f"Unexpected official ORIGINAL inventory: {originals}")
    pdf_path = official / str(originals[0]["name"])
    pdf_path.write_bytes(request_bytes(str(originals[0]["content_url"])))
    if not pdf_path.read_bytes()[:5] == PDF_MAGIC:
        raise RuntimeError("Official ORIGINAL is not a PDF")

    crossref_url = "https://api.crossref.org/works/" + urllib.parse.quote(DOI, safe="")
    datacite_url = (
        "https://api.datacite.org/dois?query="
        + urllib.parse.quote(f'titles.title:"{TITLE}"')
        + "&page[size]=100"
    )
    crossref = request_json(crossref_url)
    datacite = request_json(datacite_url)
    write_json(official / "crossref.json", crossref)
    write_json(official / "datacite_title_query.json", datacite)

    exact_datacite = []
    for entry in datacite.get("data", []):
        titles = entry.get("attributes", {}).get("titles", [])
        if any(str(title.get("title", "")).strip().casefold() == TITLE.casefold() for title in titles):
            exact_datacite.append(entry.get("id"))

    machine_suffixes = {
        ".csv", ".tsv", ".txt", ".xls", ".xlsx", ".mat", ".nc", ".zip", ".json"
    }
    machine_files = [
        row["name"]
        for row in originals
        if Path(str(row["name"])).suffix.lower() in machine_suffixes
    ]
    nearest = nearest_tree_groups()
    with (OUT / "nearest_project_tree_evidence_groups.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nearest[0]))
        writer.writeheader()
        writer.writerows(nearest)

    nearest_row = nearest[0]
    nearest_is_nyinjt = (
        nearest_row["public_identity_id"] == "Asia-NYINJT::Asia_223"
        or nearest_row["metadata_identities"] == "Asia-NYINJT::Asia_223"
    )
    decision = (
        "advance_machine_series_source_identity_audit"
        if machine_files and not nearest_is_nyinjt
        else "metadata_candidate_numeric_series_not_public_source_overlap_unresolved"
    )
    summary = {
        "round": 226,
        "candidate": {
            "title": TITLE,
            "doi": DOI,
            "target": "January-June mean temperature",
            "reconstruction_coverage": "1377-1998",
            "calibration_coverage": "1961-1998",
            "reported_explained_variance_percent": 35.0,
            "species": "Cupressus gigantea",
            "sample_count": "46 cores from 28 living trees",
            "site_latitude": SITE_LAT,
            "site_longitude": SITE_LON,
        },
        "official_repository": {
            "item_id": ITEM_ID,
            "original_count": len(originals),
            "original_files": originals,
            "machine_readable_original_files": machine_files,
            "pdf_size_bytes": pdf_path.stat().st_size,
            "pdf_sha256": sha256(pdf_path),
        },
        "repository_crosscheck": {
            "crossref_relation_count": len(
                crossref.get("message", {}).get("relation", {})
            ),
            "datacite_exact_title_ids": exact_datacite,
        },
        "identity_preflight": {
            "nearest_project_group": nearest_row,
            "nearest_is_nyinjt": nearest_is_nyinjt,
            "classification": "source_overlap_unresolved_requires_sequence_or_sample_crosswalk",
        },
        "pdf_text_extracted": False,
        "figure_digitized": False,
        "annual_values_read": False,
        "locked_period_values_read": False,
        "jionglacuo_accessed": False,
        "decision": decision,
    }
    write_json(OUT / "summary.json", summary)
    write_json(
        OUT / "source_manifest.json",
        {
            "official_pdf": {
                "path": str(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
                "sha256": sha256(pdf_path),
            },
            "ledger": {"path": str(LEDGER), "sha256": sha256(LEDGER)},
            "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        },
    )
    (OUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 226 Yang 2010 southeast-Tibet 622-year metadata recovery",
                "",
                f"- Official FAU ORIGINAL files: `{len(originals)}`; machine-readable annual files: `{len(machine_files)}`.",
                f"- Official PDF SHA256: `{sha256(pdf_path)}`.",
                f"- Nearest project group: `{nearest_row['evidence_group_id']}` / `{nearest_row['public_identity_id'] or nearest_row['metadata_identities']}` at `{nearest_row['distance_km']}` km.",
                "- The official deposit is PDF-only, and the nearest project sequence is NYINJT; source independence is unresolved without a sequence or sample crosswalk.",
                f"- Decision: `{decision}`.",
                "- No PDF text, figure, proxy value, reconstruction value, locked-period value, or Jionglacuo value was opened.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
