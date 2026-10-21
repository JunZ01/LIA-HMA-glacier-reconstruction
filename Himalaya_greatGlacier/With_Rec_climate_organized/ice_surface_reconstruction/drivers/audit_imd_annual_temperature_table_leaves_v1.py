from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v1"
RAW = OUT / "raw_api"

TITLE_QUERY = '"MEAN AND EXTREME AIR-TEMPERATURES"'
YEARS = tuple(range(1875, 1891))
STATIONS = {
    "Darjeeling": ("Darjeeling",),
    "Simla": ("Simla", "Shimla"),
    "Leh": ("Leh",),
    "Murree": ("Murree",),
    "Chakrata": ("Chakrata",),
    "Ranikhet": ("Ranikhet",),
}


@dataclass(frozen=True)
class Item:
    identifier: str
    report_years: tuple[int, ...]
    note: str = ""


ITEMS = (
    Item("india.history.resource.108631", (1875,)),
    Item("india.history.resource.108632", (1877, 1878), "bound volume"),
    Item("india.history.resource.117788", (1877,), "duplicate scan"),
    Item("india.history.resource.108633", (1879,)),
    Item("india.history.resource.108634", (1880,)),
    Item("india.history.resource.108635", (1881,)),
    Item("india.history.resource.108636", (1883,)),
    Item("india.history.resource.108637", (1884,)),
    Item("india.history.resource.108638", (1885,)),
    Item("india.history.resource.109237", (1886,)),
    Item("india.history.resource.109238", (1887,)),
    Item("india.history.resource.109240", (1888,)),
    Item("india.history.resource.109241", (1890,)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(session: requests.Session, url: str) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(1, 8):
        try:
            response = session.get(url, timeout=(30, 240))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError(f"Expected object from {url}")
            return payload
        except Exception as exc:
            error = exc
            time.sleep(min(60, 3 * attempt))
    raise RuntimeError(f"API request failed after retries: {url}") from error


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_leaves(payload: dict[str, Any]) -> dict[int, list[str]]:
    leaves: dict[int, list[str]] = {}
    for match in payload.get("matches", []):
        text = str(match.get("text", ""))
        for paragraph in match.get("par", []):
            if "page" not in paragraph:
                continue
            leaf = int(paragraph["page"])
            leaves.setdefault(leaf, []).append(text)
    return leaves


def strip_markers(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("{{{", "").replace("}}}", "")).strip()


def extract_title(texts: list[str], expected_year: int) -> str:
    for raw in texts:
        text = strip_markers(raw)
        phrase = re.search(r"MEAN\s*(?:AND|[.,-]*AND)\s*EXTREME\s+AIR[-\s]TEMPERATURES", text, re.I)
        if not phrase:
            continue
        start = max(0, text.rfind("TABLE", 0, phrase.start()))
        year = re.search(rf"\b{expected_year}\b", text[phrase.end() : phrase.end() + 140])
        if year:
            end = phrase.end() + year.end()
            return text[start:end].strip(" ,.;")
        end = min(len(text), phrase.end() + 80)
        return text[start:end].strip(" ,.;")
    return ""


def title_leaves_for_year(
    all_title_leaves: dict[int, list[str]], report_year: int, item_years: tuple[int, ...]
) -> dict[int, list[str]]:
    explicit = {
        leaf: texts
        for leaf, texts in all_title_leaves.items()
        if any(re.search(rf"\b{report_year}\b", strip_markers(text)) for text in texts)
    }
    if explicit:
        return explicit
    if len(item_years) == 1:
        return all_title_leaves
    return {}


def inside_url(metadata: dict[str, Any], identifier: str, query: str) -> str:
    node = metadata.get("d2") or metadata.get("d1") or metadata.get("server")
    directory = metadata.get("dir")
    if not node or not directory:
        raise KeyError(f"Missing datanode locator for {identifier}")
    doc = identifier.rsplit(".", 1)[-1]
    return (
        f"https://{node}/fulltext/inside.php?"
        f"item_id={quote(identifier)}&doc={quote(doc)}&path={quote(str(directory), safe='')}"
        f"&q={quote(query)}&pre_tag=%7B%7B%7B&post_tag=%7D%7D%7D"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    RAW.mkdir()
    session = requests.Session()
    session.headers.update({"User-Agent": "HMA IMD table-leaf read-only audit/1.0"})

    candidates: list[dict[str, Any]] = []
    title_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []

    for item in ITEMS:
        metadata_url = f"https://archive.org/metadata/{item.identifier}"
        metadata = get_json(session, metadata_url)
        metadata_path = RAW / f"{item.identifier}_metadata.json"
        write_json(metadata_path, metadata)
        item_rows.append(
            {
                "item_id": item.identifier,
                "report_years": ";".join(map(str, item.report_years)),
                "note": item.note,
                "metadata_url": metadata_url,
                "datanode": metadata.get("d2") or metadata.get("d1") or metadata.get("server"),
                "directory": metadata.get("dir", ""),
            }
        )

        title_url = inside_url(metadata, item.identifier, TITLE_QUERY)
        title_payload = get_json(session, title_url)
        write_json(RAW / f"{item.identifier}_title.json", title_payload)
        all_title_leaves = result_leaves(title_payload)

        station_payloads: dict[str, dict[str, Any]] = {}
        station_leaves: dict[str, dict[int, list[str]]] = {}
        for station, aliases in STATIONS.items():
            merged: dict[int, list[str]] = {}
            for alias in aliases:
                query_url = inside_url(metadata, item.identifier, alias)
                payload = get_json(session, query_url)
                station_payloads[alias] = payload
                write_json(RAW / f"{item.identifier}_{alias.lower()}_station.json", payload)
                for leaf, texts in result_leaves(payload).items():
                    merged.setdefault(leaf, []).extend(texts)
                time.sleep(0.25)
            station_leaves[station] = merged

        for report_year in item.report_years:
            year_title_leaves = title_leaves_for_year(
                all_title_leaves, report_year, item.report_years
            )
            for leaf, texts in sorted(year_title_leaves.items()):
                title_rows.append(
                    {
                        "item_id": item.identifier,
                        "report_year": report_year,
                        "title_leaf": leaf,
                        "table_title": extract_title(texts, report_year),
                        "title_match_count_on_leaf": len(texts),
                    }
                )

            for station in STATIONS:
                station_set = set(station_leaves[station])
                title_set = set(year_title_leaves)
                exact = sorted(station_set & title_set)
                rows_for_pair: list[tuple[int, int, str, int]] = [
                    (leaf, leaf, "same_leaf", 0) for leaf in exact
                ]
                if not rows_for_pair and station_set and title_set:
                    nearest: list[tuple[int, int, str, int]] = []
                    minimum = min(abs(sleaf - tleaf) for sleaf in station_set for tleaf in title_set)
                    if minimum <= 2:
                        for sleaf in sorted(station_set):
                            for tleaf in sorted(title_set):
                                if abs(sleaf - tleaf) == minimum:
                                    nearest.append((sleaf, tleaf, "adjacent_leaf_candidate", minimum))
                        rows_for_pair = nearest

                if rows_for_pair:
                    for station_leaf, title_leaf, relation, distance in rows_for_pair:
                        candidates.append(
                            {
                                "item_id": item.identifier,
                                "report_year": report_year,
                                "station": station,
                                "candidate_leaf": station_leaf,
                                "title_leaf": title_leaf,
                                "leaf_relation": relation,
                                "leaf_distance": distance,
                                "table_title": extract_title(
                                    year_title_leaves.get(title_leaf, []), report_year
                                ),
                                "status": "candidate_not_value_verified",
                            }
                        )
                    status = "candidate_located"
                elif not year_title_leaves:
                    status = "year_specific_title_not_located"
                elif not station_set:
                    status = "station_not_found_in_item"
                else:
                    status = "station_and_title_found_but_not_within_2_leaves"
                coverage_rows.append(
                    {
                        "item_id": item.identifier,
                        "report_year": report_year,
                        "station": station,
                        "status": status,
                        "candidate_count": len(rows_for_pair),
                        "station_hit_leaf_count": len(station_set),
                        "year_title_leaf_count": len(title_set),
                    }
                )
        time.sleep(0.5)

    present_years = {year for item in ITEMS for year in item.report_years}
    for report_year in YEARS:
        if report_year in present_years:
            continue
        for station in STATIONS:
            coverage_rows.append(
                {
                    "item_id": "",
                    "report_year": report_year,
                    "station": station,
                    "status": "no_ia_item_located",
                    "candidate_count": 0,
                    "station_hit_leaf_count": 0,
                    "year_title_leaf_count": 0,
                }
            )

    write_csv(
        OUT / "station_table_leaf_candidates.csv",
        [
            "item_id",
            "report_year",
            "station",
            "candidate_leaf",
            "title_leaf",
            "leaf_relation",
            "leaf_distance",
            "table_title",
            "status",
        ],
        candidates,
    )
    write_csv(
        OUT / "table_title_leaf_hits.csv",
        ["item_id", "report_year", "title_leaf", "table_title", "title_match_count_on_leaf"],
        title_rows,
    )
    write_csv(
        OUT / "station_year_coverage.csv",
        [
            "item_id",
            "report_year",
            "station",
            "status",
            "candidate_count",
            "station_hit_leaf_count",
            "year_title_leaf_count",
        ],
        coverage_rows,
    )
    write_csv(
        OUT / "item_inventory.csv",
        ["item_id", "report_years", "note", "metadata_url", "datanode", "directory"],
        item_rows,
    )

    status_counts: dict[str, int] = {}
    for row in coverage_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "experiment": OUT.name,
        "scope_years": [YEARS[0], YEARS[-1]],
        "sealed_years_accessed": False,
        "stations": list(STATIONS),
        "ia_items": len(ITEMS),
        "report_years_with_items": sorted(present_years),
        "report_years_without_located_ia_items": sorted(set(YEARS) - present_years),
        "station_year_pairs": len(coverage_rows),
        "candidate_rows": len(candidates),
        "status_counts": status_counts,
        "interpretation": (
            "Leaf candidates are page locators only. OCR temperature values were not "
            "transcribed, validated, or admitted as scientific observations."
        ),
    }
    write_json(OUT / "summary.json", summary)

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = [
        {
            "path": str(path.relative_to(OUT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    write_json(OUT / "source_manifest.json", manifest)

    report = [
        "# IMD annual temperature-table leaf audit, 1875-1890",
        "",
        "This is a read-only Internet Archive fulltext-inside page-location audit.",
        "",
        f"- IA items queried: {len(ITEMS)}",
        f"- Stations: {', '.join(STATIONS)}",
        f"- Report years with an IA item: {', '.join(map(str, sorted(present_years)))}",
        f"- No IA item located: {', '.join(map(str, sorted(set(YEARS) - present_years)))}",
        f"- Candidate rows: {len(candidates)}",
        f"- Pair-level statuses: {json.dumps(status_counts, sort_keys=True)}",
        "- Access to 1921-1939: none.",
        "- OCR numbers are not final values: no temperature value was transcribed or admitted.",
        "",
        "## Files",
        "",
        "- `station_table_leaf_candidates.csv`: requested item/year/station leaf candidates and table titles.",
        "- `station_year_coverage.csv`: complete pair-level coverage and unresolved cases.",
        "- `table_title_leaf_hits.csv`: every year-filtered title leaf returned by the exact phrase query.",
        "- `raw_api/`: unmodified metadata and fulltext JSON responses.",
        "",
        "## Interpretation boundary",
        "",
        "A same-leaf intersection means only that the station term and the exact table-title phrase occur on the same OCR leaf. An adjacent-leaf result is explicitly downgraded. Page-image inspection and independent transcription are still required before any numeric observation can be used.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
