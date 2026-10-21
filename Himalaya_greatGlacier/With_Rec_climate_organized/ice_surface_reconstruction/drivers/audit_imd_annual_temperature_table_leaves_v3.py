from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_imd_annual_temperature_table_leaves_v1 as audit  # noqa: E402


SOURCE = audit.LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v2"
OUT = audit.LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v3"
RAW = OUT / "raw_api"
ITEM_CONTENT_YEARS = {
    "india.history.resource.108631": (1875,),
    "india.history.resource.108632": (1877,),
    "india.history.resource.117788": (1878,),
    "india.history.resource.108633": (1879,),
    "india.history.resource.108634": (1880,),
    "india.history.resource.108635": (1881,),
    "india.history.resource.108636": (1883,),
    "india.history.resource.108637": (1884,),
    "india.history.resource.108638": (1885,),
    "india.history.resource.109237": (1886,),
    "india.history.resource.109238": (1887,),
    "india.history.resource.109240": (1888,),
    "india.history.resource.109241": (1890,),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main_title_cluster(leaves: dict[int, list[str]]) -> tuple[dict[int, list[str]], list[int]]:
    ordered = sorted(leaves)
    if not ordered:
        return {}, []
    clusters: list[list[int]] = [[ordered[0]]]
    for leaf in ordered[1:]:
        if leaf - clusters[-1][-1] <= 3:
            clusters[-1].append(leaf)
        else:
            clusters.append([leaf])
    chosen = max(clusters, key=lambda values: (len(values), max(values)))
    discarded = sorted(set(ordered) - set(chosen))
    return {leaf: leaves[leaf] for leaf in chosen}, discarded


def representative_title(
    title_leaves: dict[int, list[str]], year: int
) -> tuple[str, int | None]:
    choices: list[tuple[tuple[int, int, int, int], str, int]] = []
    for leaf, texts in title_leaves.items():
        title = audit.extract_title(texts, year)
        if not title:
            continue
        upper = title.upper()
        score = (
            int(str(year) in title),
            int("SHADE" in upper),
            int(upper.startswith("TABLE")),
            -abs(len(title) - 70),
        )
        choices.append((score, title, leaf))
    if not choices:
        return f"MEAN AND EXTREME AIR-TEMPERATURES [OCR heading, {year}]", None
    _, title, leaf = max(choices, key=lambda row: row[0])
    return title, leaf


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SOURCE / "raw_api", RAW)

    detail: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    title_clusters: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []

    for item_id, years in ITEM_CONTENT_YEARS.items():
        metadata = read_json(RAW / f"{item_id}_metadata.json")
        title_payload = read_json(RAW / f"{item_id}_title.json")
        metadata_title = str(metadata.get("metadata", {}).get("title", ""))
        catalog = sorted({int(x) for x in re.findall(r"\b18[7-9][0-9]\b", metadata_title)})
        title_tokens = sorted(
            {
                int(x)
                for match in title_payload.get("matches", [])
                for x in re.findall(
                    r"\b18[7-9][0-9]\b", audit.strip_markers(str(match.get("text", "")))
                )
            }
        )
        conflict = item_id == "india.history.resource.117788"
        identities.append(
            {
                "item_id": item_id,
                "ia_catalog_title": metadata_title,
                "catalog_years": ";".join(map(str, catalog)),
                "fulltext_title_year_tokens": ";".join(map(str, title_tokens)),
                "assigned_content_years": ";".join(map(str, years)),
                "catalog_content_conflict": str(conflict).lower(),
                "assignment_basis": "repeated_temperature_table_heading_year",
            }
        )

        all_titles = audit.result_leaves(title_payload)
        station_pages: dict[str, set[int]] = {}
        for station, aliases in audit.STATIONS.items():
            pages: set[int] = set()
            for alias in aliases:
                station_payload = read_json(RAW / f"{item_id}_{alias.lower()}_station.json")
                pages.update(audit.result_leaves(station_payload))
            station_pages[station] = pages

        for year in years:
            year_titles = audit.title_leaves_for_year(all_titles, year, years)
            main_titles, discarded = main_title_cluster(year_titles)
            main_set = set(main_titles)
            default_title, default_title_leaf = representative_title(main_titles, year)
            title_clusters.append(
                {
                    "item_id": item_id,
                    "report_year": year,
                    "main_title_leaves": ";".join(map(str, sorted(main_set))),
                    "discarded_isolated_title_leaves": ";".join(map(str, discarded)),
                    "representative_table_title": default_title,
                    "representative_title_leaf": "" if default_title_leaf is None else default_title_leaf,
                }
            )

            for station in audit.STATIONS:
                pages = station_pages[station]
                exact = sorted(pages & main_set)
                candidates: list[tuple[int, int, str, int]] = [
                    (leaf, leaf, "same_leaf", 0) for leaf in exact
                ]
                if not candidates and pages and main_set:
                    minimum = min(abs(page - title_leaf) for page in pages for title_leaf in main_set)
                    if minimum <= 2:
                        candidates = [
                            (page, title_leaf, "adjacent_leaf_candidate", minimum)
                            for page in sorted(pages)
                            for title_leaf in sorted(main_set)
                            if abs(page - title_leaf) == minimum
                        ]

                candidate_titles: list[str] = []
                for page, title_leaf, relation, distance in candidates:
                    title = audit.extract_title(main_titles.get(title_leaf, []), year)
                    title_source_leaf: int | str = title_leaf
                    if not title:
                        title = default_title
                        title_source_leaf = "" if default_title_leaf is None else default_title_leaf
                    candidate_titles.append(title)
                    detail.append(
                        {
                            "item_id": item_id,
                            "report_year": year,
                            "station": station,
                            "candidate_leaf": page,
                            "title_leaf": title_leaf,
                            "leaf_relation": relation,
                            "leaf_distance": distance,
                            "table_title": title,
                            "table_title_source_leaf": title_source_leaf,
                            "status": "candidate_not_value_verified",
                        }
                    )

                if candidates:
                    status = "candidate_located"
                elif not main_set:
                    status = "main_table_title_run_not_located"
                elif not pages:
                    status = "station_not_found_in_item"
                else:
                    status = "station_not_located_on_or_near_main_table_run"
                matrix.append(
                    {
                        "item_id": item_id,
                        "report_year": year,
                        "station": station,
                        "candidate_leaves": ";".join(map(str, sorted({row[0] for row in candidates}))),
                        "title_leaves": ";".join(map(str, sorted({row[1] for row in candidates}))),
                        "leaf_relations": ";".join(sorted({row[2] for row in candidates})),
                        "table_title": candidate_titles[0] if candidate_titles else default_title,
                        "status": status,
                        "numeric_value_status": "not_transcribed_not_verified",
                    }
                )

    present = {year for years in ITEM_CONTENT_YEARS.values() for year in years}
    for year in audit.YEARS:
        if year in present:
            continue
        for station in audit.STATIONS:
            matrix.append(
                {
                    "item_id": "",
                    "report_year": year,
                    "station": station,
                    "candidate_leaves": "",
                    "title_leaves": "",
                    "leaf_relations": "",
                    "table_title": "",
                    "status": "no_ia_item_located",
                    "numeric_value_status": "not_transcribed_not_verified",
                }
            )

    detail_fields = [
        "item_id", "report_year", "station", "candidate_leaf", "title_leaf",
        "leaf_relation", "leaf_distance", "table_title", "table_title_source_leaf", "status",
    ]
    matrix_fields = [
        "item_id", "report_year", "station", "candidate_leaves", "title_leaves",
        "leaf_relations", "table_title", "status", "numeric_value_status",
    ]
    write_csv(OUT / "station_table_leaf_candidates.csv", detail_fields, detail)
    write_csv(OUT / "item_year_station_leaf_audit.csv", matrix_fields, matrix)
    write_csv(
        OUT / "table_title_cluster_audit.csv",
        [
            "item_id", "report_year", "main_title_leaves",
            "discarded_isolated_title_leaves", "representative_table_title",
            "representative_title_leaf",
        ],
        title_clusters,
    )
    write_csv(
        OUT / "item_year_identity_audit.csv",
        [
            "item_id", "ia_catalog_title", "catalog_years", "fulltext_title_year_tokens",
            "assigned_content_years", "catalog_content_conflict", "assignment_basis",
        ],
        identities,
    )

    counts: dict[str, int] = {}
    for row in matrix:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "experiment": OUT.name,
        "supersedes": SOURCE.name,
        "scope_years": [1875, 1890],
        "sealed_years_accessed": False,
        "ia_items": len(ITEM_CONTENT_YEARS),
        "report_years_with_items": sorted(present),
        "report_years_without_located_ia_items": sorted(set(audit.YEARS) - present),
        "requested_pair_rows": len(matrix),
        "candidate_pair_rows": counts.get("candidate_located", 0),
        "candidate_detail_rows": len(detail),
        "status_counts": counts,
        "discarded_isolated_title_leaf_hits": sum(
            len(row["discarded_isolated_title_leaves"].split(";"))
            for row in title_clusters
            if row["discarded_isolated_title_leaves"]
        ),
        "catalog_content_conflicts": ["india.history.resource.117788"],
        "ocr_numeric_values_admitted": False,
    }
    write_json(OUT / "summary.json", summary)
    report = [
        "# IMD mean/extreme air-temperature table leaf audit, 1875-1890 (v3)",
        "",
        "Read-only audit of Internet Archive fulltext-inside metadata and leaf matches.",
        "",
        f"- Requested item/year/station rows: {len(matrix)}",
        f"- Candidate pairs: {counts.get('candidate_located', 0)}",
        f"- Candidate detail rows: {len(detail)}",
        f"- Status counts: {json.dumps(counts, sort_keys=True)}",
        f"- Missing IA report years: {', '.join(map(str, sorted(set(audit.YEARS) - present)))}",
        "- `117788` is assigned to 1878 from repeated table headings; its IA catalog title says 1877 and is flagged as conflicting.",
        "- Isolated contents/index title hits were excluded; only the largest dense run of repeated table headings was retained.",
        "- 1921-1939 accessed: no.",
        "- OCR temperature values admitted: no. Every candidate remains a page locator pending image inspection and independent transcription.",
        "",
        "The requested complete table is `item_year_station_leaf_audit.csv`; multi-hit details are in `station_table_leaf_candidates.csv`.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "source_manifest.json"):
        manifest.append(
            {
                "path": str(path.relative_to(OUT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(OUT / "source_manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
