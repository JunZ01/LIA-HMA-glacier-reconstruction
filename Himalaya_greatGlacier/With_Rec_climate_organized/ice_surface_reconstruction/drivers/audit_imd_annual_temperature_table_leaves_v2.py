from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V1_DRIVER = HERE / "audit_imd_annual_temperature_table_leaves_v1.py"


def load_v1():
    spec = importlib.util.spec_from_file_location("imd_leaf_audit_v1", V1_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V1_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_v1()
SOURCE = audit.OUT
OUT = audit.LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v2"
RAW = OUT / "raw_api"

# Content years are assigned from repeated table headings, not catalog labels.
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def fulltext_years(payload: dict[str, Any]) -> list[int]:
    years: set[int] = set()
    for match in payload.get("matches", []):
        text = audit.strip_markers(str(match.get("text", "")))
        years.update(int(value) for value in re.findall(r"\b18[7-9][0-9]\b", text))
    return sorted(years)


def catalog_years(title: str) -> list[int]:
    return sorted({int(value) for value in re.findall(r"\b18[7-9][0-9]\b", title)})


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SOURCE / "raw_api", RAW)

    candidates: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    title_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []

    for item_id, years in ITEM_CONTENT_YEARS.items():
        metadata = read_json(RAW / f"{item_id}_metadata.json")
        title_payload = read_json(RAW / f"{item_id}_title.json")
        meta_title = str(metadata.get("metadata", {}).get("title", ""))
        title_year_values = fulltext_years(title_payload)
        assigned = list(years)
        conflict = item_id == "india.history.resource.117788"
        identities.append(
            {
                "item_id": item_id,
                "ia_catalog_title": meta_title,
                "catalog_years": ";".join(map(str, catalog_years(meta_title))),
                "fulltext_title_year_tokens": ";".join(map(str, title_year_values)),
                "assigned_content_years": ";".join(map(str, assigned)),
                "assignment_basis": "repeated_MEAN_AND_EXTREME_AIR_TEMPERATURES_headings",
                "catalog_content_conflict": str(conflict).lower(),
                "note": (
                    "catalog says 1877, but repeated temperature-table headings say 1878"
                    if conflict
                    else ""
                ),
            }
        )

        all_titles = audit.result_leaves(title_payload)
        station_leaves: dict[str, dict[int, list[str]]] = {}
        for station, aliases in audit.STATIONS.items():
            merged: dict[int, list[str]] = {}
            for alias in aliases:
                payload = read_json(RAW / f"{item_id}_{alias.lower()}_station.json")
                for leaf, texts in audit.result_leaves(payload).items():
                    merged.setdefault(leaf, []).extend(texts)
            station_leaves[station] = merged

        for report_year in years:
            year_titles = audit.title_leaves_for_year(all_titles, report_year, years)
            title_set = set(year_titles)
            for leaf, texts in sorted(year_titles.items()):
                title_rows.append(
                    {
                        "item_id": item_id,
                        "report_year": report_year,
                        "title_leaf": leaf,
                        "table_title": audit.extract_title(texts, report_year),
                        "title_match_count_on_leaf": len(texts),
                    }
                )

            for station in audit.STATIONS:
                station_set = set(station_leaves[station])
                exact = sorted(station_set & title_set)
                pair_candidates = [(leaf, leaf, "same_leaf", 0) for leaf in exact]
                if not pair_candidates and station_set and title_set:
                    minimum = min(abs(s - t) for s in station_set for t in title_set)
                    if minimum <= 2:
                        pair_candidates = [
                            (s, t, "adjacent_leaf_candidate", minimum)
                            for s in sorted(station_set)
                            for t in sorted(title_set)
                            if abs(s - t) == minimum
                        ]
                for station_leaf, title_leaf, relation, distance in pair_candidates:
                    candidates.append(
                        {
                            "item_id": item_id,
                            "report_year": report_year,
                            "station": station,
                            "candidate_leaf": station_leaf,
                            "title_leaf": title_leaf,
                            "leaf_relation": relation,
                            "leaf_distance": distance,
                            "table_title": audit.extract_title(
                                year_titles.get(title_leaf, []), report_year
                            ),
                            "status": "candidate_not_value_verified",
                        }
                    )
                if pair_candidates:
                    status = "candidate_located"
                elif not title_set:
                    status = "year_specific_title_not_located"
                elif not station_set:
                    status = "station_not_found_in_item"
                else:
                    status = "station_and_title_found_but_not_within_2_leaves"
                coverage.append(
                    {
                        "item_id": item_id,
                        "report_year": report_year,
                        "station": station,
                        "status": status,
                        "candidate_count": len(pair_candidates),
                        "station_hit_leaf_count": len(station_set),
                        "year_title_leaf_count": len(title_set),
                    }
                )

    present_years = {year for years in ITEM_CONTENT_YEARS.values() for year in years}
    for report_year in audit.YEARS:
        if report_year in present_years:
            continue
        for station in audit.STATIONS:
            coverage.append(
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
        coverage,
    )
    write_csv(
        OUT / "table_title_leaf_hits.csv",
        ["item_id", "report_year", "title_leaf", "table_title", "title_match_count_on_leaf"],
        title_rows,
    )
    write_csv(
        OUT / "item_year_identity_audit.csv",
        [
            "item_id",
            "ia_catalog_title",
            "catalog_years",
            "fulltext_title_year_tokens",
            "assigned_content_years",
            "assignment_basis",
            "catalog_content_conflict",
            "note",
        ],
        identities,
    )

    counts: dict[str, int] = {}
    for row in coverage:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "experiment": OUT.name,
        "supersedes": SOURCE.name,
        "scope_years": [audit.YEARS[0], audit.YEARS[-1]],
        "sealed_years_accessed": False,
        "ia_items": len(ITEM_CONTENT_YEARS),
        "report_years_with_items": sorted(present_years),
        "report_years_without_located_ia_items": sorted(set(audit.YEARS) - present_years),
        "station_year_pairs": len(coverage),
        "candidate_rows": len(candidates),
        "status_counts": counts,
        "catalog_content_conflicts": ["india.history.resource.117788"],
        "interpretation": (
            "Leaf candidates are page locators only. OCR temperature values were not "
            "transcribed, validated, or admitted as scientific observations."
        ),
    }
    write_json(OUT / "summary.json", summary)

    report = [
        "# IMD annual temperature-table leaf audit, 1875-1890 (v2)",
        "",
        "This corrected read-only audit uses saved Internet Archive metadata and fulltext-inside responses.",
        "",
        f"- IA items audited: {len(ITEM_CONTENT_YEARS)}",
        f"- Report years with located content: {', '.join(map(str, sorted(present_years)))}",
        f"- No IA item located: {', '.join(map(str, sorted(set(audit.YEARS) - present_years)))}",
        f"- Candidate rows: {len(candidates)}",
        f"- Pair-level statuses: {json.dumps(counts, sort_keys=True)}",
        "- `117788` is assigned to content year 1878 because its repeated table headings say 1878, despite an IA catalog title saying 1877.",
        "- Access to 1921-1939: none.",
        "- No OCR temperature value was transcribed, checked, or admitted as a final observation.",
        "",
        "`station_table_leaf_candidates.csv` is the requested item/year/station locator table. Same-leaf intersections are stronger than explicitly labeled adjacent-leaf candidates. Page-image inspection remains mandatory before numeric transcription.",
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
