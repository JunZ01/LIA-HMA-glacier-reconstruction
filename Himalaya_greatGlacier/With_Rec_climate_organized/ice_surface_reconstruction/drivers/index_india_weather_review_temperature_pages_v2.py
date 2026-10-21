from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import index_india_weather_review_temperature_pages_v1 as v1


SECTION_STARTS = {
    "india.history.resource.108898": (
        (4, 1899),
        (172, 1900),
        (340, 1901),
        (504, 1902),
    ),
    "india.history.resource.108906": (
        (4, 1905),
        (132, 1906),
        (272, 1907),
        (396, 1908),
        (534, 1912),
        (656, 1914),
        (772, 1916),
        (868, 1920),
    ),
    "in.ernet.dli.2015.35323": ((3, 1906),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def year_for_leaf(identifier: str, leaf: int) -> int:
    starts = SECTION_STARTS[identifier]
    eligible = [year for start, year in starts if leaf >= start]
    return eligible[-1] if eligible else starts[0][1]


def assign_section_years(frame: pd.DataFrame, identifier: str) -> pd.DataFrame:
    result = frame.copy()
    result["assigned_year"] = [year_for_leaf(identifier, int(leaf)) for leaf in result["leaf"]]
    boundary_year = dict(SECTION_STARTS[identifier])
    result["title_year_marker"] = [boundary_year.get(int(leaf), "") for leaf in result["leaf"]]
    return result


def boundary_audit(raw_root: Path, spec: dict) -> pd.DataFrame:
    xml_path = raw_root / spec["xml"]
    metadata = v1.read_leaf_metadata(raw_root / spec["scandata"])
    objects = ET.parse(xml_path).getroot().findall(".//OBJECT")
    rows = []
    for leaf, expected_year in SECTION_STARTS[spec["identifier"]]:
        text = v1.normalize(" ".join(v1.page_words(objects[leaf])))
        years = sorted(set(re.findall(r"\b(?:18|19)\d{2}\b", text)))
        rows.append(
            {
                "identifier": spec["identifier"],
                "section_start_leaf": leaf,
                "expected_year": expected_year,
                "printed_page": metadata.get(leaf, {}).get("printed_page", ""),
                "page_type": metadata.get(leaf, {}).get("page_type", ""),
                "ocr_year_tokens": ";".join(years),
                "weather_review_phrase": bool(re.search(r"WEATHER\s+REVIEW", text, re.I)),
                "annual_summary_phrase": bool(re.search(r"ANNUAL\s+SUMMARY", text, re.I)),
                "anchor_text": text[:900],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise RuntimeError("Versioned output already completed")

    declared_years = sorted({year for starts in SECTION_STARTS.values() for _, year in starts})
    if any(v1.LOCKED_START <= year <= v1.LOCKED_END for year in declared_years):
        raise RuntimeError("A section boundary crosses the locked interval")

    inputs = [args.raw_root / spec[key] for spec in v1.SOURCE_SPECS for key in ("xml", "scandata")]
    before = {str(path.resolve()): v1.sha256(path) for path in inputs}
    page_frames = []
    candidate_frames = []
    boundary_frames = []
    for spec in v1.SOURCE_SPECS:
        pages, candidates = v1.parse_source(args.raw_root, spec)
        page_frames.append(assign_section_years(pages, spec["identifier"]))
        candidate_frames.append(assign_section_years(candidates, spec["identifier"]))
        boundary_frames.append(boundary_audit(args.raw_root, spec))

    pages = pd.concat(page_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True).sort_values(
        ["candidate_score", "identifier", "assigned_year", "leaf"],
        ascending=[False, True, True, True],
    )
    boundaries = pd.concat(boundary_frames, ignore_index=True)

    station_rows = []
    for row in candidates.itertuples(index=False):
        for station in str(row.station_hits).split(";"):
            if station:
                station_rows.append(
                    {
                        "identifier": row.identifier,
                        "assigned_year": int(row.assigned_year),
                        "station": station,
                        "leaf": int(row.leaf),
                        "printed_page": row.printed_page,
                        "candidate_score": int(row.candidate_score),
                        "temperature_terms": row.temperature_terms,
                        "archive_page_url": row.archive_page_url,
                    }
                )
    station_candidates = pd.DataFrame(station_rows).drop_duplicates()
    coverage = (
        station_candidates.groupby(["identifier", "assigned_year", "station"], as_index=False)
        .agg(candidate_page_count=("leaf", "nunique"), best_score=("candidate_score", "max"))
        .sort_values(["assigned_year", "station", "identifier"])
    )

    pages.to_csv(args.output / "page_year_index.csv", index=False)
    candidates.to_csv(args.output / "temperature_station_page_candidates.csv", index=False)
    station_candidates.to_csv(args.output / "station_page_candidates_long.csv", index=False)
    coverage.to_csv(args.output / "station_year_candidate_coverage.csv", index=False)
    boundaries.to_csv(args.output / "section_boundary_audit.csv", index=False)

    after = {str(path.resolve()): v1.sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Source state changed")
    duplicate_1906 = coverage.loc[coverage["assigned_year"].eq(1906), "identifier"].nunique() >= 2
    expected_pairs = {(identifier, year) for identifier, starts in SECTION_STARTS.items() for _, year in starts}
    observed_pairs = set(zip(pages["identifier"], pages["assigned_year"]))
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise RuntimeError(f"Missing indexed sections: {missing_pairs}")

    manifest = {
        "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
        "section_starts": SECTION_STARTS,
        "source_hashes_unchanged": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "experiment": "temperature_india_weather_review_1899_1920_ocr_recovery_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_corrected_page_index",
        "supersedes": "temperature_india_weather_review_1899_1920_ocr_recovery_v1",
        "declared_years": declared_years,
        "source_volume_count": len(v1.SOURCE_SPECS),
        "section_count": int(len(boundaries)),
        "page_count": int(len(pages)),
        "temperature_station_candidate_page_count": int(len(candidates)),
        "station_year_coverage_rows": int(len(coverage)),
        "independent_ocr_copy_available_for_1906": bool(duplicate_1906),
        "year_assignment_method": "explicit section starts verified from title or annual-summary leaves",
        "scientific_role": "same direct-observation family as GHCN-M; primary-table processing audit, not an independent climate vote",
        "next_action": "visually verify the best annual temperature-table leaves and extract allowed-year station values, starting with duplicate 1906 scans",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# India Weather Review OCR recovery v2

The v1 year detector was invalid because it treated years mentioned inside annual text as section titles. This version assigns years only from verified volume-section starts and preserves the OCR text for every boundary in `section_boundary_audit.csv`.

- Corrected sections: **{len(boundaries)}**
- Indexed pages: **{len(pages)}**
- Station and temperature candidate pages: **{len(candidates)}**
- Station-year coverage rows: **{len(coverage)}**
- Two independently scanned 1906 copies available: **{'yes' if duplicate_1906 else 'no'}**

The official printed reviews are in the same observation family as GHCN-M and therefore are not counted as independent climate evidence. They can directly test transcription and processing choices for allowed years.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 or another prohibited source accessed: **no**
- Original or production files modified: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
