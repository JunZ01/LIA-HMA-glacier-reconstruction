from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOCKED_START = 1921
LOCKED_END = 1939
STATIONS = (
    "LEH",
    "SIMLA",
    "SHIMLA",
    "GILGIT",
    "SKARDU",
    "SRINAGAR",
    "MURREE",
    "DARJEELING",
    "RANIKHET",
    "CHAKRATA",
    "MUSSOORIE",
    "MUSSOOREE",
    "KATHMANDU",
)
SOURCE_SPECS = (
    {
        "identifier": "india.history.resource.108898",
        "xml": "108898_djvu.xml",
        "scandata": "108898_scandata.xml",
        "years": (1899, 1900, 1901, 1902),
        "archive_dir": "https://ia601508.us.archive.org/26/items/india.history.resource.108898",
    },
    {
        "identifier": "india.history.resource.108906",
        "xml": "108906_djvu.xml",
        "scandata": "108906_scandata.xml",
        "years": (1905, 1906, 1907, 1908, 1912, 1914, 1916, 1920),
        "archive_dir": "https://ia601406.us.archive.org/6/items/india.history.resource.108906",
    },
    {
        "identifier": "in.ernet.dli.2015.35323",
        "xml": "dli1906_djvu.xml",
        "scandata": "dli1906_scandata.xml",
        "years": (1906,),
        "archive_dir": "https://ia601501.us.archive.org/28/items/in.ernet.dli.2015.35323",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_leaf_metadata(scandata_path: Path) -> dict[int, dict[str, str]]:
    root = ET.parse(scandata_path).getroot()
    rows = {}
    for page in root.findall(".//page"):
        leaf = int(page.attrib["leafNum"])
        rows[leaf] = {
            "printed_page": page.findtext("pageNumber") or "",
            "page_type": page.findtext("pageType") or "",
        }
    return rows


def page_words(obj: ET.Element) -> list[str]:
    return [word.text or "" for word in obj.iter("WORD") if (word.text or "").strip()]


def title_year(text_upper: str, allowed_years: tuple[int, ...]) -> int | None:
    if "WEATHER" not in text_upper or "REVIEW" not in text_upper:
        return None
    first = text_upper[:2500]
    found = []
    for year in allowed_years:
        token = str(year)
        index = first.find(token)
        if index >= 0:
            found.append((index, year))
    if not found:
        return None
    return min(found)[1]


def snippets(text: str, terms: list[str], radius: int = 220) -> str:
    upper = text.upper()
    parts = []
    for term in terms[:4]:
        index = upper.find(term)
        if index >= 0:
            parts.append(text[max(0, index - radius): index + len(term) + radius])
    return " || ".join(normalize(part) for part in parts)[:1800]


def parse_source(raw_root: Path, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    xml_path = raw_root / spec["xml"]
    leaf_meta = read_leaf_metadata(raw_root / spec["scandata"])
    root = ET.parse(xml_path).getroot()
    allowed_years = tuple(spec["years"])
    current_year = allowed_years[0]
    next_year_index = 0
    page_rows = []
    candidate_rows = []
    for leaf, obj in enumerate(root.findall(".//OBJECT")):
        text = normalize(" ".join(page_words(obj)))
        upper = text.upper()
        marker = title_year(upper, allowed_years)
        if marker is not None:
            marker_index = allowed_years.index(marker)
            if marker_index >= next_year_index:
                current_year = marker
                next_year_index = marker_index

        hits = sorted({name for name in STATIONS if re.search(rf"\b{re.escape(name)}\b", upper)})
        temperature_terms = [
            term
            for term in (
                "TEMPERATURE OF AIR",
                "MEAN TEMPERATURE",
                "TEMPERATURE DEPARTURE",
                "MEAN MAXIMUM",
                "MEAN MINIMUM",
                "MONTHLY MEAN",
            )
            if term in upper
        ]
        annual_terms = [term for term in ("ANNUAL SUMMARY", "TABLE A", "TABLE B", "TABLE II", "YEAR") if term in upper]
        info = leaf_meta.get(leaf, {})
        page_file = next(
            (
                param.attrib.get("value", "")
                for param in obj.findall("PARAM")
                if param.attrib.get("name") == "PAGE"
            ),
            "",
        )
        row = {
            "identifier": spec["identifier"],
            "leaf": leaf,
            "page_file": page_file,
            "printed_page": info.get("printed_page", ""),
            "page_type": info.get("page_type", ""),
            "assigned_year": current_year,
            "title_year_marker": marker if marker is not None else "",
            "station_hits": ";".join(hits),
            "temperature_terms": ";".join(temperature_terms),
            "annual_terms": ";".join(annual_terms),
            "word_count": len(text.split()),
            "archive_page_url": f"https://archive.org/details/{spec['identifier']}/page/n{leaf}/mode/1up",
        }
        page_rows.append(row)
        if hits and temperature_terms:
            score = 2 * len(hits) + 2 * len(temperature_terms) + len(annual_terms)
            candidate_rows.append(
                {
                    **row,
                    "candidate_score": score,
                    "text_snippet": snippets(text, hits + temperature_terms),
                }
            )
    return pd.DataFrame(page_rows), pd.DataFrame(candidate_rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise RuntimeError("Versioned output already completed")
    declared_years = sorted({year for spec in SOURCE_SPECS for year in spec["years"]})
    if any(LOCKED_START <= year <= LOCKED_END for year in declared_years):
        raise RuntimeError("A source specification crosses the locked interval")

    inputs = [args.raw_root / spec[key] for spec in SOURCE_SPECS for key in ("xml", "scandata")]
    before = {str(path.resolve()): sha256(path) for path in inputs}
    page_frames = []
    candidate_frames = []
    for spec in SOURCE_SPECS:
        pages, candidates = parse_source(args.raw_root, spec)
        page_frames.append(pages)
        candidate_frames.append(candidates)
    pages = pd.concat(page_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates = candidates.sort_values(
        ["candidate_score", "identifier", "assigned_year", "leaf"], ascending=[False, True, True, True]
    )

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

    after = {str(path.resolve()): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Source state changed")
    manifest = {
        "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
        "source_specs": SOURCE_SPECS,
        "source_hashes_unchanged": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    duplicate_1906 = coverage.loc[coverage["assigned_year"].eq(1906), "identifier"].nunique() >= 2
    summary = {
        "experiment": "temperature_india_weather_review_1899_1920_ocr_recovery_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_page_index",
        "declared_years": declared_years,
        "source_volume_count": len(SOURCE_SPECS),
        "page_count": int(len(pages)),
        "temperature_station_candidate_page_count": int(len(candidates)),
        "station_year_coverage_rows": int(len(coverage)),
        "independent_ocr_copy_available_for_1906": bool(duplicate_1906),
        "scientific_role": "same direct-observation family; primary-table processing audit and possible station gap recovery, not an independent climate vote",
        "next_action": "visually adjudicate the highest-scoring annual temperature-table leaves, then extract allowed-year station values with duplicate-1906 OCR agreement",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# India Weather Review OCR recovery

Three official historical volumes were recovered from Internet Archive OCR and indexed page by page. Their declared years are {', '.join(map(str, declared_years))}; all are outside the locked interval.

- Indexed pages: **{len(pages)}**
- Pages containing both HMA station names and temperature-table language: **{len(candidates)}**
- Station-year-page coverage rows: **{len(coverage)}**
- Independent OCR copy for 1906: **{'yes' if duplicate_1906 else 'no'}**

These records belong to the same physical observation family as GHCN-M. They cannot count as an independent climate vote. Their value is more specific: they provide official printed-table anchors that can test transcription, observation-time handling, and homogenization adjustments over many more years than the four previously audited Shimla pages.

The next action is limited to the highest-scoring annual temperature-table leaves. Values will be admitted only after visual page verification and, for 1906, agreement between the two independently scanned OCR copies.

- Locked 1921-1939 values accessed: **no**
- Prohibited source accessed: **no**
- Original or production files modified: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
