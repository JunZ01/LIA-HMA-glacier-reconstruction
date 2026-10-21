from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "temperature_historical_imd_report_ocr_archive_v1"
DRIVER = Path(__file__).resolve()

ITEMS = [
    ("india.history.resource.108631", "108631", "1875", "1875"),
    ("india.history.resource.108632", "108632", "1877-1878", "1877-1878"),
    ("india.history.resource.117788", "117788", "1877", "1877_duplicate_scan"),
    ("india.history.resource.108633", "108633", "1879", "1879"),
    ("india.history.resource.108634", "108634", "1880", "1880"),
    ("india.history.resource.108635", "108635", "1881", "1881"),
    ("india.history.resource.108636", "108636", "1883", "1883"),
    ("india.history.resource.108637", "108637", "1884", "1884"),
    ("india.history.resource.108638", "108638", "1885", "1885"),
    ("india.history.resource.109237", "109237", "1886", "1886"),
    ("india.history.resource.109238", "109238", "1887", "1887"),
    ("india.history.resource.109240", "109240", "1888", "1888"),
    ("india.history.resource.109241", "109241", "1890", "1890"),
]

STATION_PATTERNS = {
    "darjeeling": r"darj(?:e|i){1,2}ling",
    "simla": r"\bsimla\b",
    "leh": r"\bleh\b",
    "srinagar": r"\bsrinagar\b",
    "mukteshwar": r"mukte(?:s|sh)war",
    "chakrata": r"chakrata",
    "mussoorie": r"musso(?:o|u)r(?:ee|ie)",
    "ranikhet": r"ranikhet",
    "murree": r"\bmurree\b",
    "peshawar": r"peshawar",
    "quetta": r"quetta",
    "lahore": r"lahore",
    "rawalpindi": r"rawalpindi",
    "kathmandu": r"kathmandu",
    "shillong": r"shillong",
    "gauhati": r"gauhati",
}

TEMPERATURE_PATTERNS = {
    "mean_and_extreme_air_temperature": r"mean\s+and\s+extreme\s+air\s*[- ]?temperatures",
    "average_monthly_mean_temperature": r"average\s+monthly\s+mean\s+temperatures",
    "monthly_mean_temperature_comparison": r"comparison\s+of\s+monthly\s+mean\s+air\s*[- ]?temperatures",
    "air_temperature_section": r"air\s*[- ]?temperature",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 LIA historical climate audit"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def page_type(text: str) -> str:
    compact = normalized(text).lower()
    matches = [
        name
        for name, pattern in TEMPERATURE_PATTERNS.items()
        if re.search(pattern, compact, re.IGNORECASE)
    ]
    return ";".join(matches)


def station_hits(text: str) -> list[str]:
    compact = normalized(text).lower()
    return [
        station
        for station, pattern in STATION_PATTERNS.items()
        if re.search(pattern, compact, re.IGNORECASE)
    ]


def relevant_context(text: str, station: str, width: int = 900) -> str:
    compact = normalized(text)
    match = re.search(STATION_PATTERNS[station], compact, re.IGNORECASE)
    if match is None:
        return ""
    start = max(0, match.start() - width // 3)
    end = min(len(compact), match.end() + 2 * width // 3)
    return compact[start:end]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    raw = OUT / "official_ocr_and_page_metadata"
    raw.mkdir()

    inventory: list[dict[str, object]] = []
    hits: list[dict[str, object]] = []
    report_summary: list[dict[str, object]] = []

    for item_id, base, coverage, role in ITEMS:
        item_dir = raw / base
        item_dir.mkdir()
        files = [f"{base}_djvu.txt", f"{base}_scandata.xml"]
        for name in files:
            url = f"https://archive.org/download/{item_id}/{name}"
            path = item_dir / name
            download(url, path)
            inventory.append(
                {
                    "item_id": item_id,
                    "coverage": coverage,
                    "role": role,
                    "name": name,
                    "url": url,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

        text_path = item_dir / f"{base}_djvu.txt"
        text = text_path.read_text(encoding="utf-8", errors="replace")
        pages = text.split("\x0c")
        report_hit_count = 0
        for page_index, page in enumerate(pages, start=1):
            stations = station_hits(page)
            if not stations:
                continue
            kind = page_type(page)
            if not kind:
                continue
            report_hit_count += len(stations)
            for station in stations:
                hits.append(
                    {
                        "item_id": item_id,
                        "coverage": coverage,
                        "role": role,
                        "ocr_page_index": page_index,
                        "station": station,
                        "temperature_table_type": kind,
                        "context": relevant_context(page, station),
                    }
                )
        report_summary.append(
            {
                "item_id": item_id,
                "coverage": coverage,
                "role": role,
                "ocr_page_count": len(pages),
                "temperature_station_hit_count": report_hit_count,
            }
        )

    with (OUT / "official_source_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    with (OUT / "temperature_station_page_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(hits[0]))
        writer.writeheader()
        writer.writerows(hits)
    with (OUT / "report_hit_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report_summary[0]))
        writer.writeheader()
        writer.writerows(report_summary)

    station_counts: dict[str, int] = {}
    station_coverages: dict[str, list[str]] = {}
    for station in STATION_PATTERNS:
        selected = [row for row in hits if row["station"] == station]
        station_counts[station] = len(selected)
        station_coverages[station] = sorted({str(row["coverage"]) for row in selected})

    summary = {
        "experiment": "temperature_historical_imd_report_ocr_archive_v1",
        "source": "India Meteorological Department annual reports via Internet Archive institutional mirror",
        "report_count": len(ITEMS),
        "unique_nominal_years": [
            1875,
            1877,
            1878,
            1879,
            1880,
            1881,
            1883,
            1884,
            1885,
            1886,
            1887,
            1888,
            1890,
        ],
        "missing_nominal_years_1875_1890": [1876, 1882, 1889],
        "downloaded_bytes": sum(int(row["bytes"]) for row in inventory),
        "temperature_station_page_hits": len(hits),
        "station_hit_counts": station_counts,
        "station_coverage_labels": station_coverages,
        "interpretation_boundary": (
            "original_publication_archive_and_transcription_are_independent_of_"
            "ghcn_processing_but_not_independent_physical_station_observations"
        ),
        "annual_values_transcribed": False,
        "locked_period_values_read": False,
        "decision": "advance_target_table_double_transcription",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "sources": inventory,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    contract = """# Historical IMD OCR Archive Contract

- Download only official/institutional OCR text and page-mapping metadata for
  India Meteorological Department annual reports dated 1875-1890.
- Index temperature-table pages for predefined HMA and HMA-margin stations.
- Do not transcribe annual values in this experiment; OCR is a locator, not a
  final numeric authority.
- Treat the original publication and a new transcription as independent of
  GHCN processing, but never as a second independent physical observation.
- No source includes 1921-1939. Do not access production climate, Jionglacuo,
  EnKF, or OGGM.
"""
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(contract, encoding="utf-8")
    result = f"""# Historical IMD Report OCR Recovery

Recovered {len(ITEMS)} institutionally mirrored annual-report objects using
only OCR text and page metadata ({summary['downloaded_bytes']} bytes). The
archive covers 13 nominal years in 1875-1890; 1876, 1882, and 1889 remain to be
filled from other public scans.

The station index contains {len(hits)} temperature-table hits. These records
make direct historical station-table recovery operational rather than merely
conceptual. The next experiment must double-transcribe the target rows against
page images before any climate comparison.

No annual value was promoted and no locked-period value was accessed.
Decision: `advance_target_table_double_transcription`.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
