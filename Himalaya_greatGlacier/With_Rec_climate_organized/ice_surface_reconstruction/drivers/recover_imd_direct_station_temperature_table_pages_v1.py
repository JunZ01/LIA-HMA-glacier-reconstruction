from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "temperature_historical_imd_direct_station_table_recovery_v1"
DRIVER = Path(__file__).resolve()

ITEMS = [
    ("india.history.resource.108631", "108631", "1875"),
    ("india.history.resource.108632", "108632", "1877-1878"),
    ("india.history.resource.117788", "117788", "1877_duplicate"),
    ("india.history.resource.108633", "108633", "1879"),
    ("india.history.resource.108634", "108634", "1880"),
    ("india.history.resource.108635", "108635", "1881"),
    ("india.history.resource.108636", "108636", "1883"),
    ("india.history.resource.108637", "108637", "1884"),
    ("india.history.resource.108638", "108638", "1885"),
    ("india.history.resource.109237", "109237", "1886"),
    ("india.history.resource.109238", "109238", "1887"),
    ("india.history.resource.109240", "109240", "1888"),
    ("india.history.resource.109241", "109241", "1890"),
]

STATIONS = ["Darjeeling", "Simla", "Leh", "Murree", "Chakrata", "Ranikhet"]
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
EXCLUDE = [
    "solar radiation", "pressure", "evaporation", "humidity", "rainfall",
    "wind in miles", "wind velocity", "cloud proportion", "vapour tension",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_bytes(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 LIA historical climate audit"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def request_json(url: str) -> dict[str, object]:
    return json.loads(request_bytes(url).decode("utf-8", errors="replace"))


def candidate_score(text: str) -> tuple[int, int, bool, str]:
    compact = re.sub(r"\s+", " ", text).lower()
    month_count = sum(month in compact for month in MONTHS)
    excluded = any(term in compact for term in EXCLUDE)
    max_signal = bool(
        re.search(r"h[i1l][a-z]{2,10}\s+max", compact)
        or re.search(r"highest\s+max", compact)
    )
    min_signal = bool(
        re.search(r"low[a-z]{2,10}\s+min", compact)
        or re.search(r"lowest\s+min", compact)
    )
    title_signal = "mean and extreme air" in compact
    year_signal = bool(re.search(r"\byear\b", compact))
    corrected_signal = "corrected" in compact
    score = (
        5 * int(title_signal)
        + 3 * int(max_signal)
        + 3 * int(min_signal)
        + 2 * int(year_signal)
        + min(month_count, 12)
        + int(corrected_signal)
        - 20 * int(excluded)
    )
    accepted = (
        not excluded
        and month_count >= 8
        and year_signal
        and max_signal
        and min_signal
    )
    signals = ";".join(
        name
        for name, value in [
            ("title", title_signal),
            ("highest_maximum", max_signal),
            ("lowest_minimum", min_signal),
            ("year", year_signal),
            ("corrected", corrected_signal),
            ("excluded_other_table", excluded),
        ]
        if value
    )
    return score, month_count, accepted, signals


def search_url(server: str, item_id: str, base: str, item_path: str, station: str) -> str:
    query = urllib.parse.urlencode(
        {
            "item_id": item_id,
            "doc": base,
            "path": item_path,
            "q": station,
            "pre_tag": "{{{",
            "post_tag": "}}}",
        }
    )
    return f"https://{server}/fulltext/inside.php?{query}"


def image_url(server: str, item_id: str, base: str, item_path: str, leaf: int) -> str:
    query = urllib.parse.urlencode(
        {
            "zip": f"{item_path}/{base}_jp2.zip",
            "file": f"{base}_jp2/{base}_{leaf:04d}.jp2",
            "id": item_id,
            "scale": 2,
            "rotate": 0,
        }
    )
    return f"https://{server}/BookReader/BookReaderImages.php?{query}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    raw_json = OUT / "official_fulltext_search_json"
    page_images = OUT / "official_table_page_images"
    raw_json.mkdir()
    page_images.mkdir()

    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    image_keys: set[tuple[str, int]] = set()

    for item_id, base, coverage in ITEMS:
        metadata_url = f"https://archive.org/metadata/{item_id}"
        metadata_bytes = request_bytes(metadata_url)
        metadata_path = raw_json / f"{base}_metadata.json"
        metadata_path.write_bytes(metadata_bytes)
        metadata = json.loads(metadata_bytes.decode("utf-8", errors="replace"))
        server = str(metadata.get("server") or metadata.get("d2") or metadata["d1"])
        item_path = str(metadata["dir"])
        sources.append(
            {
                "kind": "item_metadata",
                "item_id": item_id,
                "coverage": coverage,
                "url": metadata_url,
                "path": str(metadata_path),
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256(metadata_path),
            }
        )

        item_json_dir = raw_json / base
        item_image_dir = page_images / base
        item_json_dir.mkdir()
        item_image_dir.mkdir()
        for station in STATIONS:
            url = search_url(server, item_id, base, item_path, station)
            payload_bytes = request_bytes(url)
            payload_path = item_json_dir / f"{station.lower()}.json"
            payload_path.write_bytes(payload_bytes)
            payload = json.loads(payload_bytes.decode("utf-8", errors="replace"))
            sources.append(
                {
                    "kind": "fulltext_search_json",
                    "item_id": item_id,
                    "coverage": coverage,
                    "station": station,
                    "url": url,
                    "path": str(payload_path),
                    "bytes": payload_path.stat().st_size,
                    "sha256": sha256(payload_path),
                }
            )
            for match_index, match in enumerate(payload.get("matches", []), start=1):
                text = str(match.get("text", ""))
                paragraphs = match.get("par") or []
                leaves = sorted(
                    {
                        int(paragraph["page"])
                        for paragraph in paragraphs
                        if "page" in paragraph
                    }
                )
                score, month_count, accepted, signals = candidate_score(text)
                for leaf in leaves:
                    row = {
                        "item_id": item_id,
                        "base": base,
                        "coverage": coverage,
                        "station": station,
                        "match_index": match_index,
                        "leaf": leaf,
                        "candidate_score": score,
                        "month_count": month_count,
                        "accepted_temperature_table": accepted,
                        "signals": signals,
                        "ocr_text": re.sub(r"\s+", " ", text).strip(),
                    }
                    rows.append(row)
                    if accepted and (base, leaf) not in image_keys:
                        img_url = image_url(server, item_id, base, item_path, leaf)
                        image_path = item_image_dir / f"{base}_leaf_{leaf:04d}.jpg"
                        image_path.write_bytes(request_bytes(img_url))
                        image_keys.add((base, leaf))
                        sources.append(
                            {
                                "kind": "official_page_image",
                                "item_id": item_id,
                                "coverage": coverage,
                                "leaf": leaf,
                                "url": img_url,
                                "path": str(image_path),
                                "bytes": image_path.stat().st_size,
                                "sha256": sha256(image_path),
                            }
                        )

    fieldnames = [
        "item_id", "base", "coverage", "station", "match_index", "leaf",
        "candidate_score", "month_count", "accepted_temperature_table",
        "signals", "ocr_text",
    ]
    with (OUT / "station_search_page_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accepted_rows = [row for row in rows if row["accepted_temperature_table"]]
    accepted_keys = sorted(
        {
            (str(row["coverage"]), str(row["station"]), int(row["leaf"]))
            for row in accepted_rows
        }
    )
    with (OUT / "accepted_station_table_pages.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["coverage", "station", "leaf"]
        )
        writer.writeheader()
        writer.writerows(
            {"coverage": coverage, "station": station, "leaf": leaf}
            for coverage, station, leaf in accepted_keys
        )

    summary = {
        "experiment": "temperature_historical_imd_direct_station_table_recovery_v1",
        "report_objects": len(ITEMS),
        "target_stations": STATIONS,
        "search_result_rows": len(rows),
        "accepted_station_table_rows": len(accepted_rows),
        "accepted_station_year_leaf_keys": len(accepted_keys),
        "downloaded_unique_table_page_images": len(image_keys),
        "annual_values_promoted": False,
        "locked_period_values_read": False,
        "interpretation": (
            "page images are primary-publication transcription evidence; "
            "they are not additional independent physical observations"
        ),
        "decision": "advance_double_transcription_of_accepted_pages",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "sources": sources,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    contract = """# Direct IMD Station Table Recovery Contract

- Recover only 1875-1890 report search metadata and scan pages.
- Use OCR only to locate candidate pages; do not promote OCR values directly.
- Exclude non-air-temperature tables before downloading candidate pages.
- Preserve every response and page hash in this independent result directory.
- Treat publication/transcription independence separately from physical-observation independence.
- Do not access 1921-1939, production reconstruction, EnKF, or OGGM.
"""
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(contract, encoding="utf-8")
    result = f"""# Direct Historical IMD Station Table Recovery

Queried {len(ITEMS)} institutionally archived IMD report objects for six HMA
and HMA-margin stations. The archive returned {len(rows)} station-search page
matches. Strict table-structure screening accepted {len(accepted_keys)} unique
station/year/leaf candidates and recovered {len(image_keys)} unique primary
scan pages for visual transcription.

No OCR number has been promoted as a temperature value. The accepted pages are
now ready for double transcription and comparison with GHCN-M source values.
No source covers 1921-1939.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
