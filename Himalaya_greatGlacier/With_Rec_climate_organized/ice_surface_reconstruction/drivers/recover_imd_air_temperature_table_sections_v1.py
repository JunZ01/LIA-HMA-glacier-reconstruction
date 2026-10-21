from __future__ import annotations

import csv
import difflib
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
OUT = LOOP / "temperature_historical_imd_air_temperature_table_sections_v1"
DRIVER = Path(__file__).resolve()

ITEMS = [
    ("india.history.resource.108631", "108631", "1875", [1875]),
    ("india.history.resource.108632", "108632", "1877-1878", [1877, 1878]),
    ("india.history.resource.117788", "117788", "1877_duplicate", [1877]),
    ("india.history.resource.108633", "108633", "1879", [1879]),
    ("india.history.resource.108634", "108634", "1880", [1880]),
    ("india.history.resource.108635", "108635", "1881", [1881]),
    ("india.history.resource.108636", "108636", "1883", [1883]),
    ("india.history.resource.108637", "108637", "1884", [1884]),
    ("india.history.resource.108638", "108638", "1885", [1885]),
    ("india.history.resource.109237", "109237", "1886", [1886]),
    ("india.history.resource.109238", "109238", "1887", [1887]),
    ("india.history.resource.109240", "109240", "1888", [1888]),
    ("india.history.resource.109241", "109241", "1890", [1890]),
]
STATIONS = ["darjeeling", "simla", "leh", "murree", "chakrata", "ranikhet"]
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, attempts: int = 4) -> bytes:
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


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_air_temperature_table(text: str, years: list[int]) -> tuple[bool, int, str]:
    value = compact(text).lower().replace("{{{", "").replace("}}}", "")
    month_count = sum(month in value for month in MONTHS)
    air_signal = bool(re.search(r"air\W{0,3}temp", value))
    table_signal = bool(re.search(r"mean.{0,25}extreme", value))
    year_signal = any(str(year) in value for year in years)
    accepted = air_signal and table_signal and year_signal and month_count >= 8
    return accepted, month_count, ";".join(
        name
        for name, present in [
            ("air_temperature", air_signal),
            ("mean_extreme", table_signal),
            ("year", year_signal),
            ("months_ge_8", month_count >= 8),
        ]
        if present
    )


def fuzzy_station_scores(text: str) -> dict[str, float]:
    value = compact(text).lower().replace("{{{", "").replace("}}}", "")
    words = re.findall(r"[a-z]{3,18}", value)
    scores: dict[str, float] = {}
    for station in STATIONS:
        scores[station] = max(
            (difflib.SequenceMatcher(None, station, word).ratio() for word in words),
            default=0.0,
        )
    return scores


def search_url(server: str, item_id: str, base: str, item_path: str) -> str:
    query = urllib.parse.urlencode(
        {
            "item_id": item_id,
            "doc": base,
            "path": item_path,
            "q": "extreme",
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
    raw_dir = OUT / "official_title_search_json"
    image_dir = OUT / "official_air_temperature_table_pages"
    raw_dir.mkdir()
    image_dir.mkdir()

    pages: list[dict[str, object]] = []
    fuzzy_rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []

    for item_id, base, coverage, years in ITEMS:
        metadata_url = f"https://archive.org/metadata/{item_id}"
        metadata_bytes = fetch(metadata_url)
        metadata_path = raw_dir / f"{base}_metadata.json"
        metadata_path.write_bytes(metadata_bytes)
        metadata = json.loads(metadata_bytes.decode("utf-8", errors="replace"))
        server = str(metadata.get("server") or metadata.get("d2") or metadata["d1"])
        item_path = str(metadata["dir"])

        url = search_url(server, item_id, base, item_path)
        payload_bytes = fetch(url)
        payload_path = raw_dir / f"{base}_extreme_search.json"
        payload_path.write_bytes(payload_bytes)
        payload = json.loads(payload_bytes.decode("utf-8", errors="replace"))
        sources.extend(
            [
                {
                    "kind": "item_metadata",
                    "item_id": item_id,
                    "coverage": coverage,
                    "url": metadata_url,
                    "path": str(metadata_path),
                    "bytes": metadata_path.stat().st_size,
                    "sha256": sha256(metadata_path),
                },
                {
                    "kind": "fulltext_title_search_json",
                    "item_id": item_id,
                    "coverage": coverage,
                    "url": url,
                    "path": str(payload_path),
                    "bytes": payload_path.stat().st_size,
                    "sha256": sha256(payload_path),
                },
            ]
        )

        unique: dict[int, str] = {}
        for match in payload.get("matches", []):
            text = str(match.get("text", ""))
            accepted, month_count, signals = is_air_temperature_table(text, years)
            if not accepted:
                continue
            for paragraph in match.get("par") or []:
                if "page" in paragraph:
                    leaf = int(paragraph["page"])
                    prior = unique.get(leaf, "")
                    if len(text) > len(prior):
                        unique[leaf] = text

        target_dir = image_dir / base
        target_dir.mkdir()
        for leaf, text in sorted(unique.items()):
            accepted, month_count, signals = is_air_temperature_table(text, years)
            url = image_url(server, item_id, base, item_path, leaf)
            path = target_dir / f"{base}_leaf_{leaf:04d}.jpg"
            path.write_bytes(fetch(url))
            pages.append(
                {
                    "item_id": item_id,
                    "coverage": coverage,
                    "leaf": leaf,
                    "month_count": month_count,
                    "signals": signals,
                    "image_path": str(path),
                    "image_bytes": path.stat().st_size,
                    "image_sha256": sha256(path),
                    "ocr_text": compact(text),
                }
            )
            sources.append(
                {
                    "kind": "official_air_temperature_table_page",
                    "item_id": item_id,
                    "coverage": coverage,
                    "leaf": leaf,
                    "url": url,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
            scores = fuzzy_station_scores(text)
            for station, score in scores.items():
                if score >= 0.70:
                    fuzzy_rows.append(
                        {
                            "item_id": item_id,
                            "coverage": coverage,
                            "leaf": leaf,
                            "station": station,
                            "fuzzy_score": round(score, 6),
                            "manual_verification_required": True,
                        }
                    )

    page_fields = [
        "item_id", "coverage", "leaf", "month_count", "signals",
        "image_path", "image_bytes", "image_sha256", "ocr_text",
    ]
    with (OUT / "air_temperature_table_pages.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=page_fields)
        writer.writeheader()
        writer.writerows(pages)
    with (OUT / "fuzzy_station_page_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "item_id", "coverage", "leaf", "station", "fuzzy_score",
                "manual_verification_required",
            ],
        )
        writer.writeheader()
        writer.writerows(fuzzy_rows)

    coverage_counts = {
        coverage: sum(row["coverage"] == coverage for row in pages)
        for _, _, coverage, _ in ITEMS
    }
    summary = {
        "experiment": "temperature_historical_imd_air_temperature_table_sections_v1",
        "report_objects": len(ITEMS),
        "accepted_primary_scan_pages": len(pages),
        "accepted_page_counts_by_coverage": coverage_counts,
        "fuzzy_station_page_candidates": len(fuzzy_rows),
        "annual_values_promoted": False,
        "locked_period_values_read": False,
        "decision": "advance_station_identity_and_numeric_double_transcription",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {"driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)}, "sources": sources},
            indent=2,
            ensure_ascii=True,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        """# Historical IMD Air-Temperature Table Section Contract

- Locate complete air-temperature table sections by table title, not station OCR.
- Download only reports dated 1875-1890; do not access 1921-1939.
- Preserve primary page images and source responses with hashes.
- Fuzzy station matches are locators only and require manual verification.
- Do not promote OCR numbers, alter production data, run EnKF, or run OGGM.
""",
        encoding="utf-8",
    )
    (OUT / "FINAL_RESULT.md").write_text(
        f"""# Historical IMD Air-Temperature Table Sections

Recovered {len(pages)} primary scan pages from complete mean/extreme
air-temperature table sections in {len(ITEMS)} report objects. This title-led
route is robust to OCR damage in station names and supersedes station-name-only
page discovery for section completeness.

Fuzzy matching produced {len(fuzzy_rows)} station-page candidates. These are
locators, not accepted station identities or temperature values. The next
stage is station-heading verification and independent transcription of annual
means from the page images.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
