from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STATIONS = (
    "DARJEELING",
    "TONGLO",
    "FALUT",
    "KATHMANDU",
    "KATMANDU",
    "MUSSOORIE",
    "SIMLA",
    "LAHOL",
    "KASHMIR",
    "MURREE",
    "LEH",
)
TEMPERATURE_TERMS = (
    "TEMPERATURE",
    "THERMOMETER",
    "THERMOMETRICAL",
    "MAXIMUM",
    "MINIMUM",
    "MEAN",
    "FAHRENHEIT",
    "CENTIGRADE",
)
MONTHS = (
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alto_text(path: Path) -> str:
    root = ET.parse(path).getroot()
    words = [node.attrib.get("CONTENT", "") for node in root.findall(".//{*}String")]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def image_services(manifest_path: Path) -> dict[int, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {}
    for canvas in manifest["sequences"][0]["canvases"]:
        label = str(canvas.get("label", ""))
        if not label.isdigit():
            continue
        resource = canvas["images"][0]["resource"]
        service = resource.get("service", {})
        if isinstance(service, list):
            service = service[0] if service else {}
        rows[int(label)] = str(service.get("@id", ""))
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise RuntimeError("Versioned output already completed")
    files = sorted(args.ocr_root.glob("page_*.xml"), key=lambda path: int(path.stem.split("_")[-1]))
    inputs = [args.manifest, args.mets, *files]
    before = {str(path.resolve()): sha256(path) for path in inputs}
    services = image_services(args.manifest)

    rows = []
    for path in files:
        page = int(path.stem.split("_")[-1])
        text = alto_text(path)
        upper = text.upper()
        station_hits = [name for name in STATIONS if re.search(rf"\b{re.escape(name)}\b", upper)]
        temperature_hits = [term for term in TEMPERATURE_TERMS if term in upper]
        month_hits = [month for month in MONTHS if month in upper]
        numeric_tokens = re.findall(r"(?<![A-Z])[-+]?\d+(?:[.\u00b7']\d+)?", upper)
        score = (
            4 * len(station_hits)
            + 2 * len(temperature_hits)
            + len(month_hits)
            + min(len(numeric_tokens), 80) / 10
        )
        service = services.get(page, "")
        rows.append(
            {
                "printed_page": page,
                "ocr_path": str(path.resolve()),
                "station_hits": ";".join(station_hits),
                "temperature_terms": ";".join(temperature_hits),
                "month_hits": ";".join(month_hits),
                "numeric_token_count": len(numeric_tokens),
                "word_count": len(text.split()),
                "candidate_score": score,
                "iiif_service": service,
                "iiif_full_jpg": f"{service}/full/2000,/0/default.jpg" if service else "",
                "text_snippet": text[:1200],
            }
        )
    pages = pd.DataFrame(rows)
    candidates = pages[
        pages.station_hits.ne("")
        & pages.temperature_terms.ne("")
        & pages.numeric_token_count.ge(8)
    ].sort_values(["candidate_score", "printed_page"], ascending=[False, True])
    pages.to_csv(args.output / "page_ocr_index.csv", index=False)
    candidates.to_csv(args.output / "temperature_table_candidates.csv", index=False)

    after = {str(path.resolve()): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Source state changed")
    summary = {
        "experiment": "temperature_schlagintweit_1854_1858_direct_observation_recovery_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_ocr_page_index",
        "ocr_page_count": len(pages),
        "temperature_table_candidate_count": len(candidates),
        "candidate_printed_pages": candidates.printed_page.astype(int).tolist(),
        "scientific_role": "independent short historical expedition observations; local month and elevation anchors only, not regional annual truth",
        "next_action": "download and visually adjudicate the highest-scoring table pages, then recover station-date-elevation-temperature tuples with instrument metadata",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
        "source_hashes_unchanged": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = f"""# Schlagintweit High Asia 1854-1858 OCR recovery

The official Heidelberg IIIF manifest, METS record, and ALTO OCR for printed pages 465-534 were recovered and indexed without touching any production file.

- OCR pages indexed: **{len(pages)}**
- Pages matching station names, temperature language, and numeric content: **{len(candidates)}**

These expedition observations are independent of the later IMD/GHCN processing chain. Their scientific role is deliberately limited to short local month/elevation anchors after visual and instrument audit; they cannot define a regional annual correction by themselves.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 or another prohibited source accessed: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
