from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


PRIMARY_LEAVES = (134, 144, 157, 249, 252, 253, 254)
DUPLICATE_LEAVES = (3, 13, 28, 118)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-pdf", type=Path, required=True)
    parser.add_argument("--primary-xml", type=Path, required=True)
    parser.add_argument("--duplicate-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_texts(path: Path) -> list[str]:
    objects = ET.parse(path).getroot().findall(".//OBJECT")
    return [" ".join(word.text or "" for word in obj.iter("WORD")) for obj in objects]


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+", text.upper())


def similarity(left: str, right: str) -> tuple[float, float, int]:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union = left_set | right_set
    jaccard = len(left_set & right_set) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio()
    return jaccard, sequence, len(left_set & right_set)


def extract_images(pdf_path: Path, output: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) <= max(PRIMARY_LEAVES):
        raise RuntimeError("Primary PDF has fewer pages than expected")
    rows = []
    image_root = output / "verified_pages"
    image_root.mkdir(parents=True, exist_ok=True)
    for leaf in PRIMARY_LEAVES:
        images = list(reader.pages[leaf].images)
        if len(images) != 1:
            raise RuntimeError(f"Expected one scan image on PDF leaf {leaf}, found {len(images)}")
        image = images[0]
        suffix = Path(image.name).suffix or ".bin"
        destination = image_root / f"108906_leaf_{leaf:04d}{suffix}"
        destination.write_bytes(image.data)
        rows.append(
            {
                "identifier": "india.history.resource.108906",
                "leaf": leaf,
                "pdf_page_index_zero_based": leaf,
                "image_path": str(destination.resolve()),
                "image_bytes": destination.stat().st_size,
                "image_sha256": sha256(destination),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise RuntimeError("Versioned output already completed")
    inputs = (args.primary_pdf, args.primary_xml, args.duplicate_xml)
    before = {str(path.resolve()): sha256(path) for path in inputs}

    primary_text = page_texts(args.primary_xml)
    duplicate_text = page_texts(args.duplicate_xml)
    pair_rows = []
    for duplicate_leaf in DUPLICATE_LEAVES:
        for primary_leaf in PRIMARY_LEAVES:
            jaccard, sequence, shared = similarity(
                duplicate_text[duplicate_leaf], primary_text[primary_leaf]
            )
            pair_rows.append(
                {
                    "duplicate_leaf": duplicate_leaf,
                    "primary_leaf": primary_leaf,
                    "token_jaccard": jaccard,
                    "token_sequence_ratio": sequence,
                    "shared_unique_tokens": shared,
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs["combined_score"] = 0.5 * pairs["token_jaccard"] + 0.5 * pairs["token_sequence_ratio"]
    pairs["duplicate_leaf_rank"] = pairs.groupby("duplicate_leaf")["combined_score"].rank(
        method="first", ascending=False
    ).astype(int)
    best = pairs.loc[pairs["duplicate_leaf_rank"].eq(1)].sort_values("duplicate_leaf")
    pairs.sort_values(["duplicate_leaf", "duplicate_leaf_rank"]).to_csv(
        args.output / "duplicate_scan_page_alignment.csv", index=False
    )
    best.to_csv(args.output / "best_duplicate_scan_page_matches.csv", index=False)

    image_rows = extract_images(args.primary_pdf, args.output)
    pd.DataFrame(image_rows).to_csv(args.output / "extracted_primary_page_images.csv", index=False)

    after = {str(path.resolve()): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Source state changed")
    manifest = {
        "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
        "source_hashes_unchanged": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "experiment": "temperature_india_weather_review_1906_duplicate_scan_verification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_ocr_alignment_and_primary_page_extraction",
        "primary_page_images_extracted": len(image_rows),
        "duplicate_scan_candidate_pages": len(DUPLICATE_LEAVES),
        "primary_scan_candidate_pages": len(PRIMARY_LEAVES),
        "best_page_matches": best.to_dict(orient="records"),
        "interpretation_limit": "OCR agreement can identify corresponding pages but printed numeric values require visual verification",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = """# India Weather Review 1906 duplicate-scan verification

The two independently scanned 1906 volumes were aligned using page-level OCR tokens. Candidate table pages from the complete primary PDF were also extracted as original embedded scan images for visual verification.

This experiment does not treat the two scans as independent climate evidence. They are duplicate representations of the same official observations and are used only to detect OCR or transcription errors.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 or another prohibited source accessed: **no**
- Original or production files modified: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
