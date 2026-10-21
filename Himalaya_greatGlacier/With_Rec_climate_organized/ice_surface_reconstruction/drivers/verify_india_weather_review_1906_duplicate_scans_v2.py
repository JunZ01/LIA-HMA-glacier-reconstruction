from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium

import verify_india_weather_review_1906_duplicate_scans_v1 as v1


def extract_images(pdf_path: Path, output: Path) -> list[dict]:
    document = pdfium.PdfDocument(str(pdf_path))
    if len(document) <= max(v1.PRIMARY_LEAVES):
        raise RuntimeError("Primary PDF has fewer pages than expected")
    rows = []
    image_root = output / "verified_pages"
    image_root.mkdir(parents=True, exist_ok=True)
    for leaf in v1.PRIMARY_LEAVES:
        destination = image_root / f"108906_leaf_{leaf:04d}.png"
        bitmap = document[leaf].render(scale=2.5)
        image = bitmap.to_pil()
        image.save(destination, format="PNG", optimize=True)
        rows.append(
            {
                "identifier": "india.history.resource.108906",
                "leaf": leaf,
                "pdf_page_index_zero_based": leaf,
                "image_path": str(destination.resolve()),
                "image_width": image.width,
                "image_height": image.height,
                "image_bytes": destination.stat().st_size,
                "image_sha256": v1.sha256(destination),
            }
        )
    return rows


def main() -> None:
    v1.extract_images = extract_images
    v1.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = "temperature_india_weather_review_1906_duplicate_scan_verification_v2"
    summary["supersedes"] = "temperature_india_weather_review_1906_duplicate_scan_verification_v1_incomplete"
    summary["render_method"] = "pypdfium2 at 2.5x scale"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    inventory = pd.read_csv(output / "extracted_primary_page_images.csv")
    if (inventory["image_width"] < 1500).any() or (inventory["image_height"] < 1500).any():
        raise RuntimeError("Rendered scan image resolution is too low")


if __name__ == "__main__":
    main()
