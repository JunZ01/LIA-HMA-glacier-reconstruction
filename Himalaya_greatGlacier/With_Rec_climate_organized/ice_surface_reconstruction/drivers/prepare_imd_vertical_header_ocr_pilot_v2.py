from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULT_ROOT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
PAGE_ROOT = RESULT_ROOT / "imd_ghcn_overlap_pages_v2/official_full_resolution_station_pages"
OUT = RESULT_ROOT / "imd_vertical_header_ocr_pilot_v2"

# Coordinates are in the original 2700 x 3801 Internet Archive page image.
CASES = (
    {
        "case": "darjeeling_1886",
        "page": PAGE_ROOT / "india.history.resource.109237/leaf_0307.jpg",
        "crop": (1320, 390, 2600, 900),
    },
    {
        "case": "simla_1886",
        "page": PAGE_ROOT / "india.history.resource.109237/leaf_0301.jpg",
        "crop": (100, 390, 1390, 900),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for spec in CASES:
        page = Path(spec["page"])
        if not page.exists():
            raise FileNotFoundError(page)

        with Image.open(page) as source:
            source.load()
            original_size = source.size
            crop = source.crop(spec["crop"])

        gray = ImageOps.grayscale(crop)
        enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
        variants = {
            "unrotated": enhanced,
            "clockwise": enhanced.rotate(-90, expand=True),
            "counterclockwise": enhanced.rotate(90, expand=True),
        }

        for orientation, image in variants.items():
            enlarged = image.resize(
                (image.width * 2, image.height * 2),
                resample=Image.Resampling.LANCZOS,
            )
            output = OUT / f"{spec['case']}_{orientation}.png"
            enlarged.save(output, optimize=True)
            rows.append(
                {
                    "case": spec["case"],
                    "orientation": orientation,
                    "source_page": str(page),
                    "source_sha256": sha256(page),
                    "source_width": original_size[0],
                    "source_height": original_size[1],
                    "crop_left": spec["crop"][0],
                    "crop_top": spec["crop"][1],
                    "crop_right": spec["crop"][2],
                    "crop_bottom": spec["crop"][3],
                    "output": str(output),
                    "output_sha256": sha256(output),
                    "output_width": enlarged.width,
                    "output_height": enlarged.height,
                }
            )

    manifest = OUT / "image_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "imd_vertical_header_ocr_pilot_v2",
        "purpose": "Identify the vertical IMD table headers before extracting adopted annual means.",
        "cases": len(CASES),
        "images": len(rows),
        "source_files_modified": False,
        "locked_period_accessed": False,
        "note": "No temperature value is promoted by this image-preparation step.",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
