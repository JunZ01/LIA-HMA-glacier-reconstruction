from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
SOURCE = LOOP / "imd_ghcn_adopted_temperature_extract_v1/station_year_extraction.csv"
OUT = LOOP / "imd_station_header_panel_v1"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    image_dir = OUT / "header_images"
    image_dir.mkdir()

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))

    manifest: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for row in source_rows:
        heading_y = as_float(row.get("station_heading_y", ""))
        side = row.get("station_side", "")
        if math.isnan(heading_y) or side not in {"left", "right"}:
            missing.append(
                {
                    "report_year": row["report_year"],
                    "station": row["imd_station"],
                    "page_key": row.get("page_key", ""),
                    "reason": "station_heading_not_located_by_frozen_full_page_ocr",
                }
            )
            continue

        page = Path(row["image_path"])
        with Image.open(page) as source:
            source.load()
            width, height = source.size
            left = 40 if side == "left" else width // 2 - 40
            right = width // 2 + 40 if side == "left" else width - 40
            top = max(0, int(round(heading_y)) + 15)
            bottom = min(height, top + 590)
            crop_box = (left, top, right, bottom)
            crop = source.crop(crop_box)

        enhanced = ImageEnhance.Contrast(ImageOps.grayscale(crop)).enhance(2.0)
        case = f"{int(row['report_year'])}_{row['imd_station'].lower()}_{row['page_key']}"
        for orientation, image in {
            "clockwise": enhanced.rotate(-90, expand=True),
            "counterclockwise": enhanced.rotate(90, expand=True),
        }.items():
            enlarged = image.resize(
                (image.width * 2, image.height * 2),
                resample=Image.Resampling.LANCZOS,
            )
            output = image_dir / f"{case}_{orientation}.png"
            enlarged.save(output, optimize=True)
            manifest.append(
                {
                    "report_year": row["report_year"],
                    "station": row["imd_station"],
                    "page_key": row["page_key"],
                    "side": side,
                    "orientation": orientation,
                    "source_page": str(page),
                    "source_sha256": sha256(page),
                    "crop_left": left,
                    "crop_top": top,
                    "crop_right": right,
                    "crop_bottom": bottom,
                    "output": str(output),
                    "output_sha256": sha256(output),
                }
            )

    with (OUT / "header_image_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (OUT / "unlocated_station_years.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(missing[0]))
        writer.writeheader()
        writer.writerows(missing)

    summary = {
        "experiment": "imd_station_header_panel_v1",
        "requested_station_years": len(source_rows),
        "located_station_years": len(manifest) // 2,
        "unlocated_station_years": len(missing),
        "header_images": len(manifest),
        "source_files_modified": False,
        "locked_period_accessed": False,
        "decision": "ocr_actual_headers_before_column_selection",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "frozen_station_geometry": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
