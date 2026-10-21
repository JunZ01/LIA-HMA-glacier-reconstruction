from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
PAGES = LOOP / "imd_ghcn_overlap_pages_v2/official_full_resolution_station_pages"
OUT = LOOP / "imd_vertical_header_ocr_pilot_v1"
DRIVER = Path(__file__).resolve()

PILOTS = [
    {
        "name": "darjeeling_1886_right",
        "source": PAGES / "india.history.resource.109237/leaf_0307.jpg",
        "crop": (1350, 480, 2550, 760),
    },
    {
        "name": "simla_1886_left",
        "source": PAGES / "india.history.resource.109237/leaf_0301.jpg",
        "crop": (180, 470, 1320, 760),
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = []
    for pilot in PILOTS:
        source = Path(pilot["source"])
        with Image.open(source) as image:
            crop = image.crop(pilot["crop"])
            gray = ImageOps.grayscale(crop)
            enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
            for direction, angle in [("cw", -90), ("ccw", 90)]:
                rotated = enhanced.rotate(angle, expand=True, fillcolor=255)
                rotated = rotated.resize(
                    (rotated.width * 2, rotated.height * 2),
                    Image.Resampling.LANCZOS,
                )
                path = OUT / f"{pilot['name']}_{direction}.png"
                rotated.save(path)
                manifest.append(
                    {
                        "pilot": pilot["name"],
                        "direction": direction,
                        "source": str(source),
                        "source_sha256": sha256(source),
                        "crop": list(pilot["crop"]),
                        "output": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    summary = {
        "experiment": "imd_vertical_header_ocr_pilot_v1",
        "processed_headers": len(PILOTS),
        "rotated_images": len(manifest),
        "locked_period_values_read": False,
        "decision": "run_matlab_ocr_on_both_rotations",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "images": manifest,
            },
            indent=2,
            ensure_ascii=True,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        """# IMD Vertical Header OCR Pilot Contract

- Crop only table-header regions from 1886 primary scan pages.
- Rotate both directions and enhance contrast solely for OCR legibility.
- Preserve source hashes and do not alter the source pages.
- Do not promote any temperature value in this header-only experiment.
- Do not access 1921-1939, production data, EnKF, or OGGM.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
