"""Extract official Upper Indus supplementary figures without altering the DOCX."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
SOURCE = PROJECT / "validation_data/sources/external_hma_temperature_upper_indus_mxd_v1/upper_indus_S0048969724075508_mmc1.docx"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_upper_indus_figure_extraction_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = sha256(SOURCE)
    extracted = []
    with zipfile.ZipFile(SOURCE) as archive:
        for member in ["word/media/image1.tif", "word/media/image2.png"]:
            target = OUTPUT / Path(member).name
            target.write_bytes(archive.read(member))
            with Image.open(target) as image:
                info = {
                    "archive_member": member,
                    "output": str(target),
                    "width_px": image.width,
                    "height_px": image.height,
                    "mode": image.mode,
                    "format": image.format,
                }
                if image.format == "TIFF":
                    png = OUTPUT / "image1_rendered.png"
                    image.convert("RGB").save(png, dpi=(600, 600))
                    info["rendered_png"] = str(png)
                extracted.append(info)
    after = sha256(SOURCE)
    summary = {
        "status": "complete",
        "source_sha256": before,
        "source_hash_unchanged": before == after,
        "figures": extracted,
        "original_source_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
