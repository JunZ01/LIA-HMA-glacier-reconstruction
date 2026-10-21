"""Extract only the through-1900 portion of the Hirpora reconstruction figure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_temperature_hirpora_independent_pre1901_validation_v1"
)
PDF = OUTPUT / "source_full_article_do_not_analyze_post1900.pdf"
CONTRACT = OUTPUT / "ROUND_CONTRACT.md"
CROP = OUTPUT / "hirpora_figure7_through_1900_only.png"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if CROP.exists():
        raise FileExistsError(CROP)
    before = {"pdf": sha256(PDF), "contract": sha256(CONTRACT)}
    reader = PdfReader(PDF)
    if len(reader.pages) != 12:
        raise RuntimeError(f"unexpected-page-count:{len(reader.pages)}")
    images = reader.pages[6].images
    if len(images) != 1:
        raise RuntimeError(f"unexpected-page7-image-count:{len(images)}")
    image = images[0].image.convert("RGB")
    gray = np.asarray(image.convert("L"))
    if gray.shape != (618, 1238):
        raise RuntimeError(f"unexpected-figure-shape:{gray.shape}")

    dark = gray < 80
    x_axis = int(np.argmax(dark.sum(axis=1)))
    y_axis = int(np.argmax(dark.sum(axis=0)))
    if abs(x_axis - 539) > 2 or abs(y_axis - 114) > 2:
        raise RuntimeError(f"axis-detection-failed:{x_axis}:{y_axis}")

    plot_right = 1163
    expected_1900 = y_axis + (1900 - 1773) / (2012 - 1773) * (plot_right - y_axis)
    search = np.arange(int(expected_1900) - 12, int(expected_1900) + 13)
    tick_1900 = int(search[np.argmax(dark[:, search].sum(axis=0))])
    if abs(tick_1900 - expected_1900) > 8:
        raise RuntimeError(f"year-1900-gridline-detection-failed:{expected_1900}:{tick_1900}")

    crop_right = tick_1900 + 5
    image.crop((0, 0, crop_right, image.height)).save(CROP)
    metadata = {
        "status": "complete",
        "pdf_page_1based": 7,
        "embedded_image_shape": [int(gray.shape[0]), int(gray.shape[1])],
        "detected_x_axis_row": x_axis,
        "detected_y_axis_column": y_axis,
        "assumed_plot_year_bounds": [1773, 2012],
        "assumed_plot_x_bounds": [y_axis, plot_right],
        "expected_1900_x": float(expected_1900),
        "detected_1900_gridline_x": tick_1900,
        "saved_crop_x_bounds": [0, crop_right],
        "post_1900_curve_pixels_saved": False,
        "full_embedded_figure_saved": False,
    }
    (OUTPUT / "pre1901_crop_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    after = {"pdf": sha256(PDF), "contract": sha256(CONTRACT)}
    if before != after:
        raise RuntimeError("source-state-changed")
    (OUTPUT / "pre1901_crop_source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": True}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
