"""Extract the Hirpora Figure 7 through year 1900 using detected tick labels."""

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
CROP = OUTPUT / "hirpora_figure7_through_1900_only_v2.png"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contiguous(values: np.ndarray) -> list[np.ndarray]:
    return [group for group in np.split(values, np.where(np.diff(values) > 1)[0] + 1) if len(group)]


def main() -> int:
    if CROP.exists():
        raise FileExistsError(CROP)
    before = {"pdf": sha256(PDF), "contract": sha256(CONTRACT)}
    image = PdfReader(PDF).pages[6].images[0].image.convert("RGB")
    gray = np.asarray(image.convert("L"))
    if gray.shape != (618, 1238):
        raise RuntimeError(f"unexpected-figure-shape:{gray.shape}")

    dark = gray < 80
    score = dark[535:570].sum(axis=0)
    glyphs = contiguous(np.flatnonzero(score >= 8))
    glyph_bounds = [(int(group[0]), int(group[-1])) for group in glyphs]
    clusters: list[list[tuple[int, int]]] = []
    for bounds in glyph_bounds:
        if not clusters or bounds[0] - clusters[-1][-1][1] > 80:
            clusters.append([bounds])
        else:
            clusters[-1].append(bounds)
    label_bounds = [(group[0][0], group[-1][1]) for group in clusters]
    if len(label_bounds) != 5:
        raise RuntimeError(f"unexpected-x-label-count:{label_bounds}")
    centers = np.asarray([(left + right) / 2 for left, right in label_bounds], dtype=float)
    years = np.asarray([1800, 1850, 1900, 1950, 2000], dtype=float)
    slope, intercept = np.polyfit(years, centers, 1)
    residual = centers - (slope * years + intercept)
    if slope < 4.0 or slope > 4.4 or np.max(np.abs(residual)) > 2.5:
        raise RuntimeError(f"x-axis-calibration-failed:{slope}:{residual.tolist()}")
    x1900 = float(slope * 1900 + intercept)
    crop_right = int(np.ceil(x1900 + slope / 2))
    image.crop((0, 0, crop_right, image.height)).save(CROP)

    metadata = {
        "status": "complete",
        "supersedes_v1_crop": True,
        "v1_failure": "used data coverage as axis bounds and stopped near 1895; conservative but incomplete",
        "pdf_page_1based": 7,
        "detected_tick_label_bounds": label_bounds,
        "assigned_tick_years": years.astype(int).tolist(),
        "tick_centers_x": centers.tolist(),
        "pixels_per_year": float(slope),
        "maximum_tick_fit_residual_pixels": float(np.max(np.abs(residual))),
        "fitted_1900_center_x": x1900,
        "saved_crop_x_bounds": [0, crop_right],
        "post_1900_curve_pixels_saved": False,
        "full_embedded_figure_saved": False,
    }
    (OUTPUT / "pre1901_crop_metadata_v2.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    after = {"pdf": sha256(PDF), "contract": sha256(CONTRACT)}
    if before != after:
        raise RuntimeError("source-state-changed")
    (OUTPUT / "pre1901_crop_source_manifest_v2.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": True}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
