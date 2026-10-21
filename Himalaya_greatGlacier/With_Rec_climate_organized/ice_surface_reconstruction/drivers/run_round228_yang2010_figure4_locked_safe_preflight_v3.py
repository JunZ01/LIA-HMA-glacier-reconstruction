from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pypdfium2 as pdfium
from PIL import ImageDraw


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
PDF = LOOP / "round_226_yang2010_setp_622y_metadata_recovery_v1/official_sources_unmodified/2972_yang_622_year_4223.pdf"
V2 = LOOP / "round_228_yang2010_figure4_locked_safe_preflight_v2"
OUT = LOOP / "round_228_yang2010_figure4_locked_safe_preflight_v3"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
CONTRACT = DRIVER.with_name("round228_yang2010_figure4_locked_safe_preflight_v3_contract.md")
FRAME = {"left": 354, "right": 1757, "top": 1021, "bottom": 1623}
YEAR_START = 1377
YEAR_END = 1998
MASK_FROM_YEAR = 1900
CROP_MARGIN = {"left": 100, "right": 45, "top": 45, "bottom": 100}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    required = [PDF, FACTS, DRIVER, CONTRACT, V2 / "summary.json"]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("Missing safe-preflight input: " + "; ".join(missing))
    facts_before = sha256(FACTS)
    OUT.mkdir(parents=True)

    document = pdfium.PdfDocument(str(PDF))
    try:
        image = document[5].render(scale=250 / 72).to_pil().convert("RGB")
    finally:
        document.close()
    mask_global_x = round(
        FRAME["left"]
        + (MASK_FROM_YEAR - YEAR_START)
        / (YEAR_END - YEAR_START)
        * (FRAME["right"] - FRAME["left"])
    )
    crop_box = (
        FRAME["left"] - CROP_MARGIN["left"],
        FRAME["top"] - CROP_MARGIN["top"],
        FRAME["right"] + CROP_MARGIN["right"] + 1,
        FRAME["bottom"] + CROP_MARGIN["bottom"] + 1,
    )
    crop = image.crop(crop_box)
    mask_crop_x = mask_global_x - crop_box[0]
    ImageDraw.Draw(crop).rectangle((mask_crop_x, 0, crop.width, crop.height), fill=(0, 0, 0))
    preview = OUT / "figure4b_pre1900_masked.png"
    crop.save(preview, format="PNG", optimize=True)
    compact = crop.copy()
    compact.thumbnail((1100, 700))
    compact_path = OUT / "figure4b_pre1900_masked_preview.jpg"
    compact.save(compact_path, format="JPEG", quality=65, optimize=True)

    facts_after = sha256(FACTS)
    if facts_after != facts_before:
        raise RuntimeError("Canonical facts changed during read-only preflight")
    summary = {
        "round": 228,
        "version": 3,
        "source_frame_px": FRAME,
        "source_year_range": [YEAR_START, YEAR_END],
        "mask_from_year": MASK_FROM_YEAR,
        "mask_global_x_px": mask_global_x,
        "crop_box_px": list(crop_box),
        "mask_crop_x_px": mask_crop_x,
        "saved_crop_size_px": [crop.width, crop.height],
        "unmasked_raster_saved": False,
        "curve_digitized": False,
        "locked_period_values_read": False,
        "jionglacuo_accessed": False,
        "production_files_modified": False,
        "facts_hash_unchanged": facts_after == facts_before,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "pdf": {"path": str(PDF), "sha256": sha256(PDF)},
                "v2_summary": {"path": str(V2 / "summary.json"), "sha256": sha256(V2 / "summary.json")},
                "facts": {"path": str(FACTS), "sha256": facts_after},
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
                "masked_png": {"path": str(preview), "sha256": sha256(preview)},
                "masked_jpg": {"path": str(compact_path), "sha256": sha256(compact_path)},
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(CONTRACT, OUT / "ROUND_CONTRACT.md")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
