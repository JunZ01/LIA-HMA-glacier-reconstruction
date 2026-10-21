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
V4 = LOOP / "round_228_yang2010_figure4_locked_safe_preflight_v4"
OUT = LOOP / "round_228_yang2010_figure4_locked_safe_preflight_v5"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
CONTRACT = DRIVER.with_name("round228_yang2010_figure4_locked_safe_preflight_v5_contract.md")
FRAME = {"left": 354, "right": 1757, "top": 1021, "bottom": 1623}
AXIS_YEAR_START, AXIS_YEAR_END = 1350, 2000
DATA_YEAR_START, DATA_YEAR_END = 1377, 1998
MASK_START_YEAR, MASK_END_YEAR = 1900, 1939
CROP = {"left": 254, "top": 976, "right": 1803, "bottom": 1724}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def year_x(year: int) -> int:
    return round(
        FRAME["left"]
        + (year - AXIS_YEAR_START)
        / (AXIS_YEAR_END - AXIS_YEAR_START)
        * (FRAME["right"] - FRAME["left"])
    )


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    required = [PDF, FACTS, DRIVER, CONTRACT, V4 / "summary.json"]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("Missing safe-preflight input: " + "; ".join(missing))
    facts_before = sha256(FACTS)
    OUT.mkdir(parents=True)

    document = pdfium.PdfDocument(str(PDF))
    try:
        image = document[5].render(scale=250 / 72).to_pil().convert("RGB")
    finally:
        document.close()
    crop = image.crop((CROP["left"], CROP["top"], CROP["right"], CROP["bottom"]))
    mask_left = year_x(MASK_START_YEAR) - CROP["left"]
    mask_right = year_x(MASK_END_YEAR + 1) - CROP["left"]
    ImageDraw.Draw(crop).rectangle((mask_left, 0, mask_right, crop.height), fill=(0, 0, 0))
    safe_image = OUT / "figure4b_axis1350_2000_mask1900_1939.png"
    crop.save(safe_image, format="PNG", optimize=True)

    facts_after = sha256(FACTS)
    if facts_after != facts_before:
        raise RuntimeError("Canonical facts changed during safe preflight")
    summary = {
        "round": 228,
        "version": 5,
        "supersedes_v4_axis_mapping": str(V4 / "summary.json"),
        "source_frame_px": FRAME,
        "axis_year_range": [AXIS_YEAR_START, AXIS_YEAR_END],
        "data_year_range": [DATA_YEAR_START, DATA_YEAR_END],
        "saved_crop_box_px": CROP,
        "masked_year_range": [MASK_START_YEAR, MASK_END_YEAR],
        "mask_global_x_px": [year_x(MASK_START_YEAR), year_x(MASK_END_YEAR + 1)],
        "mask_crop_x_px": [mask_left, mask_right],
        "visible_data_year_segments": [[DATA_YEAR_START, 1899], [1940, DATA_YEAR_END]],
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
                "v4_summary": {"path": str(V4 / "summary.json"), "sha256": sha256(V4 / "summary.json")},
                "facts": {"path": str(FACTS), "sha256": facts_after},
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
                "safe_image": {"path": str(safe_image), "sha256": sha256(safe_image)},
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
