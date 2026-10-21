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
V1_FAILED = LOOP / "round_228_yang2010_figure4_locked_safe_preflight_v1"
OUT = LOOP / "round_228_yang2010_figure4_locked_safe_preflight_v2"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
CONTRACT = DRIVER.with_name("round228_yang2010_figure4_locked_safe_preflight_v2_contract.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    required = [PDF, FACTS, DRIVER, CONTRACT]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("Missing preflight input: " + "; ".join(missing))
    facts_before = sha256(FACTS)
    OUT.mkdir(parents=True)

    document = pdfium.PdfDocument(str(PDF))
    try:
        page = document[5]
        bitmap = page.render(scale=250 / 72)
        image = bitmap.to_pil().convert("RGB")
    finally:
        document.close()
    width, height = image.size
    mask_start_x = int(round(width * 0.70))
    ImageDraw.Draw(image).rectangle((mask_start_x, 0, width, height), fill=(0, 0, 0))
    preview = OUT / "page6_right30pct_masked.png"
    image.save(preview, format="PNG", optimize=True)

    facts_after = sha256(FACTS)
    if facts_after != facts_before:
        raise RuntimeError("Canonical facts changed during read-only preflight")
    summary = {
        "round": 228,
        "version": 2,
        "supersedes_v1_status": "v1_failed_before_rendering_poppler_stdout_wrapper",
        "v1_failed_path_retained": str(V1_FAILED),
        "pdf_page_rendered": 6,
        "render_dpi": 250,
        "image_width_px": width,
        "image_height_px": height,
        "mask_start_x_px": mask_start_x,
        "masked_fraction_from_right": 0.30,
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
                "facts": {"path": str(FACTS), "sha256": facts_after},
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
                "masked_preview": {"path": str(preview), "sha256": sha256(preview)},
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
