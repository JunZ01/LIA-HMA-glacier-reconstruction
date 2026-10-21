from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image


EXPECTED_SHA256 = "8ffc24620f1189d4ed49e441dda4332552a93ec744abbd1cf0e13823d55462c8"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / (
        "round_247_npl065_official_supplement_acquisition_v2/raw_unopened/"
        "1-s2.0-S1125786520301387-mmc1.docx"
    )
    output = loop / "round_249_npl065_supplement_modern_figure_audit_v1"
    figures = output / "figures_png"
    figures.mkdir(parents=True, exist_ok=False)
    source_bytes = source.read_bytes()
    if digest(source_bytes) != EXPECTED_SHA256:
        raise RuntimeError("supplement-hash-changed")
    records = []
    with zipfile.ZipFile(BytesIO(source_bytes), "r") as archive:
        media_names = sorted(name for name in archive.namelist() if name.startswith("word/media/"))
        if len(media_names) != 5:
            raise RuntimeError(f"media-count-changed:{len(media_names)}")
        for index, media_name in enumerate(media_names, start=1):
            payload = archive.read(media_name)
            with Image.open(BytesIO(payload)) as image:
                converted = image.convert("RGB")
                target = figures / f"figure_s{index}.png"
                converted.save(target, format="PNG", optimize=True)
                records.append({
                    "figure": f"S{index}",
                    "source_part": media_name,
                    "source_bytes": len(payload),
                    "source_sha256": digest(payload),
                    "source_format": image.format,
                    "pixel_width": image.width,
                    "pixel_height": image.height,
                    "png_path": str(target),
                    "png_sha256": digest(target.read_bytes()),
                })
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "figures": records,
        "figure_time_scope_from_captions": {
            "S1": "site_and_samples_no_time_series",
            "S2": "1975-2013",
            "S3": "1960-2013",
            "S4": "1960-2013",
            "S5": "1967-2013",
        },
        "locked_1921_1939_values_accessed": False,
        "paleo_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_docx_unchanged": digest(source.read_bytes()) == EXPECTED_SHA256,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 249 NPL065 supplement modern-figure audit\n\n"
        "Five official figures were extracted losslessly from the fixed-hash DOCX and converted to PNG for visual inspection.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
