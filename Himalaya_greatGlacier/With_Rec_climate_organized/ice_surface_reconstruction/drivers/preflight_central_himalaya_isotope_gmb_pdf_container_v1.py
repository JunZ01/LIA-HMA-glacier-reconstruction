from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "central_himalaya_isotope_gmb_pdf_container_preflight_v1"
)
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__)
URL = "https://tc.copernicus.org/articles/15/95/2021/tc-15-95-2021-supplement.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    before = {"facts": sha256(FACTS), "driver": sha256(DRIVER)}
    pdf_path = OUTPUT / "tc-15-95-2021-supplement.pdf"
    request = urllib.request.Request(URL, headers={"User-Agent": "LIA-HMA-climate-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
    pdf_path.write_bytes(payload)

    reader = PdfReader(str(pdf_path))
    attachment_names = sorted(reader.attachments.keys())
    metadata_keys = sorted(str(key) for key in (reader.metadata or {}).keys())
    after = {"facts": sha256(FACTS), "driver": sha256(DRIVER)}
    summary = {
        "experiment": "central_himalaya_isotope_gmb_pdf_container_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "doi": "10.5194/tc-15-95-2021",
        "url": URL,
        "content_type": content_type,
        "size_bytes": len(payload),
        "sha256": sha256(pdf_path),
        "page_count_container_only": len(reader.pages),
        "attachment_names": attachment_names,
        "metadata_keys": metadata_keys,
        "page_text_extracted": False,
        "page_images_rendered": False,
        "annual_values_accessed": False,
        "sealed_1921_1939_accessed": False,
        "source_hashes_unchanged": before == after,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "facts": {"path": str(FACTS), "sha256_before": before["facts"], "sha256_after": after["facts"]},
                "driver": {"path": str(DRIVER), "sha256_before": before["driver"], "sha256_after": after["driver"]},
                "download": {"path": str(pdf_path), "sha256": summary["sha256"], "url": URL},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("Read-only source changed")


if __name__ == "__main__":
    main()
