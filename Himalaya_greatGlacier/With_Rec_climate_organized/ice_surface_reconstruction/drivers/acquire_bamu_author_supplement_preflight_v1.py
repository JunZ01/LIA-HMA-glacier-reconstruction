from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "temperature_bamu_author_supplement_preflight_v1"
)
SOURCE_DIR = OUTPUT / "source"
SOURCE = SOURCE_DIR / "bamu_co_author_supplement_mmc1.docx"
URL = "https://ars.els-cdn.com/content/image/1-s2.0-S0031018223004418-mmc1.docx"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    SOURCE_DIR.mkdir()
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, SOURCE.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    if SOURCE.stat().st_size < 1000:
        raise RuntimeError("download-too-small")
    with zipfile.ZipFile(SOURCE) as archive:
        members = sorted(archive.namelist())
    embeddings = [name for name in members if name.startswith("word/embeddings/")]
    spreadsheet_members = [name for name in members if name.lower().endswith((".xlsx", ".xls", ".csv"))]
    summary = {
        "experiment": "temperature_bamu_author_supplement_preflight_v1",
        "status": "container_inventory_complete_document_content_not_decoded",
        "paper_doi": "10.1016/j.palaeo.2023.111823",
        "source_url": URL,
        "source_path": str(SOURCE),
        "source_size_bytes": SOURCE.stat().st_size,
        "source_sha256": sha256(SOURCE),
        "zip_member_count": len(members),
        "embedded_object_members": embeddings,
        "spreadsheet_members": spreadsheet_members,
        "document_xml_opened": False,
        "numeric_cells_decoded": False,
        "sealed_period_values_used": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "container_members.json").write_text(json.dumps(members, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Bamu Co author supplement preflight\n\n"
        f"The official Elsevier DOCX was downloaded to E and inventoried as an OOXML container. It contains {len(embeddings)} embedded object members and {len(spreadsheet_members)} direct spreadsheet members. "
        "The document XML and all numeric content were deliberately left unopened, so no sealed-period value was decoded.\n\n"
        "Production was not changed, EnKF and OGGM were not run, and 20CRv3 was not accessed.\n",
        encoding="utf-8",
    )
    manifest = {
        "source": {"url": URL, "path": str(SOURCE), "sha256": sha256(SOURCE)},
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
