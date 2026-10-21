from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


URL = (
    "https://ir.xtbg.ac.cn/bitstream/353005/12150/1/"
    "Ring%20widths%20of%20Rhododendron%20shrubs%20reveal%20a%20persistent%20"
    "winter%20warming%20in%20the%20central%20Himalaya.pdf"
)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_242_npl065_paper_method_recovery_v1"
    )
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw_unopened"
    text_dir = output / "extracted_text"
    raw.mkdir()
    text_dir.mkdir()
    inputs = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round242_npl065_paper_method_recovery_v1_contract.md",
        "driver": Path(__file__).resolve(),
    }
    before = {name: sha256(path) for name, path in inputs.items()}

    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        status = response.status
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        last_modified = response.headers.get("Last-Modified", "")
    if status != 200 or not payload.startswith(b"%PDF"):
        raise RuntimeError(f"invalid-pdf-response:{status}:{content_type}:{payload[:16]!r}")
    pdf = raw / "Panthi_2020_Dendrochronologia_125799.pdf"
    pdf.write_bytes(payload)
    reader = PdfReader(pdf)
    page_records = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        path = text_dir / f"page_{page_number:03d}.txt"
        path.write_text(text, encoding="utf-8")
        page_records.append({
            "page": page_number,
            "text_path": str(path),
            "character_count": len(text),
        })
    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("input-hash-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "doi": "10.1016/j.dendro.2020.125799",
        "download": {
            "url": URL,
            "final_url": final_url,
            "http_status": status,
            "content_type": content_type,
            "last_modified": last_modified,
            "pdf_path": str(pdf),
            "size_bytes": pdf.stat().st_size,
            "sha256": sha256(pdf),
            "page_count": len(reader.pages),
        },
        "pages": page_records,
        "project_proxy_or_climate_values_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_inputs_unchanged": True,
    }
    (output / "acquisition_manifest.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 242 NPL065 paper method recovery\n\n"
        f"- PDF pages extracted: `{len(reader.pages)}`.\n"
        f"- PDF SHA256: `{result['download']['sha256']}`.\n"
        "- Method interpretation: pending page-text audit.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
