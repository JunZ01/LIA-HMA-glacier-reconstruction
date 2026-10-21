from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DOI = "10.1007/s11676-026-02104-0"
ARTICLE_URL = f"https://link.springer.com/article/{DOI}"
SUPPLEMENT_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1007%2Fs11676-026-02104-0/MediaObjects/"
    "11676_2026_2104_MOESM1_ESM.pdf"
)
LOCKED_YEARS = tuple(str(year) for year in range(1921, 1940))


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


def download(url: str, path: Path) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
        final_url = response.geturl()
    path.write_bytes(payload)
    return {
        "requested_url": url,
        "final_url": final_url,
        "content_type": content_type,
        "size_bytes": len(payload),
        "sha256": sha256(path),
    }


def embedded_files(reader) -> list[str]:
    names = []
    attachments = getattr(reader, "attachments", None)
    if attachments:
        names.extend(str(name) for name in attachments)
    root = reader.trailer.get("/Root", {})
    name_tree = root.get("/Names") if hasattr(root, "get") else None
    if name_tree and hasattr(name_tree, "get") and name_tree.get("/EmbeddedFiles"):
        names.append("pdf-name-tree-present")
    return sorted(set(names))


def sanitized_inventory(reader) -> tuple[list[str], int]:
    retained: list[str] = []
    locked_lines_discarded = 0
    patterns = (
        re.compile(r"^(?:table|fig(?:ure)?|data availability|supplementary)", re.I),
        re.compile(r"(?:chronolog|reconstruct|dataset|data are|data is|available)", re.I),
    )
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for line in text.splitlines():
            cleaned = " ".join(line.split())
            if not cleaned:
                continue
            if any(year in cleaned for year in LOCKED_YEARS):
                locked_lines_discarded += 1
                continue
            if any(pattern.search(cleaned) for pattern in patterns):
                retained.append(f"p{page_number}: {cleaned[:500]}")
    return retained, locked_lines_discarded


def main() -> int:
    from pypdf import PdfReader

    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_168_central_hma_mxd_public_data_recovery_v1"
    )
    raw = output / "raw_unopened"
    raw.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)

    files = {
        "article_html": download(ARTICLE_URL, raw / "article.html"),
        "supplement_pdf": download(SUPPLEMENT_URL, raw / "supplementary_file1.pdf"),
    }
    pdf_path = raw / "supplementary_file1.pdf"
    if not pdf_path.read_bytes().startswith(b"%PDF"):
        raise RuntimeError("supplement-not-pdf")
    reader = PdfReader(pdf_path)
    headings, discarded = sanitized_inventory(reader)
    attachments = embedded_files(reader)

    article_text = (raw / "article.html").read_text(encoding="utf-8", errors="ignore")
    links = sorted(set(re.findall(r"https?://[^\"'<> ]+", article_text)))
    numeric_extensions = (".csv", ".txt", ".xlsx", ".xls", ".rwl", ".crn", ".zip", ".nc")
    public_numeric_links = [
        link for link in links if link.lower().split("?", 1)[0].endswith(numeric_extensions)
    ]
    annual_numeric_public = bool(attachments or public_numeric_links)
    decision = (
        "advance_public_annual_series_to_identity_audit"
        if annual_numeric_public
        else "retain_as_metadata_only_external_evidence"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "doi": DOI,
        "files": files,
        "pdf_page_count": len(reader.pages),
        "embedded_files": attachments,
        "public_numeric_links": public_numeric_links,
        "sanitized_heading_and_availability_inventory": headings,
        "locked_lines_discarded_before_output_or_numeric_conversion": discarded,
        "locked_1921_1939_numeric_values_converted": False,
        "curves_digitized_or_ocr_run": False,
        "annual_numeric_series_publicly_recovered": annual_numeric_public,
        "decision": decision,
        "paleo_comparison_run": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    result = output / "FINAL_RESULT.md"
    result.write_text(
        "# Round 168 Central-HMA MXD Public-Data Recovery\n\n"
        f"- Supplement pages: `{len(reader.pages)}`.\n"
        f"- Embedded files: `{len(attachments)}`.\n"
        f"- Public numeric links: `{len(public_numeric_links)}`.\n"
        f"- Locked text lines discarded: `{discarded}`.\n"
        f"- Annual numeric series recovered: `{str(annual_numeric_public).lower()}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
