from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


DOI = "10.1016/j.quascirev.2025.109266"
SUPPLEMENT_URL = (
    "https://ars.els-cdn.com/content/image/"
    "1-s2.0-S0277379125000861-mmc1.docx"
)
LOCKED_YEAR_TEXT = {str(year) for year in range(1921, 1940)}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


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
        metadata = {
            "requested_url": url,
            "final_url": response.geturl(),
            "content_type": response.headers.get("Content-Type", ""),
            "size_bytes": len(payload),
        }
    path.write_bytes(payload)
    metadata["sha256"] = sha256(path)
    return metadata


def text_of(element: ET.Element) -> str:
    return " ".join(
        "".join(node.text or "" for node in element.iter(f"{W}t")).split()
    )


def classify_tables(document_xml: bytes) -> tuple[list[dict[str, object]], int]:
    root = ET.fromstring(document_xml)
    inventory = []
    locked_rows_discarded = 0
    for table_index, table in enumerate(root.iter(f"{W}tbl"), start=1):
        rows = list(table.findall(f"{W}tr"))
        column_count = max(
            (len(row.findall(f"{W}tc")) for row in rows), default=0
        )
        safe_years: list[int] = []
        nonnumeric_header = ""
        for row_index, row in enumerate(rows):
            cells = [text_of(cell) for cell in row.findall(f"{W}tc")]
            first = cells[0].strip() if cells else ""
            normalized = re.sub(r"^(?:AD|CE)\s*", "", first, flags=re.I).strip()
            if normalized in LOCKED_YEAR_TEXT:
                locked_rows_discarded += 1
                continue
            if re.fullmatch(r"-?\d{1,4}", normalized):
                safe_years.append(int(normalized))
            elif row_index < 3 and first and not nonnumeric_header:
                nonnumeric_header = " | ".join(cells)[:500]
        unique_years = sorted(set(safe_years))
        annual_like = len(unique_years) >= 20
        inventory.append(
            {
                "table_1based": table_index,
                "row_count": len(rows),
                "maximum_column_count": column_count,
                "nonnumeric_header_only": nonnumeric_header,
                "allowed_first_column_year_count": len(unique_years),
                "allowed_first_column_year_min": min(unique_years) if unique_years else None,
                "allowed_first_column_year_max": max(unique_years) if unique_years else None,
                "annual_like_first_column": annual_like,
            }
        )
    return inventory, locked_rows_discarded


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_169_huang2025_mxd_public_data_recovery_v1"
    )
    raw = output / "raw_unopened"
    raw.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    source = raw / "huang2025_supplement_mmc1.docx"
    source_metadata = download(SUPPLEMENT_URL, source)
    if not zipfile.is_zipfile(source):
        raise RuntimeError("supplement-not-ooxml-zip")

    with zipfile.ZipFile(source) as archive:
        members = archive.namelist()
        document_xml = archive.read("word/document.xml")
    inventory, locked_rows = classify_tables(document_xml)
    embedded_members = sorted(
        name for name in members if name.startswith("word/embeddings/")
    )
    media_members = sorted(
        name for name in members if name.startswith("word/media/")
    )
    annual_tables = [row for row in inventory if row["annual_like_first_column"]]
    annual_numeric_found = bool(annual_tables or embedded_members)
    decision = (
        "advance_huang2025_annual_series_to_source_identity_audit"
        if annual_numeric_found
        else "close_huang2025_anonymous_public_numeric_recovery"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "doi": DOI,
        "source": source_metadata,
        "ooxml_member_count": len(members),
        "table_inventory": inventory,
        "annual_like_table_count": len(annual_tables),
        "embedded_members": embedded_members,
        "media_member_count": len(media_members),
        "locked_rows_discarded_before_numeric_conversion_or_output": locked_rows,
        "locked_1921_1939_numeric_values_converted": False,
        "annual_values_exported": False,
        "figures_rendered_or_digitized": False,
        "annual_numeric_series_publicly_recovered": annual_numeric_found,
        "decision": decision,
        "paleo_trajectory_inspected": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    (output / "table_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 169 Huang 2025 MXD Public-Data Recovery\n\n"
        f"- OOXML tables: `{len(inventory)}`.\n"
        f"- Annual-like tables: `{len(annual_tables)}`.\n"
        f"- Embedded members: `{len(embedded_members)}`.\n"
        f"- Media members: `{len(media_members)}`.\n"
        f"- Locked rows discarded: `{locked_rows}`.\n"
        f"- Annual values exported: `false`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
