"""Inventory the official Upper Indus supplementary DOCX without extracting data."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
SOURCE = PROJECT / "validation_data/sources/external_hma_temperature_upper_indus_mxd_v1/upper_indus_S0048969724075508_mmc1.docx"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_upper_indus_supplement_preflight_v2"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_text(cell: ET.Element) -> str:
    return "".join(node.text or "" for node in cell.findall(".//w:t", NS)).strip()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = sha256(SOURCE)
    with zipfile.ZipFile(SOURCE) as archive:
        names = archive.namelist()
        root = ET.fromstring(archive.read("word/document.xml"))
        media = [
            {"name": name, "size_bytes": archive.getinfo(name).file_size}
            for name in names
            if name.startswith("word/media/")
        ]

    paragraph_text = []
    for paragraph in root.findall(".//w:p", NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()
        if text and len(text) <= 240:
            paragraph_text.append(text)

    tables = []
    for table_index, table in enumerate(root.findall(".//w:tbl", NS), start=1):
        rows = table.findall("./w:tr", NS)
        row_lengths = [len(row.findall("./w:tc", NS)) for row in rows]
        first = set(range(min(4, len(rows))))
        last = set(range(max(0, len(rows) - 4), len(rows)))
        previews = []
        for row_index in sorted(first | last):
            cells = rows[row_index].findall("./w:tc", NS)
            previews.append({"row_index_1based": row_index + 1, "cells": [cell_text(cell) for cell in cells]})
        tables.append(
            {
                "table_index_1based": table_index,
                "row_count": len(rows),
                "min_column_count": min(row_lengths) if row_lengths else 0,
                "max_column_count": max(row_lengths) if row_lengths else 0,
                "safe_first_last_row_previews": previews,
            }
        )

    inventory = {
        "source": str(SOURCE),
        "source_sha256": before,
        "table_count": len(tables),
        "tables": tables,
        "media": media,
        "short_paragraph_text": paragraph_text,
        "series_values_extracted": False,
        "locked_1921_1939_values_accessed": False,
        "v1_failure_fixed": "set union expression only",
    }
    (OUTPUT / "preflight_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    after = sha256(SOURCE)
    if before != after:
        raise RuntimeError("Source hash changed")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
