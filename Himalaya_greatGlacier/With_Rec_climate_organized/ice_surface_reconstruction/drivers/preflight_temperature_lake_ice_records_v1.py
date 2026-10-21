"""Schema-only preflight for new lake and ice temperature evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from docx import Document
from openpyxl import load_workbook


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
SOURCE = ROOT / "validation_data/sources/external_hma_temperature_candidates_lake_ice_v1"
OUTPUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_temperature_lake_ice_external_preflight_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def inspect_workbook(path: Path) -> dict[str, object]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for sheet in workbook.worksheets:
        first_values = []
        skipped_locked = 0
        text_rows = []
        nonempty = 0
        for row_index, row in enumerate(sheet.iter_rows(), start=1):
            if not row:
                continue
            first = as_number(row[0].value)
            if first is not None and 1921 <= int(first) <= 1939 and abs(first - int(first)) < 1e-9:
                skipped_locked += 1
                continue
            values = [cell.value for cell in row]
            if not any(value is not None for value in values):
                continue
            nonempty += 1
            if first is not None:
                first_values.append(first)
            if len(text_rows) < 20:
                labels = [str(value).strip() for value in values if isinstance(value, str) and value.strip()]
                if labels:
                    text_rows.append({"row": row_index, "text": labels[:20]})
        sheets.append(
            {
                "sheet": sheet.title,
                "max_row": int(sheet.max_row),
                "max_column": int(sheet.max_column),
                "nonempty_rows_excluding_locked": nonempty,
                "first_column_numeric_min_excluding_locked": min(first_values) if first_values else None,
                "first_column_numeric_max_excluding_locked": max(first_values) if first_values else None,
                "first_column_numeric_count_excluding_locked": len(first_values),
                "locked_year_rows_skipped_before_remaining_cell_access": skipped_locked,
                "text_header_preview": text_rows,
            }
        )
    workbook.close()
    return {"file": path.name, "kind": "xlsx", "sheets": sheets}


def inspect_docx(path: Path) -> dict[str, object]:
    document = Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    tables = []
    for index, table in enumerate(document.tables, start=1):
        header = [cell.text.strip() for cell in table.rows[0].cells] if table.rows else []
        tables.append(
            {
                "table": index,
                "row_count": len(table.rows),
                "column_count": len(table.columns),
                "header_text": header,
                "body_values_read": False,
            }
        )
    return {
        "file": path.name,
        "kind": "docx",
        "paragraph_count": len(paragraphs),
        "paragraph_text_preview": paragraphs[:50],
        "table_count": len(tables),
        "tables": tables,
        "inline_shape_count": len(document.inline_shapes),
        "table_body_values_read": False,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    if not contract.exists():
        raise FileNotFoundError("Contract must exist before preflight")
    paths = {
        "chongce": SOURCE / "chongce_zenodo_5766235.xlsx",
        "jiang_s3": SOURCE / "jiang_co_S2095927323002943_mmc4.xlsx",
        "bamu": SOURCE / "bamu_co_S0031018223004418_mmc1.docx",
        "download_manifest": SOURCE / "source_manifest.json",
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}
    inventory = [
        inspect_workbook(paths["chongce"]),
        inspect_workbook(paths["jiang_s3"]),
        inspect_docx(paths["bamu"]),
    ]
    after = {name: sha256(path) for name, path in paths.items()}
    result = {
        "status": "complete",
        "scope": "schema_and_chronology_preflight_only",
        "inventory": inventory,
        "scientific_values_reported": False,
        "scientific_score_calculated": False,
        "docx_table_body_values_read": False,
        "source_hashes_unchanged": before == after,
    }
    (OUTPUT / "preflight_inventory.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]}
                    for name in paths
                },
                "all_input_hashes_unchanged": before == after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
