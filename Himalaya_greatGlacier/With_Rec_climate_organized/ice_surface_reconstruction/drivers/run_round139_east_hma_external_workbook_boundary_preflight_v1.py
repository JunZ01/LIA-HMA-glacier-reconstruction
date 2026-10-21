"""Safely export allowed rows from two unopened hydroclimate workbooks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_RE = re.compile(r"([A-Z]+)([0-9]+)")
LOCKED_START = 1921
LOCKED_END = 1939


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def column_index(reference: str) -> int:
    match = CELL_RE.fullmatch(reference)
    if match is None:
        raise RuntimeError(f"invalid-cell-reference:{reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def first_numeric_year(row: ET.Element) -> int | None:
    cells = row.findall(f"{{{MAIN_NS}}}c")
    if not cells:
        return None
    cell = min(cells, key=lambda item: column_index(item.attrib["r"]))
    if cell.attrib.get("t", "n") not in {"n", ""}:
        return None
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return None
    try:
        number = float(value.text)
    except ValueError:
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9 or not 500 <= rounded <= 2500:
        return None
    return rounded


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(name))
    output = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        output.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
    return output


def decode_cell(cell: ET.Element, strings: list[str]) -> object:
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t"))
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return ""
    raw = value.text
    if cell_type == "s":
        return strings[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer():
        return int(number)
    return number


def workbook_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation_map = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    output = []
    sheets = workbook.find(f"{{{MAIN_NS}}}sheets")
    if sheets is None:
        raise RuntimeError("workbook-has-no-sheets")
    for sheet in sheets:
        relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
        target = PurePosixPath(relation_map[relation_id])
        if str(target).startswith("/"):
            xml_path = str(target).lstrip("/")
        else:
            xml_path = str(PurePosixPath("xl") / target)
        output.append((sheet.attrib["name"], xml_path))
    return output


def rows_from_xml(xml_bytes: bytes):
    for _, element in ET.iterparse(io.BytesIO(xml_bytes), events=("end",)):
        if element.tag == f"{{{MAIN_NS}}}row":
            yield element
            element.clear()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def inspect_sheet(
    xml_bytes: bytes,
    strings: list[str],
    workbook_label: str,
    sheet_name: str,
    output: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    year_min = None
    year_max = None
    year_rows = 0
    locked_rows = 0
    locked_non_numeric_types = 0
    maximum_column = -1
    predata_rows: list[ET.Element] = []
    first_year_seen = False

    for row in rows_from_xml(xml_bytes):
        cells = row.findall(f"{{{MAIN_NS}}}c")
        if cells:
            maximum_column = max(
                maximum_column,
                max(column_index(cell.attrib["r"]) for cell in cells),
            )
        year = first_numeric_year(row)
        if year is None:
            if not first_year_seen:
                predata_rows.append(ET.fromstring(ET.tostring(row)))
            continue
        first_year_seen = True
        year_rows += 1
        year_min = year if year_min is None else min(year_min, year)
        year_max = year if year_max is None else max(year_max, year)
        if LOCKED_START <= year <= LOCKED_END:
            locked_rows += 1
            cells_sorted = sorted(cells, key=lambda item: column_index(item.attrib["r"]))
            locked_non_numeric_types += sum(
                cell.attrib.get("t", "n") in {"s", "inlineStr", "str"}
                for cell in cells_sorted[1:]
            )
    if locked_non_numeric_types:
        raise RuntimeError(
            f"locked-row-nonnumeric-cell-type:{workbook_label}:{sheet_name}:"
            f"{locked_non_numeric_types}"
        )
    if not year_rows:
        raise RuntimeError(f"no-year-rows:{workbook_label}:{sheet_name}")

    header_row = max(
        predata_rows,
        key=lambda row: len(row.findall(f"{{{MAIN_NS}}}c")),
        default=None,
    )
    width = maximum_column + 1
    headers = [f"column_{index + 1}" for index in range(width)]
    header_source_row = None
    if header_row is not None:
        header_source_row = int(header_row.attrib.get("r", "0"))
        for cell in header_row.findall(f"{{{MAIN_NS}}}c"):
            index = column_index(cell.attrib["r"])
            decoded = str(decode_cell(cell, strings)).strip()
            if decoded:
                headers[index] = decoded
    seen: dict[str, int] = {}
    for index, header in enumerate(headers):
        normalized = header or f"column_{index + 1}"
        seen[normalized] = seen.get(normalized, 0) + 1
        headers[index] = (
            normalized
            if seen[normalized] == 1
            else f"{normalized}_{seen[normalized]}"
        )

    allowed_rows: list[list[object]] = []
    for row in rows_from_xml(xml_bytes):
        year = first_numeric_year(row)
        if year is None:
            continue
        if LOCKED_START <= year <= LOCKED_END:
            continue
        if not ((1000 <= year <= 1920) or (1940 <= year <= 2000)):
            continue
        values: list[object] = [""] * width
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            values[column_index(cell.attrib["r"])] = decode_cell(cell, strings)
        allowed_rows.append(values)
    if not allowed_rows:
        raise RuntimeError(f"no-allowed-rows:{workbook_label}:{sheet_name}")

    csv_name = f"{safe_name(workbook_label)}__{safe_name(sheet_name)}__allowed.csv"
    csv_path = output / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(allowed_rows)
    allowed_years = [int(row[0]) for row in allowed_rows]
    inventory = {
        "workbook": workbook_label,
        "sheet": sheet_name,
        "year_row_count": year_rows,
        "source_first_year": year_min,
        "source_last_year": year_max,
        "locked_row_count_skipped_without_value_decode": locked_rows,
        "locked_non_numeric_value_type_count": locked_non_numeric_types,
        "column_count": width,
        "header_source_row": header_source_row,
        "allowed_row_count": len(allowed_rows),
        "allowed_first_year": min(allowed_years),
        "allowed_last_year": max(allowed_years),
        "allowed_csv": csv_name,
    }
    preview = {
        "workbook": workbook_label,
        "sheet": sheet_name,
        "headers": headers,
        "first_three_allowed_rows": allowed_rows[:3],
        "last_three_allowed_rows": allowed_rows[-3:],
    }
    return inventory, preview


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_139_east_hma_external_workbook_boundary_preflight_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round139_east_hma_external_workbook_boundary_preflight_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    raw = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_proxy_discovery_20260818_v1/raw_unopened"
    )
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "salween_brahmaputra": raw / "salween_brahmaputra_precipitation.xlsx",
        "three_rivers": raw / "three_rivers_streamflow_reconstruction.xlsx",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    inventory_rows = []
    previews = []
    for key in ("salween_brahmaputra", "three_rivers"):
        path = paths[key]
        with zipfile.ZipFile(path) as archive:
            sheets = workbook_sheets(archive)
            sheet_bytes = [(name, xml_path, archive.read(xml_path)) for name, xml_path in sheets]
            for sheet_name, _, xml_bytes in sheet_bytes:
                for row in rows_from_xml(xml_bytes):
                    year = first_numeric_year(row)
                    if year is None or not LOCKED_START <= year <= LOCKED_END:
                        continue
                    cells = sorted(
                        row.findall(f"{{{MAIN_NS}}}c"),
                        key=lambda item: column_index(item.attrib["r"]),
                    )
                    if any(
                        cell.attrib.get("t", "n") in {"s", "inlineStr", "str"}
                        for cell in cells[1:]
                    ):
                        raise RuntimeError(
                            f"locked-shared-string-preflight-failed:{key}:{sheet_name}"
                        )
            strings = shared_strings(archive)
            for sheet_name, _, xml_bytes in sheet_bytes:
                inventory, preview = inspect_sheet(
                    xml_bytes, strings, path.name, sheet_name, output
                )
                inventory_rows.append(inventory)
                previews.append(preview)

    inventory_path = output / "sheet_boundary_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)
    (output / "allowed_data_previews.json").write_text(
        json.dumps(previews, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-file-changed")
    summary = {
        "experiment_id": "round_139_east_hma_external_workbook_boundary_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workbook_count": 2,
        "sheet_count": len(inventory_rows),
        "allowed_csv_count": len(inventory_rows),
        "locked_rows_skipped_without_value_decode": int(
            sum(row["locked_row_count_skipped_without_value_decode"] for row in inventory_rows)
        ),
        "locked_non_numeric_value_type_count": int(
            sum(row["locked_non_numeric_value_type_count"] for row in inventory_rows)
        ),
        "locked_1921_1939_values_decoded": 0,
        "frozen_external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
        "decision": "advance_allowed_exports_to_identity_and_role_closure",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    sheet_lines = "\n".join(
        f"- {row['workbook']} / {row['sheet']}: source "
        f"{row['source_first_year']}-{row['source_last_year']}, "
        f"{row['column_count']} columns, {row['allowed_row_count']} allowed rows, "
        f"{row['locked_row_count_skipped_without_value_decode']} locked rows skipped."
        for row in inventory_rows
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 139 Final Result\n\n"
        f"{sheet_lines}\n\n"
        "All locked rows were rejected before non-year values were decoded. "
        "Allowed exports may proceed to a separate identity and scientific-role closure.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(sheet_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
