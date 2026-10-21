from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF = re.compile(r"([A-Z]+)([0-9]+)")


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


def normalize_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def rwl_identity(path: Path) -> dict[str, Any]:
    samples: set[str] = set()
    starts: list[int] = []
    segment_count = 0
    for raw_line in path.read_bytes().splitlines():
        prefix = raw_line[:12].decode("ascii", errors="ignore")
        if len(prefix) < 12:
            continue
        sample = prefix[:8].strip()
        year_token = prefix[8:12].strip()
        if not sample or not re.fullmatch(r"-?[0-9]+", year_token):
            continue
        samples.add(sample)
        starts.append(int(year_token))
        segment_count += 1
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "sample_count": len(samples),
        "segment_count": segment_count,
        "minimum_segment_start_year": min(starts) if starts else None,
        "maximum_segment_start_year": max(starts) if starts else None,
        "sample_ids": sorted(samples),
        "normalized_sample_ids": sorted({normalize_id(value) for value in samples}),
        "ring_width_tokens_decoded": False,
    }


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(name))
    values: list[str] = []
    for item in root.findall(f"{{{NS_MAIN}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t")))
    return values


def workbook_structure(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(f"{{{NS_PKG_REL}}}Relationship")
        }
        strings = shared_strings(archive)
        sheets: list[dict[str, Any]] = []
        for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
            title = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{NS_REL}}}id"]
            target = targets[rel_id].replace("\\", "/")
            if target.startswith("/"):
                sheet_path = target.lstrip("/")
            elif target.startswith("xl/"):
                sheet_path = target
            else:
                sheet_path = f"xl/{target}"
            xml = ET.fromstring(archive.read(sheet_path))
            dimension = xml.find(f"{{{NS_MAIN}}}dimension")
            type_counts: Counter[str] = Counter()
            cell_count = 0
            row_count = 0
            header_strings: list[dict[str, str]] = []
            for row in xml.findall(f".//{{{NS_MAIN}}}row"):
                row_count += 1
                row_number = int(row.attrib.get("r", "0") or 0)
                for cell in row.findall(f"{{{NS_MAIN}}}c"):
                    cell_count += 1
                    cell_type = cell.attrib.get("t", "numeric")
                    type_counts[cell_type] += 1
                    if row_number > 10 or cell_type not in {"s", "inlineStr"}:
                        continue
                    value = ""
                    if cell_type == "s":
                        node = cell.find(f"{{{NS_MAIN}}}v")
                        if node is not None and node.text is not None:
                            index = int(node.text)
                            value = strings[index] if 0 <= index < len(strings) else ""
                    else:
                        value = "".join(
                            node.text or "" for node in cell.iter(f"{{{NS_MAIN}}}t")
                        )
                    if value:
                        header_strings.append(
                            {"cell": cell.attrib.get("r", ""), "text": value}
                        )
            sheets.append(
                {
                    "title": title,
                    "xml_path": sheet_path,
                    "dimension": dimension.attrib.get("ref", "") if dimension is not None else "",
                    "row_element_count": row_count,
                    "cell_count": cell_count,
                    "cell_type_counts": dict(type_counts),
                    "first_10_row_string_cells": header_strings,
                    "numeric_cell_values_read": False,
                }
            )
        return {
            "filename": path.name,
            "sha256": sha256(path),
            "zip_member_count": len(names),
            "shared_string_count": len(strings),
            "sheet_count": len(sheets),
            "sheets": sheets,
            "numeric_cell_values_read": False,
        }


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    source = loop / "round_178_mendeley_east_hma_temperature_candidates_v1"
    raw = source / "raw_unopened"
    output = loop / "round_179_mendeley_temperature_structure_identity_v1"
    output.mkdir(parents=True, exist_ok=False)
    source_files = sorted(path for path in raw.rglob("*") if path.is_file())
    hashes_before = {str(path): sha256(path) for path in source_files}

    xlsx_files = [path for path in source_files if path.suffix.lower() == ".xlsx"]
    rwl_files = [path for path in source_files if path.suffix.lower() != ".xlsx"]
    workbook_results = [workbook_structure(path) for path in xlsx_files]
    rwl_results = [rwl_identity(path) for path in rwl_files]

    pairwise_overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(rwl_results):
        left_ids = set(left["normalized_sample_ids"])
        for right in rwl_results[left_index + 1 :]:
            overlap = sorted(left_ids & set(right["normalized_sample_ids"]))
            pairwise_overlaps.append(
                {
                    "left": left["filename"],
                    "right": right["filename"],
                    "overlap_count": len(overlap),
                    "overlap_ids": overlap,
                }
            )

    existing_rwl_matches: list[dict[str, Any]] = []
    candidate_union = {
        value for item in rwl_results for value in item["normalized_sample_ids"]
    }
    for existing in root.rglob("*.rwl"):
        if source in existing.parents:
            continue
        identity = rwl_identity(existing)
        overlap = sorted(candidate_union & set(identity["normalized_sample_ids"]))
        if overlap:
            existing_rwl_matches.append(
                {
                    "existing_path": str(existing),
                    "existing_sha256": identity["sha256"],
                    "overlap_count": len(overlap),
                    "overlap_ids": overlap,
                }
            )

    structure = {
        "workbooks": workbook_results,
        "raw_tree_ring_files": rwl_results,
        "candidate_pairwise_sample_overlaps": pairwise_overlaps,
        "existing_project_rwl_matches": existing_rwl_matches,
    }
    (output / "structure_identity.json").write_text(
        json.dumps(structure, indent=2, ensure_ascii=True), encoding="ascii"
    )
    hashes_after = {str(path): sha256(path) for path in source_files}
    if hashes_before != hashes_after:
        raise RuntimeError("round178-source-hash-changed")
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "workbook_count": len(workbook_results),
        "raw_tree_ring_file_count": len(rwl_results),
        "raw_tree_ring_sample_count_total": sum(
            int(item["sample_count"]) for item in rwl_results
        ),
        "candidate_pairwise_overlap_count_total": sum(
            int(item["overlap_count"]) for item in pairwise_overlaps
        ),
        "existing_project_rwl_match_file_count": len(existing_rwl_matches),
        "numeric_spreadsheet_values_read": False,
        "ring_width_tokens_decoded": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "decision": "use_sheet_structure_to_contract_targeted_boundary_safe_value_audit",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 179 Mendeley Temperature Structure and Identity\n\n"
        f"- Workbooks structurally inventoried: `{len(workbook_results)}`.\n"
        f"- Raw tree-ring files inventoried: `{len(rwl_results)}`.\n"
        "- Numeric spreadsheet values read: `false`.\n"
        "- Ring-width tokens decoded: `false`.\n"
        "- Locked-period values accessed: `false`.\n\n"
        "Decision: `use_sheet_structure_to_contract_targeted_boundary_safe_value_audit`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
