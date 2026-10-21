"""Audit Round 121 public downloads without opening candidate data values."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def workbook_structure(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{NS_PKG_REL}}}Relationship")
        }
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t"))
                for item in root.findall(f"{{{NS_MAIN}}}si")
            ]
        sheets = []
        for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
            target = targets[sheet.attrib[f"{{{NS_REL}}}id"]].replace("\\", "/")
            if target.startswith("/"):
                target = target.lstrip("/")
            elif not target.startswith("xl/"):
                target = f"xl/{target}"
            root = ET.fromstring(archive.read(target))
            dimension = root.find(f"{{{NS_MAIN}}}dimension")
            first_row = root.find(f".//{{{NS_MAIN}}}row")
            text_headers = []
            numeric_first_row_cells_skipped = 0
            if first_row is not None:
                for cell in first_row.findall(f"{{{NS_MAIN}}}c"):
                    cell_type = cell.attrib.get("t", "n")
                    ref = cell.attrib.get("r", "")
                    if cell_type == "s":
                        node = cell.find(f"{{{NS_MAIN}}}v")
                        if node is not None:
                            text_headers.append({"cell": ref, "text": strings[int(node.text or "0")]})
                    elif cell_type == "inlineStr":
                        inline = cell.find(f"{{{NS_MAIN}}}is")
                        text = "" if inline is None else "".join(
                            node.text or "" for node in inline.iter(f"{{{NS_MAIN}}}t")
                        )
                        text_headers.append({"cell": ref, "text": text})
                    else:
                        numeric_first_row_cells_skipped += 1
            sheets.append(
                {
                    "sheet": sheet.attrib["name"],
                    "dimension": None if dimension is None else dimension.attrib.get("ref"),
                    "first_row_text_cells": text_headers,
                    "numeric_first_row_cells_skipped": numeric_first_row_cells_skipped,
                }
            )
    return {"path": str(path), "sheets": sheets, "candidate_numeric_values_decoded": 0}


def noaa_text_structure(path: Path) -> dict:
    metadata = []
    column_header = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip("\r\n")
            if stripped.startswith("#"):
                if any(
                    token in stripped
                    for token in (
                        "Study_Name:",
                        "Investigators:",
                        "DOI:",
                        "Site_Name:",
                        "Earliest_Year:",
                        "Most_Recent_Year:",
                        "## ",
                    )
                ):
                    metadata.append(stripped)
                continue
            if not stripped.strip():
                continue
            if re.search(r"[A-Za-z_]", stripped):
                column_header = stripped
            break
    if column_header is None:
        raise RuntimeError(f"no-column-header-before-data:{path}")
    return {
        "path": str(path),
        "selected_metadata_lines": metadata,
        "column_header": column_header,
        "candidate_numeric_values_decoded": 0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_proxy_spatial_gap_discovery_20260818_v1"
    )
    raw = output / "raw_unopened"
    files = {
        "eastern_tibet_814y_temperature": raw / "eastern_tibet_814y_temperature_reconstruction.xlsx",
        "northwestern_qtp_845_2023_precipitation": raw / "northwestern_qtp_845_2023_precipitation_REC.xlsx",
        "sun2026_tibet_temperature": raw / "sun2026_tibet_temperature_reconstruction.txt",
        "tibet2019_temperature": raw / "tibet2019_temperature_reconstructions.txt",
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    expected = {
        "eastern_tibet_814y_temperature": (
            "sha256",
            "6eeb2b65c512ad718f5e848a7e778b74df4f3443185aeba4dfd62fb2e5780812",
        ),
        "northwestern_qtp_845_2023_precipitation": (
            "md5",
            "1cca4d3ec93668e6ecc45eaaef1dba02",
        ),
    }
    checksums = {}
    for name, path in files.items():
        algorithm, expected_value = expected.get(name, ("sha256", None))
        observed = digest(path, algorithm)
        checksums[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "algorithm": algorithm,
            "observed": observed,
            "expected": expected_value,
            "matches_repository": expected_value is None or observed == expected_value,
        }
        if expected_value is not None and observed != expected_value:
            raise RuntimeError(f"repository-checksum-failed:{name}")
    structures = {
        "eastern_tibet_814y_temperature": workbook_structure(
            files["eastern_tibet_814y_temperature"]
        ),
        "northwestern_qtp_845_2023_precipitation": workbook_structure(
            files["northwestern_qtp_845_2023_precipitation"]
        ),
        "sun2026_tibet_temperature": noaa_text_structure(files["sun2026_tibet_temperature"]),
        "tibet2019_temperature": noaa_text_structure(files["tibet2019_temperature"]),
    }
    inventory = [
        {
            "candidate": "eastern_tibet_814y_temperature",
            "climate_variable": "temperature",
            "archive_input_available": "no",
            "acquired_role": "external_low_frequency_benchmark",
            "reason": "repository supplies reconstructed June-September temperature only",
        },
        {
            "candidate": "sun2026_tibet_temperature",
            "climate_variable": "temperature",
            "archive_input_available": "no",
            "acquired_role": "external_low_frequency_benchmark",
            "reason": "NOAA file supplies reconstructed JJA Tmax, not underlying tree-ring d18O",
        },
        {
            "candidate": "tibet2019_temperature",
            "climate_variable": "temperature",
            "archive_input_available": "no",
            "acquired_role": "external_low_frequency_benchmark",
            "reason": "NOAA file supplies NETP annual and SETP JJA reconstructed temperature",
        },
        {
            "candidate": "northwestern_qtp_845_2023_precipitation",
            "climate_variable": "precipitation",
            "archive_input_available": "pending_header_audit",
            "acquired_role": "pending_structure_classification",
            "reason": "repository description mentions an ensemble chronology and reconstruction",
        },
    ]
    write_csv(output / "downloaded_candidate_role_inventory.csv", inventory)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "checksums": checksums,
        "structures": structures,
        "candidate_numeric_values_decoded": 0,
        "locked_1921_1939_candidate_values_decoded": 0,
        "original_sources_modified": False,
    }
    (output / "download_structure_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
