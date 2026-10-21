from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.io import loadmat

from audit_external_proxy_identity_closure_v1 import (
    COMBINED_SOURCES,
    NS_MAIN,
    NS_PKG_REL,
    NS_REL,
    OUTPUT,
    RAW,
    sha256,
)


SAFE_ROW_SELECTIONS = {
    "hengduan_upload_information.xlsx": {
        "Sampling sites information": range(1, 7),
    },
    "haizi_temperature_supplement.xlsx": {
        "Sheet1": range(1, 6),
    },
    "salween_brahmaputra_precipitation.xlsx": {
        "Streamflow": range(1, 6),
        "Biru reconstruction": range(1, 6),
        "Jiali reconstruction": range(1, 6),
    },
}


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall(f"{{{NS_MAIN}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t")))
    return values


def sheet_targets(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{NS_PKG_REL}}}Relationship")
    }
    result = {}
    for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
        target = targets[sheet.attrib[f"{{{NS_REL}}}id"]].replace("\\", "/")
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = f"xl/{target}"
        result[sheet.attrib["name"]] = target
    return result


def selected_rows(path: Path, selections: dict[str, range]) -> list[dict]:
    output = []
    with zipfile.ZipFile(path) as archive:
        strings = shared_strings(archive)
        targets = sheet_targets(archive)
        for sheet_name, safe_rows in selections.items():
            safe_set = set(safe_rows)
            with archive.open(targets[sheet_name]) as stream:
                for event, row in ET.iterparse(stream, events=("end",)):
                    if row.tag != f"{{{NS_MAIN}}}row":
                        continue
                    row_index = int(row.attrib["r"])
                    if row_index not in safe_set:
                        row.clear()
                        continue
                    for cell in row.findall(f"{{{NS_MAIN}}}c"):
                        ref = cell.attrib.get("r", "")
                        cell_type = cell.attrib.get("t", "n")
                        value_node = cell.find(f"{{{NS_MAIN}}}v")
                        inline_node = cell.find(f"{{{NS_MAIN}}}is")
                        value = ""
                        if cell_type == "s" and value_node is not None:
                            value = strings[int(value_node.text)]
                        elif cell_type == "inlineStr" and inline_node is not None:
                            value = "".join(
                                node.text or ""
                                for node in inline_node.iter(f"{{{NS_MAIN}}}t")
                            )
                        elif value_node is not None:
                            value = value_node.text or ""
                        output.append(
                            {
                                "workbook": path.name,
                                "sheet": sheet_name,
                                "row": row_index,
                                "cell": ref,
                                "cell_type": cell_type,
                                "value": value,
                                "selection_reason": "metadata_or_earliest_allowed_year_rows",
                            }
                        )
                    row.clear()
    return output


def matlab_text(value) -> str:
    item = np.asarray(value).squeeze()
    if item.dtype.kind in {"U", "S"}:
        if item.ndim == 0:
            return str(item.item()).strip()
        return "".join(str(part) for part in item.flat).strip()
    if item.dtype == object:
        return matlab_text(item.item()) if item.size == 1 else "|".join(matlab_text(x) for x in item.flat)
    return str(item.item()) if item.ndim == 0 else "|".join(str(x) for x in item.flat)


def combined_metadata(path: Path) -> list[dict]:
    selected = ["archive", "dataName", "p_lat", "p_lon", "source_flag"]
    mat = loadmat(path, variable_names=selected, squeeze_me=False, chars_as_strings=False)
    lat = np.asarray(mat["p_lat"], dtype=float).reshape(-1)
    lon = np.asarray(mat["p_lon"], dtype=float).reshape(-1)
    flags = np.asarray(mat["source_flag"], dtype=float).reshape(-1)
    archives = np.asarray(mat["archive"], dtype=object).reshape(-1)

    names_raw = np.asarray(mat["dataName"])
    if names_raw.dtype.kind in {"U", "S"} and names_raw.shape[0] == lat.size:
        names = ["".join(str(x) for x in names_raw[i].flat).strip() for i in range(lat.size)]
    else:
        names = [matlab_text(names_raw)] * lat.size

    rows = []
    for index in range(lat.size):
        rows.append(
            {
                "source": str(path),
                "source_sha256": sha256(path),
                "record_1based": index + 1,
                "latitude": lat[index],
                "longitude": lon[index],
                "source_flag": flags[index],
                "data_name": names[index],
                "archive": matlab_text(archives[index]),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not OUTPUT.exists():
        raise RuntimeError("Run metadata-only closure preflight first")

    selected = []
    for workbook_name, selections in SAFE_ROW_SELECTIONS.items():
        selected.extend(selected_rows(RAW / workbook_name, selections))
    write_rows(OUTPUT / "selected_safe_workbook_cells.csv", selected)

    source = COMBINED_SOURCES[1]
    metadata = combined_metadata(source)
    write_rows(OUTPUT / "combined_proxy_metadata_only.csv", metadata)

    summary = {
        "locked_interval": [1921, 1939],
        "selected_workbook_rows": {
            workbook: {sheet: [min(rows), max(rows)] for sheet, rows in sheets.items()}
            for workbook, sheets in SAFE_ROW_SELECTIONS.items()
        },
        "candidate_time_series_full_load": False,
        "combined_proxy_time_series_variable_loaded": False,
        "combined_metadata_variables_loaded": [
            "archive",
            "dataName",
            "p_lat",
            "p_lon",
            "source_flag",
        ],
        "combined_record_count": len(metadata),
        "source_files_modified": False,
    }
    (OUTPUT / "metadata_extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
