from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from scipy.io import whosmat


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DISCOVERY = PROJECT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_external_proxy_discovery_20260818_v1"
)
RAW = DISCOVERY / "raw_unopened"
OUTPUT = PROJECT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_external_proxy_identity_closure_20260818_v1"
)

COMBINED_SOURCES = [
    Path(r"D:\Users\JUNZ\Desktop\select\PHYDA\combined_proxydata6.mat"),
    Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat"),
]

WORKBOOKS = [
    RAW / "hengduan_upload_information.xlsx",
    RAW / "haizi_temperature_supplement.xlsx",
    RAW / "salween_brahmaputra_precipitation.xlsx",
]

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workbook_structure(path: Path) -> dict:
    """Read package metadata only; worksheet cell records are never parsed."""
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{NS_PKG_REL}}}Relationship")
        }
        sheets = []
        for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
            rel_id = sheet.attrib[f"{{{NS_REL}}}id"]
            target = targets[rel_id].replace("\\", "/")
            if target.startswith("/"):
                sheet_path = target.lstrip("/")
            elif target.startswith("xl/"):
                sheet_path = target
            else:
                sheet_path = f"xl/{target}"

            dimension = None
            row_count = None
            column_count = None
            # iterparse sees only structural start tags and clears every element;
            # no cell value text is retained or emitted.
            with archive.open(sheet_path) as stream:
                for event, element in ET.iterparse(stream, events=("start",)):
                    local = element.tag.rsplit("}", 1)[-1]
                    if local == "dimension":
                        dimension = element.attrib.get("ref")
                    elif local == "row":
                        row_count = int(element.attrib.get("r", row_count or 0))
                    elif local == "c":
                        ref = element.attrib.get("r", "")
                        letters = "".join(ch for ch in ref if ch.isalpha())
                        value = 0
                        for ch in letters.upper():
                            value = value * 26 + ord(ch) - 64
                        column_count = max(column_count or 0, value)
                    element.clear()

            sheets.append(
                {
                    "name": sheet.attrib["name"],
                    "state": sheet.attrib.get("state", "visible"),
                    "package_path": sheet_path,
                    "dimension": dimension,
                    "last_row_index": row_count,
                    "max_column_index": column_count,
                }
            )

        return {
            "file": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "sheet_count": len(sheets),
            "sheets": sheets,
            "cell_values_parsed": False,
            "locked_interval_values_accessed": False,
        }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)

    structures = [workbook_structure(path) for path in WORKBOOKS]
    (OUTPUT / "workbook_structure.json").write_text(
        json.dumps(structures, indent=2), encoding="utf-8"
    )

    mat_rows = []
    for source in COMBINED_SOURCES:
        if not source.exists():
            mat_rows.append(
                {
                    "source": str(source),
                    "source_exists": False,
                    "source_sha256": "",
                    "variable": "",
                    "shape": "",
                    "matlab_class": "",
                }
            )
            continue
        source_hash = sha256(source)
        for name, shape, matlab_class in whosmat(source):
            mat_rows.append(
                {
                    "source": str(source),
                    "source_exists": True,
                    "source_sha256": source_hash,
                    "variable": name,
                    "shape": "x".join(str(value) for value in shape),
                    "matlab_class": matlab_class,
                }
            )
    write_csv(
        OUTPUT / "combined_proxy_mat_variable_inventory.csv",
        mat_rows,
        [
            "source",
            "source_exists",
            "source_sha256",
            "variable",
            "shape",
            "matlab_class",
        ],
    )

    source_rows = []
    for path in WORKBOOKS + [RAW / name for name in ["hengduan_DLL", "hengduan_FGH", "hengduan_KQH", "hengduan_XYS", "hengduan_YPL"]]:
        source_rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256(path) if path.exists() else "",
            }
        )
    write_csv(
        OUTPUT / "source_manifest.csv",
        source_rows,
        ["path", "exists", "size_bytes", "sha256"],
    )

    protocol = {
        "phase": "metadata_only_preflight",
        "locked_interval": [1921, 1939],
        "worksheet_cell_values_parsed": False,
        "proxy_time_series_loaded": False,
        "sequence_fingerprints_computed": False,
        "source_files_modified": False,
        "next_gate": (
            "Only metadata sheets or explicitly allowed-year sequence segments may be "
            "opened after sheet roles and MAT metadata variables are identified."
        ),
    }
    (OUTPUT / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
