"""Boundary-safe release of unsealed Hengduan and Haizi temperature candidates."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


LOCKED_START = 1921
LOCKED_END = 1939


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allowed_year(year: int) -> bool:
    return 1000 <= year <= 1920 or 1940 <= year <= 2000


def parse_rwl(path: Path, output: Path):
    rows = []
    locked_slots = 0
    decoded_slots = 0
    identifiers = set()
    with path.open("r", encoding="ascii", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if len(line) < 12:
                raise RuntimeError(f"short-rwl-line:{path.name}:{line_number}")
            identifier = line[:8].strip()
            year_text = line[8:12].strip()
            if not year_text.lstrip("-").isdigit():
                raise RuntimeError(f"invalid-rwl-year:{path.name}:{line_number}:{year_text}")
            base_year = int(year_text)
            identifiers.add(identifier)
            value_area = line[12:]
            slot_count = (len(value_area) + 5) // 6
            for slot in range(slot_count):
                year = base_year + slot
                chunk = value_area[slot * 6 : (slot + 1) * 6]
                if LOCKED_START <= year <= LOCKED_END:
                    locked_slots += 1
                    continue
                if not allowed_year(year):
                    continue
                token = chunk.strip()
                if not token:
                    continue
                try:
                    value = int(token)
                except ValueError as exc:
                    raise RuntimeError(f"invalid-rwl-value:{path.name}:{line_number}:{slot}") from exc
                decoded_slots += 1
                if value in {-9999, 9999}:
                    continue
                rows.append({"series_id": identifier, "year": year, "measurement_integer": value})
    csv_name = f"{path.name.lower()}__allowed_long.csv"
    with (output / csv_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["series_id", "year", "measurement_integer"])
        writer.writeheader()
        writer.writerows(rows)
    years = [row["year"] for row in rows]
    return {
        "source_file": path.name,
        "format": "tucson_rwl_fixed_width",
        "series_count": len(identifiers),
        "allowed_value_count": len(rows),
        "decoded_allowed_or_outside_locked_slot_count": decoded_slots,
        "allowed_first_year": min(years) if years else None,
        "allowed_last_year": max(years) if years else None,
        "locked_slots_skipped_before_value_decode": locked_slots,
        "allowed_csv": csv_name,
    }


def inspect_workbook(path: Path, label: str, output: Path, base):
    rows = []
    previews = []
    with zipfile.ZipFile(path) as archive:
        sheets = base.workbook_sheets(archive)
        sheet_bytes = [(name, xml_path, archive.read(xml_path)) for name, xml_path in sheets]
        year_counts = {}
        for sheet_name, _, xml_bytes in sheet_bytes:
            year_count = 0
            locked_count = 0
            locked_nonnumeric = 0
            for row in base.rows_from_xml(xml_bytes):
                year = base.first_numeric_year(row)
                if year is None:
                    continue
                year_count += 1
                if LOCKED_START <= year <= LOCKED_END:
                    locked_count += 1
                    cells = sorted(
                        row.findall(f"{{{base.MAIN_NS}}}c"),
                        key=lambda item: base.column_index(item.attrib["r"]),
                    )
                    locked_nonnumeric += sum(
                        cell.attrib.get("t", "n") in {"s", "inlineStr", "str"}
                        for cell in cells[1:]
                    )
            if locked_nonnumeric:
                raise RuntimeError(f"locked-workbook-string-value:{label}:{sheet_name}:{locked_nonnumeric}")
            year_counts[sheet_name] = (year_count, locked_count)
        strings = base.shared_strings(archive)
        for sheet_name, _, xml_bytes in sheet_bytes:
            year_count, locked_count = year_counts[sheet_name]
            if year_count == 0:
                rows.append({
                    "workbook": label,
                    "sheet": sheet_name,
                    "role": "metadata_or_nonannual_sheet_not_decoded",
                    "year_row_count": 0,
                    "source_first_year": None,
                    "source_last_year": None,
                    "locked_row_count_skipped_without_value_decode": 0,
                    "locked_non_numeric_value_type_count": 0,
                    "column_count": None,
                    "header_source_row": None,
                    "allowed_row_count": 0,
                    "allowed_first_year": None,
                    "allowed_last_year": None,
                    "allowed_csv": None,
                })
                continue
            inventory, preview = base.inspect_sheet(xml_bytes, strings, label, sheet_name, output)
            if inventory["locked_row_count_skipped_without_value_decode"] != locked_count:
                raise RuntimeError(f"locked-row-count-mismatch:{label}:{sheet_name}")
            inventory["role"] = "annual_allowed_export"
            rows.append(inventory)
            previews.append(preview)
    return rows, previews


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_144_temperature_candidate_boundary_preflight_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round144_temperature_candidate_boundary_preflight_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    base = load_module(
        "round144_xlsx",
        driver / "run_round139_east_hma_external_workbook_boundary_preflight_v1.py",
    )
    raw = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_external_proxy_discovery_20260818_v1/raw_unopened"
    rwl_names = ("hengduan_DLL", "hengduan_FGH", "hengduan_KQH", "hengduan_XYS", "hengduan_YPL")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "haizi": raw / "haizi_temperature_supplement.xlsx",
        "hengduan_info": raw / "hengduan_upload_information.xlsx",
        "jionglacuo_sealed": raw / "jionglacuo_temperature_reconstruction.xlsx",
    }
    for name in rwl_names:
        paths[name] = raw / name
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: base.fingerprint(path) for name, path in paths.items()}

    rwl_inventory = [parse_rwl(paths[name], output) for name in rwl_names]
    workbook_rows = []
    previews = []
    for key, label in (("haizi", "haizi"), ("hengduan_info", "hengduan_info")):
        rows, items = inspect_workbook(paths[key], label, output, base)
        workbook_rows.extend(rows)
        previews.extend(items)

    with (output / "rwl_boundary_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rwl_inventory[0]))
        writer.writeheader()
        writer.writerows(rwl_inventory)
    with (output / "workbook_boundary_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(workbook_rows[0]))
        writer.writeheader()
        writer.writerows(workbook_rows)
    (output / "allowed_workbook_previews.json").write_text(json.dumps(previews, indent=2, ensure_ascii=True), encoding="utf-8")

    after = {name: base.fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_144_temperature_candidate_boundary_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "rwl_file_count": len(rwl_inventory),
        "rwl_series_count": int(sum(row["series_count"] for row in rwl_inventory)),
        "rwl_allowed_value_count": int(sum(row["allowed_value_count"] for row in rwl_inventory)),
        "rwl_locked_slots_skipped_before_value_decode": int(sum(row["locked_slots_skipped_before_value_decode"] for row in rwl_inventory)),
        "workbook_sheet_count": len(workbook_rows),
        "workbook_annual_export_count": int(sum(row["role"] == "annual_allowed_export" for row in workbook_rows)),
        "workbook_locked_rows_skipped_without_value_decode": int(sum(row["locked_row_count_skipped_without_value_decode"] for row in workbook_rows)),
        "locked_1921_1939_values_decoded": 0,
        "jionglacuo_external_validation_file_opened": False,
        "source_fingerprints_unchanged": unchanged,
        "enkf_run": False,
        "oggm_run": False,
        "decision": "advance_hengduan_haizi_allowed_exports_to_identity_chronology_closure",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 144 Temperature Candidate Boundary Preflight\n\n"
        f"- Five Hengduan RWL files contain `{summary['rwl_series_count']}` raw series and `{summary['rwl_allowed_value_count']}` allowed values.\n"
        f"- `{summary['rwl_locked_slots_skipped_before_value_decode']}` locked RWL slots and `{summary['workbook_locked_rows_skipped_without_value_decode']}` workbook rows were skipped before value decode.\n"
        f"- Jionglacuo sealed external validation file opened: `{summary['jionglacuo_external_validation_file_opened']}`.\n"
        f"- Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
