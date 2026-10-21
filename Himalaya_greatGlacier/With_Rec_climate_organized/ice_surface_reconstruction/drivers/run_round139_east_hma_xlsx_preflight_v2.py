"""Path-shortened execution of the Round 139 workbook boundary preflight."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_139_east_hma_xlsx_preflight_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round139_east_hma_xlsx_preflight_v2_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    base = load_module(
        "round139_v1",
        driver / "run_round139_east_hma_external_workbook_boundary_preflight_v1.py",
    )

    raw = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_proxy_discovery_20260818_v1/raw_unopened"
    )
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "sb": raw / "salween_brahmaputra_precipitation.xlsx",
        "tr": raw / "three_rivers_streamflow_reconstruction.xlsx",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: base.fingerprint(path) for name, path in paths.items()}

    inventory_rows = []
    previews = []
    for key in ("sb", "tr"):
        path = paths[key]
        with zipfile.ZipFile(path) as archive:
            sheets = base.workbook_sheets(archive)
            sheet_bytes = [
                (name, xml_path, archive.read(xml_path)) for name, xml_path in sheets
            ]
            for sheet_name, _, xml_bytes in sheet_bytes:
                for row in base.rows_from_xml(xml_bytes):
                    year = base.first_numeric_year(row)
                    if year is None or not base.LOCKED_START <= year <= base.LOCKED_END:
                        continue
                    cells = sorted(
                        row.findall(f"{{{base.MAIN_NS}}}c"),
                        key=lambda item: base.column_index(item.attrib["r"]),
                    )
                    if any(
                        cell.attrib.get("t", "n") in {"s", "inlineStr", "str"}
                        for cell in cells[1:]
                    ):
                        raise RuntimeError(
                            f"locked-shared-string-preflight-failed:{key}:{sheet_name}"
                        )
            strings = base.shared_strings(archive)
            for sheet_name, _, xml_bytes in sheet_bytes:
                inventory, preview = base.inspect_sheet(
                    xml_bytes, strings, key, sheet_name, output
                )
                inventory_rows.append(inventory)
                previews.append(preview)

    with (output / "sheet_boundary_inventory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)
    (output / "allowed_data_previews.json").write_text(
        json.dumps(previews, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    after = {name: base.fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-file-changed")
    summary = {
        "experiment_id": "round_139_east_hma_xlsx_preflight_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v1_status": "stopped_on_windows_output_path_length_before_statistics",
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
        "# Round 139 V2 Final Result\n\n"
        f"{sheet_lines}\n\n"
        "All locked rows were rejected before non-year values were decoded. "
        "Allowed exports may proceed to identity and scientific-role closure.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(sheet_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
