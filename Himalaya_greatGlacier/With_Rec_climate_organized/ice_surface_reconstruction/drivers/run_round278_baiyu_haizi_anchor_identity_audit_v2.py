from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import run_round278_baiyu_haizi_anchor_identity_audit_v1 as audit


audit.OUT = audit.LOOP / "round_278_baiyu_haizi_eastern_hma_anchor_identity_audit_v2"
audit.DRIVER = Path(__file__).resolve()


def read_series_closed(self, sheet_name, value_column, is_allowed):
    output = {}
    locked_rows_skipped = 0
    outside_rows_skipped = 0
    with self.archive.open(self.sheet_paths[sheet_name]) as stream:
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag.rsplit("}", 1)[-1] != "row":
                continue
            cells = {
                audit.column_name(cell.attrib.get("r", "")): cell
                for cell in row.findall(f"{{{audit.NS_MAIN}}}c")
            }
            if "A" not in cells:
                row.clear()
                continue
            year_token = self._decode_cell(cells["A"], None, "A")
            try:
                year = int(float(str(year_token)))
            except ValueError:
                row.clear()
                continue
            if audit.LOCKED_START <= year <= audit.LOCKED_END:
                locked_rows_skipped += 1
                row.clear()
                continue
            if not is_allowed(year):
                outside_rows_skipped += 1
                row.clear()
                continue
            if value_column not in cells:
                raise RuntimeError(f"missing-value-cell:{sheet_name}:{year}:{value_column}")
            value = float(self._decode_cell(cells[value_column], year, value_column))
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite-value:{sheet_name}:{year}:{value_column}")
            if year in output:
                raise RuntimeError(f"duplicate-year:{sheet_name}:{year}")
            output[year] = value
            row.clear()
    return output, {
        "locked_rows_skipped_before_value_decode": locked_rows_skipped,
        "outside_allowed_rows_skipped_before_value_decode": outside_rows_skipped,
        "allowed_value_cells_decoded": len(output),
    }


audit.YearFirstXlsx.read_series = read_series_closed


if __name__ == "__main__":
    raise SystemExit(audit.main())
