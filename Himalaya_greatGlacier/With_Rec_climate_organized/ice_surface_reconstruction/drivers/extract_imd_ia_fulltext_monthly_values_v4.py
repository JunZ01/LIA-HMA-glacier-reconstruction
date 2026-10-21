from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "extract_imd_ia_fulltext_monthly_values_v3.py"
SPEC = importlib.util.spec_from_file_location("imd_ia_v3", V3_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {V3_PATH}")
V3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V3)
V1 = V3.V1

V1.OUT = V1.LOOP / "imd_ia_fulltext_monthly_values_v4"
V1.DRIVER = Path(__file__).resolve()


def parse_temperatures(text: str):
    cleaned = V3.normalize_ocr(text)
    imperfect = "IMPERFECT" in cleaned
    parsed = []
    occupied = []

    def overlaps(span):
        return any(not (span[1] <= prior[0] or span[0] >= prior[1]) for prior in occupied)

    decimal = re.compile(r"(?<!\d)(-?\d{1,3})\s*[°'’`:.-]\s*(\d)")
    for match in decimal.finditer(cleaned):
        whole = int(match.group(1))
        sign = -1 if whole < 0 else 1
        value = sign * (abs(whole) + int(match.group(2)) / 10.0)
        if -20.0 <= value <= 120.0:
            parsed.append((match.start(), value))
            occupied.append(match.span())
    for match in re.finditer(r"(?<!\d)(1\d{2})\s*°(?!\d)", cleaned):
        if not overlaps(match.span()):
            parsed.append((match.start(), float(int(match.group(1)) - 100)))
            occupied.append(match.span())
    for match in re.finditer(r"(?<!\d)(\d{3})(?!\d)", cleaned):
        if not overlaps(match.span()):
            value = int(match.group(1)) / 10.0
            if -20.0 <= value <= 120.0:
                parsed.append((match.start(), value))
                occupied.append(match.span())
    values = [value for _, value in sorted(parsed)]
    return ([999.0] + values) if imperfect else values


V1.parse_temperatures = parse_temperatures
V1.split_station_group = V3.split_station_group


def main() -> None:
    V1.main()
    path = V1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = "imd_ia_fulltext_monthly_values_v4"
    summary["supersedes"] = "imd_ia_fulltext_monthly_values_v2"
    summary["v3_status"] = "not_executed_ordering_draft"
    summary["tokens_sorted_by_original_text_position"] = True
    summary["imperfect_left_table_months_excluded"] = True
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
