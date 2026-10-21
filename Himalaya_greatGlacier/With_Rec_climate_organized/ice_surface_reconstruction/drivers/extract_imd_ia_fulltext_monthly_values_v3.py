from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
V2_PATH = HERE / "extract_imd_ia_fulltext_monthly_values_v2.py"
SPEC = importlib.util.spec_from_file_location("imd_ia_v2", V2_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {V2_PATH}")
V2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V2)
V1 = V2.V1

V1.OUT = V1.LOOP / "imd_ia_fulltext_monthly_values_v3"
V1.DRIVER = Path(__file__).resolve()


def normalize_ocr(text: str) -> str:
    text = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)\b", " ", text, flags=re.I)
    text = text.replace("§", "5")
    text = re.sub(r"(?<=\d)[gG](?=[°'’`:.-])", "9", text)
    text = re.sub(r"\b[Gg](?=\d)", "6", text)
    text = re.sub(r"(?<=\d)[rR](?=[°'’`:.-])", "1", text)
    text = text.upper().replace("O", "0")
    text = re.sub(r"(?<=[°'’`:.-])S\b", "5", text)
    return text


def parse_temperatures(text: str):
    cleaned = normalize_ocr(text)
    values = [999.0] if "IMPERFECT" in cleaned else []
    occupied = []
    decimal_pattern = re.compile(r"(?<!\d)(-?\d{1,3})\s*[°'’`:.-]\s*(\d)")
    for match in decimal_pattern.finditer(cleaned):
        whole = int(match.group(1))
        sign = -1 if whole < 0 else 1
        value = sign * (abs(whole) + int(match.group(2)) / 10.0)
        if -20.0 <= value <= 120.0:
            values.append(value)
            occupied.append(match.span())

    def overlaps(span):
        return any(not (span[1] <= prior[0] or span[0] >= prior[1]) for prior in occupied)

    for match in re.finditer(r"(?<!\d)(1\d{2})\s*°(?!\d)", cleaned):
        if not overlaps(match.span()):
            values.append(float(int(match.group(1)) - 100))
            occupied.append(match.span())
    for match in re.finditer(r"(?<!\d)(\d{3})(?!\d)", cleaned):
        if not overlaps(match.span()):
            value = int(match.group(1)) / 10.0
            if -20.0 <= value <= 120.0:
                values.append(value)
                occupied.append(match.span())
    return values


def split_station_group(values, side):
    imperfect_left = bool(values and values[0] == 999.0)
    if imperfect_left:
        values = values[1:]
        if side == "left":
            return []
    if side == "left":
        return values
    if len(values) >= 15:
        return values[9:]
    return values[len(values) // 2:]


V1.parse_temperatures = parse_temperatures
V1.split_station_group = split_station_group


def main() -> None:
    V1.main()
    path = V1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = "imd_ia_fulltext_monthly_values_v3"
    summary["supersedes"] = "imd_ia_fulltext_monthly_values_v2"
    summary["ocr_character_substitutions"] = ["S->5", "section_sign->5", "numeric_g/r/O corrections", "bare_3_digit_tenths"]
    summary["imperfect_left_table_months_excluded"] = True
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
