from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "extract_imd_ia_fulltext_monthly_values_v1.py"
SPEC = importlib.util.spec_from_file_location("imd_ia_v1", V1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {V1_PATH}")
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

V1.OUT = V1.LOOP / "imd_ia_fulltext_monthly_values_v2"
V1.DRIVER = Path(__file__).resolve()


def parse_temperatures(text: str):
    cleaned = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)\b", " ", text, flags=re.I)
    cleaned = cleaned.upper().replace("O", "0")
    pattern = re.compile(r"(?<!\d)(-?\d{1,3})\s*[°'’`:.-]\s*(\d)")
    values = []
    for match in pattern.finditer(cleaned):
        whole = int(match.group(1))
        sign = -1 if whole < 0 else 1
        value = sign * (abs(whole) + int(match.group(2)) / 10.0)
        if -20.0 <= value <= 120.0:
            values.append(value)
    return values


def split_station_group(values, side):
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
    summary["experiment"] = "imd_ia_fulltext_monthly_values_v2"
    summary["supersedes"] = "imd_ia_fulltext_monthly_values_v1"
    summary["trailing_table_digit_tolerated"] = True
    summary["left_table_uses_first_six_tokens"] = True
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
