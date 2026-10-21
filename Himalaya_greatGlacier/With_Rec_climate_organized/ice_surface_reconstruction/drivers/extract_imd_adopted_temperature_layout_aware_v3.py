from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "extract_imd_adopted_temperature_layout_aware_v2.py"
SPEC = importlib.util.spec_from_file_location("imd_layout_v2", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

BASE.OUT = BASE.LOOP / "imd_ghcn_adopted_temperature_layout_aware_v3"
BASE.DRIVER = Path(__file__).resolve()


def split_compact_digits(raw: str, x: float, width: float):
    normalized = raw.upper().replace("O", "0")
    results = []
    for match in re.finditer(r"\d{4,8}", normalized):
        digits = match.group(0)
        splits = []
        if len(digits) == 4:
            splits = [[digits[:2], digits[2:]]]
        elif len(digits) == 5:
            splits = [[digits[:2], digits[2:]], [digits[:3], digits[3:]]]
        elif len(digits) == 6:
            splits = [[digits[:3], digits[3:]]]
        elif len(digits) == 8:
            splits = [[digits[i:i + 2] for i in range(0, 8, 2)]]
        valid = []
        for pieces in splits:
            values = []
            for piece in pieces:
                value = int(piece) / 10.0 if len(piece) == 3 else float(int(piece))
                values.append(value)
            if all(-20.0 <= value <= 120.0 for value in values):
                valid.append(values)
        if not valid:
            continue
        values = min(valid, key=lambda values: max(values) - min(values))
        for index, value in enumerate(values):
            fraction = (match.start() + (index + 0.5) * len(digits) / len(values)) / max(len(normalized), 1)
            results.append((x + width * fraction, value, digits))
    return results


def numeric_parts(raw: str, x: float, width: float):
    compact = split_compact_digits(raw, x, width)
    if compact:
        return compact
    return BASE.numeric_parts_original(raw, x, width)


BASE.numeric_parts_original = BASE.numeric_parts
BASE.numeric_parts = numeric_parts


def row_stat_values(words, row_y, side, page_width):
    if side == "left":
        low_x, high_x = page_width * 0.12, page_width * 0.36
    else:
        low_x, high_x = page_width * 0.515, page_width * 0.72
    candidates = []
    tolerance = max(30.0, page_width * 0.014)
    for word in words:
        if abs(float(word["y"]) - row_y) > tolerance:
            continue
        for center_x, value, token in BASE.numeric_parts(
            str(word["word"]), float(word["x"]), float(word["width"])
        ):
            if low_x <= center_x <= high_x:
                candidates.append((center_x, value, token, float(word["confidence"])))
    candidates.sort(key=lambda item: item[0])
    return candidates[:8]


BASE.row_stat_values = row_stat_values


def main() -> None:
    BASE.main()
    summary_path = BASE.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = "imd_ghcn_adopted_temperature_layout_aware_v3"
    summary["supersedes"].append("imd_ghcn_adopted_temperature_layout_aware_v2")
    summary["compact_ocr_digit_split"] = True
    summary["statistical_region_boundary_tightened"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
