from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
PANEL = LOOP / "imd_ghcn_overlap_pages_v2"
OCR = LOOP / "imd_ghcn_overlap_matlab_ocr_v4"
OUT = LOOP / "imd_ghcn_adopted_temperature_layout_aware_v2"
DRIVER = Path(__file__).resolve()

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
ALIASES = {
    "Darjeeling": ["darjeeling", "darjiling"],
    "Simla": ["simla", "shimla"],
    "Leh": ["leh"],
    "Murree": ["murree"],
}
# Table V layout and 1886 method pages establish these station-specific columns.
ADOPTED_COLUMN = {"Darjeeling": 3, "Simla": 1, "Murree": 1, "Leh": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def similarity(value: str, targets: list[str]) -> float:
    normalized = normalize_word(value)
    if not normalized:
        return 0.0
    return max(difflib.SequenceMatcher(None, normalized, target).ratio() for target in targets)


def numeric_parts(raw: str, x: float, width: float) -> list[tuple[float, float, str]]:
    normalized = raw.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    pattern = re.compile(r"(?<!\d)(-?\d{1,3})(?:[.'’`:°](\d))?(?!\d)")
    results: list[tuple[float, float, str]] = []
    denominator = max(len(normalized), 1)
    for match in pattern.finditer(normalized):
        whole = int(match.group(1))
        decimal = match.group(2)
        if decimal is not None:
            sign = -1.0 if whole < 0 else 1.0
            value = sign * (abs(whole) + int(decimal) / 10.0)
        elif abs(whole) >= 100 and abs(whole) <= 999:
            value = whole / 10.0
        else:
            value = float(whole)
        if not -20.0 <= value <= 120.0:
            continue
        center_fraction = (match.start() + match.end()) / (2.0 * denominator)
        results.append((x + width * center_fraction, value, match.group(0)))
    return results


def locate_heading(words: list[dict[str, object]], station: str) -> tuple[dict[str, object] | None, float]:
    ranked = []
    for word in words:
        if station == "Leh" and normalize_word(str(word["word"])) != "leh":
            continue
        score = similarity(str(word["word"]), ALIASES[station])
        if score >= 0.72:
            ranked.append((score, float(word["confidence"]), -float(word["y"]), word))
    if not ranked:
        return None, 0.0
    score, _, _, word = max(ranked, key=lambda item: item[:3])
    return word, score


def label_rows(words: list[dict[str, object]], label: str, low: float, high: float) -> list[dict[str, object]]:
    selected = []
    for word in words:
        y = float(word["y"])
        if not low < y < high:
            continue
        score = difflib.SequenceMatcher(None, normalize_word(str(word["word"])), label).ratio()
        if score >= 0.72:
            selected.append(word)
    return sorted(selected, key=lambda word: float(word["y"]))


def row_stat_values(
    words: list[dict[str, object]], row_y: float, side: str, page_width: int
) -> list[tuple[float, float, str, float]]:
    if side == "left":
        low_x, high_x = page_width * 0.09, page_width * 0.36
    else:
        low_x, high_x = page_width * 0.50, page_width * 0.72
    candidates = []
    tolerance = max(30.0, page_width * 0.014)
    for word in words:
        if abs(float(word["y"]) - row_y) > tolerance:
            continue
        for center_x, value, raw in numeric_parts(
            str(word["word"]), float(word["x"]), float(word["width"])
        ):
            if low_x <= center_x <= high_x:
                candidates.append((center_x, value, raw, float(word["confidence"])))
    candidates.sort(key=lambda item: item[0])
    return candidates[:8]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    panel = read_csv(PANEL / "station_year_page_panel.csv")
    ghcn_rows = read_csv(PANEL / "ghcn_annual_overlap.csv")
    ghcn = {
        (row["station_name"], row["series"], int(row["year"])): float(row["annual_mean_c"])
        for row in ghcn_rows
    }
    results: list[dict[str, object]] = []
    monthly_output: list[dict[str, object]] = []

    for source in panel:
        station = source["imd_station"]
        year = int(source["report_year"])
        suffix = source["item_id"].rsplit(".", 1)[-1]
        leaf = int(source["leaf"])
        page_key = f"{suffix}_{leaf:04d}"
        word_path = OCR / "pages" / page_key / "words.csv"
        words = []
        for row in read_csv(word_path):
            words.append(
                {
                    "word": row["word"], "x": float(row["x"]), "y": float(row["y"]),
                    "width": float(row["width"]), "height": float(row["height"]),
                    "confidence": float(row["confidence"]),
                }
            )
        with Image.open(source["image_path"]) as image:
            page_width, page_height = image.size

        heading, heading_score = locate_heading(words, station)
        if heading is None:
            results.append({**source, "page_key": page_key, "status": "heading_not_found"})
            continue
        heading_x = float(heading["x"]) + float(heading["width"]) / 2.0
        heading_y = float(heading["y"])
        side = "left" if heading_x < page_width / 2 else "right"
        years = label_rows(words, "year", heading_y + page_height * 0.05, page_height)
        same_side_years = [
            word for word in years
            if (float(word["x"]) < page_width / 2) == (side == "left")
        ]
        year_word = (same_side_years or years)[0] if years else None
        if year_word is None:
            results.append({**source, "page_key": page_key, "status": "year_row_not_found"})
            continue
        year_y = float(year_word["y"])
        adopted_index = ADOPTED_COLUMN[station] - 1
        annual_candidates = row_stat_values(words, year_y, side, page_width)
        annual = annual_candidates[adopted_index] if len(annual_candidates) > adopted_index else None

        month_values = []
        month_details = []
        for month_index, month in enumerate(MONTHS, start=1):
            labels = label_rows(words, month, heading_y, year_y)
            same_side = [
                word for word in labels
                if (float(word["x"]) < page_width / 2) == (side == "left")
            ]
            label = (same_side or labels)[0] if labels else None
            if label is None:
                continue
            candidates = row_stat_values(words, float(label["y"]), side, page_width)
            if len(candidates) <= adopted_index:
                continue
            selected = candidates[adopted_index]
            month_values.append(selected[1])
            detail = {
                "report_year": year,
                "station": station,
                "page_key": page_key,
                "month": month_index,
                "month_name": month,
                "adopted_column": adopted_index + 1,
                "value_f": selected[1],
                "raw": selected[2],
                "x": selected[0],
                "confidence": selected[3],
                "row_candidates": ";".join(f"{v:.1f}@{x:.1f}" for x, v, _, _ in candidates),
            }
            month_details.append(detail)
            monthly_output.append(detail)

        monthly_mean = sum(month_values) / len(month_values) if month_values else math.nan
        annual_value = annual[1] if annual is not None else math.nan
        closure = monthly_mean - annual_value if annual is not None and month_values else math.nan
        accepted = (
            heading_score >= 0.72
            and len(month_values) >= 10
            and annual is not None
            and abs(closure) <= 0.5
        )
        annual_c = (annual_value - 32.0) / 1.8 if accepted else math.nan
        output = {
            **source,
            "page_key": page_key,
            "station_side": side,
            "heading_score": heading_score,
            "adopted_column": adopted_index + 1,
            "method_rule": (
                "10h_16h_minimum_plus_range_factor" if station == "Darjeeling"
                else "four_synoptic_hours_mean" if station == "Leh"
                else "mean_of_16h_and_minimum"
            ),
            "annual_value_f": annual_value,
            "annual_raw": annual[2] if annual else "",
            "annual_x": annual[0] if annual else math.nan,
            "readable_months": len(month_values),
            "monthly_recomputed_mean_f": monthly_mean,
            "monthly_minus_printed_annual_f": closure,
            "status": "accepted" if accepted else "failed_quality_gate",
        }
        for series in ("qcu", "qcf"):
            ghcn_value = ghcn.get((source["ghcn_station_name"], series, year), math.nan)
            output[f"ghcn_{series}_annual_c"] = ghcn_value
            output[f"imd_minus_{series}_c"] = (
                annual_c - ghcn_value
                if accepted and not math.isnan(ghcn_value)
                else math.nan
            )
        results.append(output)

    fields = []
    for row in results:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / "station_year_extraction.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    with (OUT / "monthly_extraction.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(monthly_output[0]))
        writer.writeheader()
        writer.writerows(monthly_output)

    accepted_rows = [row for row in results if row["status"] == "accepted"]
    comparisons = [
        row for row in accepted_rows
        if not math.isnan(float(row["ghcn_qcu_annual_c"]))
        or not math.isnan(float(row["ghcn_qcf_annual_c"]))
    ]
    station_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in comparisons:
        station_groups[str(row["imd_station"])].append(row)
    station_summary = []
    for station, group in sorted(station_groups.items()):
        record: dict[str, object] = {"station": station, "comparison_years": len(group)}
        for series in ("qcu", "qcf"):
            values = [
                float(row[f"imd_minus_{series}_c"])
                for row in group if not math.isnan(float(row[f"imd_minus_{series}_c"]))
            ]
            record[f"n_{series}"] = len(values)
            record[f"mean_imd_minus_{series}_c"] = sum(values) / len(values) if values else math.nan
            record[f"min_imd_minus_{series}_c"] = min(values) if values else math.nan
            record[f"max_imd_minus_{series}_c"] = max(values) if values else math.nan
        station_summary.append(record)
    if station_summary:
        with (OUT / "station_difference_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(station_summary[0]))
            writer.writeheader()
            writer.writerows(station_summary)

    all_qcu = [float(row["imd_minus_qcu_c"]) for row in comparisons if not math.isnan(float(row["imd_minus_qcu_c"]))]
    all_qcf = [float(row["imd_minus_qcf_c"]) for row in comparisons if not math.isnan(float(row["imd_minus_qcf_c"]))]
    summary = {
        "experiment": "imd_ghcn_adopted_temperature_layout_aware_v2",
        "requested_station_years": len(results),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn": len(comparisons),
        "qcu_comparisons": len(all_qcu),
        "qcf_comparisons": len(all_qcf),
        "mean_imd_minus_qcu_c": sum(all_qcu) / len(all_qcu) if all_qcu else None,
        "mean_imd_minus_qcf_c": sum(all_qcf) / len(all_qcf) if all_qcf else None,
        "locked_period_accessed": False,
        "source_files_modified": False,
        "supersedes": [
            "imd_ghcn_adopted_temperature_extract_v1",
            "imd_matlab_ia_ocr_ghcn_adjudication_v1",
        ],
        "decision": "inspect_failed_rows_and_independently_adjudicate_accepted_values",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "station_panel": str(PANEL / "station_year_page_panel.csv"),
                "matlab_ocr": str(OCR / "summary.json"),
                "method_evidence": str(LOOP / "imd_1886_air_temperature_mean_method_v1"),
            },
            indent=2,
        ) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
