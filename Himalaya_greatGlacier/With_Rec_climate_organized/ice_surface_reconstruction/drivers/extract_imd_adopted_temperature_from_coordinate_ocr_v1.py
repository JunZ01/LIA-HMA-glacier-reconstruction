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
OUT = LOOP / "imd_ghcn_adopted_temperature_extract_v1"
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


def word_similarity(value: str, targets: list[str]) -> float:
    normalized = normalize_word(value)
    if not normalized:
        return 0.0
    return max(difflib.SequenceMatcher(None, normalized, target).ratio() for target in targets)


def parse_fahrenheit(value: str) -> float | None:
    text = value.strip().upper()
    text = text.replace("O", "0").replace("S", "5")
    text = text.replace("I", "1").replace("L", "1")
    match = re.search(r"(?<!\d)(\d{1,3})(?:[.'’`:°-](\d))?(?!\d)", text)
    if match is None:
        return None
    whole = match.group(1)
    decimal = match.group(2)
    if decimal is not None:
        value_f = float(f"{int(whole)}.{decimal}")
    elif len(whole) == 3:
        value_f = int(whole) / 10.0
    elif len(whole) <= 2:
        value_f = float(int(whole))
    else:
        return None
    if not 15.0 <= value_f <= 100.0:
        return None
    return value_f


def closest_row_value(
    words: list[dict[str, object]], y: float, target_x: float, width: int
) -> tuple[float | None, str, float, float]:
    candidates: list[tuple[float, float, str, float]] = []
    for word in words:
        if abs(float(word["y"]) - y) > max(24.0, width * 0.012):
            continue
        parsed = parse_fahrenheit(str(word["word"]))
        if parsed is None:
            continue
        distance = abs(float(word["x"]) - target_x)
        candidates.append((distance, parsed, str(word["word"]), float(word["confidence"])))
    if not candidates:
        return None, "", math.inf, math.nan
    distance, value, raw, confidence = min(candidates, key=lambda row: row[0])
    if distance > width * 0.04:
        return None, raw, distance, confidence
    return value, raw, distance, confidence


def find_station_heading(
    words: list[dict[str, object]], station: str
) -> tuple[dict[str, object] | None, float]:
    aliases = ALIASES[station]
    ranked: list[tuple[float, float, float, dict[str, object]]] = []
    for word in words:
        score = word_similarity(str(word["word"]), aliases)
        if station == "Leh" and normalize_word(str(word["word"])) != "leh":
            continue
        if score < 0.72:
            continue
        ranked.append(
            (
                score,
                float(word["confidence"]),
                -float(word["y"]),
                word,
            )
        )
    if not ranked:
        return None, 0.0
    score, _, _, word = max(ranked, key=lambda row: row[:3])
    return word, score


def find_label_rows(
    words: list[dict[str, object]], label: str, minimum_y: float, maximum_y: float
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for word in words:
        y = float(word["y"])
        if not minimum_y < y < maximum_y:
            continue
        normalized = normalize_word(str(word["word"]))
        score = difflib.SequenceMatcher(None, normalized, label).ratio()
        if score >= 0.72:
            selected.append(word)
    return sorted(selected, key=lambda row: float(row["y"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    station_panel = read_csv(PANEL / "station_year_page_panel.csv")
    ghcn = read_csv(PANEL / "ghcn_annual_overlap.csv")
    ghcn_lookup = {
        (row["station_name"], row["series"], int(row["year"])): row
        for row in ghcn
    }

    results: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for row in station_panel:
        item_id = row["item_id"]
        leaf = int(row["leaf"])
        station = row["imd_station"]
        year = int(row["report_year"])
        suffix = item_id.rsplit(".", 1)[-1]
        page_key = f"{suffix}_{leaf:04d}"
        word_path = OCR / "pages" / page_key / "words.csv"
        image_path = Path(row["image_path"])
        words_raw = read_csv(word_path)
        words: list[dict[str, object]] = []
        for value in words_raw:
            words.append(
                {
                    "word": value["word"],
                    "x": float(value["x"]),
                    "y": float(value["y"]),
                    "width": float(value["width"]),
                    "height": float(value["height"]),
                    "confidence": float(value["confidence"]),
                }
            )
        with Image.open(image_path) as image:
            width, height = image.size

        heading, heading_score = find_station_heading(words, station)
        if heading is None:
            results.append(
                {
                    **row,
                    "page_key": page_key,
                    "image_width": width,
                    "station_heading_score": heading_score,
                    "extraction_status": "station_heading_not_found",
                }
            )
            continue

        station_x = float(heading["x"]) + float(heading["width"]) / 2.0
        station_y = float(heading["y"])
        side = "left" if station_x < width / 2 else "right"
        target_x = width * (0.1956 if side == "left" else 0.5874)
        year_words = find_label_rows(words, "year", station_y + height * 0.05, height)
        if not year_words:
            results.append(
                {
                    **row,
                    "page_key": page_key,
                    "image_width": width,
                    "station_heading_score": heading_score,
                    "station_heading_word": heading["word"],
                    "station_side": side,
                    "extraction_status": "year_row_not_found",
                }
            )
            continue
        year_word = year_words[0]
        year_y = float(year_word["y"])
        annual_f, annual_raw, annual_x_distance, annual_confidence = closest_row_value(
            words, year_y, target_x, width
        )

        next_year_y = height
        all_year_words = find_label_rows(words, "year", 0, height)
        for candidate in all_year_words:
            candidate_y = float(candidate["y"])
            if candidate_y > year_y + height * 0.03:
                next_year_y = candidate_y
                break

        month_values: list[float] = []
        month_y_seen: set[int] = set()
        for month_index, month in enumerate(MONTHS, start=1):
            labels = find_label_rows(words, month, station_y, year_y)
            if not labels:
                continue
            label = labels[0]
            label_y = float(label["y"])
            key = round(label_y)
            if key in month_y_seen:
                continue
            value_f, raw, distance, confidence = closest_row_value(
                words, label_y, target_x, width
            )
            if value_f is None:
                continue
            month_y_seen.add(key)
            month_values.append(value_f)
            monthly_rows.append(
                {
                    "report_year": year,
                    "station": station,
                    "page_key": page_key,
                    "month": month_index,
                    "month_name": month,
                    "value_f": value_f,
                    "raw_word": raw,
                    "x_distance": distance,
                    "confidence": confidence,
                }
            )

        monthly_mean_f = (
            sum(month_values) / len(month_values) if month_values else math.nan
        )
        closure_f = (
            monthly_mean_f - annual_f
            if annual_f is not None and month_values
            else math.nan
        )
        accepted = (
            annual_f is not None
            and len(month_values) >= 10
            and abs(closure_f) <= 0.25
            and annual_x_distance <= width * 0.025
            and heading_score >= 0.72
        )
        annual_c = (annual_f - 32.0) / 1.8 if annual_f is not None else math.nan
        ghcn_values: dict[str, float] = {}
        for series in ["qcu", "qcf"]:
            match = ghcn_lookup.get((row["ghcn_station_name"], series, year))
            ghcn_values[f"ghcn_{series}_annual_c"] = (
                float(match["annual_mean_c"]) if match is not None else math.nan
            )
            ghcn_values[f"imd_minus_{series}_c"] = (
                annual_c - ghcn_values[f"ghcn_{series}_annual_c"]
                if accepted and not math.isnan(ghcn_values[f"ghcn_{series}_annual_c"])
                else math.nan
            )
        results.append(
            {
                **row,
                "page_key": page_key,
                "image_width": width,
                "image_height": height,
                "station_heading_word": heading["word"],
                "station_heading_score": heading_score,
                "station_side": side,
                "station_heading_x": station_x,
                "station_heading_y": station_y,
                "target_adopted_mean_x": target_x,
                "year_row_y": year_y,
                "annual_adopted_mean_raw": annual_raw,
                "annual_adopted_mean_f": annual_f,
                "annual_adopted_mean_c": annual_c,
                "annual_word_confidence": annual_confidence,
                "annual_x_distance": annual_x_distance,
                "readable_months": len(month_values),
                "monthly_recomputed_mean_f": monthly_mean_f,
                "monthly_minus_annual_f": closure_f,
                **ghcn_values,
                "extraction_status": "accepted" if accepted else "failed_quality_gate",
            }
        )

    all_fields: list[str] = []
    for row in results:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    with (OUT / "station_year_extraction.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(results)
    monthly_fields = list(monthly_rows[0]) if monthly_rows else []
    with (OUT / "monthly_column_extraction.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=monthly_fields)
        if monthly_fields:
            writer.writeheader()
            writer.writerows(monthly_rows)

    accepted_rows = [row for row in results if row["extraction_status"] == "accepted"]
    by_station = defaultdict(list)
    for row in accepted_rows:
        by_station[row["imd_station"]].append(row)
    station_summary = []
    for station, rows in sorted(by_station.items()):
        qcf_diffs = [float(row["imd_minus_qcf_c"]) for row in rows if not math.isnan(float(row["imd_minus_qcf_c"]))]
        qcu_diffs = [float(row["imd_minus_qcu_c"]) for row in rows if not math.isnan(float(row["imd_minus_qcu_c"]))]
        station_summary.append(
            {
                "station": station,
                "accepted_years": len(rows),
                "mean_imd_minus_qcf_c": sum(qcf_diffs) / len(qcf_diffs) if qcf_diffs else math.nan,
                "mean_imd_minus_qcu_c": sum(qcu_diffs) / len(qcu_diffs) if qcu_diffs else math.nan,
            }
        )
    with (OUT / "station_bias_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(station_summary[0]) if station_summary else [])
        if station_summary:
            writer.writeheader()
            writer.writerows(station_summary)

    summary = {
        "experiment": "imd_ghcn_adopted_temperature_extract_v1",
        "requested_station_years": len(results),
        "accepted_station_years": len(accepted_rows),
        "failed_station_years": len(results) - len(accepted_rows),
        "minimum_readable_months": 10,
        "maximum_internal_closure_f": 0.25,
        "accepted_station_counts": {key: len(value) for key, value in by_station.items()},
        "locked_period_values_read": False,
        "production_files_modified": False,
        "decision": "inspect_extraction_and_add_ia_ocr_agreement_gate",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "station_panel": str(PANEL / "station_year_page_panel.csv"),
                "ghcn_panel": str(PANEL / "ghcn_annual_overlap.csv"),
                "matlab_ocr_summary": str(OCR / "summary.json"),
            },
            indent=2,
            ensure_ascii=True,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        """# IMD Adopted Temperature Coordinate Extraction Contract

- Extract the adopted mean temperature from the third numeric table column.
- Require station-heading identity, Year-row geometry, and monthly-to-annual closure.
- Accept at least 10 readable months and at most 0.25 F closure error.
- Compare only accepted station-years with matching Round 277 qcu/qcf annual means.
- This stage remains provisional until IA OCR agreement is added.
- Do not access 1921-1939, production data, EnKF, or OGGM.
""",
        encoding="utf-8",
    )
    (OUT / "FINAL_RESULT.md").write_text(
        f"""# IMD Adopted Temperature Coordinate Extraction

Processed {len(results)} same-leaf station-year identities. {len(accepted_rows)}
passed station, geometry, monthly-support, and annual-closure gates. The values
remain provisional until a second-engine IA OCR agreement gate is applied.

No locked-period value was read and no production file was modified.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
