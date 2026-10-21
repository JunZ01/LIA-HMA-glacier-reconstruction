from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
PANEL = LOOP / "imd_ghcn_overlap_pages_v2"
IA = LOOP / "temperature_historical_imd_direct_station_table_recovery_v1/official_fulltext_search_json"
MATLAB = LOOP / "imd_ghcn_adopted_temperature_formula_checked_v5/station_year_extraction.csv"
OUT = LOOP / "imd_ia_fulltext_monthly_values_v1"
DRIVER = Path(__file__).resolve()
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_temperatures(text: str) -> list[float]:
    cleaned = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)\b", " ", text, flags=re.I)
    cleaned = cleaned.upper().replace("O", "0")
    pattern = re.compile(r"(?<!\d)(-?\d{1,3})\s*[°'’`:.-]\s*(\d)(?!\d)")
    values = []
    for match in pattern.finditer(cleaned):
        whole = int(match.group(1))
        sign = -1 if whole < 0 else 1
        value = sign * (abs(whole) + int(match.group(2)) / 10.0)
        if -20.0 <= value <= 120.0:
            values.append(value)
    return values


def table_text(item_id: str, station: str, leaf: int) -> tuple[str, str]:
    suffix = item_id.rsplit(".", 1)[-1]
    path = IA / suffix / f"{station.lower()}.json"
    if not path.exists():
        return "", "search_json_missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for match in payload.get("matches", []):
        leaves = {int(par.get("page", -1)) for par in (match.get("par") or [])}
        text = str(match.get("text", ""))
        if leaf in leaves and sum(month in text.lower() for month in MONTHS) >= 8:
            matches.append(text)
    if not matches:
        return "", "table_match_missing"
    return max(matches, key=len), "located"


def month_segment(text: str, month_index: int) -> str:
    month = MONTHS[month_index]
    match = re.search(rf"\b{month}\w*\b", text, flags=re.I)
    if match is None:
        return ""
    start = match.end()
    ends = []
    for later in MONTHS[month_index + 1:]:
        later_match = re.search(rf"\b{later}\w*\b", text[start:], flags=re.I)
        if later_match is not None:
            ends.append(start + later_match.start())
    year_match = re.search(r"\byear\b", text[start:], flags=re.I)
    if year_match is not None:
        ends.append(start + year_match.start())
    end = min(ends) if ends else len(text)
    return text[start:end]


def split_station_group(values: list[float], side: str) -> list[float]:
    if len(values) >= 18:
        boundary = 9
    else:
        boundary = len(values) // 2
    return values[:boundary] if side == "left" else values[boundary:]


def validate_month(station: str, columns: list[float]) -> tuple[float, str]:
    if len(columns) < 6:
        return math.nan, "fewer_than_six_statistics"
    adopted, hour16, mean_maximum, mean_minimum = columns[0], columns[2], columns[3], columns[5]
    if station in {"Simla", "Murree"}:
        computed = (hour16 + mean_minimum) / 2.0
        if abs(adopted - computed) <= 0.8:
            return adopted, "formula_and_printed_agree"
        return math.nan, "formula_disagreement"
    if station == "Darjeeling":
        adopted = columns[2]
        if mean_minimum <= adopted <= mean_maximum:
            return adopted, "adopted_between_mean_extremes"
        return math.nan, "adopted_outside_mean_extremes"
    if station == "Leh":
        synoptic = columns[1:5]
        computed = sum(synoptic) / 4.0
        if abs(adopted - computed) <= 0.8:
            return adopted, "four_hour_formula_and_printed_agree"
        return math.nan, "four_hour_formula_disagreement"
    return math.nan, "unknown_station"


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    panel = read_csv(PANEL / "station_year_page_panel.csv")
    ghcn_rows = read_csv(PANEL / "ghcn_annual_overlap.csv")
    ghcn = {
        (row["station_name"], row["series"], int(row["year"])): float(row["annual_mean_c"])
        for row in ghcn_rows
    }
    matlab_rows = read_csv(MATLAB)
    matlab = {(row["report_year"], row["imd_station"]): row for row in matlab_rows}
    station_results = []
    month_results = []

    for source in panel:
        year = int(source["report_year"])
        station = source["imd_station"]
        text, text_status = table_text(source["item_id"], station, int(source["leaf"]))
        geometry = matlab.get((str(year), station), {})
        side = geometry.get("station_side", "")
        if not text or side not in {"left", "right"}:
            station_results.append({**source, "text_status": text_status, "side": side, "status": "table_or_side_missing"})
            continue
        accepted_values = []
        for index, month in enumerate(MONTHS):
            segment = month_segment(text, index)
            values = parse_temperatures(segment)
            group = split_station_group(values, side)
            columns = group[:6]
            selected, status = validate_month(station, columns)
            if not math.isnan(selected):
                accepted_values.append(selected)
            month_results.append(
                {
                    "report_year": year,
                    "station": station,
                    "month": index + 1,
                    "month_name": month,
                    "side": side,
                    "all_temperature_tokens": ";".join(f"{value:.1f}" for value in values),
                    "station_group_tokens": ";".join(f"{value:.1f}" for value in group),
                    "first_six_statistics": ";".join(f"{value:.1f}" for value in columns),
                    "selected_value_f": selected,
                    "status": status,
                }
            )
        annual_f = sum(accepted_values) / len(accepted_values) if accepted_values else math.nan
        accepted = len(accepted_values) >= 10
        annual_c = (annual_f - 32.0) / 1.8 if accepted else math.nan
        matlab_annual = as_float(geometry.get("reconstructed_annual_f", ""))
        output = {
            **source,
            "text_status": text_status,
            "side": side,
            "accepted_months": len(accepted_values),
            "ia_reconstructed_annual_f": annual_f,
            "matlab_reconstructed_annual_f": matlab_annual,
            "ia_minus_matlab_annual_f": annual_f - matlab_annual if not math.isnan(matlab_annual) else math.nan,
            "status": "accepted" if accepted else "failed_quality_gate",
        }
        for series in ("qcu", "qcf"):
            observed = ghcn.get((source["ghcn_station_name"], series, year), math.nan)
            output[f"ghcn_{series}_annual_c"] = observed
            output[f"imd_minus_{series}_c"] = (
                annual_c - observed if accepted and not math.isnan(observed) else math.nan
            )
        station_results.append(output)

    fields = []
    for row in station_results:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / "station_year_extraction.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(station_results)
    with (OUT / "monthly_extraction.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(month_results[0]))
        writer.writeheader()
        writer.writerows(month_results)

    accepted_rows = [row for row in station_results if row["status"] == "accepted"]
    comparisons = [row for row in accepted_rows if not math.isnan(as_float(row.get("ghcn_qcu_annual_c", "")))]
    qcu = [as_float(row["imd_minus_qcu_c"]) for row in comparisons]
    qcf = [as_float(row["imd_minus_qcf_c"]) for row in comparisons]
    summary = {
        "experiment": "imd_ia_fulltext_monthly_values_v1",
        "requested_station_years": len(station_results),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn": len(comparisons),
        "mean_imd_minus_qcu_c": sum(qcu) / len(qcu) if qcu else None,
        "mean_imd_minus_qcf_c": sum(qcf) / len(qcf) if qcf else None,
        "locked_period_accessed": False,
        "source_files_modified": False,
        "decision": "inspect_ia_group_boundaries_and_cross_engine_agreement",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "ia_search_root": str(IA),
                "station_panel": str(PANEL / "station_year_page_panel.csv"),
                "matlab_comparator": str(MATLAB),
            },
            indent=2,
        ) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
