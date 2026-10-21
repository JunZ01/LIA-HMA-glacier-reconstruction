from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
IA_MONTHLY = LOOP / "imd_ia_fulltext_monthly_values_v5/monthly_extraction.csv"
IA_STATIONS = LOOP / "imd_ia_fulltext_monthly_values_v5/station_year_extraction.csv"
PANEL = LOOP / "imd_ghcn_overlap_pages_v2"
OUT = LOOP / "imd_ia_formula_recovery_v6"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_list(text: str) -> list[float]:
    values = []
    for token in (text or "").split(";"):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def recover_month(station: str, tokens: list[float]) -> tuple[float, str, float, float]:
    candidates = []
    for start in range(min(4, max(0, len(tokens) - 4))):
        values = tokens[start:]
        if station in {"Simla", "Murree"}:
            if len(values) >= 6:
                mean, hour16, maximum, daily_range, minimum = values[0], values[2], values[3], values[4], values[5]
                range_error = abs((maximum - minimum) - daily_range)
                formula_mean = (hour16 + minimum) / 2.0
                formula_error = abs(mean - formula_mean)
                candidates.append((range_error + 0.25 * formula_error, range_error, formula_error, formula_mean, f"complete_start_{start}"))
            if len(values) >= 5:
                hour16, maximum, daily_range, minimum = values[1], values[2], values[3], values[4]
                range_error = abs((maximum - minimum) - daily_range)
                formula_mean = (hour16 + minimum) / 2.0
                candidates.append((range_error + 0.5, range_error, math.nan, formula_mean, f"missing_printed_mean_start_{start}"))
        elif station == "Darjeeling":
            if len(values) >= 6:
                adopted, maximum, daily_range, minimum = values[2], values[3], values[4], values[5]
                range_error = abs((maximum - minimum) - daily_range)
                physical_error = 0.0 if minimum <= adopted <= maximum else 10.0
                candidates.append((range_error + physical_error, range_error, math.nan, adopted, f"complete_start_{start}"))
            if len(values) >= 5:
                adopted, maximum, daily_range, minimum = values[1], values[2], values[3], values[4]
                range_error = abs((maximum - minimum) - daily_range)
                physical_error = 0.0 if minimum <= adopted <= maximum else 10.0
                candidates.append((range_error + physical_error + 0.5, range_error, math.nan, adopted, f"missing_first_observation_start_{start}"))
    if not candidates:
        return math.nan, "no_layout_candidate", math.nan, math.nan
    _, range_error, formula_error, selected, layout = min(candidates, key=lambda row: row[0])
    if range_error > 2.0 or not -20.0 <= selected <= 100.0:
        return math.nan, "range_closure_failed", range_error, formula_error
    return selected, layout, range_error, formula_error


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    monthly = read_csv(IA_MONTHLY)
    station_meta = {(row["report_year"], row["imd_station"]): row for row in read_csv(IA_STATIONS)}
    recovered = []
    values_by_station = defaultdict(list)
    for row in monthly:
        key = (row["report_year"], row["station"])
        side = station_meta.get(key, {}).get("side", "")
        tokens = parse_list(row["all_temperature_tokens"])
        if side != "left" or row["station"] == "Leh" or not tokens:
            value, status, range_error, formula_error = math.nan, "not_supported_left_layout", math.nan, math.nan
        else:
            value, status, range_error, formula_error = recover_month(row["station"], tokens)
        if not math.isnan(value):
            values_by_station[key].append(value)
        recovered.append(
            {
                "report_year": row["report_year"],
                "station": row["station"],
                "month": row["month"],
                "side": side,
                "tokens": row["all_temperature_tokens"],
                "selected_value_f": value,
                "layout_status": status,
                "range_closure_error_f": range_error,
                "formula_printed_error_f": formula_error,
            }
        )

    panel = read_csv(PANEL / "station_year_page_panel.csv")
    ghcn_rows = read_csv(PANEL / "ghcn_annual_overlap.csv")
    ghcn = {
        (row["station_name"], row["series"], int(row["year"])): float(row["annual_mean_c"])
        for row in ghcn_rows
    }
    station_output = []
    for source in panel:
        key = (source["report_year"], source["imd_station"])
        values = values_by_station[key]
        annual_f = sum(values) / len(values) if values else math.nan
        accepted = len(values) >= 10
        annual_c = (annual_f - 32.0) / 1.8 if accepted else math.nan
        output = {
            **source,
            "accepted_months": len(values),
            "formula_recovered_annual_f": annual_f,
            "formula_recovered_annual_c": annual_c,
            "status": "accepted" if accepted else "failed_quality_gate",
        }
        for series in ("qcu", "qcf"):
            observed = ghcn.get((source["ghcn_station_name"], series, int(source["report_year"])), math.nan)
            output[f"ghcn_{series}_annual_c"] = observed
            output[f"imd_minus_{series}_c"] = (
                annual_c - observed if accepted and not math.isnan(observed) else math.nan
            )
        station_output.append(output)

    with (OUT / "monthly_formula_recovery.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(recovered[0]))
        writer.writeheader()
        writer.writerows(recovered)
    fields = []
    for row in station_output:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / "station_year_formula_recovery.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(station_output)

    accepted_rows = [row for row in station_output if row["status"] == "accepted"]
    comparisons = [row for row in accepted_rows if not math.isnan(as_float(row["ghcn_qcu_annual_c"]))]
    qcu = [as_float(row["imd_minus_qcu_c"]) for row in comparisons]
    qcf = [as_float(row["imd_minus_qcf_c"]) for row in comparisons]
    summary = {
        "experiment": "imd_ia_formula_recovery_v6",
        "requested_station_years": len(station_output),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn": len(comparisons),
        "mean_imd_minus_qcu_c": sum(qcu) / len(qcu) if qcu else None,
        "mean_imd_minus_qcf_c": sum(qcf) / len(qcf) if qcf else None,
        "supported_layout": "left_half Simla/Murree/Darjeeling",
        "locked_period_accessed": False,
        "source_files_modified": False,
        "decision": "compare_formula_recovery_with_dual_ocr_adjudication",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "ia_monthly": {"path": str(IA_MONTHLY), "sha256": sha256(IA_MONTHLY)},
                "ia_station_meta": {"path": str(IA_STATIONS), "sha256": sha256(IA_STATIONS)},
            },
            indent=2,
        ) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
