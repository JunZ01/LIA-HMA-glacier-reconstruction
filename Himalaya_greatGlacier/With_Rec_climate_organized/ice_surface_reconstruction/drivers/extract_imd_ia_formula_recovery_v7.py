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
OUT = LOOP / "imd_ia_formula_recovery_v7"
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


def simla_murree_candidates(tokens: list[float]) -> list[tuple[float, float, float, str]]:
    """Return score, selected mean, closure error and template.

    The left-hand station starts at token zero. Templates account only for a
    single omitted OCR field; allowing arbitrary offsets created false matches
    in the adjacent station columns.
    """
    candidates = []

    def add(mean: float, hour16: float, maximum: float, daily_range: float, minimum: float,
            template: str, printed: bool = True) -> None:
        range_error = abs((maximum - minimum) - daily_range)
        formula_mean = (hour16 + minimum) / 2.0
        formula_error = abs(mean - formula_mean) if printed else 0.0
        physical = 0.0 if minimum <= mean <= maximum and minimum <= hour16 <= maximum + 10 else 25.0
        score = range_error + formula_error + physical
        if range_error <= 1.0 and formula_error <= 1.0 and physical == 0.0:
            candidates.append((score, formula_mean, max(range_error, formula_error), template))

    t = tokens
    if len(t) >= 6:
        add(t[0], t[2], t[3], t[4], t[5], "complete")
    if len(t) >= 5:
        # Adopted mean omitted: 10h, 16h, max, range, min.
        add((t[1] + t[4]) / 2.0, t[1], t[2], t[3], t[4], "missing_adopted", False)
        # 10h omitted: mean, 16h, max, range, min.
        add(t[0], t[1], t[2], t[3], t[4], "missing_10h")
        # Daily range omitted: mean, 10h, 16h, max, min.
        add(t[0], t[2], t[3], t[3] - t[4], t[4], "missing_range")
        # Minimum omitted: infer it from max - range.
        inferred_min = t[3] - t[4]
        add(t[0], t[2], t[3], t[4], inferred_min, "missing_minimum")
        # Maximum omitted: infer it from min + range.
        inferred_max = t[4] + t[3]
        add(t[0], t[2], inferred_max, t[3], t[4], "missing_maximum")
    if len(t) >= 4:
        # Both adopted mean and 10h omitted: 16h, max, range, min.
        add((t[0] + t[3]) / 2.0, t[0], t[1], t[2], t[3], "missing_adopted_and_10h", False)
    return candidates


def recover_month(station: str, tokens: list[float]) -> tuple[float, str, float]:
    if station in {"Simla", "Murree"}:
        candidates = simla_murree_candidates(tokens)
    elif station == "Darjeeling":
        candidates = []
        if len(tokens) >= 6:
            adopted, maximum, daily_range, minimum = tokens[2], tokens[3], tokens[4], tokens[5]
            error = abs((maximum - minimum) - daily_range)
            if error <= 1.0 and minimum <= adopted <= maximum:
                candidates.append((error, adopted, error, "complete"))
        if len(tokens) >= 5:
            # 10h omitted: 16h, adopted mean, max, range, min.
            adopted, maximum, daily_range, minimum = tokens[1], tokens[2], tokens[3], tokens[4]
            error = abs((maximum - minimum) - daily_range)
            if error <= 1.0 and minimum <= adopted <= maximum:
                candidates.append((error + 0.1, adopted, error, "missing_10h"))
    else:
        candidates = []
    if not candidates:
        return math.nan, "no_closed_row_start_template", math.nan
    score, selected, closure, template = min(candidates, key=lambda row: row[0])
    if not -20.0 <= selected <= 100.0:
        return math.nan, "implausible_temperature", closure
    return selected, template, closure


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
            value, status, closure = math.nan, "unsupported_layout", math.nan
        else:
            value, status, closure = recover_month(row["station"], tokens)
        if not math.isnan(value):
            values_by_station[key].append(value)
        recovered.append({
            "report_year": row["report_year"], "station": row["station"],
            "month": row["month"], "side": side,
            "tokens": row["all_temperature_tokens"], "selected_value_f": value,
            "layout_status": status, "closure_error_f": closure,
        })

    panel = read_csv(PANEL / "station_year_page_panel.csv")
    ghcn_rows = read_csv(PANEL / "ghcn_annual_overlap.csv")
    ghcn = {(r["station_name"], r["series"], int(r["year"])): float(r["annual_mean_c"]) for r in ghcn_rows}
    station_output = []
    for source in panel:
        key = (source["report_year"], source["imd_station"])
        values = values_by_station[key]
        annual_f = sum(values) / len(values) if values else math.nan
        accepted = len(values) >= 10
        annual_c = (annual_f - 32.0) / 1.8 if accepted else math.nan
        output = {**source, "accepted_months": len(values), "formula_recovered_annual_f": annual_f,
                  "formula_recovered_annual_c": annual_c,
                  "status": "accepted" if accepted else "failed_quality_gate"}
        for series in ("qcu", "qcf"):
            observed = ghcn.get((source["ghcn_station_name"], series, int(source["report_year"])), math.nan)
            output[f"ghcn_{series}_annual_c"] = observed
            output[f"imd_minus_{series}_c"] = annual_c - observed if accepted and not math.isnan(observed) else math.nan
        station_output.append(output)

    with (OUT / "monthly_formula_recovery.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(recovered[0]))
        writer.writeheader(); writer.writerows(recovered)
    fields = list(dict.fromkeys(key for row in station_output for key in row))
    with (OUT / "station_year_formula_recovery.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(station_output)

    accepted_rows = [row for row in station_output if row["status"] == "accepted"]
    comparisons = [row for row in accepted_rows if not math.isnan(as_float(row["ghcn_qcu_annual_c"]))]
    qcu = [as_float(row["imd_minus_qcu_c"]) for row in comparisons]
    qcf = [as_float(row["imd_minus_qcf_c"]) for row in comparisons]
    summary = {
        "experiment": "imd_ia_formula_recovery_v7",
        "requested_station_years": len(station_output),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn": len(comparisons),
        "mean_imd_minus_qcu_c": sum(qcu) / len(qcu) if qcu else None,
        "mean_imd_minus_qcf_c": sum(qcf) / len(qcf) if qcf else None,
        "locked_period_accessed": False, "source_files_modified": False,
        "decision": "retain_only_station_years_consistent_with_dual_ocr_and_printed_annual_rows",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "source_manifest.json").write_text(json.dumps({
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "ia_monthly": {"path": str(IA_MONTHLY), "sha256": sha256(IA_MONTHLY)},
        "ia_station_meta": {"path": str(IA_STATIONS), "sha256": sha256(IA_STATIONS)},
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
