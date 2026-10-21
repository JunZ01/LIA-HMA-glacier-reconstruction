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
PANEL = LOOP / "imd_ghcn_overlap_pages_v2"
IA = LOOP / "imd_ia_fulltext_monthly_values_v4/monthly_extraction.csv"
MATLAB = LOOP / "imd_ghcn_adopted_temperature_formula_checked_v5/monthly_formula_audit.csv"
OUT = LOOP / "imd_dual_ocr_monthly_temperature_adjudication_v1"
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


def as_float(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return math.nan


def first_statistic(text: str) -> float:
    if not text:
        return math.nan
    return as_float(text.split(";", 1)[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    panel = read_csv(PANEL / "station_year_page_panel.csv")
    ghcn_rows = read_csv(PANEL / "ghcn_annual_overlap.csv")
    ghcn = {
        (row["station_name"], row["series"], int(row["year"])): float(row["annual_mean_c"])
        for row in ghcn_rows
    }
    ia_rows = read_csv(IA)
    matlab_rows = read_csv(MATLAB)
    ia = {(row["report_year"], row["station"], row["month"]): row for row in ia_rows}
    matlab = {(row["report_year"], row["station"], row["month"]): row for row in matlab_rows}

    month_output = []
    accepted_by_station_year = defaultdict(list)
    for source in panel:
        year = source["report_year"]
        station = source["imd_station"]
        for month in range(1, 13):
            key = (year, station, str(month))
            ia_row = ia.get(key, {})
            matlab_row = matlab.get(key, {})
            ia_value = as_float(ia_row.get("selected_value_f"))
            matlab_value = as_float(matlab_row.get("selected_value_f"))
            ia_status = ia_row.get("status", "missing")
            matlab_status = matlab_row.get("rule_status", "missing")
            ia_exact = ia_status in {
                "formula_and_printed_agree",
                "four_hour_formula_and_printed_agree",
                "adopted_between_mean_extremes",
            }
            matlab_exact = matlab_status in {
                "formula_and_printed_agree",
                "four_hour_formula_and_printed_agree",
                "printed_adopted_mean_with_extreme_check",
            }
            ia_printed = first_statistic(ia_row.get("first_six_statistics", ""))
            selected = math.nan
            decision = "rejected"
            if ia_exact and matlab_exact:
                if abs(ia_value - matlab_value) <= 0.8:
                    selected = ia_value
                    decision = "dual_engine_agreement"
                else:
                    decision = "dual_engine_conflict"
            elif ia_exact:
                selected = ia_value
                decision = "ia_internal_formula_or_extreme_gate"
            elif matlab_exact:
                selected = matlab_value
                decision = "matlab_internal_formula_or_extreme_gate"
            elif (
                not math.isnan(ia_printed)
                and not math.isnan(matlab_value)
                and "formula_authoritative" in matlab_status
                and abs(ia_printed - matlab_value) <= 0.8
            ):
                selected = (ia_printed + matlab_value) / 2.0
                decision = "matlab_formula_matches_ia_printed_mean"

            if not math.isnan(selected):
                accepted_by_station_year[(year, station)].append(selected)
            month_output.append(
                {
                    "report_year": year,
                    "station": station,
                    "month": month,
                    "ia_value_f": ia_value,
                    "ia_status": ia_status,
                    "ia_printed_first_statistic_f": ia_printed,
                    "matlab_value_f": matlab_value,
                    "matlab_status": matlab_status,
                    "selected_value_f": selected,
                    "decision": decision,
                }
            )

    station_output = []
    for source in panel:
        key = (source["report_year"], source["imd_station"])
        values = accepted_by_station_year[key]
        accepted = len(values) >= 10
        annual_f = sum(values) / len(values) if values else math.nan
        annual_c = (annual_f - 32.0) / 1.8 if accepted else math.nan
        output = {
            **source,
            "accepted_months": len(values),
            "adjudicated_annual_f": annual_f,
            "adjudicated_annual_c": annual_c,
            "status": "accepted" if accepted else "failed_quality_gate",
        }
        for series in ("qcu", "qcf"):
            observed = ghcn.get((source["ghcn_station_name"], series, int(source["report_year"])), math.nan)
            output[f"ghcn_{series}_annual_c"] = observed
            output[f"imd_minus_{series}_c"] = (
                annual_c - observed if accepted and not math.isnan(observed) else math.nan
            )
        station_output.append(output)

    with (OUT / "monthly_adjudication.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(month_output[0]))
        writer.writeheader()
        writer.writerows(month_output)
    fields = []
    for row in station_output:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / "station_year_adjudication.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(station_output)

    accepted_rows = [row for row in station_output if row["status"] == "accepted"]
    comparisons = [row for row in accepted_rows if not math.isnan(as_float(str(row["ghcn_qcu_annual_c"])))]
    by_station = defaultdict(list)
    for row in comparisons:
        by_station[row["imd_station"]].append(row)
    station_summary = []
    for station, rows in sorted(by_station.items()):
        record = {"station": station, "comparison_years": len(rows)}
        for series in ("qcu", "qcf"):
            diffs = [as_float(str(row[f"imd_minus_{series}_c"])) for row in rows]
            diffs = [value for value in diffs if not math.isnan(value)]
            record[f"n_{series}"] = len(diffs)
            record[f"mean_imd_minus_{series}_c"] = sum(diffs) / len(diffs) if diffs else math.nan
            record[f"min_imd_minus_{series}_c"] = min(diffs) if diffs else math.nan
            record[f"max_imd_minus_{series}_c"] = max(diffs) if diffs else math.nan
        station_summary.append(record)
    if station_summary:
        with (OUT / "station_difference_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(station_summary[0]))
            writer.writeheader()
            writer.writerows(station_summary)

    qcu = [as_float(str(row["imd_minus_qcu_c"])) for row in comparisons]
    qcf = [as_float(str(row["imd_minus_qcf_c"])) for row in comparisons]
    decisions = defaultdict(int)
    for row in month_output:
        decisions[row["decision"]] += 1
    summary = {
        "experiment": "imd_dual_ocr_monthly_temperature_adjudication_v1",
        "requested_station_years": len(station_output),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn": len(comparisons),
        "mean_imd_minus_qcu_c": sum(qcu) / len(qcu) if qcu else None,
        "mean_imd_minus_qcf_c": sum(qcf) / len(qcf) if qcf else None,
        "monthly_decisions": dict(decisions),
        "locked_period_accessed": False,
        "source_files_modified": False,
        "decision": "inspect_accepted_station_years_and_compare_original_publication_to_qcu_qcf",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "ia_monthly": {"path": str(IA), "sha256": sha256(IA)},
                "matlab_monthly": {"path": str(MATLAB), "sha256": sha256(MATLAB)},
                "station_panel": str(PANEL / "station_year_page_panel.csv"),
            },
            indent=2,
        ) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
