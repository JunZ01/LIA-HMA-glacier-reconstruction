from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "extract_imd_adopted_temperature_layout_aware_v3.py"
SPEC = importlib.util.spec_from_file_location("imd_layout_v3", V3_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {V3_PATH}")
V3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V3)
B = V3.BASE

OUT = B.LOOP / "imd_ghcn_adopted_temperature_formula_checked_v4"
DRIVER = Path(__file__).resolve()


def all_stat_candidates(words, row_y, side, page_width):
    if side == "left":
        low_x, high_x = page_width * 0.12, page_width * 0.32
    else:
        low_x, high_x = page_width * 0.515, page_width * 0.70
    tolerance = max(30.0, page_width * 0.014)
    candidates = []
    for word in words:
        if abs(float(word["y"]) - row_y) > tolerance:
            continue
        for center_x, value, token in V3.numeric_parts(
            str(word["word"]), float(word["x"]), float(word["width"])
        ):
            if low_x <= center_x <= high_x:
                candidates.append((center_x, value, token, float(word["confidence"])))
    return sorted(candidates, key=lambda item: item[0])


def cluster_columns(rows, page_width):
    points = sorted(candidate[0] for row in rows for candidate in row)
    tolerance = page_width * 0.018
    clusters = []
    for point in points:
        if not clusters or point - sum(clusters[-1]) / len(clusters[-1]) > tolerance:
            clusters.append([point])
        else:
            clusters[-1].append(point)
    viable = [cluster for cluster in clusters if len(cluster) >= 4]
    viable.sort(key=lambda cluster: sum(cluster) / len(cluster))
    return [sum(cluster) / len(cluster) for cluster in viable[:6]]


def align_columns(candidates, centers, page_width):
    aligned = [None] * len(centers)
    used = set()
    for column, center in enumerate(centers):
        ranked = sorted(
            (
                (abs(candidate[0] - center), index, candidate)
                for index, candidate in enumerate(candidates)
                if index not in used
            ),
            key=lambda item: item[0],
        )
        if ranked and ranked[0][0] <= page_width * 0.025:
            _, index, candidate = ranked[0]
            aligned[column] = candidate
            used.add(index)
    return aligned


def selected_month_value(station, columns):
    def value(index):
        return columns[index][1] if len(columns) > index and columns[index] else math.nan

    if station in {"Simla", "Murree"}:
        hour16, minimum = value(2), value(5)
        if math.isnan(hour16) or math.isnan(minimum):
            return math.nan, "missing_16h_or_minimum"
        computed = (hour16 + minimum) / 2.0
        printed = value(0)
        status = (
            "formula_and_printed_agree"
            if not math.isnan(printed) and abs(printed - computed) <= 0.7
            else "formula_authoritative_printed_ocr_disagrees_or_missing"
        )
        return computed, status
    if station == "Darjeeling":
        adopted, maximum, minimum = value(2), value(3), value(5)
        if math.isnan(adopted):
            return math.nan, "missing_adopted_mean_column"
        if not math.isnan(maximum) and not math.isnan(minimum) and not minimum <= adopted <= maximum:
            return math.nan, "adopted_mean_outside_mean_extremes"
        return adopted, "printed_adopted_mean_with_extreme_check"
    if station == "Leh":
        printed = value(0)
        synoptic = [value(index) for index in range(1, 5)]
        if all(not math.isnan(item) for item in synoptic):
            computed = sum(synoptic) / 4.0
            if not math.isnan(printed) and abs(printed - computed) <= 0.7:
                return printed, "four_hour_formula_and_printed_agree"
            return computed, "four_hour_formula_authoritative"
        return printed, "printed_mean_four_hour_check_incomplete"
    return math.nan, "unknown_station_rule"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    panel = B.read_csv(B.PANEL / "station_year_page_panel.csv")
    ghcn_rows = B.read_csv(B.PANEL / "ghcn_annual_overlap.csv")
    ghcn = {
        (row["station_name"], row["series"], int(row["year"])): float(row["annual_mean_c"])
        for row in ghcn_rows
    }
    station_rows = []
    month_rows = []

    for source in panel:
        station = source["imd_station"]
        year = int(source["report_year"])
        suffix = source["item_id"].rsplit(".", 1)[-1]
        leaf = int(source["leaf"])
        page_key = f"{suffix}_{leaf:04d}"
        words = []
        for row in B.read_csv(B.OCR / "pages" / page_key / "words.csv"):
            words.append(
                {
                    "word": row["word"], "x": float(row["x"]), "y": float(row["y"]),
                    "width": float(row["width"]), "height": float(row["height"]),
                    "confidence": float(row["confidence"]),
                }
            )
        with Image.open(source["image_path"]) as image:
            page_width, page_height = image.size
        heading, score = B.locate_heading(words, station)
        if heading is None:
            station_rows.append({**source, "page_key": page_key, "status": "heading_not_found"})
            continue
        heading_x = float(heading["x"]) + float(heading["width"]) / 2.0
        heading_y = float(heading["y"])
        side = "left" if heading_x < page_width / 2 else "right"
        years = B.label_rows(words, "year", heading_y + page_height * 0.05, page_height)
        same_side_years = [word for word in years if (float(word["x"]) < page_width / 2) == (side == "left")]
        year_word = (same_side_years or years)[0] if years else None
        if year_word is None:
            station_rows.append({**source, "page_key": page_key, "status": "year_row_not_found"})
            continue
        year_y = float(year_word["y"])

        raw_months = []
        for month_index, month in enumerate(B.MONTHS, start=1):
            labels = B.label_rows(words, month, heading_y, year_y)
            same_side = [word for word in labels if (float(word["x"]) < page_width / 2) == (side == "left")]
            label = (same_side or labels)[0] if labels else None
            if label is None:
                continue
            candidates = all_stat_candidates(words, float(label["y"]), side, page_width)
            raw_months.append((month_index, month, candidates))
        centers = cluster_columns([row[2] for row in raw_months], page_width)
        if len(centers) < 6:
            station_rows.append(
                {**source, "page_key": page_key, "station_side": side,
                 "detected_columns": len(centers), "status": "fewer_than_six_columns"}
            )
            continue

        accepted_values = []
        formula_agree = 0
        for month_index, month, candidates in raw_months:
            columns = align_columns(candidates, centers, page_width)
            selected, rule_status = selected_month_value(station, columns)
            if not math.isnan(selected) and -20.0 <= selected <= 100.0:
                accepted_values.append(selected)
            if "agree" in rule_status:
                formula_agree += 1
            month_rows.append(
                {
                    "report_year": year,
                    "station": station,
                    "page_key": page_key,
                    "month": month_index,
                    "month_name": month,
                    "selected_value_f": selected,
                    "rule_status": rule_status,
                    "column_centers": ";".join(f"{center:.1f}" for center in centers),
                    "aligned_values": ";".join(
                        "" if value is None else f"{value[1]:.1f}@{value[0]:.1f}"
                        for value in columns
                    ),
                }
            )

        annual_candidates = all_stat_candidates(words, year_y, side, page_width)
        annual_columns = align_columns(annual_candidates, centers, page_width)
        printed_index = 2 if station == "Darjeeling" else 0
        printed_annual_f = (
            annual_columns[printed_index][1]
            if len(annual_columns) > printed_index and annual_columns[printed_index]
            else math.nan
        )
        reconstructed_annual_f = (
            sum(accepted_values) / len(accepted_values) if accepted_values else math.nan
        )
        closure = reconstructed_annual_f - printed_annual_f if not math.isnan(printed_annual_f) else math.nan
        accepted = len(accepted_values) >= 10 and score >= 0.72
        annual_c = (reconstructed_annual_f - 32.0) / 1.8 if accepted else math.nan
        output = {
            **source,
            "page_key": page_key,
            "station_side": side,
            "heading_score": score,
            "detected_columns": len(centers),
            "column_centers": ";".join(f"{center:.1f}" for center in centers),
            "accepted_months": len(accepted_values),
            "formula_printed_agreement_months": formula_agree,
            "reconstructed_annual_f": reconstructed_annual_f,
            "printed_annual_f": printed_annual_f,
            "reconstructed_minus_printed_f": closure,
            "status": "accepted" if accepted else "failed_quality_gate",
        }
        for series in ("qcu", "qcf"):
            observed = ghcn.get((source["ghcn_station_name"], series, year), math.nan)
            output[f"ghcn_{series}_annual_c"] = observed
            output[f"imd_minus_{series}_c"] = (
                annual_c - observed if accepted and not math.isnan(observed) else math.nan
            )
        station_rows.append(output)

    fields = []
    for row in station_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / "station_year_extraction.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(station_rows)
    with (OUT / "monthly_formula_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(month_rows[0]))
        writer.writeheader()
        writer.writerows(month_rows)

    accepted_rows = [row for row in station_rows if row["status"] == "accepted"]
    comparisons = [row for row in accepted_rows if not math.isnan(float(row.get("ghcn_qcu_annual_c", math.nan)))]
    by_station = defaultdict(list)
    for row in comparisons:
        by_station[row["imd_station"]].append(row)
    station_summary = []
    for station, rows in sorted(by_station.items()):
        record = {"station": station, "comparison_years": len(rows)}
        for series in ("qcu", "qcf"):
            values = [float(row[f"imd_minus_{series}_c"]) for row in rows if not math.isnan(float(row[f"imd_minus_{series}_c"]))]
            record[f"n_{series}"] = len(values)
            record[f"mean_imd_minus_{series}_c"] = sum(values) / len(values) if values else math.nan
        station_summary.append(record)
    if station_summary:
        with (OUT / "station_difference_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(station_summary[0]))
            writer.writeheader()
            writer.writerows(station_summary)

    all_qcu = [float(row["imd_minus_qcu_c"]) for row in comparisons]
    all_qcf = [float(row["imd_minus_qcf_c"]) for row in comparisons]
    summary = {
        "experiment": "imd_ghcn_adopted_temperature_formula_checked_v4",
        "requested_station_years": len(station_rows),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn": len(comparisons),
        "mean_imd_minus_qcu_c": sum(all_qcu) / len(all_qcu) if all_qcu else None,
        "mean_imd_minus_qcf_c": sum(all_qcf) / len(all_qcf) if all_qcf else None,
        "locked_period_accessed": False,
        "source_files_modified": False,
        "supersedes": "imd_ghcn_adopted_temperature_layout_aware_v3",
        "decision": "inspect_formula_audit_and_add_ia_year_row_adjudication",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": B.sha256(DRIVER)},
                "station_panel": str(B.PANEL / "station_year_page_panel.csv"),
                "matlab_ocr": str(B.OCR / "summary.json"),
                "method_evidence": str(B.LOOP / "imd_1886_air_temperature_mean_method_v1"),
            },
            indent=2,
        ) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
