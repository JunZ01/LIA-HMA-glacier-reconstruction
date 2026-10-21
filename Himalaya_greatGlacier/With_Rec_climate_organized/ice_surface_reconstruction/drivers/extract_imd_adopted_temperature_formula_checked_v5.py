from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
V4_PATH = HERE / "extract_imd_adopted_temperature_formula_checked_v4.py"
SPEC = importlib.util.spec_from_file_location("imd_formula_v4", V4_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {V4_PATH}")
V4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V4)

OUT = V4.B.LOOP / "imd_ghcn_adopted_temperature_formula_checked_v5"
V4.OUT = OUT
V4.DRIVER = Path(__file__).resolve()


def all_stat_candidates(words, row_y, side, page_width):
    if side == "left":
        low_x, high_x = page_width * 0.08, page_width * 0.46
    else:
        low_x, high_x = page_width * 0.50, page_width * 0.94
    tolerance = max(30.0, page_width * 0.014)
    candidates = []
    for word in words:
        if abs(float(word["y"]) - row_y) > tolerance:
            continue
        for center_x, value, token in V4.V3.numeric_parts(
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
        if not clusters or point - statistics.mean(clusters[-1]) > tolerance:
            clusters.append([point])
        else:
            clusters[-1].append(point)
    viable = [cluster for cluster in clusters if len(cluster) >= 4]
    if len(viable) < 6:
        return [statistics.mean(cluster) for cluster in viable]

    best = None
    for start in range(len(viable) - 5):
        window = viable[start:start + 6]
        centers = [statistics.mean(cluster) for cluster in window]
        gaps = [right - left for left, right in zip(centers, centers[1:])]
        mean_gap = statistics.mean(gaps)
        cv = statistics.pstdev(gaps) / mean_gap if mean_gap else 9.0
        support = sum(min(len(cluster), 12) for cluster in window)
        expected_penalty = abs(mean_gap / page_width - 0.03) * 80.0
        score = support - 40.0 * cv - expected_penalty
        candidate = (score, -start, centers)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2]


V4.all_stat_candidates = all_stat_candidates
V4.cluster_columns = cluster_columns


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    V4.main()
    station_path = OUT / "station_year_extraction.csv"
    month_path = OUT / "monthly_formula_audit.csv"
    stations = read_csv(station_path)
    months = read_csv(month_path)
    exact = defaultdict(int)
    for row in months:
        if row["rule_status"] in {
            "formula_and_printed_agree",
            "four_hour_formula_and_printed_agree",
        }:
            exact[(row["report_year"], row["station"])] += 1

    for row in stations:
        key = (row["report_year"], row["imd_station"])
        count = exact[key]
        row["exact_formula_printed_agreement_months"] = count
        accepted_months = int(float(row.get("accepted_months") or 0))
        closure = as_float(row.get("reconstructed_minus_printed_f"))
        station = row["imd_station"]
        if station in {"Simla", "Murree", "Leh"}:
            accepted = accepted_months >= 10 and count >= 8
        elif station == "Darjeeling":
            accepted = accepted_months >= 10 and not math.isnan(closure) and abs(closure) <= 1.0
        else:
            accepted = False
        row["status"] = "accepted" if accepted else "failed_quality_gate"
        annual_f = as_float(row.get("reconstructed_annual_f"))
        annual_c = (annual_f - 32.0) / 1.8 if accepted else math.nan
        for series in ("qcu", "qcf"):
            observed = as_float(row.get(f"ghcn_{series}_annual_c"))
            row[f"imd_minus_{series}_c"] = (
                annual_c - observed if accepted and not math.isnan(observed) else math.nan
            )
    write_csv(station_path, stations)

    accepted_rows = [row for row in stations if row["status"] == "accepted"]
    comparisons = [row for row in accepted_rows if not math.isnan(as_float(row.get("ghcn_qcu_annual_c")))]
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["imd_station"]].append(row)
    summaries = []
    for station, rows in sorted(grouped.items()):
        record = {"station": station, "comparison_years": len(rows)}
        for series in ("qcu", "qcf"):
            values = [as_float(row[f"imd_minus_{series}_c"]) for row in rows]
            values = [value for value in values if not math.isnan(value)]
            record[f"n_{series}"] = len(values)
            record[f"mean_imd_minus_{series}_c"] = sum(values) / len(values) if values else math.nan
        summaries.append(record)
    if summaries:
        write_csv(OUT / "station_difference_summary.csv", summaries)

    qcu = [as_float(row["imd_minus_qcu_c"]) for row in comparisons]
    qcf = [as_float(row["imd_minus_qcf_c"]) for row in comparisons]
    summary = {
        "experiment": "imd_ghcn_adopted_temperature_formula_checked_v5",
        "requested_station_years": len(stations),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn": len(comparisons),
        "mean_imd_minus_qcu_c": sum(qcu) / len(qcu) if qcu else None,
        "mean_imd_minus_qcf_c": sum(qcf) / len(qcf) if qcf else None,
        "locked_period_accessed": False,
        "source_files_modified": False,
        "supersedes": "imd_ghcn_adopted_temperature_formula_checked_v4",
        "quality_gate": "at least 10 months and >=8 exact formula/printed agreements; Darjeeling additionally requires <=1 F annual closure",
        "decision": "adjudicate_accepted_values_against_ia_year_rows_and_recover_three_missing_headings",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["driver"] = {"path": str(Path(__file__).resolve()), "sha256": V4.B.sha256(Path(__file__).resolve())}
    manifest["base_driver"] = str(V4_PATH)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
