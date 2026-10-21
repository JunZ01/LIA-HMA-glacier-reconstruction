from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
EXTRACT = LOOP / "imd_ghcn_adopted_temperature_extract_v1"
IA = LOOP / "temperature_historical_imd_direct_station_table_recovery_v1"
OUT = LOOP / "imd_matlab_ia_ocr_ghcn_adjudication_v1"
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


def numeric_tokens(text: str) -> list[float]:
    normalized = text.upper().replace("O", "0")
    tokens = re.findall(r"(?<!\d)\d{1,3}(?:[.'’`:°-]\d)?(?!\d)", normalized)
    values: list[float] = []
    for token in tokens:
        pieces = re.split(r"[.'’`:°-]", token)
        if len(pieces) == 2:
            value = float(f"{int(pieces[0])}.{pieces[1]}")
        else:
            digits = pieces[0]
            if len(digits) == 3:
                value = int(digits) / 10.0
            else:
                value = float(int(digits))
        if 15.0 <= value <= 100.0:
            values.append(value)
    return values


def ia_year_row(
    item_id: str, station: str, leaf: int
) -> tuple[str, list[float], str]:
    suffix = item_id.rsplit(".", 1)[-1]
    path = IA / "official_fulltext_search_json" / suffix / f"{station.lower()}.json"
    if not path.exists():
        return "", [], "ia_station_search_json_missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        match
        for match in payload.get("matches", [])
        if any(int(par.get("page", -1)) == leaf for par in match.get("par") or [])
    ]
    if not matches:
        return "", [], "ia_leaf_match_missing"
    text = max((str(match.get("text", "")) for match in matches), key=len)
    marker = text.find("{{{")
    if marker < 0:
        marker = 0
    year_match = re.search(r"\by[eo]ar\b", text[marker:], re.IGNORECASE)
    if year_match is None:
        return "", [], "ia_year_row_missing"
    start = marker + year_match.start()
    tail = text[start : start + 900]
    stop_candidates = []
    for pattern in [r"\*\s*Mean of", r"Mean of \d+ days", r"\d+\.\s+[A-Z]{3,}"]:
        match = re.search(pattern, tail[40:], re.IGNORECASE)
        if match is not None:
            stop_candidates.append(40 + match.start())
    end = min(stop_candidates) if stop_candidates else min(len(tail), 500)
    row_text = tail[:end]
    return row_text, numeric_tokens(row_text), "located"


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return math.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows = read_csv(EXTRACT / "station_year_extraction.csv")
    adjudicated: list[dict[str, object]] = []
    for row in rows:
        annual_f = as_float(row, "annual_adopted_mean_f")
        leaf = int(row["leaf"])
        ia_text, ia_values, ia_status = ia_year_row(
            row["item_id"], row["imd_station"], leaf
        )
        agreement_count = (
            sum(abs(value - annual_f) <= 0.051 for value in ia_values)
            if not math.isnan(annual_f)
            else 0
        )
        ia_unique_agreement = agreement_count == 1
        heading_score = as_float(row, "station_heading_score")
        x_distance = as_float(row, "annual_x_distance")
        width = as_float(row, "image_width")
        readable_months = int(row.get("readable_months") or 0)
        closure = as_float(row, "monthly_minus_annual_f")
        geometry_ok = (
            not math.isnan(annual_f)
            and heading_score >= 0.72
            and not math.isnan(x_distance)
            and x_distance <= width * 0.025
        )
        monthly_support_ok = readable_months >= 10
        strict_closure = not math.isnan(closure) and abs(closure) <= 0.25
        relaxed_closure = not math.isnan(closure) and abs(closure) <= 1.0
        accepted = (
            geometry_ok
            and monthly_support_ok
            and (
                strict_closure
                or (ia_unique_agreement and relaxed_closure)
            )
        )
        output = dict(row)
        output.update(
            {
                "ia_year_row_status": ia_status,
                "ia_year_row_text": ia_text,
                "ia_numeric_values": ";".join(f"{value:.1f}" for value in ia_values),
                "ia_matlab_value_agreement_count": agreement_count,
                "ia_unique_value_agreement": ia_unique_agreement,
                "geometry_ok": geometry_ok,
                "monthly_support_ok": monthly_support_ok,
                "strict_monthly_closure": strict_closure,
                "relaxed_monthly_closure": relaxed_closure,
                "final_numeric_status": "accepted" if accepted else "not_accepted",
            }
        )
        for series in ["qcu", "qcf"]:
            ghcn = as_float(row, f"ghcn_{series}_annual_c")
            annual_c = as_float(row, "annual_adopted_mean_c")
            output[f"final_imd_minus_{series}_c"] = (
                annual_c - ghcn
                if accepted and not math.isnan(ghcn)
                else math.nan
            )
        adjudicated.append(output)

    fields: list[str] = []
    for row in adjudicated:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / "station_year_adjudication.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(adjudicated)

    accepted_rows = [row for row in adjudicated if row["final_numeric_status"] == "accepted"]
    comparison_rows = [
        row
        for row in accepted_rows
        if not math.isnan(float(row["final_imd_minus_qcf_c"]))
        or not math.isnan(float(row["final_imd_minus_qcu_c"]))
    ]
    station_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in comparison_rows:
        station_groups[str(row["imd_station"])].append(row)
    station_summary: list[dict[str, object]] = []
    for station, group in sorted(station_groups.items()):
        record: dict[str, object] = {
            "station": station,
            "accepted_comparison_years": len(group),
        }
        for series in ["qcu", "qcf"]:
            values = [
                float(row[f"final_imd_minus_{series}_c"])
                for row in group
                if not math.isnan(float(row[f"final_imd_minus_{series}_c"]))
            ]
            record[f"years_with_{series}"] = len(values)
            record[f"mean_imd_minus_{series}_c"] = (
                sum(values) / len(values) if values else math.nan
            )
            record[f"min_imd_minus_{series}_c"] = min(values) if values else math.nan
            record[f"max_imd_minus_{series}_c"] = max(values) if values else math.nan
        station_summary.append(record)
    if station_summary:
        with (OUT / "station_difference_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(station_summary[0]))
            writer.writeheader()
            writer.writerows(station_summary)

    qcf_values = [
        float(row["final_imd_minus_qcf_c"])
        for row in comparison_rows
        if not math.isnan(float(row["final_imd_minus_qcf_c"]))
    ]
    qcu_values = [
        float(row["final_imd_minus_qcu_c"])
        for row in comparison_rows
        if not math.isnan(float(row["final_imd_minus_qcu_c"]))
    ]
    summary = {
        "experiment": "imd_matlab_ia_ocr_ghcn_adjudication_v1",
        "requested_station_years": len(adjudicated),
        "accepted_station_years": len(accepted_rows),
        "accepted_with_ghcn_comparison": len(comparison_rows),
        "ia_unique_agreement_station_years": sum(
            bool(row["ia_unique_value_agreement"]) for row in adjudicated
        ),
        "qcf_comparisons": len(qcf_values),
        "qcu_comparisons": len(qcu_values),
        "mean_imd_minus_qcf_c": sum(qcf_values) / len(qcf_values) if qcf_values else None,
        "mean_imd_minus_qcu_c": sum(qcu_values) / len(qcu_values) if qcu_values else None,
        "locked_period_values_read": False,
        "interpretation": (
            "publication transcription evidence tests GHCN processing but is "
            "not a second independent physical observation"
        ),
        "decision": "evaluate_station_and_year_dependence_before_bias_claim",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "coordinate_extraction": str(EXTRACT / "station_year_extraction.csv"),
                "ia_search_root": str(IA / "official_fulltext_search_json"),
            },
            indent=2,
            ensure_ascii=True,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        """# MATLAB-IA OCR and GHCN Adjudication Contract

- Preserve the frozen MATLAB coordinate extraction and station geometry.
- Require strict monthly closure or unique IA Year-row digit agreement plus <=1 F closure.
- Require at least 10 readable months and the existing coordinate-distance gate.
- Count IMD publication and GHCN as two processing chains of the same observation.
- Do not access 1921-1939, production data, EnKF, or OGGM.
""",
        encoding="utf-8",
    )
    result = f"""# MATLAB-IA OCR and GHCN Adjudication

Accepted {len(accepted_rows)} of {len(adjudicated)} station-years after geometry,
monthly support, and two-engine/closure adjudication. {len(comparison_rows)} have
matching GHCN annual values. Mean accepted differences are
{summary['mean_imd_minus_qcf_c']} C versus qcf and
{summary['mean_imd_minus_qcu_c']} C versus qcu.

These differences diagnose source processing. They are not independent climate
votes and are not yet a production correction.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
