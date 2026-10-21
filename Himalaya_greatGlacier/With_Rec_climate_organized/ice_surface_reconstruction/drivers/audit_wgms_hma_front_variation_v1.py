from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


GENERAL_SUFFIX = "A-GENERAL-INFORMATION.csv"
FRONT_SUFFIX = "C-FRONT-VARIATION.csv"
HMA_COUNTRIES = {"AF", "BT", "CN", "IN", "KG", "NP", "PK", "TJ"}
LOCKED_START = 1921
LOCKED_END = 1939


def iter_cr_records(stream, chunk_size: int = 65536):
    """Yield records without decoding bytes beyond the current record."""
    pending = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        pending += chunk
        parts = pending.split(b"\r")
        pending = parts.pop()
        for part in parts:
            part = part.lstrip(b"\n")
            if part:
                yield part
    pending = pending.lstrip(b"\n")
    if pending:
        yield pending


def parse_csv_record(record: bytes) -> list[str]:
    text = record.decode("latin1")
    return next(csv.reader([text], delimiter=";", quotechar='"'))


def parse_year(token: bytes) -> int | None:
    match = re.search(rb"(?:18|19|20)\d{2}", token.strip())
    return int(match.group(0)) if match else None


def parse_float(token: str) -> float | None:
    token = token.strip().replace(",", ".")
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix} member, found {matches}")
    return matches[0]


def region_for(longitude: float) -> str:
    if longitude < 80:
        return "west"
    if longitude < 90:
        return "central"
    return "east"


def main(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        general_member = locate_member(names, GENERAL_SUFFIX)
        front_member = locate_member(names, FRONT_SUFFIX)

        glaciers: dict[str, dict[str, object]] = {}
        with archive.open(general_member) as stream:
            records = iter_cr_records(stream)
            header = parse_csv_record(next(records))
            indices = {name: index for index, name in enumerate(header)}
            required = {"POLITICAL_UNIT", "NAME", "WGMS_ID", "LATITUDE", "LONGITUDE"}
            if not required.issubset(indices):
                raise RuntimeError(f"Missing general fields: {required - set(indices)}")
            for record in records:
                row = parse_csv_record(record)
                if len(row) < len(header):
                    row.extend([""] * (len(header) - len(row)))
                country = row[indices["POLITICAL_UNIT"]].strip()
                glacier_id = row[indices["WGMS_ID"]].strip()
                latitude = parse_float(row[indices["LATITUDE"]])
                longitude = parse_float(row[indices["LONGITUDE"]])
                if (
                    country not in HMA_COUNTRIES
                    or not glacier_id
                    or latitude is None
                    or longitude is None
                    or not (25.0 <= latitude <= 42.0 and 65.0 <= longitude <= 105.0)
                ):
                    continue
                glaciers[glacier_id] = {
                    "political_unit": country,
                    "name": row[indices["NAME"]].strip(),
                    "wgms_id": glacier_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "region": region_for(longitude),
                }

        audit = {
            "records_seen": 0,
            "locked_survey_year_skipped_before_value_decode": 0,
            "locked_reference_year_skipped_before_value_decode": 0,
            "cross_locked_interval_skipped_before_value_decode": 0,
            "post_1940_unknown_reference_skipped_before_value_decode": 0,
            "outside_allowed_period_skipped_before_value_decode": 0,
            "non_hma_skipped_before_value_decode": 0,
            "allowed_rows_decoded": 0,
        }
        allowed_rows: list[dict[str, object]] = []
        with archive.open(front_member) as stream:
            records = iter_cr_records(stream)
            header_record = next(records)
            header = parse_csv_record(header_record)
            if header[:10] != [
                "POLITICAL_UNIT",
                "NAME",
                "WGMS_ID",
                "YEAR",
                "FRONT_VARIATION",
                "FRONT_VAR_ACY",
                "QUALITATIVE_VARIATION",
                "SURVEY_DATE",
                "SURVEY_METHOD",
                "REFERENCE_DATE",
            ]:
                raise RuntimeError(f"Unexpected front-variation header: {header[:10]}")

            for record in records:
                audit["records_seen"] += 1
                fields = record.split(b";", 13)
                if len(fields) < 10:
                    continue
                glacier_id = fields[2].strip().decode("ascii", "ignore")
                survey_year = parse_year(fields[3])
                reference_year = parse_year(fields[9])

                if survey_year is not None and LOCKED_START <= survey_year <= LOCKED_END:
                    audit["locked_survey_year_skipped_before_value_decode"] += 1
                    continue
                if reference_year is not None and LOCKED_START <= reference_year <= LOCKED_END:
                    audit["locked_reference_year_skipped_before_value_decode"] += 1
                    continue
                if (
                    survey_year is not None
                    and reference_year is not None
                    and reference_year <= 1920
                    and survey_year >= 1940
                ):
                    audit["cross_locked_interval_skipped_before_value_decode"] += 1
                    continue
                if survey_year is None or not (
                    1850 <= survey_year <= 1920 or 1940 <= survey_year <= 1950
                ):
                    audit["outside_allowed_period_skipped_before_value_decode"] += 1
                    continue
                if survey_year >= 1940 and reference_year is None:
                    audit["post_1940_unknown_reference_skipped_before_value_decode"] += 1
                    continue
                if glacier_id not in glaciers:
                    audit["non_hma_skipped_before_value_decode"] += 1
                    continue

                row = parse_csv_record(record)
                if len(row) < len(header):
                    row.extend([""] * (len(header) - len(row)))
                meta = glaciers[glacier_id]
                front_variation = parse_float(row[4])
                qualitative = row[6].strip()
                period = "1850_1920" if survey_year <= 1920 else "1940_1950"
                allowed_rows.append(
                    {
                        **meta,
                        "period": period,
                        "survey_year": survey_year,
                        "reference_year": reference_year,
                        "front_variation_m": front_variation,
                        "front_variation_accuracy_m": parse_float(row[5]),
                        "qualitative_variation": qualitative,
                        "survey_method": row[8].strip(),
                        "publication": row[10].strip() if len(row) > 10 else "",
                    }
                )
                audit["allowed_rows_decoded"] += 1

    glacier_fields = [
        "political_unit",
        "name",
        "wgms_id",
        "latitude",
        "longitude",
        "region",
    ]
    with (output_dir / "hma_glacier_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=glacier_fields)
        writer.writeheader()
        writer.writerows(sorted(glaciers.values(), key=lambda row: (row["region"], row["name"])))

    allowed_fields = glacier_fields + [
        "period",
        "survey_year",
        "reference_year",
        "front_variation_m",
        "front_variation_accuracy_m",
        "qualitative_variation",
        "survey_method",
        "publication",
    ]
    with (output_dir / "allowed_front_variation_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=allowed_fields)
        writer.writeheader()
        writer.writerows(
            sorted(allowed_rows, key=lambda row: (row["period"], row["region"], row["name"], row["survey_year"]))
        )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in allowed_rows:
        grouped[(str(row["wgms_id"]), str(row["period"]))].append(row)

    summaries = []
    for (glacier_id, period), rows in grouped.items():
        values = [float(row["front_variation_m"]) for row in rows if row["front_variation_m"] is not None]
        meta = glaciers[glacier_id]
        summaries.append(
            {
                **meta,
                "period": period,
                "first_survey_year": min(int(row["survey_year"]) for row in rows),
                "last_survey_year": max(int(row["survey_year"]) for row in rows),
                "row_count": len(rows),
                "quantitative_row_count": len(values),
                "advance_row_count": sum(value > 0 for value in values),
                "retreat_row_count": sum(value < 0 for value in values),
                "stationary_row_count": sum(value == 0 for value in values),
                "sum_front_variation_m": sum(values) if values else None,
                "direction": (
                    "retreat" if values and sum(values) < 0 else "advance" if values and sum(values) > 0 else "neutral_or_unknown"
                ),
            }
        )

    summary_fields = glacier_fields + [
        "period",
        "first_survey_year",
        "last_survey_year",
        "row_count",
        "quantitative_row_count",
        "advance_row_count",
        "retreat_row_count",
        "stationary_row_count",
        "sum_front_variation_m",
        "direction",
    ]
    with (output_dir / "glacier_period_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda row: (row["period"], row["region"], row["name"])))

    quantitative_summaries = [row for row in summaries if row["quantitative_row_count"]]
    result = {
        "experiment": "wgms_hma_front_variation_audit_v1",
        "source_zip": str(zip_path),
        "source_sha256": sha256(zip_path),
        "hma_bbox": {"latitude": [25.0, 42.0], "longitude": [65.0, 105.0]},
        "allowed_periods": [[1850, 1920], [1940, 1950]],
        "locked_period": [LOCKED_START, LOCKED_END],
        "hma_glacier_count": len(glaciers),
        "allowed_front_variation_row_count": len(allowed_rows),
        "glacier_period_summary_count": len(summaries),
        "quantitative_glacier_period_count": len(quantitative_summaries),
        "retreating_glacier_period_count": sum(row["direction"] == "retreat" for row in quantitative_summaries),
        "advancing_glacier_period_count": sum(row["direction"] == "advance" for row in quantitative_summaries),
        "audit": audit,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# WGMS HMA front-variation audit v1",
        "",
        f"- HMA-bbox glacier inventory: `{len(glaciers)}`.",
        f"- Allowed front-variation rows: `{len(allowed_rows)}`.",
        f"- Glacier-period summaries with quantitative change: `{len(quantitative_summaries)}`.",
        f"- Retreating/advancing summaries: `{result['retreating_glacier_period_count']}` / `{result['advancing_glacier_period_count']}`.",
        f"- Locked survey rows skipped before value decode: `{audit['locked_survey_year_skipped_before_value_decode']}`.",
        f"- Locked reference rows skipped before value decode: `{audit['locked_reference_year_skipped_before_value_decode']}`.",
        f"- Cross-locked intervals skipped before value decode: `{audit['cross_locked_interval_skipped_before_value_decode']}`.",
        "",
        "This is an independent glacier-response direction audit. It is not a direct air-temperature reconstruction and cannot define a uniform temperature offset.",
    ]
    (output_dir / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: audit_wgms_hma_front_variation_v1.py SOURCE_ZIP OUTPUT_DIR")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
