from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USER_AGENT = "Mozilla/5.0 HMA-climate-time-axis-audit/1.0"
TIME_NAME = re.compile(r"(?:age|year|yr|date)", re.IGNORECASE)
DIRECT_TEMPERATURE = "temperature variable>temperature"
DELTA_18O = "delta 18O"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str) -> tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type", "")


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return normalized[:120] or "unnamed"


def split_tokens(line: str) -> list[str]:
    if "\t" in line:
        return [part.strip() for part in line.rstrip("\r\n").split("\t")]
    if "," in line:
        return [part.strip() for part in line.rstrip("\r\n").split(",")]
    return re.split(r"\s+", line.strip())


def data_lines(payload: bytes) -> tuple[list[str], list[str]]:
    text = payload.decode("utf-8", errors="replace")
    lines = text.splitlines()
    marker = None
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("# data:"):
            marker = index
    if marker is None:
        return [], []
    body = [line for line in lines[marker + 1 :] if line.strip() and not line.lstrip().startswith("#")]
    if not body:
        return [], []
    return split_tokens(body[0]), body[1:]


def time_column(headers: list[str]) -> int | None:
    candidates = [
        index
        for index, name in enumerate(headers)
        if TIME_NAME.search(name) and "depth" not in name.lower()
    ]
    return candidates[0] if candidates else None


def to_ce(value: float, time_unit: str) -> float:
    if "bp" in time_unit.lower():
        return 1950.0 - value
    return value


def audit_time_axis(
    payload: bytes, time_unit: str
) -> tuple[list[str], int | None, dict[str, Any]]:
    headers, rows = data_lines(payload)
    index = time_column(headers)
    result: dict[str, Any] = {
        "data_header_count": len(headers),
        "data_row_count": len(rows),
        "time_header": "",
        "time_tokens_decoded": 0,
        "invalid_time_tokens": 0,
        "locked_rows_skipped_before_non_time_conversion": 0,
        "modern_1940_2000_observation_count": 0,
        "modern_1940_2000_unique_integer_year_count": 0,
        "minimum_time_ce": None,
        "maximum_time_ce": None,
    }
    if index is None:
        return headers, None, result
    result["time_header"] = headers[index]
    allowed_times: list[float] = []
    modern_times: list[float] = []
    for line in rows:
        tokens = split_tokens(line)
        if index >= len(tokens):
            result["invalid_time_tokens"] += 1
            continue
        try:
            time_value = float(tokens[index])
        except ValueError:
            result["invalid_time_tokens"] += 1
            continue
        ce = to_ce(time_value, time_unit)
        result["time_tokens_decoded"] += 1
        if 1921 <= ce <= 1939:
            result["locked_rows_skipped_before_non_time_conversion"] += 1
            continue
        allowed_times.append(ce)
        if 1940 <= ce <= 2000:
            modern_times.append(ce)
    if allowed_times:
        result["minimum_time_ce"] = min(allowed_times)
        result["maximum_time_ce"] = max(allowed_times)
    result["modern_1940_2000_observation_count"] = len(modern_times)
    result["modern_1940_2000_unique_integer_year_count"] = len(
        {int(round(value)) for value in modern_times}
    )
    return headers, index, result


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    source_round = loop / "round_175_noaa_hma_non_tree_temperature_inventory_v2"
    inventory_path = source_round / "candidate_inventory.csv"
    output = loop / "round_176_noaa_non_tree_time_axis_audit_v2"
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw_unopened"
    raw.mkdir()
    source_hash_before = sha256(inventory_path)

    with inventory_path.open("r", encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    table_time_units: dict[tuple[str, str], str] = {}
    for metadata_path in (source_round / "study_metadata_unmodified").glob("*.json"):
        study = json.loads(metadata_path.read_text(encoding="utf-8"))["study"][0]
        study_id = str(study.get("NOAAStudyId", ""))
        for site in study.get("site", []) or []:
            for table in site.get("paleoData", []) or []:
                key = (study_id, str(table.get("dataTableName", "")))
                table_time_units[key] = str(table.get("timeUnit", ""))
    selected: list[dict[str, str]] = []
    for row in inventory:
        if row.get("disposition") != "metadata_candidate_needs_raw_identity_frequency_audit":
            continue
        variables = row.get("variable_descriptions", "")
        if DIRECT_TEMPERATURE not in variables and DELTA_18O not in variables:
            continue
        for url in row.get("data_file_urls", "").split(";"):
            if url.strip().lower().endswith(".txt"):
                item = dict(row)
                item["selected_url"] = url.strip()
                item["time_unit"] = table_time_units.get(
                    (row["noaa_study_id"], row["table_name"]), ""
                )
                selected.append(item)

    url_payloads: dict[str, tuple[Path, bytes, str, str]] = {}
    manifest: list[dict[str, Any]] = []
    for row in selected:
        url = row["selected_url"]
        if url in url_payloads:
            continue
        payload, final_url, content_type = fetch(url)
        basename = Path(urllib.parse.urlparse(url).path).name
        filename = f"{row['noaa_study_id']}__{safe_name(basename)}"
        path = raw / filename
        path.write_bytes(payload)
        url_payloads[url] = (path, payload, final_url, content_type)
        manifest.append(
            {
                "url": url,
                "final_url": final_url,
                "content_type": content_type,
                "filename": filename,
                "size_bytes": len(payload),
                "sha256": sha256(path),
            }
        )

    audits: list[dict[str, Any]] = []
    for row in selected:
        path, payload, _, _ = url_payloads[row["selected_url"]]
        headers, index, metrics = audit_time_axis(payload, row.get("time_unit", ""))
        variables = row.get("variable_descriptions", "")
        direct_temperature = DIRECT_TEMPERATURE in variables
        oxygen_isotope = DELTA_18O in variables
        hydro_interpretation = any(
            term in (row.get("study_name", "") + " " + variables).lower()
            for term in ("precipitation", "hydroclim", "lake level")
        )
        modern_count = int(metrics["modern_1940_2000_observation_count"])
        if "100yr" in row.get("table_name", "").lower():
            disposition = "exclude_declared_centennial_average"
        elif index is None:
            disposition = "exclude_no_row_time_axis"
        elif modern_count < 20:
            disposition = "exclude_fewer_than_20_modern_observations"
        elif direct_temperature:
            disposition = "direct_temperature_frequency_pass_needs_identity_oof"
        elif hydro_interpretation:
            disposition = "exclude_hydroclimate_isotope_interpretation"
        elif oxygen_isotope:
            disposition = "isotope_frequency_pass_needs_physical_orientation_identity"
        else:
            disposition = "exclude_no_temperature_interpretation"
        audits.append(
            {
                "noaa_study_id": row["noaa_study_id"],
                "study_name": row["study_name"],
                "site_name": row["site_name"],
                "table_name": row["table_name"],
                "time_unit": row.get("time_unit", ""),
                "source_filename": path.name,
                "source_sha256": sha256(path),
                "data_headers": ";".join(headers),
                "direct_temperature_metadata": direct_temperature,
                "oxygen_isotope_metadata": oxygen_isotope,
                "hydroclimate_interpretation": hydro_interpretation,
                **metrics,
                "non_time_numeric_tokens_decoded": False,
                "disposition": disposition,
            }
        )

    audits.sort(key=lambda row: (row["disposition"], row["noaa_study_id"], row["table_name"]))
    with (output / "time_axis_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audits[0]) if audits else [])
        if audits:
            writer.writeheader()
            writer.writerows(audits)
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="ascii"
    )
    counts: dict[str, int] = {}
    for row in audits:
        key = str(row["disposition"])
        counts[key] = counts.get(key, 0) + 1
    advancing = [row for row in audits if "pass_needs" in str(row["disposition"])]
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "downloaded_file_count": len(manifest),
        "audited_table_count": len(audits),
        "disposition_counts": counts,
        "advancing_table_keys": [
            f"{row['noaa_study_id']}::{row['table_name']}" for row in advancing
        ],
        "annual_files_downloaded_unmodified": True,
        "only_time_tokens_converted": True,
        "non_time_numeric_values_decoded": False,
        "locked_1921_1939_non_time_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_inventory_sha256_before": source_hash_before,
        "source_inventory_sha256_after": sha256(inventory_path),
        "decision": (
            "advance_frequency_qualified_tables"
            if advancing
            else "close_noaa_non_tree_candidates_on_frequency_and_physics"
        ),
    }
    if summary["source_inventory_sha256_before"] != summary["source_inventory_sha256_after"]:
        raise RuntimeError("round175-inventory-hash-changed")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 176 NOAA Non-Tree Time-Axis Audit\n\n"
        f"- Official text files preserved unchanged: `{len(manifest)}`.\n"
        f"- Tables audited: `{len(audits)}`.\n"
        f"- Frequency-qualified tables: `{len(advancing)}`.\n"
        "- Non-time numeric values decoded: `false`.\n"
        "- Locked-period non-time values accessed: `false`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
