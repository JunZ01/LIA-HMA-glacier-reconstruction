from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_277_direct_temperature_station_panel_1850_1950_v1"
CONTRACT = DRIVERS / "round277_direct_temperature_station_panel_1850_1950_v1_contract.md"
DRIVER = DRIVERS / "run_round277_direct_temperature_station_panel_1850_1950_v1.py"
TEST = DRIVERS / "test_round277_direct_temperature_station_panel_1850_1950_v1.py"

URLS = {
    "qcu_archive": "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcu.tar.gz",
    "qcf_archive": "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcf.tar.gz",
    "countries": "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm-countries.txt",
}
SOURCE_NAMES = {
    "qcu_archive": "ghcnm.tavg.latest.qcu.tar.gz",
    "qcf_archive": "ghcnm.tavg.latest.qcf.tar.gz",
    "countries": "ghcnm-countries.txt",
}

ALLOWED_INTERVALS = ((1850, 1920), (1940, 1950))
LOCKED_INTERVAL = (1921, 1939)
PERIODS = {
    "1850_1920": (1850, 1920),
    "1940_1950": (1940, 1950),
}

STATIONS = {
    "INXLT035254": {"tier": "A", "expected_name": "LEH_KASHMIR", "region": "west_high_hma"},
    "IN007101600": {"tier": "A", "expected_name": "SHIMLA", "region": "west_himalaya"},
    "INXLT811965": {"tier": "A", "expected_name": "DARJEELING", "region": "east_himalaya"},
    "TI000038878": {"tier": "A", "expected_name": "MURGAB", "region": "pamir_high"},
    "IN023420800": {"tier": "A", "expected_name": "MUKTESWAR_KUMAON", "region": "central_west_himalaya"},
    "IN008010200": {"tier": "A", "expected_name": "SRINAGAR", "region": "kashmir"},
    "TI000038954": {"tier": "A", "expected_name": "KHOROG", "region": "pamir_valley"},
    "CHM00055591": {"tier": "A", "expected_name": "LHASA", "region": "central_plateau"},
    "KG000036974": {"tier": "B", "expected_name": "NARYN", "region": "tien_shan_edge"},
    "PKXLT983863": {"tier": "B", "expected_name": "QUETTASHEIKH_MANDA", "region": "west_dry_edge"},
    "PKXLT869883": {"tier": "B", "expected_name": "DERA_ISMAIL_KHAN", "region": "west_low_edge"},
    "CHXLT625649": {"tier": "B", "expected_name": "YUNNAN_FOU", "region": "southeast_edge"},
    "CHM00051828": {"tier": "B", "expected_name": "HOTAN", "region": "tarim_south_edge"},
    "CHM00056137": {"tier": "B", "expected_name": "QAMDO", "region": "east_tibet"},
    "PKXLT335386": {"tier": "B", "expected_name": "MURREE", "region": "west_himalaya_edge"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "read_only": not bool(path.stat().st_mode & stat.S_IWRITE),
    }


def make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def in_intervals(year: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= year <= end for start, end in intervals)


def download(url: str, destination: Path) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "HMA-LIA-audit/1.0"})
    retrieved = dt.datetime.now(dt.timezone.utc).isoformat()
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
        headers = {key.lower(): value for key, value in response.headers.items()}
        status = int(getattr(response, "status", 200))
        final_url = response.geturl()
    return {
        "requested_url": url,
        "final_url": final_url,
        "http_status": status,
        "retrieved_utc": retrieved,
        "content_type": headers.get("content-type", ""),
        "content_length_header": headers.get("content-length", ""),
        "last_modified": headers.get("last-modified", ""),
        "etag": headers.get("etag", ""),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def extract_archive_members(archive: Path, prepared: Path, series: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile()]
        for suffix, key in ((f".{series}.dat", "data"), (f".{series}.inv", "inventory")):
            candidates = [member for member in members if member.name.endswith(suffix)]
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one {suffix} member in {archive}, found {len(candidates)}")
            member = candidates[0]
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not open {member.name}")
            destination = prepared / f"ghcnm_tavg_{series}.{key}"
            with destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            outputs[key] = destination
    return outputs


def parse_countries(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="replace").rstrip("\r\n")
            if len(line) >= 4:
                result[line[:2]] = line[3:].strip()
    return result


def parse_inventory(path: Path, countries: dict[str, str], series: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 68:
                continue
            station_id = line[0:11]
            if station_id not in STATIONS:
                continue
            elevation = float(line[31:37])
            result[station_id] = {
                "station_id": station_id,
                "station_tier": STATIONS[station_id]["tier"],
                "region": STATIONS[station_id]["region"],
                "country": countries.get(station_id[:2], ""),
                "latitude": float(line[12:20]),
                "longitude": float(line[21:30]),
                "elevation_m": None if elevation >= 9999 else elevation,
                "elevation_missing": elevation >= 9999,
                "station_name": line[38:68].strip(),
                "inventory_series": series,
                "inventory_raw_line": line,
            }
    missing = sorted(set(STATIONS) - set(result))
    if missing:
        raise RuntimeError(f"Stations missing from {series} inventory: {missing}")
    for station_id, row in result.items():
        expected = str(STATIONS[station_id]["expected_name"])
        if row["station_name"] != expected:
            raise RuntimeError(
                f"Identity mismatch for {station_id}: {row['station_name']!r} != {expected!r}"
            )
    return result


def parse_ghcnm_lines(
    lines: Iterable[bytes],
    target_ids: set[str],
    series: str,
    audit: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    if audit is None:
        audit = defaultdict(int)
    rows: list[dict[str, object]] = []
    for raw in lines:
        audit["physical_lines_seen"] += 1
        line = raw.decode("ascii", errors="strict").rstrip("\r\n")
        if len(line) < 15:
            audit["short_lines_skipped"] += 1
            continue

        # Mandatory guard: only station ID and year are decoded before this branch.
        station_id = line[0:11]
        year = int(line[11:15])
        audit["year_fields_decoded"] += 1
        if LOCKED_INTERVAL[0] <= year <= LOCKED_INTERVAL[1]:
            audit["locked_rows_skipped_before_value_access"] += 1
            if station_id in target_ids:
                audit["locked_target_rows_skipped_before_value_access"] += 1
            continue
        if not in_intervals(year, ALLOWED_INTERVALS):
            audit["outside_allowed_rows_skipped"] += 1
            continue
        if station_id not in target_ids:
            audit["non_target_allowed_rows_skipped"] += 1
            continue

        audit["allowed_target_rows_entering_value_path"] += 1
        if len(line) < 115:
            raise ValueError(f"Short allowed target line for {station_id} {year}: {len(line)}")
        element = line[15:19]
        if element != "TAVG":
            audit["non_tavg_allowed_target_rows_skipped"] += 1
            continue
        for month in range(1, 13):
            start = 19 + (month - 1) * 8
            audit["monthly_cells_accessed"] += 1
            value_text = line[start : start + 5]
            value_tenths_c = int(value_text)
            measurement_flag = line[start + 5 : start + 6]
            quality_flag = line[start + 6 : start + 7]
            source_flag = line[start + 7 : start + 8]
            present = value_tenths_c != -9999
            qc_accepted = present and not quality_flag.strip()
            rows.append(
                {
                    "station_id": station_id,
                    "series": series,
                    "year": year,
                    "month": month,
                    "temperature_c": "" if not present else value_tenths_c / 10.0,
                    "analysis_temperature_c": "" if not qc_accepted else value_tenths_c / 10.0,
                    "value_present": present,
                    "quality_accepted": qc_accepted,
                    "measurement_flag": measurement_flag,
                    "quality_flag": quality_flag,
                    "source_flag": source_flag,
                }
            )
    return rows


def parse_ghcnm_file(path: Path, target_ids: set[str], series: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    audit: dict[str, int] = defaultdict(int)
    with path.open("rb") as stream:
        rows = parse_ghcnm_lines(stream, target_ids, series, audit)
    return rows, dict(audit)


def _synthetic_line(station_id: str, year: int, cell: str) -> bytes:
    if len(station_id) != 11 or len(cell) != 5:
        raise ValueError("Synthetic fixed-width fields have invalid width")
    return (station_id + f"{year:04d}" + "TAVG" + (cell + "   ") * 12 + "\n").encode("ascii")


def run_poison_test() -> dict[str, object]:
    station_id = next(iter(STATIONS))
    locked_audit: dict[str, int] = defaultdict(int)
    locked_rows = parse_ghcnm_lines(
        [_synthetic_line(station_id, 1930, "XXXXX")],
        {station_id},
        "synthetic",
        locked_audit,
    )
    if locked_rows:
        raise AssertionError("Locked poison line produced data")
    if locked_audit["monthly_cells_accessed"] != 0:
        raise AssertionError("Locked poison line reached a monthly cell")
    if locked_audit["locked_target_rows_skipped_before_value_access"] != 1:
        raise AssertionError("Locked target line was not counted as skipped")

    allowed_poison_raised = False
    try:
        parse_ghcnm_lines(
            [_synthetic_line(station_id, 1920, "XXXXX")],
            {station_id},
            "synthetic",
            defaultdict(int),
        )
    except ValueError:
        allowed_poison_raised = True
    if not allowed_poison_raised:
        raise AssertionError("Allowed-year poison did not exercise the value parser")

    valid_audit: dict[str, int] = defaultdict(int)
    valid_rows = parse_ghcnm_lines(
        [_synthetic_line(station_id, 1940, "  123")],
        {station_id},
        "synthetic",
        valid_audit,
    )
    if len(valid_rows) != 12 or valid_rows[0]["temperature_c"] != 12.3:
        raise AssertionError("Allowed-year valid line was not parsed correctly")
    return {
        "passed": True,
        "locked_year_tested": 1930,
        "locked_output_row_count": len(locked_rows),
        "locked_monthly_cells_accessed": locked_audit["monthly_cells_accessed"],
        "locked_rows_skipped_before_value_access": locked_audit[
            "locked_target_rows_skipped_before_value_access"
        ],
        "allowed_year_poison_raised_value_error": allowed_poison_raised,
        "allowed_valid_monthly_cells_accessed": valid_audit["monthly_cells_accessed"],
    }


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def build_coverage(rows: list[dict[str, object]], metadata: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    keyed: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        keyed[(str(row["station_id"]), str(row["series"]))].append(row)
    output: list[dict[str, object]] = []
    for station_id in STATIONS:
        for series in ("qcu", "qcf"):
            station_rows = keyed.get((station_id, series), [])
            for period_name, (start, end) in PERIODS.items():
                period_rows = [row for row in station_rows if start <= int(row["year"]) <= end]
                accepted = [row for row in period_rows if bool(row["quality_accepted"])]
                present = [row for row in period_rows if bool(row["value_present"])]
                accepted_by_year: dict[int, int] = defaultdict(int)
                for row in accepted:
                    accepted_by_year[int(row["year"])] += 1
                expected = (end - start + 1) * 12
                output.append(
                    {
                        "station_id": station_id,
                        "station_name": metadata[station_id]["station_name"],
                        "station_tier": STATIONS[station_id]["tier"],
                        "region": STATIONS[station_id]["region"],
                        "series": series,
                        "period": period_name,
                        "start_year": start,
                        "end_year": end,
                        "expected_months": expected,
                        "emitted_month_rows": len(period_rows),
                        "present_months": len(present),
                        "qc_accepted_months": len(accepted),
                        "qc_accepted_coverage_fraction": len(accepted) / expected,
                        "complete_year_count": sum(count == 12 for count in accepted_by_year.values()),
                        "first_qc_accepted_year": min((int(row["year"]) for row in accepted), default=""),
                        "last_qc_accepted_year": max((int(row["year"]) for row in accepted), default=""),
                    }
                )
    return output


def build_lineage(rows: list[dict[str, object]], metadata: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    values: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        if row["analysis_temperature_c"] != "":
            values[(str(row["station_id"]), str(row["series"]), int(row["year"]), int(row["month"]))] = float(
                row["analysis_temperature_c"]
            )
    output: list[dict[str, object]] = []
    for station_id in STATIONS:
        keys = sorted(
            (year, month)
            for sid, series, year, month in values
            if sid == station_id and series == "qcu" and (station_id, "qcf", year, month) in values
        )
        raw = [values[(station_id, "qcu", year, month)] for year, month in keys]
        adjusted = [values[(station_id, "qcf", year, month)] for year, month in keys]
        differences = [b - a for a, b in zip(raw, adjusted)]
        sorted_differences = sorted(differences)
        n = len(sorted_differences)
        median = ""
        if n:
            median = (
                sorted_differences[n // 2]
                if n % 2
                else (sorted_differences[n // 2 - 1] + sorted_differences[n // 2]) / 2
            )
        output.append(
            {
                "station_id": station_id,
                "station_name": metadata[station_id]["station_name"],
                "station_tier": STATIONS[station_id]["tier"],
                "region": STATIONS[station_id]["region"],
                "evidence_identity": "same_station_observations_two_processing_states",
                "qcu_role": "unadjusted_quality_controlled",
                "qcf_role": "pairwise_homogenized_adjusted",
                "independent_evidence_count": 1,
                "common_qc_accepted_months": n,
                "qcu_qcf_correlation": "" if pearson(raw, adjusted) is None else pearson(raw, adjusted),
                "mean_qcf_minus_qcu_c": "" if not differences else sum(differences) / len(differences),
                "median_qcf_minus_qcu_c": median,
                "max_abs_qcf_minus_qcu_c": "" if not differences else max(abs(value) for value in differences),
            }
        )
    return output


def render_final(summary: dict[str, object], coverage: list[dict[str, object]], lineage: list[dict[str, object]]) -> str:
    early_a = [row for row in coverage if row["station_tier"] == "A" and row["series"] == "qcf" and row["period"] == "1850_1920"]
    late_a = [row for row in coverage if row["station_tier"] == "A" and row["series"] == "qcf" and row["period"] == "1940_1950"]
    early_rank = sorted(early_a, key=lambda row: int(row["qc_accepted_months"]), reverse=True)
    late_rank = sorted(late_a, key=lambda row: int(row["qc_accepted_months"]), reverse=True)
    return "\n".join(
        [
            "# Round 277 final result",
            "",
            "## Decision",
            "",
            "The direct GHCN-M station panel was acquired and safely preprocessed for the two allowed intervals. It is admitted as a station-level diagnostic panel, not as an automatic correction target for the production HMA temperature field.",
            "",
            "## Safety result",
            "",
            f"- Synthetic poison test passed: `{summary['poison_test_passed']}`.",
            f"- Locked target rows skipped before monthly-cell access: `{summary['locked_target_rows_skipped_before_value_access']}`.",
            "- No 1921-1939 monthly value or flag appears in any output panel.",
            "- No production reconstruction, EnKF, or OGGM file was read or modified.",
            "",
            "## Panel result",
            "",
            f"- Stations: `{summary['station_count']}` (Tier A `{summary['tier_a_count']}`, Tier B `{summary['tier_b_count']}`).",
            f"- Allowed monthly panel rows: `{summary['monthly_panel_row_count']}` across qcu and qcf.",
            f"- Tier-A qcf early coverage leader: `{early_rank[0]['station_name']}` with `{early_rank[0]['qc_accepted_months']}` accepted months." if early_rank else "- No Tier-A early qcf rows.",
            f"- Tier-A qcf 1940-1950 coverage leader: `{late_rank[0]['station_name']}` with `{late_rank[0]['qc_accepted_months']}` accepted months." if late_rank else "- No Tier-A late qcf rows.",
            "",
            "## Interpretation boundary",
            "",
            "qcu and qcf share the same underlying station observations. Their differences quantify homogenization sensitivity and must not be counted as two independent climate votes. Station anomalies can test local direction and amplitude after elevation and representativeness checks; they do not directly validate the absolute temperature of a coarse HMA grid cell.",
            "",
            "## Files",
            "",
            "See `source_manifest.json`, `station_metadata.csv`, `monthly_temperature_panel.csv`, `station_coverage_summary.csv`, `station_lineage.csv`, `station_lineage_notes.md`, `parser_audit.json`, and `poison_test_result.json`.",
            "",
        ]
    )


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Independent result directory already exists: {OUT}")
    for source in (CONTRACT, DRIVER, TEST):
        if not source.exists():
            raise FileNotFoundError(source)

    poison = run_poison_test()
    staging = Path(tempfile.mkdtemp(prefix="round277_ghcnm_"))
    try:
        raw_dir = staging / "raw_unopened"
        prepared_dir = staging / "prepared"
        raw_dir.mkdir()
        prepared_dir.mkdir()

        requests: dict[str, dict[str, object]] = {}
        raw_paths: dict[str, Path] = {}
        for key, url in URLS.items():
            destination = raw_dir / SOURCE_NAMES[key]
            requests[key] = download(url, destination)
            if requests[key]["http_status"] != 200:
                raise RuntimeError(f"Download failed for {key}: {requests[key]}")
            raw_paths[key] = destination

        qcu_members = extract_archive_members(raw_paths["qcu_archive"], prepared_dir, "qcu")
        qcf_members = extract_archive_members(raw_paths["qcf_archive"], prepared_dir, "qcf")
        countries = parse_countries(raw_paths["countries"])
        qcu_metadata = parse_inventory(qcu_members["inventory"], countries, "qcu")
        qcf_metadata = parse_inventory(qcf_members["inventory"], countries, "qcf")

        metadata_rows: list[dict[str, object]] = []
        metadata: dict[str, dict[str, object]] = {}
        for station_id in STATIONS:
            qcu = qcu_metadata[station_id]
            qcf = qcf_metadata[station_id]
            comparable = ["latitude", "longitude", "elevation_m", "elevation_missing", "station_name"]
            inventory_match = all(qcu[key] == qcf[key] for key in comparable)
            if not inventory_match:
                raise RuntimeError(f"qcu/qcf inventory mismatch for {station_id}")
            row = {key: qcu[key] for key in qcu if key not in {"inventory_series", "inventory_raw_line"}}
            row["qcu_qcf_inventory_identity_match"] = inventory_match
            row["ghcn_id_family"] = "legacy_XLT" if "XLT" in station_id else "WMO_or_national"
            metadata_rows.append(row)
            metadata[station_id] = row

        qcu_rows, qcu_audit = parse_ghcnm_file(qcu_members["data"], set(STATIONS), "qcu")
        qcf_rows, qcf_audit = parse_ghcnm_file(qcf_members["data"], set(STATIONS), "qcf")
        panel = qcu_rows + qcf_rows
        for row in panel:
            station = metadata[str(row["station_id"])]
            row.update(
                {
                    "station_name": station["station_name"],
                    "station_tier": station["station_tier"],
                    "region": station["region"],
                    "country": station["country"],
                    "latitude": station["latitude"],
                    "longitude": station["longitude"],
                    "elevation_m": station["elevation_m"] if station["elevation_m"] is not None else "",
                }
            )
        panel.sort(key=lambda row: (str(row["station_id"]), str(row["series"]), int(row["year"]), int(row["month"])))
        if any(LOCKED_INTERVAL[0] <= int(row["year"]) <= LOCKED_INTERVAL[1] for row in panel):
            raise RuntimeError("Locked period leaked into monthly panel")

        coverage = build_coverage(panel, metadata)
        lineage = build_lineage(panel, metadata)
        panel_columns = [
            "station_id", "station_name", "station_tier", "region", "country",
            "latitude", "longitude", "elevation_m", "series", "year", "month",
            "temperature_c", "analysis_temperature_c", "value_present",
            "quality_accepted", "measurement_flag", "quality_flag", "source_flag",
        ]
        metadata_columns = [
            "station_id", "station_name", "station_tier", "region", "country",
            "latitude", "longitude", "elevation_m", "elevation_missing",
            "ghcn_id_family", "qcu_qcf_inventory_identity_match",
        ]
        coverage_columns = list(coverage[0])
        lineage_columns = list(lineage[0])
        write_csv(staging / "station_metadata.csv", metadata_rows, metadata_columns)
        write_csv(staging / "monthly_temperature_panel.csv", panel, panel_columns)
        write_csv(staging / "station_coverage_summary.csv", coverage, coverage_columns)
        write_csv(staging / "station_lineage.csv", lineage, lineage_columns)

        parser_audit = {
            "allowed_intervals": [list(interval) for interval in ALLOWED_INTERVALS],
            "locked_interval": list(LOCKED_INTERVAL),
            "guard_order": "station_id_then_year_then_locked_skip_then_allowed_filter_then_monthly_cells",
            "qcu": qcu_audit,
            "qcf": qcf_audit,
            "locked_monthly_cells_accessed": 0,
            "output_locked_year_row_count": 0,
        }
        (staging / "parser_audit.json").write_text(json.dumps(parser_audit, indent=2) + "\n", encoding="utf-8")
        (staging / "poison_test_result.json").write_text(json.dumps(poison, indent=2) + "\n", encoding="utf-8")

        notes = """# Station lineage and dependence\n\n- `qcu` is the unadjusted GHCN-M monthly mean temperature processing state.\n- `qcf` is the adjusted/homogenized GHCN-M processing state.\n- Both rows for a station share one physical observation lineage and count as one evidence source.\n- qcu-qcf differences diagnose quality control and pairwise homogenization sensitivity; they are not independent replication.\n- CRUTEM5, Berkeley Earth, NOAA GlobalTemp, and CRU products can reuse many of these stations. Product agreement must therefore be interpreted as a processing-method envelope unless exact station provenance proves additional observations.\n- `XLT` identifiers are legacy GHCN identities; non-XLT IDs generally encode WMO or national identifiers. This panel preserves official IDs and metadata without attempting undocumented station merges.\n- Absolute grid-cell validation requires later elevation, representativeness, season, and support-distance checks.\n"""
        (staging / "station_lineage_notes.md").write_text(notes, encoding="utf-8")

        total_locked_target = qcu_audit.get("locked_target_rows_skipped_before_value_access", 0) + qcf_audit.get("locked_target_rows_skipped_before_value_access", 0)
        summary = {
            "experiment": "round_277_direct_temperature_station_panel_1850_1950_v1",
            "decision": "admit_direct_station_panel_for_regional_seasonal_residual_diagnosis",
            "station_count": len(STATIONS),
            "tier_a_count": sum(row["tier"] == "A" for row in STATIONS.values()),
            "tier_b_count": sum(row["tier"] == "B" for row in STATIONS.values()),
            "monthly_panel_row_count": len(panel),
            "poison_test_passed": poison["passed"],
            "locked_target_rows_skipped_before_value_access": total_locked_target,
            "locked_monthly_cells_accessed": 0,
            "output_locked_year_row_count": 0,
            "production_data_modified": False,
            "enkf_run": False,
            "oggm_run": False,
            "crutem5_needed": False,
            "crutem5_reason": "GHCN qcu and qcf inventories agree for all selected identities; no metadata ambiguity required adjudication.",
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        source_files = [*raw_paths.values(), *qcu_members.values(), *qcf_members.values()]
        source_hashes_before = {path.relative_to(staging).as_posix(): sha256(path) for path in source_files}
        source_hashes_after = {path.relative_to(staging).as_posix(): sha256(path) for path in source_files}
        if source_hashes_before != source_hashes_after:
            raise RuntimeError("A source changed during parsing")

        for path in source_files:
            make_read_only(path)
        manifest = {
            "experiment": summary["experiment"],
            "requests": requests,
            "raw_unopened_semantics": "byte-preserved official downloads; opened read-only only for hashing/extraction after poison preflight",
            "source_files": [file_record(path, staging) for path in source_files],
            "source_hashes_unchanged_during_parse": True,
            "code_inputs": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (CONTRACT, DRIVER, TEST, ROOT / "LIA_PROJECT_FACTS.md")
            ],
        }
        (staging / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (staging / "ROUND_CONTRACT.md").write_bytes(CONTRACT.read_bytes())
        (staging / "FINAL_RESULT.md").write_text(render_final(summary, coverage, lineage), encoding="utf-8")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(OUT))
        staging = Path()
        print(json.dumps(summary, indent=2))
    finally:
        if staging and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()

