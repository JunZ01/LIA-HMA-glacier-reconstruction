from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = Path(__file__).resolve().parent
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_233_daocheng_ghcnm_documented_target_coverage_v1"
CONTRACT = DRIVERS / "round233_daocheng_ghcnm_documented_target_coverage_v1_contract.md"
GHCN_DIR = LOOP / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4"
DATA = GHCN_DIR / "ghcnm_qcu_temperature.dat"
INVENTORY = GHCN_DIR / "ghcnm_qcu_inventory.inv"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATION_ID = "CHXLT574247"
BLOCKS = ((1959, 1970), (1971, 1982), (1983, 1994), (1995, 2006), (2007, 2017))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def longest_false_run(years: range, complete: dict[int, bool]) -> int:
    longest = 0
    current = 0
    for year in years:
        if complete.get(year, False):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": FACTS,
        "contract": CONTRACT,
        "driver": Path(__file__).resolve(),
        "ghcnm_data": DATA,
        "ghcnm_inventory": INVENTORY,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in sources.items()}

    station_line = None
    with INVENTORY.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if line.startswith(STATION_ID):
                station_line = line
                break
    if station_line is None:
        raise RuntimeError("Daocheng station missing from inventory")
    station_metadata = {
        "station_id": station_line[0:11],
        "latitude": float(station_line[12:20]),
        "longitude": float(station_line[21:30]),
        "elevation_m": float(station_line[31:37]),
        "name": station_line[38:68].strip(),
    }
    if station_metadata != {
        "station_id": STATION_ID,
        "latitude": 29.05,
        "longitude": 100.3,
        "elevation_m": 3729.0,
        "name": "DAOCHENG",
    }:
        raise RuntimeError(f"Unexpected Daocheng metadata: {station_metadata}")

    month_valid: dict[tuple[int, int], bool] = {}
    locked_lines_skipped = 0
    with DATA.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 115 or line[0:11] != STATION_ID:
                continue
            year = int(line[11:15])
            if 1921 <= year <= 1939:
                locked_lines_skipped += 1
                continue
            if year < 1958 or year > 2017 or line[15:19] != "TAVG":
                continue
            for month in range(1, 13):
                start = 19 + (month - 1) * 8
                raw_value = line[start : start + 5]
                qcflag = line[start + 6 : start + 7]
                month_valid[(year, month)] = raw_value != "-9999" and not qcflag.strip()

    complete: dict[int, bool] = {}
    for year in range(1959, 2018):
        months = [(year - 1, month) for month in (10, 11, 12)]
        months.extend((year, month) for month in range(1, 10))
        complete[year] = all(month_valid.get(key, False) for key in months)

    complete_years = [year for year, valid in complete.items() if valid]
    block_counts = {
        f"{start}-{end}": sum(complete.get(year, False) for year in range(start, end + 1))
        for start, end in BLOCKS
    }
    early_count = sum(complete.get(year, False) for year in range(1959, 1989))
    late_count = sum(complete.get(year, False) for year in range(1989, 2018))
    max_gap = longest_false_run(range(1959, 2018), complete)
    gates = {
        "at_least_45_complete_targets": len(complete_years) >= 45,
        "at_least_18_complete_targets_each_half": early_count >= 18 and late_count >= 18,
        "at_least_6_complete_targets_each_block": all(count >= 6 for count in block_counts.values()),
        "maximum_missing_run_at_most_3_years": max_gap <= 3,
    }
    decision = (
        "advance_daocheng_ghcnm_to_direct_station_blocked_oof"
        if all(gates.values())
        else "close_daocheng_direct_station_route_on_coverage"
    )

    after = {name: state(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("A source fingerprint changed")
    result = {
        "round": 233,
        "status": "complete",
        "station": station_metadata,
        "target": "previous October through current September mean temperature",
        "candidate_years": [1959, 2017],
        "complete_target_count": len(complete_years),
        "complete_target_years": complete_years,
        "early_complete_count_1959_1988": early_count,
        "late_complete_count_1989_2017": late_count,
        "block_complete_counts": block_counts,
        "maximum_consecutive_missing_target_years": max_gap,
        "gates": gates,
        "decision": decision,
        "locked_station_lines_skipped_before_month_fields": locked_lines_skipped,
        "temperature_magnitudes_converted_or_retained": False,
        "paleo_chronology_values_accessed": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 233 Result\n\n"
        f"Decision: `{decision}`.\n\n"
        f"Complete documented-season targets: `{len(complete_years)}/59`; "
        f"early/late counts: `{early_count}/{late_count}`; maximum missing run: `{max_gap}` years.\n\n"
        "No station temperature magnitude or sealed-period value was converted or retained.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
