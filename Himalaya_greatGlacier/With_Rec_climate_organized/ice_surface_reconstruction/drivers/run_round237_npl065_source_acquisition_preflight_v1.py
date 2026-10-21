from __future__ import annotations

import csv
import hashlib
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


URL = "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/npl065.rwl"
SITE_LAT = 27.89
SITE_LON = 86.37
METADATA_FIRST_YEAR = 1787
METADATA_LAST_YEAR = 2013
LOCKED_START = 1921
LOCKED_END = 1939


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


def number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def structural_inventory(path: Path) -> dict[str, object]:
    sample_ids: set[str] = set()
    coverage: dict[int, int] = {year: 0 for year in range(1940, 2001)}
    locked_slots_skipped = 0
    paleo_slots_skipped = 0
    modern_slots_counted = 0
    marker_count = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip(b"\r\n")
            if not line.strip() or line == b"\x1a" or len(line) < 12:
                continue
            year_text = line[8:12].strip()
            if not year_text.lstrip(b"-").isdigit():
                continue
            sample = line[:8].strip().decode("ascii", errors="strict")
            start_year = int(year_text)
            sample_ids.add(sample)
            for offset, token in enumerate(line[12:].split()):
                year = start_year + offset
                if token in {b"999", b"-9999", b"9999"}:
                    marker_count += 1
                    break
                if LOCKED_START <= year <= LOCKED_END:
                    locked_slots_skipped += 1
                    continue
                if year < 1940:
                    paleo_slots_skipped += 1
                    continue
                if 1940 <= year <= 2000:
                    coverage[year] += 1
                    modern_slots_counted += 1
    missing_years = [year for year, count in coverage.items() if count == 0]
    return {
        "sample_count": len(sample_ids),
        "sample_ids": sorted(sample_ids),
        "structural_modern_slot_count": modern_slots_counted,
        "structural_1940_2000_min_depth": min(coverage.values()),
        "structural_1940_2000_max_depth": max(coverage.values()),
        "structural_1940_2000_missing_years": missing_years,
        "end_marker_count": marker_count,
        "locked_slots_skipped_before_numeric_conversion": locked_slots_skipped,
        "paleo_slots_skipped_before_numeric_conversion": paleo_slots_skipped,
        "ring_width_numeric_values_converted": False,
    }


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_237_npl065_source_acquisition_preflight_v1"
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / "raw_unopened"
    raw_dir.mkdir()
    inputs = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": Path(__file__).with_name("round237_npl065_source_acquisition_preflight_v1_contract.md"),
        "round236_metadata": loop / "round_236_central_east_hma_noaa_candidate_metadata_audit_v2/raw/npl065.json",
        "ledger": loop / "round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
    }
    before = {name: sha256(path) for name, path in inputs.items()}

    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
        status = response.status
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        last_modified = response.headers.get("Last-Modified", "")
    source = raw_dir / "npl065.rwl"
    source.write_bytes(payload)
    inventory = structural_inventory(source)

    with inputs["ledger"].open("r", encoding="utf-8", newline="") as stream:
        ledger = [row for row in csv.DictReader(stream) if row.get("network") == "proxy6"]
    distances: list[dict[str, object]] = []
    exact_bound_matches: list[str] = []
    for row in ledger:
        lat = number(row.get("latitude", ""))
        lon = number(row.get("longitude", ""))
        if lat is None or lon is None:
            continue
        distance = haversine_km(SITE_LAT, SITE_LON, lat, lon)
        distances.append({
            "record_id": row.get("record_id", ""),
            "formal_metadata_identity": row.get("formal_metadata_identity", ""),
            "distance_km": distance,
            "first_year": row.get("first_year", ""),
            "last_year": row.get("last_year", ""),
        })
        row_first = number(row.get("first_year", ""))
        row_last = number(row.get("last_year", ""))
        if (
            distance <= 2.0
            and row_first is not None
            and row_last is not None
            and abs(row_first - METADATA_FIRST_YEAR) <= 5
            and abs(row_last - METADATA_LAST_YEAR) <= 5
        ):
            exact_bound_matches.append(str(row.get("record_id", "")))
    distances.sort(key=lambda row: float(row["distance_km"]))
    nearest = distances[:10]
    with (output / "nearest_existing_proxy_metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nearest[0]))
        writer.writeheader()
        writer.writerows(nearest)

    gates = {
        "http_success_and_nonempty": bool(status == 200 and len(payload) > 0),
        "at_least_30_distinct_series": bool(int(inventory["sample_count"]) >= 30),
        "complete_structural_1940_2000_coverage": not inventory["structural_1940_2000_missing_years"],
        "metadata_reaches_2000": METADATA_LAST_YEAR >= 2000,
        "no_exact_coordinate_chronology_bound_match": not exact_bound_matches,
    }
    passes = bool(all(gates.values()))
    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("input-hash-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate": {
            "code": "NPL065",
            "site": "Gaurishankar Conservation Area",
            "latitude": SITE_LAT,
            "longitude": SITE_LON,
            "species": "Rhododendron campanulatum",
            "metadata_years": [METADATA_FIRST_YEAR, METADATA_LAST_YEAR],
            "documented_target": "December-February minimum temperature",
            "published_calibration": "1960-2013",
            "published_correlation": 0.781,
            "published_explained_variance_fraction": 0.61,
        },
        "download": {
            "url": URL,
            "final_url": final_url,
            "http_status": status,
            "content_type": content_type,
            "last_modified": last_modified,
            "size_bytes": len(payload),
            "sha256": sha256(source),
        },
        "structural_inventory": inventory,
        "nearest_existing_proxy_metadata": nearest,
        "exact_coordinate_chronology_bound_matches": exact_bound_matches,
        "gates": gates,
        "passes": passes,
        "decision": "advance_npl065_to_boundary_safe_modern_oof" if passes else "close_npl065_on_source_preflight",
        "paleo_trajectory_exported_or_inspected": False,
        "ring_width_numeric_values_converted": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "input_hashes_unchanged": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"inputs_before": before, "inputs_after": after, "download": result["download"]}, indent=2),
        encoding="ascii",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 237 NPL065 Source Acquisition Preflight\n\n"
        f"- Source SHA256: `{result['download']['sha256']}`.\n"
        f"- Distinct series: `{inventory['sample_count']}`.\n"
        f"- 1940--2000 structural minimum depth: `{inventory['structural_1940_2000_min_depth']}`.\n"
        f"- Exact existing source matches: `{len(exact_bound_matches)}`.\n"
        "- Ring-width magnitudes converted: `false`.\n\n"
        f"Decision: `{result['decision']}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

