from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPECTED_SHA256 = "d7f9d588e1362745857ac2aea3ca7c0686cee8b6ecc410e20af23a97d81d7bcd"
MODERN_START = 1941
MODERN_END = 2013
LOCKED_START = 1921
LOCKED_END = 1939


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_254_npl065_structural_age_offset_audit_v1"
    output.mkdir(parents=True, exist_ok=False)
    source = loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round254_npl065_structural_age_offset_audit_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "rwl": source,
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["rwl"]["sha256"] != EXPECTED_SHA256:
        raise RuntimeError("npl065-source-hash-changed")
    occupied_years: dict[str, set[int]] = {}
    line_records = []
    payload_tokens_decoded = 0
    locked_occupied_slots = 0
    for line_number, raw in enumerate(source.read_bytes().splitlines(), start=1):
        line = raw.rstrip(b"\r\n")
        if not line.strip() or line == b"\x1a" or len(line) < 12:
            continue
        year_field = line[8:12].strip()
        if not year_field.lstrip(b"-").isdigit():
            continue
        sample = line[:8].strip().decode("ascii", errors="strict")
        start_year = int(year_field.decode("ascii"))
        payload = line[12:]
        chunks = [payload[index:index + 6] for index in range(0, len(payload), 6)]
        if chunks and len(chunks[-1]) < 6 and chunks[-1].strip():
            raise RuntimeError(f"partial-nonblank-field:{line_number}")
        data_fields = 0
        marker_seen = False
        for offset, chunk in enumerate(chunks):
            stripped = chunk.strip()
            if not stripped:
                continue
            if stripped in (b"999", b"-9999", b"9999"):
                marker_seen = True
                break
            year = start_year + offset
            occupied_years.setdefault(sample, set()).add(year)
            data_fields += 1
            if LOCKED_START <= year <= LOCKED_END:
                locked_occupied_slots += 1
        line_records.append({
            "line_number": line_number,
            "sample": sample,
            "line_start_year": start_year,
            "six_character_field_count": len(chunks),
            "occupied_data_field_count": data_fields,
            "end_marker_seen": marker_seen,
        })
    rows = []
    modern_depth = {year: 0 for year in range(MODERN_START, MODERN_END + 1)}
    for sample in sorted(occupied_years):
        years = occupied_years[sample]
        first_year = min(years)
        last_year = max(years)
        for year in modern_depth:
            modern_depth[year] += int(year in years)
        rows.append({
            "sample": sample,
            "first_occupied_year": first_year,
            "last_occupied_year": last_year,
            "occupied_year_count": len(years),
            "structural_ring_index_at_1941_one_based": MODERN_START - first_year + 1,
            "ads30_stiffness_at_1941": 30 + MODERN_START - first_year,
            "modern_1941_2013_complete": all(year in years for year in modern_depth),
        })
    frame = pd.DataFrame(rows)
    depth_frame = pd.DataFrame({"year": list(modern_depth), "structural_sample_depth": list(modern_depth.values())})
    frame.to_csv(output / "sample_structural_age_offsets.csv", index=False)
    depth_frame.to_csv(output / "modern_structural_depth.csv", index=False)
    pd.DataFrame(line_records).to_csv(output / "rwl_line_structure.csv", index=False)
    global_first = int(frame.first_occupied_year.min())
    global_last = int(frame.last_occupied_year.max())
    gates = {
        "sample_count_53": len(frame) == 53,
        "global_coverage_1787_2013": global_first == 1787 and global_last == 2013,
        "minimum_modern_depth_at_least_41": int(depth_frame.structural_sample_depth.min()) >= 41,
        "no_payload_tokens_converted_to_numbers": payload_tokens_decoded == 0,
    }
    passes = all(gates.values())
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "sample_count": len(frame),
        "global_first_occupied_year": global_first,
        "global_last_occupied_year": global_last,
        "first_year_min_median_max": [int(frame.first_occupied_year.min()), float(frame.first_occupied_year.median()), int(frame.first_occupied_year.max())],
        "ads30_stiffness_at_1941_min_median_max": [int(frame.ads30_stiffness_at_1941.min()), float(frame.ads30_stiffness_at_1941.median()), int(frame.ads30_stiffness_at_1941.max())],
        "modern_depth_min_max": [int(depth_frame.structural_sample_depth.min()), int(depth_frame.structural_sample_depth.max())],
        "locked_occupied_slots_counted_structurally": locked_occupied_slots,
        "ring_width_payload_tokens_decoded_to_numbers": payload_tokens_decoded,
        "gates": gates,
        "passes": passes,
        "decision": "admit_structural_age_offsets_to_modern_only_ads_sensitivity" if passes else "reject_structural_age_offsets",
        "locked_1921_1939_numeric_values_decoded": False,
        "paleo_proxy_numeric_values_decoded": False,
        "jionglacuo_opened_or_hashed": False,
        "binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 254 NPL065 structural age-offset audit\n\n"
        f"- Samples: `{len(frame)}`; global coverage: `{global_first}-{global_last}`.\n"
        f"- ADS30 stiffness at 1941 min/median/max: `{int(frame.ads30_stiffness_at_1941.min())}/{frame.ads30_stiffness_at_1941.median():.0f}/{int(frame.ads30_stiffness_at_1941.max())}`.\n"
        f"- Modern depth min/max: `{int(depth_frame.structural_sample_depth.min())}/{int(depth_frame.structural_sample_depth.max())}`.\n"
        "- Ring-width payload tokens decoded to numbers: `0`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0 if passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
