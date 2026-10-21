"""Boundary-safe parser for the acquired NOAA Sun 2026 temperature text."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


YEAR_RE = re.compile(r"^\s*([+-]?\d{3,4})(?:\s+|$)")


def fingerprint(path: Path):
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    acquisition = root / "validation_data/external_sources/hma_temperature_public_acquisition_20260819_v1"
    raw = acquisition / "raw_unopened/sun2026-tibet-temp.txt"
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_148_sun2026_boundary_preflight_v1"
    contract = driver / "round148_sun2026_boundary_preflight_v1_contract.md"
    manifest = acquisition / "acquisition_manifest.json"
    paths = {"facts": root / "LIA_PROJECT_FACTS.md", "contract": contract, "driver": Path(__file__).resolve(), "raw": raw, "manifest": manifest}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    before = {name: fingerprint(path) for name, path in paths.items()}
    acquired = json.loads(manifest.read_text(encoding="utf-8"))
    entry = next(item for item in acquired["files"] if item["candidate_id"] == "T_SUN2026")
    if before["raw"]["sha256"].upper() != entry["sha256"].upper() or entry["values_opened"]:
        raise RuntimeError("acquisition-manifest-mismatch")

    headers = []
    allowed_rows = []
    locked_rows = 0
    outside_rows = 0
    source_first = None
    source_last = None
    width = None
    with raw.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            match = YEAR_RE.match(line)
            if match is None:
                headers.append(line)
                continue
            year = int(match.group(1))
            if not 500 <= year <= 2500:
                headers.append(line)
                continue
            source_first = year if source_first is None else min(source_first, year)
            source_last = year if source_last is None else max(source_last, year)
            if 1921 <= year <= 1939:
                locked_rows += 1
                continue
            if not (1000 <= year <= 1920 or 1940 <= year <= 2000):
                outside_rows += 1
                continue
            remainder = line[match.end():]
            tokens = remainder.split()
            if width is None:
                width = len(tokens)
            if len(tokens) != width:
                raise RuntimeError(f"data-width-changed:{line_number}:{len(tokens)}:{width}")
            values = []
            for column, token in enumerate(tokens, start=2):
                try:
                    values.append(float(token))
                except ValueError as exc:
                    raise RuntimeError(f"nonnumeric-allowed-field:{line_number}:{column}") from exc
            allowed_rows.append([year, *values])
    if not allowed_rows or width is None:
        raise RuntimeError("no-allowed-data-rows")

    columns = ["year", *[f"value_{index}" for index in range(1, width + 1)]]
    with (output / "sun2026_allowed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(allowed_rows)
    (output / "header_metadata.txt").write_text("\n".join(headers) + "\n", encoding="utf-8")

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_148_sun2026_boundary_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_first_year": source_first,
        "source_last_year": source_last,
        "allowed_first_year": int(allowed_rows[0][0]),
        "allowed_last_year": int(allowed_rows[-1][0]),
        "allowed_row_count": len(allowed_rows),
        "numeric_value_column_count": width,
        "locked_rows_skipped_before_value_decode": locked_rows,
        "outside_allowed_rows_skipped": outside_rows,
        "locked_1921_1939_values_decoded": 0,
        "provisional_role": "temperature_psm_target_not_independent_validation",
        "jionglacuo_opened": False,
        "source_fingerprints_unchanged": unchanged,
        "decision": "advance_allowed_sun2026_to_metadata_and_operator_feasibility",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 148 Sun 2026 Boundary Preflight\n\n"
        f"- Source coverage: `{source_first}-{source_last}`.\n"
        f"- Allowed rows: `{len(allowed_rows)}` with `{width}` numeric value columns.\n"
        f"- Locked rows skipped before value decode: `{locked_rows}`.\n"
        f"- Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
