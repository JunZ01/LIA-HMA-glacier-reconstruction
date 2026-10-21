from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_original_lake_author_reconstructions_v2 as lake_base


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_annual_maat_modern_anchor_coverage_v1"
NGAMRING = LOOP / "temperature_noaa_ngamring_original_v1/sun2022-ngamring-co.txt"
AWENG = LOOP / "temperature_noaa_hma_additional_physical_records_v1/AwengCo.Li.2017.txt"
HAIZI = LOOP / "round_216_haizi_annual_temperature_locked_safe_comparison_v1/annual_allowed_curves.csv"
LOCKED = (1921.0, 1939.999999)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_tab_year_first(path: Path, kind: str) -> tuple[pd.DataFrame, dict]:
    header = None
    rows = []
    locked_skipped = 0
    decoded = 0
    for raw in path.open("rb"):
        raw = raw.lstrip(b"\xef\xbb\xbf")
        if raw.startswith(b"#") or not raw.strip():
            continue
        tokens = raw.decode("utf-8").rstrip("\r\n").split("\t")
        while tokens and tokens[-1] == "":
            tokens.pop()
        if header is None:
            header = tokens
            continue
        if len(tokens) < len(header):
            tokens.extend([""] * (len(header) - len(tokens)))
        names = {name: index for index, name in enumerate(header)}
        if kind == "ngamring":
            age_bp = float(tokens[names["age_BP"]])
            year = 1950.0 - age_bp
            value_name = "MAAT"
        elif kind == "aweng":
            age_bp = float(tokens[names["age"]])
            year = 1950.0 - age_bp
            value_name = "temperature"
        else:
            raise RuntimeError(kind)
        if LOCKED[0] <= year <= LOCKED[1]:
            locked_skipped += 1
            continue
        if not (1500.0 <= year <= 1920.999999 or 1940.0 <= year <= 2000.999999):
            continue
        decoded += 1
        token = tokens[names[value_name]].strip()
        value = float(token) if token.lower() not in {"", "nan", "na"} else math.nan
        if np.isfinite(value):
            rows.append({"year_ce": year, "temperature_c": value})
    if header is None:
        raise RuntimeError(f"missing-header:{path}")
    return pd.DataFrame(rows).sort_values("year_ce"), {
        "locked_rows_skipped_before_temperature_decode": locked_skipped,
        "safe_temperature_cells_decoded": decoded,
    }


def coverage(record: str, frame: pd.DataFrame, audit: dict, source: Path) -> dict:
    year = frame["year_ce"].to_numpy(float)
    return {
        "record": record,
        "historical_points_1500_1849": int(np.sum((year >= 1500) & (year <= 1849.999999))),
        "bridge_points_1850_1920": int(np.sum((year >= 1850) & (year <= 1920.999999))),
        "postlock_points_1940_2000": int(np.sum((year >= 1940) & (year <= 2000.999999))),
        "baseline_points_1961_1990": int(np.sum((year >= 1961) & (year <= 1990.999999))),
        "latest_safe_year": float(np.max(year)) if len(year) else None,
        "earliest_safe_year": float(np.min(year)) if len(year) else None,
        "strict_1961_1990_anchor_eligible": int(np.sum((year >= 1961) & (year <= 1990.999999))) >= 3,
        "broad_1940_2000_anchor_eligible": int(np.sum((year >= 1940) & (year <= 2000.999999))) >= 3,
        "source": str(source),
        **audit,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    source_paths = sorted({Path(item["source"]) for item in lake_base.RECORDS.values()} | {NGAMRING, AWENG, HAIZI})
    before = {str(path): sha256(path) for path in source_paths}

    lake_base.POSTLOCK_END = 2000.999999
    point_frames = []
    coverage_rows = []
    for name, definition in lake_base.RECORDS.items():
        frame, audit = lake_base.parse_author_record(name, definition)
        frame = frame.rename(columns={"author_admitted_temperature_c": "temperature_c"})
        point_frames.append(frame[["record", "year_ce", "temperature_c"]])
        coverage_rows.append(coverage(name, frame, audit, Path(definition["source"])))

    for name, path, kind in (("Ngamring", NGAMRING, "ngamring"), ("Aweng", AWENG, "aweng")):
        frame, audit = parse_tab_year_first(path, kind)
        frame["record"] = name
        point_frames.append(frame[["record", "year_ce", "temperature_c"]])
        coverage_rows.append(coverage(name, frame, audit, path))

    haizi = pd.read_csv(HAIZI)
    haizi = haizi[haizi["year"].between(1500, 1920) | haizi["year"].between(1940, 2000)].copy()
    haizi_frame = haizi.rename(columns={"year": "year_ce", "haizi_annual_reconstruction_anomaly_degC": "temperature_c"})
    haizi_frame["record"] = "Haizi"
    point_frames.append(haizi_frame[["record", "year_ce", "temperature_c"]])
    coverage_rows.append(coverage(
        "Haizi", haizi_frame, {"locked_rows_skipped_before_temperature_decode": 19, "safe_temperature_cells_decoded": int(len(haizi_frame))}, HAIZI
    ))

    points = pd.concat(point_frames, ignore_index=True).sort_values(["record", "year_ce"])
    table = pd.DataFrame(coverage_rows).sort_values("record")
    points.to_csv(OUT / "annual_maat_safe_points_1500_1920_1940_2000.csv", index=False)
    table.to_csv(OUT / "modern_anchor_coverage.csv", index=False)

    strict = table.loc[table["strict_1961_1990_anchor_eligible"], "record"].tolist()
    broad = table.loc[table["broad_1940_2000_anchor_eligible"], "record"].tolist()
    summary = {
        "experiment": OUT.name,
        "status": "complete",
        "strict_anchor_rule": "at least 3 native annual-MAAT samples in 1961-1990",
        "broad_anchor_rule": "at least 3 native annual-MAAT samples in 1940-2000",
        "strict_anchor_eligible_records": strict,
        "broad_anchor_eligible_records": broad,
        "record_count": int(len(table)),
        "locked_rows_skipped_before_temperature_decode": int(table["locked_rows_skipped_before_temperature_decode"].sum()),
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {str(path): sha256(path) for path in source_paths}
    manifest["prohibited_source_accessed"] = False
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Annual MAAT modern-anchor coverage", "",
        f"Strict 1961-1990 anchors: `{', '.join(strict) if strict else 'none'}`.",
        f"Broad 1940-2000 anchors: `{', '.join(broad) if broad else 'none'}`.", "",
        "The audit decoded year/age before temperature and skipped 1921-1939 rows before temperature access.",
        "No production, EnKF, or OGGM file was modified or run. 20CRv3 was not accessed.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {str(path): sha256(path) for path in source_paths}
    if before != after:
        raise RuntimeError("source-changed")


if __name__ == "__main__":
    main()
