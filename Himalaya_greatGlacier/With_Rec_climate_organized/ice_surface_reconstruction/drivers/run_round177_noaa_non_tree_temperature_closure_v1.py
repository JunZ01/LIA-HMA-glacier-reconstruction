from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


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


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    round175 = loop / "round_175_noaa_hma_non_tree_temperature_inventory_v2"
    round176 = loop / "round_176_noaa_non_tree_time_axis_audit_v3"
    output = loop / "round_177_noaa_non_tree_temperature_closure_v1"
    output.mkdir(parents=True, exist_ok=False)

    audit_path = round176 / "time_axis_audit.csv"
    summary176_path = round176 / "summary.json"
    metadata_paths = sorted((round175 / "study_metadata_unmodified").glob("*.json"))
    tracked = [audit_path, summary176_path, *metadata_paths]
    hashes_before = {str(path): sha256(path) for path in tracked}

    with audit_path.open("r", encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))
    advancing = [row for row in audit if "pass_needs" in row["disposition"]]
    if {(row["noaa_study_id"], row["table_name"]) for row in advancing} != {
        ("24710", "Shenqi2018SQ1-SQ7"),
        ("24710", "Shenqi2018SQ7"),
    }:
        raise RuntimeError("unexpected-round176-advancing-set")

    shenqi = None
    for path in metadata_paths:
        study = json.loads(path.read_text(encoding="utf-8"))["study"][0]
        if str(study.get("NOAAStudyId")) == "24710":
            shenqi = study
            break
    if shenqi is None:
        raise RuntimeError("shenqi-metadata-not-found")
    notes = str(shenqi.get("studyNotes", ""))
    publications = shenqi.get("publication", []) or []
    publication_text = " ".join(
        f"{item.get('title', '')} {item.get('abstract', '')}" for item in publications
    )
    physical_text = f"{notes} {publication_text}".lower()
    if "precipitation" not in physical_text or "rainfall" not in physical_text:
        raise RuntimeError("shenqi-hydroclimate-interpretation-not-closed")

    closure_rows = []
    for row in advancing:
        closure_rows.append(
            {
                "noaa_study_id": row["noaa_study_id"],
                "table_name": row["table_name"],
                "modern_observation_count": row["modern_1940_2000_observation_count"],
                "modern_unique_integer_year_count": row[
                    "modern_1940_2000_unique_integer_year_count"
                ],
                "documented_physical_target": "monsoon_precipitation",
                "temperature_candidate": False,
                "disposition": "exclude_explicit_precipitation_proxy",
            }
        )
    with (output / "candidate_physical_closure.csv").open(
        "w", encoding="ascii", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(closure_rows[0]))
        writer.writeheader()
        writer.writerows(closure_rows)

    guliya = [
        row
        for row in audit
        if row["noaa_study_id"] == "44694"
        and "d18O, deuterium excess" in row["table_name"]
    ]
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "round176_advancing_count_before_physical_closure": len(advancing),
        "shenqi_advancing_table_count": len(closure_rows),
        "shenqi_documented_target": "monsoon_precipitation",
        "guliya_corrected_time_unit": (
            guliya[0]["time_unit_resolved"] if guliya else None
        ),
        "guliya_corrected_modern_observation_count": (
            int(guliya[0]["modern_1940_2000_observation_count"])
            if guliya
            else None
        ),
        "new_noaa_non_tree_temperature_candidate_count": 0,
        "annual_non_time_values_decoded": False,
        "locked_1921_1939_non_time_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_hashes_before": hashes_before,
        "source_hashes_after": {str(path): sha256(path) for path in tracked},
        "decision": "close_current_noaa_non_tree_catalogue_as_new_temperature_source",
    }
    if summary["source_hashes_before"] != summary["source_hashes_after"]:
        raise RuntimeError("source-hash-changed")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 177 NOAA Non-Tree Temperature Closure\n\n"
        "- Round 176 frequency-qualified tables: `2`.\n"
        "- Both are Shenqi speleothem d18O tables explicitly interpreted as "
        "monsoon precipitation.\n"
        "- New non-tree temperature candidates: `0`.\n"
        "- Non-time numeric values decoded: `false`.\n\n"
        "Decision: `close_current_noaa_non_tree_catalogue_as_new_temperature_source`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
