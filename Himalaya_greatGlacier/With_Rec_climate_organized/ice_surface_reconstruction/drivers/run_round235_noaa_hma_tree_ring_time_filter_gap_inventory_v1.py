from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


QUERY = (
    "https://www.ncei.noaa.gov/access/paleo-search/study/search.json?"
    "dataTypeId=18&reconstructionsOnly=N&minLat=25&maxLat=40&"
    "minLon=80&maxLon=105&headersOnly=true"
)
CODE = re.compile(r"ITRDB\s+([A-Z]{2,5}\d{3})", re.IGNORECASE)
KEYWORDS = (
    "mxd", "density", "temperature", "nahtz", "rara", "chola",
    "cholangpati", "nepal", "himalaya", "tibet", "bhutan", "hengduan",
)


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
    output = loop / "round_235_noaa_hma_tree_ring_time_filter_gap_inventory_v1"
    output.mkdir(parents=True, exist_ok=False)
    old_inventory = loop / "round_171_noaa_hma_temperature_raw_inventory_v1/candidate_inventory.csv"
    facts = root / "LIA_PROJECT_FACTS.md"
    contract = Path(__file__).with_name(
        "round235_noaa_hma_tree_ring_time_filter_gap_inventory_v1_contract.md"
    )
    inputs = {"facts": facts, "contract": contract, "round171_inventory": old_inventory}
    before = {name: sha256(path) for name, path in inputs.items()}

    with old_inventory.open("r", encoding="ascii", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    old_ids = {str(row["noaa_study_id"]) for row in old_rows}
    old_codes = {str(row["itrdb_code"]).upper() for row in old_rows if row["itrdb_code"]}

    request = urllib.request.Request(QUERY, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
    raw_path = output / "noaa_headers_only_response_unmodified.json"
    raw_path.write_bytes(payload)
    api = json.loads(payload.decode("utf-8"))
    studies = api.get("study", [])

    rows: list[dict[str, object]] = []
    for study in studies:
        study_id = str(study.get("NOAAStudyId", ""))
        name = str(study.get("studyName", ""))
        match = CODE.search(name)
        code = match.group(1).upper() if match else ""
        lowered = name.lower()
        matched_keywords = [keyword for keyword in KEYWORDS if keyword in lowered]
        rows.append({
            "noaa_study_id": study_id,
            "study_name": name,
            "itrdb_code": code,
            "contribution_date": study.get("contributionDate", ""),
            "present_in_round171_by_study_id": study_id in old_ids,
            "code_present_in_round171": bool(code and code in old_codes),
            "matched_review_keywords": ";".join(matched_keywords),
            "time_filter_gap_candidate": bool(study_id not in old_ids),
            "priority_metadata_review": bool(study_id not in old_ids and matched_keywords),
        })
    rows.sort(key=lambda row: (not bool(row["priority_metadata_review"]), str(row["study_name"])))
    new_rows = [row for row in rows if row["time_filter_gap_candidate"]]
    priority = [row for row in rows if row["priority_metadata_review"]]

    with (output / "time_filter_gap_inventory.csv").open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "query": QUERY,
        "api_final_url": final_url,
        "api_content_type": content_type,
        "api_response_sha256": sha256(raw_path),
        "round171_study_count": len(old_ids),
        "unfiltered_study_header_count": len(studies),
        "time_filter_gap_study_count": len(new_rows),
        "priority_metadata_review_count": len(priority),
        "priority_metadata_review": priority,
        "annual_proxy_or_climate_values_downloaded_or_parsed": False,
        "linked_noaa_data_files_opened": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_hashes_unchanged": True,
        "decision": "review_time_filter_gap_metadata" if priority else "no_priority_time_filter_gap_candidate",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    lines = [
        "# Round 235 NOAA HMA Tree-Ring Time-Filter Gap Inventory",
        "",
        f"- Round 171 study IDs: `{len(old_ids)}`.",
        f"- No-time-filter study headers: `{len(studies)}`.",
        f"- Additional study IDs: `{len(new_rows)}`.",
        f"- Priority metadata reviews: `{len(priority)}`.",
        "- Linked annual data files opened: `false`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

