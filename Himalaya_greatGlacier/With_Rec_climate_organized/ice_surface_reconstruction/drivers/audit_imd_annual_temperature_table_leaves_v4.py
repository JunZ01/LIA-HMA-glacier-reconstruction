from __future__ import annotations

import json
from pathlib import Path

import audit_imd_annual_temperature_table_leaves_v3 as audit


audit.SOURCE = audit.audit.LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v3"
audit.OUT = audit.audit.LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v4"
audit.RAW = audit.OUT / "raw_api"


def main_title_cluster(leaves: dict[int, list[str]]) -> tuple[dict[int, list[str]], list[int]]:
    ordered = sorted(leaves)
    if not ordered:
        return {}, []
    clusters: list[list[int]] = [[ordered[0]]]
    for leaf in ordered[1:]:
        if leaf - clusters[-1][-1] <= 10:
            clusters[-1].append(leaf)
        else:
            clusters.append([leaf])
    chosen = max(clusters, key=lambda values: (len(values), max(values)))
    discarded = sorted(set(ordered) - set(chosen))
    return {leaf: leaves[leaf] for leaf in chosen}, discarded


audit.main_title_cluster = main_title_cluster
audit.main()

summary_path = audit.OUT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["experiment"] = audit.OUT.name
summary["supersedes"] = audit.SOURCE.name
summary["main_table_cluster_max_internal_gap_leaves"] = 10
audit.write_json(summary_path, summary)

report = [
    "# IMD mean/extreme air-temperature table leaf audit, 1875-1890 (v4)",
    "",
    "Read-only audit of Internet Archive fulltext-inside metadata and leaf matches.",
    "",
    f"- Requested item/year/station rows: {summary['requested_pair_rows']}",
    f"- Candidate pairs: {summary['candidate_pair_rows']}",
    f"- Candidate detail rows: {summary['candidate_detail_rows']}",
    f"- Status counts: {json.dumps(summary['status_counts'], sort_keys=True)}",
    f"- Missing IA report years: {', '.join(map(str, summary['report_years_without_located_ia_items']))}",
    "- `117788` is assigned to 1878 from repeated table headings; its IA catalog title says 1877 and is flagged as conflicting.",
    "- Isolated contents/index hits were excluded. The retained main table run allows up to 10 missing OCR-title leaves between repeated headings.",
    "- 1921-1939 accessed: no.",
    "- OCR temperature values admitted: no. Every candidate remains a page locator pending page-image inspection and independent transcription.",
    "",
    "The complete requested table is `item_year_station_leaf_audit.csv`; multi-hit details are in `station_table_leaf_candidates.csv`.",
]
(audit.OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

manifest = []
for path in sorted(
    path
    for path in audit.OUT.rglob("*")
    if path.is_file() and path.name != "source_manifest.json"
):
    manifest.append(
        {
            "path": str(path.relative_to(audit.OUT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": audit.sha256(path),
        }
    )
audit.write_json(audit.OUT / "source_manifest.json", manifest)
print(json.dumps(summary, indent=2))
