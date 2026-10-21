from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_199_east_asia_proxy_identity_metadata_v3"
)
V2_METADATA = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_199_east_asia_proxy_identity_metadata_v2/external_proxy_metadata.csv"
)
PROJECT_LEDGER = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_108_proxy_metadata_reconciliation_v2/proxy_evidence_master_ledger_reconciled.csv"
)


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def year(value: object) -> int | None:
    value_number = number(value)
    return int(round(value_number)) if value_number is not None else None


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    OUT.mkdir(parents=True)

    with V2_METADATA.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    corrected: list[dict[str, object]] = []
    for row in source:
        corrected.append(
            {
                "external_column_1based": int(row["external_column_1based"]),
                "generic_id": row["identity"],
                "source_id": row["metadata_row4"],
                "source_class_code": row["metadata_row2"],
                "geometry_type": row["metadata_row3"],
                "latitude": number(row["longitude"]),
                "longitude": number(row["elevation"]),
                "elevation_m": number(row["metadata_row8"]),
                "resolution_years": number(row["first_year"]),
                "first_year": year(row["last_year"]),
                "last_year": year(row["metadata_row11"]),
                "archive_type": row["proxy_type"],
                "measurement": row["metadata_row13"],
                "publication_title": row["metadata_row14"],
            }
        )

    with PROJECT_LEDGER.open(newline="", encoding="utf-8-sig") as handle:
        project = list(csv.DictReader(handle))
    project_identity_fields = (
        "record_id", "data_name", "site_name", "public_identity_id",
        "formal_metadata_identity", "metadata_identities",
    )
    project_title_fields = (
        "public_identity_citation", "publication_cite_key", "publication_author",
    )
    identity_lookup: dict[str, set[int]] = {}
    title_lookup: dict[str, set[int]] = {}
    for index, row in enumerate(project):
        for field in project_identity_fields:
            token = norm(row.get(field))
            if token:
                identity_lookup.setdefault(token, set()).add(index)
        for field in project_title_fields:
            token = norm(row.get(field))
            if token:
                title_lookup.setdefault(token, set()).add(index)

    overlaps: list[dict[str, object]] = []
    matched_external: set[str] = set()
    strong_external: set[str] = set()
    for ext in corrected:
        key = str(ext["source_id"] or ext["generic_id"])
        identity_indices: set[int] = set()
        for token in (norm(ext["source_id"]), norm(ext["generic_id"])):
            identity_indices.update(identity_lookup.get(token, set()))
        title_indices = title_lookup.get(norm(ext["publication_title"]), set())
        coordinate_indices: set[int] = set()
        lat = number(ext["latitude"])
        lon = number(ext["longitude"])
        for index, row in enumerate(project):
            plat = number(row.get("latitude"))
            plon = number(row.get("longitude"))
            if lat is not None and lon is not None and plat is not None and plon is not None:
                if abs(plat - lat) <= 0.02 and abs(plon - lon) <= 0.02:
                    coordinate_indices.add(index)
        candidates = sorted(identity_indices | title_indices | coordinate_indices)
        if not candidates:
            overlaps.append(
                {
                    "external_source_id": key,
                    "match_type": "none",
                    "project_network": "",
                    "project_record": "",
                    "project_identity": "",
                    "project_first_year": "",
                    "project_last_year": "",
                    "coordinate_distance_degree": "",
                }
            )
            continue
        for index in candidates:
            row = project[index]
            plat = number(row.get("latitude"))
            plon = number(row.get("longitude"))
            distance = None
            if lat is not None and lon is not None and plat is not None and plon is not None:
                distance = ((plat - lat) ** 2 + (plon - lon) ** 2) ** 0.5
            same_bounds = (
                year(ext["first_year"]) is not None and year(ext["last_year"]) is not None
                and year(row.get("first_year")) is not None and year(row.get("last_year")) is not None
                and abs(year(ext["first_year"]) - year(row.get("first_year"))) <= 1
                and abs(year(ext["last_year"]) - year(row.get("last_year"))) <= 1
            )
            if index in identity_indices:
                match_type = "exact_normalized_identity"
            elif index in title_indices:
                match_type = "exact_normalized_publication"
            elif same_bounds:
                match_type = "coordinate_and_time_bounds"
            else:
                match_type = "coordinate_only"
            matched_external.add(key)
            if match_type != "coordinate_only":
                strong_external.add(key)
            overlaps.append(
                {
                    "external_source_id": key,
                    "match_type": match_type,
                    "project_network": row.get("network", ""),
                    "project_record": row.get("record_id", ""),
                    "project_identity": row.get("public_identity_id") or row.get("formal_metadata_identity") or row.get("data_name"),
                    "project_first_year": row.get("first_year", ""),
                    "project_last_year": row.get("last_year", ""),
                    "coordinate_distance_degree": distance,
                }
            )

    hma_long = [
        row for row in corrected
        if number(row["latitude"]) is not None and number(row["longitude"]) is not None
        and 25 <= number(row["latitude"]) <= 40 and 70 <= number(row["longitude"]) <= 105
        and year(row["first_year"]) is not None and year(row["first_year"]) <= 1500
        and year(row["last_year"]) is not None and year(row["last_year"]) >= 1940
    ]
    hma_long_ids = {str(row["source_id"] or row["generic_id"]) for row in hma_long}

    with (OUT / "external_proxy_metadata_corrected.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corrected[0]))
        writer.writeheader()
        writer.writerows(corrected)
    with (OUT / "project_proxy_metadata_overlap_corrected.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlaps[0]))
        writer.writeheader()
        writer.writerows(overlaps)
    with (OUT / "hma_long_proxy_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corrected[0]))
        writer.writeheader()
        writer.writerows(hma_long)

    summary = {
        "round": 199,
        "version": 3,
        "external_proxy_count": len(corrected),
        "external_with_any_project_metadata_match": len(matched_external),
        "external_with_strong_metadata_match": len(strong_external),
        "hma_long_candidate_count": len(hma_long),
        "hma_long_with_any_project_match": len(hma_long_ids & matched_external),
        "hma_long_with_strong_project_match": len(hma_long_ids & strong_external),
        "deposited_matlab_comment_row_mismatch": True,
        "deposited_matlab_xlsread_semantics_reproduced": False,
        "annual_proxy_values_decoded": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "decision": "require_matlab_xlsread_metadata_closure_before_field_use",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "v2_metadata_path": str(V2_METADATA),
                "v2_metadata_sha256": hashlib.sha256(V2_METADATA.read_bytes()).hexdigest(),
                "project_ledger_path": str(PROJECT_LEDGER),
                "project_ledger_sha256": hashlib.sha256(PROJECT_LEDGER.read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = [
        "# Round 199 V3 corrected proxy metadata audit",
        "",
        f"- External proxies: `{len(corrected)}`.",
        f"- Any / strong project metadata matches: `{len(matched_external)}` / `{len(strong_external)}`.",
        f"- HMA long candidates: `{len(hma_long)}`; any / strong matches: `{len(hma_long_ids & matched_external)}` / `{len(hma_long_ids & strong_external)}`.",
        "- The physical worksheet row layout conflicts with the deposited MATLAB comments.",
        "- Exact MATLAB R2024b `xlsread` metadata semantics must close before field use.",
        "- No annual proxy, climate, locked-period, or Jionglacuo value was accessed.",
        "",
        "Decision: `require_matlab_xlsread_metadata_closure_before_field_use`.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
