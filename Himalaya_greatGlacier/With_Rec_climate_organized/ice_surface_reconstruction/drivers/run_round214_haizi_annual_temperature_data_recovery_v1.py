from __future__ import annotations

import csv
import hashlib
import json
import math
import urllib.request
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_214_haizi_annual_temperature_data_recovery_v1"
DRIVER = Path(__file__).resolve()
LEDGER = (
    LOOP
    / "round_109_cross_network_identity_closure_v1"
    / "proxy_evidence_master_ledger_identity_closed.csv"
)
URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41598-024-79096-6/MediaObjects/"
    "41598_2024_79096_MOESM1_ESM.xlsx"
)
WORKBOOK_NAME = "41598_2024_79096_MOESM1_ESM.xlsx"
SITE_LAT = (29.03 + 30.13) / 2
SITE_LON = (100.50 + 100.80) / 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 LIA climate evidence audit"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def nearest_tree_groups() -> list[dict[str, object]]:
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates: list[tuple[float, dict[str, str]]] = []
    seen: set[str] = set()
    for row in rows:
        archive = " ".join(
            [row.get("archive_type", ""), row.get("canonical_archive_class", "")]
        ).lower()
        lat = as_float(row.get("latitude", ""))
        lon = as_float(row.get("longitude", ""))
        group = row.get("evidence_group_id", "")
        if "tree" not in archive or lat is None or lon is None or not group or group in seen:
            continue
        seen.add(group)
        candidates.append((haversine_km(SITE_LAT, SITE_LON, lat, lon), row))
    candidates.sort(key=lambda item: item[0])
    output = []
    for rank, (distance, row) in enumerate(candidates[:15], start=1):
        output.append(
            {
                "rank": rank,
                "distance_km": round(distance, 3),
                "evidence_group_id": row.get("evidence_group_id", ""),
                "record_id": row.get("record_id", ""),
                "latitude": row.get("latitude", ""),
                "longitude": row.get("longitude", ""),
                "first_year": row.get("first_year", ""),
                "last_year": row.get("last_year", ""),
                "public_identity_id": row.get("public_identity_id", ""),
                "metadata_identities": row.get("metadata_identities", ""),
                "publication_cite_key": row.get("publication_cite_key", ""),
            }
        )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    workbook_path = OUT / WORKBOOK_NAME
    download(URL, workbook_path)
    if workbook_path.read_bytes()[:4] != b"PK\x03\x04":
        raise RuntimeError("Official supplement is not an XLSX ZIP container")

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheets = [
        {
            "sheet": sheet_name,
            "max_row": workbook[sheet_name].max_row,
            "max_column": workbook[sheet_name].max_column,
        }
        for sheet_name in workbook.sheetnames
    ]
    workbook.close()
    with (OUT / "workbook_structure.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sheets[0]))
        writer.writeheader()
        writer.writerows(sheets)

    nearest = nearest_tree_groups()
    with (OUT / "nearest_project_tree_evidence_groups.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nearest[0]))
        writer.writeheader()
        writer.writerows(nearest)

    summary = {
        "round": 214,
        "paper_doi": "10.1038/s41598-024-79096-6",
        "official_supplement": {
            "url": URL,
            "bytes": workbook_path.stat().st_size,
            "sha256": sha256(workbook_path),
            "sheet_count": len(sheets),
            "sheets": sheets,
        },
        "target": "previous-October to current-September mean temperature",
        "reconstruction_coverage": "1115-2017",
        "instrumental_target": "Daocheng station, 1959-2017",
        "study_area": {
            "latitude_range": [29.03, 30.13],
            "longitude_range": [100.50, 100.80],
            "audit_centroid": [SITE_LAT, SITE_LON],
        },
        "identity_preflight": {
            "nearest": nearest[0],
            "groups_within_35_km_among_nearest_15": int(
                sum(float(row["distance_km"]) <= 35 for row in nearest)
            ),
            "groups_within_75_km_among_nearest_15": int(
                sum(float(row["distance_km"]) <= 75 for row in nearest)
            ),
            "classification": "post_2017_source_requires_exact_identity_audit",
        },
        "numeric_values_read": False,
        "locked_period_values_read": False,
        "decision": "admit_to_locked_safe_annual_temperature_comparison",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "official_supplement": {
            "url": URL,
            "path": str(workbook_path),
            "bytes": workbook_path.stat().st_size,
            "sha256": sha256(workbook_path),
        },
        "ledger": {"path": str(LEDGER), "sha256": sha256(LEDGER)},
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    contract = """# Round 214 Contract

- Recover the official Scientific Reports XLSX supplement for DOI
  `10.1038/s41598-024-79096-6` from the Springer static-content mirror.
- Preserve and hash the byte-exact workbook.
- Read workbook structure and project-ledger metadata only. Do not read annual
  reconstruction, chronology, station, or locked-period values in this round.
- Do not access Jionglacuo, run EnKF, change production climate, or run OGGM.
"""
    (OUT / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")
    result = f"""# Round 214 Result

The official supplement was recovered with SHA256
`{summary['official_supplement']['sha256']}` and contains {len(sheets)}
worksheets. The nearest current project tree evidence group is
{nearest[0]['distance_km']} km from the study-area audit centroid.

No annual numeric value was read. Decision:
`admit_to_locked_safe_annual_temperature_comparison`.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
