from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_211_wscp_temperature_data_recovery_v1"
DRIVER = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
    / "run_round211_wscp_temperature_data_recovery_v1.py"
)
LEDGER = (
    LOOP
    / "round_109_cross_network_identity_closure_v1"
    / "proxy_evidence_master_ledger_identity_closed.csv"
)
PAGE_URL = "https://www.geodoi.ac.cn/edoi.aspx?DOI=10.3974%2Fgeodb.2022.02.03.V1"
ARCHIVE_URL = "https://www.geodoi.ac.cn/down.aspx?fileID=6140"
ARCHIVE_NAME = "TemReconWSCP_1383-2005.rar"
SITE_LAT = 31.15
SITE_LON = 99.81
BSDTAR = Path(r"D:\ProgramData\Anaconda3\Library\bin\bsdtar.exe")


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


def nearest_tree_records() -> list[dict[str, object]]:
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    candidates: list[tuple[float, dict[str, str]]] = []
    seen_groups: set[str] = set()
    for row in rows:
        archive = " ".join(
            [row.get("archive_type", ""), row.get("canonical_archive_class", "")]
        ).lower()
        lat = as_float(row.get("latitude", ""))
        lon = as_float(row.get("longitude", ""))
        group = row.get("evidence_group_id", "")
        if "tree" not in archive or lat is None or lon is None or not group:
            continue
        if group in seen_groups:
            continue
        seen_groups.add(group)
        candidates.append((haversine_km(SITE_LAT, SITE_LON, lat, lon), row))

    candidates.sort(key=lambda item: item[0])
    output = []
    for rank, (distance, row) in enumerate(candidates[:12], start=1):
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
                "site_name": row.get("site_name", ""),
                "public_identity_id": row.get("public_identity_id", ""),
                "metadata_identities": row.get("metadata_identities", ""),
                "publication_cite_key": row.get("publication_cite_key", ""),
            }
        )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    extracted = OUT / "official_archive_extracted"
    extracted.mkdir()

    page = OUT / "official_geodoi_page.html"
    archive = OUT / ARCHIVE_NAME
    download(PAGE_URL, page)
    page_text = page.read_text(encoding="utf-8", errors="replace")
    required_page_tokens = [
        "10.3974/geodb.2022.02.03.V1",
        ARCHIVE_NAME,
        "a(6140,2006)",
        "1383-2005",
    ]
    missing = [token for token in required_page_tokens if token not in page_text]
    if missing:
        raise RuntimeError(f"Official page is missing frozen tokens: {missing}")

    download(ARCHIVE_URL, archive)
    if not archive.read_bytes()[:8].startswith(b"Rar!\x1a\x07"):
        raise RuntimeError("Downloaded file does not have a RAR signature")
    if not BSDTAR.exists():
        raise RuntimeError(f"Required read-only archive tool is absent: {BSDTAR}")

    listing = subprocess.run(
        [str(BSDTAR), "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.splitlines()
    subprocess.run(
        [str(BSDTAR), "-xf", str(archive), "-C", str(extracted)],
        check=True,
    )

    inventory = []
    for path in sorted(item for item in extracted.rglob("*") if item.is_file()):
        inventory.append(
            {
                "relative_path": path.relative_to(extracted).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "suffix": path.suffix.lower(),
            }
        )
    with (OUT / "archive_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    nearest = nearest_tree_records()
    with (OUT / "nearest_project_tree_evidence_groups.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nearest[0]))
        writer.writeheader()
        writer.writerows(nearest)

    close_10 = sum(float(row["distance_km"]) <= 10 for row in nearest)
    close_35 = sum(float(row["distance_km"]) <= 35 for row in nearest)
    summary = {
        "round": 211,
        "dataset_doi": "10.3974/geodb.2022.02.03.V1",
        "paper_doi": "10.1002/joc.7336",
        "site": {"latitude": SITE_LAT, "longitude": SITE_LON},
        "target": "June-July mean temperature",
        "coverage": "1383-2005",
        "archive": {
            "url": ARCHIVE_URL,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "listed_entry_count": len(listing),
            "extracted_file_count": len(inventory),
        },
        "identity_preflight": {
            "groups_within_10_km_among_nearest_12": close_10,
            "groups_within_35_km_among_nearest_12": close_35,
            "nearest": nearest[0],
            "classification": (
                "spatially_close_identity_requires_source_crosswalk"
                if close_35
                else "no_spatially_close_project_tree_group"
            ),
        },
        "numeric_values_read": False,
        "locked_period_values_read": False,
        "decision": "admit_to_locked_safe_identity_and_curve_audit",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "official_page": {"url": PAGE_URL, "sha256": sha256(page)},
        "official_archive": {
            "url": ARCHIVE_URL,
            "sha256": sha256(archive),
            "bytes": archive.stat().st_size,
        },
        "ledger": {"path": str(LEDGER), "sha256": sha256(LEDGER)},
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    contract = """# Round 211 Contract

- Recover the official Western Sichuan Plateau 1383-2005 June-July
  temperature archive from GeoDOI file ID 6140.
- Preserve the downloaded archive byte-for-byte and hash every extracted file.
- Read file metadata and project-ledger metadata only; do not read annual
  temperature values in this round.
- Do not access any Jionglacuo source, run EnKF, alter production climate,
  or run OGGM.
"""
    (OUT / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")
    result = f"""# Round 211 Result

The official archive was recovered from dataset DOI
`10.3974/geodb.2022.02.03.V1` and preserved with SHA256
`{summary['archive']['sha256']}`. It contains {len(inventory)} extracted files.

The nearest project tree evidence group is {nearest[0]['distance_km']} km from
the published Baiyu sampling site. Spatial proximity alone is not an identity
match, so the next round must inspect source-native chronology metadata before
calling the reconstruction independent.

No annual temperature value or locked-period value was read. Decision:
`admit_to_locked_safe_identity_and_curve_audit`.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
