from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2"
)
V1 = LOOP / "round_226_yang2010_setp_622y_metadata_recovery_v1"
OUT = LOOP / "round_226_yang2010_setp_622y_metadata_recovery_v2"
LEDGER = (
    LOOP
    / "round_110_unresolved_proxy_metadata_recovery_v4"
    / "proxy_evidence_master_ledger_metadata_recovered.csv"
)
DRIVER = Path(__file__).resolve()
SITE_LAT = 29 + 22 / 60
SITE_LON = 94 + 16 / 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: object) -> float | None:
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


def first_text(row: dict[str, str], *fields: str) -> str:
    for field in fields:
        value = row.get(field, "").strip()
        if value:
            return value
    return ""


def nearest_tree_groups() -> list[dict[str, object]]:
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates: list[tuple[float, dict[str, str]]] = []
    seen: set[str] = set()
    for row in rows:
        archive = " ".join(
            [row.get("archive_type", ""), row.get("canonical_archive_class", "")]
        ).lower()
        lat = as_float(row.get("latitude"))
        lon = as_float(row.get("longitude"))
        group = row.get("evidence_group_id", "")
        if "tree" not in archive or lat is None or lon is None or not group or group in seen:
            continue
        seen.add(group)
        candidates.append((haversine_km(SITE_LAT, SITE_LON, lat, lon), row))
    candidates.sort(key=lambda item: item[0])
    output: list[dict[str, object]] = []
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
                "formal_identity": first_text(
                    row,
                    "formal_metadata_identity",
                    "metadata_identities",
                    "public_identity_id",
                ),
                "formal_source": first_text(
                    row, "formal_metadata_source", "data_name", "source_family"
                ),
                "publication_cite_key": row.get("publication_cite_key", ""),
                "publication_author": row.get("publication_author", ""),
                "publication_year": row.get("publication_year", ""),
            }
        )
    return output


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    required = [V1 / "summary.json", V1 / "source_manifest.json", LEDGER]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("Missing locked-safe inputs: " + "; ".join(missing))
    OUT.mkdir(parents=True)

    v1 = json.loads((V1 / "summary.json").read_text(encoding="utf-8"))
    nearest = nearest_tree_groups()
    with (OUT / "nearest_project_tree_evidence_groups_corrected.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nearest[0]))
        writer.writeheader()
        writer.writerows(nearest)

    mainling = next(row for row in nearest if "MAINJT" in str(row["formal_identity"]))
    nyinjt = next(row for row in nearest if "NYINJT" in str(row["formal_identity"]))
    summary = {
        "round": 226,
        "version": 2,
        "supersedes_identity_preflight_only": str(V1 / "summary.json"),
        "candidate": v1["candidate"],
        "official_repository": v1["official_repository"],
        "repository_crosscheck": v1["repository_crosscheck"],
        "corrected_identity_preflight": {
            "nearest_project_group": mainling,
            "second_project_group": nyinjt,
            "spatial_context": "candidate_site_between_mainling_and_nyingchi",
            "source_independence": "unresolved",
            "reason": (
                "formal identities are closed, but metadata do not provide sample IDs "
                "or a sequence crosswalk to the 46-core Yang 2010 collection"
            ),
        },
        "annual_values_read": False,
        "locked_period_values_read": False,
        "jionglacuo_accessed": False,
        "decision": "metadata_candidate_numeric_series_not_public_sample_overlap_unresolved",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "v1_summary": {"path": str(V1 / "summary.json"), "sha256": sha256(V1 / "summary.json")},
                "v1_manifest": {"path": str(V1 / "source_manifest.json"), "sha256": sha256(V1 / "source_manifest.json")},
                "ledger": {"path": str(LEDGER), "sha256": sha256(LEDGER)},
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "ROUND_CONTRACT.md").write_text(
        "# Round 226 v2 identity-field correction\n\n"
        "Correct only the v1 nearest-project identity lookup using the formal "
        "metadata fields. Read locked-safe metadata outputs and the proxy ledger "
        "only. Do not open the PDF, proxy series, reconstruction series, locked "
        "period, Jionglacuo, EnKF, production climate, MATLAB, or OGGM.\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 226 v2 corrected identity preflight\n\n"
        f"- Nearest group: `{mainling['evidence_group_id']}` / `{mainling['formal_identity']}` "
        f"at `{mainling['distance_km']}` km.\n"
        f"- Second group: `{nyinjt['evidence_group_id']}` / `{nyinjt['formal_identity']}` "
        f"at `{nyinjt['distance_km']}` km.\n"
        "- Both project identities are formally closed, but their sample/sequence relationship "
        "to Yang 2010 remains unresolved.\n"
        "- Decision: `metadata_candidate_numeric_series_not_public_sample_overlap_unresolved`.\n"
        "- No annual, locked-period, or Jionglacuo value was read.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
