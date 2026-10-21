"""Validate the current versioned frozen registry without stale hardcoded IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
REGISTRY = (
    ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
    / "lia_191_frozen_curve_registry_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    registry = pd.read_csv(REGISTRY / "frozen_candidates.csv")
    declared = json.loads((REGISTRY / "registry_summary.json").read_text(encoding="utf-8"))
    actual = sorted(registry["lia_unique_id"].astype(int).tolist())
    unique = len(actual) == len(set(actual))
    declared_systems = sorted(int(value) for value in declared["systems"])
    rows: list[dict] = []
    for system_id in actual:
        system_root = REGISTRY / f"system_{system_id}"
        hash_path = system_root / "file_hashes.json"
        if not hash_path.exists():
            rows.append({"lia_unique_id": system_id, "file": "file_hashes.json", "exists": False, "bytes_match": False, "sha256_match": False})
            continue
        hashes = json.loads(hash_path.read_text(encoding="utf-8"))
        for name, recorded in hashes.items():
            path = system_root / name
            exists = path.exists()
            size = path.stat().st_size if exists else None
            rows.append(
                {
                    "lia_unique_id": system_id,
                    "file": name,
                    "exists": exists,
                    "recorded_bytes": int(recorded["bytes"]),
                    "actual_bytes": size,
                    "bytes_match": exists and size == int(recorded["bytes"]),
                    "sha256_match": exists and _sha256(path) == recorded["sha256"],
                }
            )
    detail = pd.DataFrame(rows)
    output = REGISTRY / "registry_validation_v2_dynamic"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "file_validation.csv", index=False)
    membership = (
        unique
        and int(declared["candidate_count"]) == len(actual)
        and declared_systems == actual
    )
    summary = {
        "schema_version": 2,
        "registry_membership_match": membership,
        "candidate_count": len(actual),
        "systems": actual,
        "recorded_file_count": len(detail),
        "missing_file_count": int((~detail["exists"]).sum()),
        "byte_mismatch_count": int((~detail["bytes_match"]).sum()),
        "sha256_mismatch_count": int((~detail["sha256_match"]).sum()),
        "all_pass": bool(
            membership
            and detail["exists"].all()
            and detail["bytes_match"].all()
            and detail["sha256_match"].all()
        ),
        "launched_oggm": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
