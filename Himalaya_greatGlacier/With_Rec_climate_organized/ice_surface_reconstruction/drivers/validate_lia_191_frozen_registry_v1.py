"""Read-only hash and membership validation for the frozen curve registry."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .build_lia_191_iterative_loop import project_root


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_frozen_curve_registry_v1"
    )
    registry = pd.read_csv(root / "frozen_candidates.csv")
    expected = {991, 1077, 1546, 1616, 2266, 2337, 2815, 2902, 2928, 2939, 4319}
    actual = set(registry["lia_unique_id"].astype(int))
    rows = []
    for system_id in sorted(actual):
        system_root = root / f"system_{system_id}"
        hashes = json.loads((system_root / "file_hashes.json").read_text(encoding="utf-8"))
        for name, recorded in hashes.items():
            path = system_root / name
            exists = path.exists()
            size = path.stat().st_size if exists else None
            digest = _sha256(path) if exists else None
            rows.append(
                {
                    "lia_unique_id": system_id,
                    "file": name,
                    "exists": exists,
                    "recorded_bytes": int(recorded["bytes"]),
                    "actual_bytes": size,
                    "bytes_match": exists and size == int(recorded["bytes"]),
                    "sha256_match": exists and digest == recorded["sha256"],
                }
            )
    detail = pd.DataFrame(rows)
    output = root / "registry_validation_v1"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "file_validation.csv", index=False)
    summary = {
        "schema_version": 1,
        "registry_membership_match": actual == expected,
        "candidate_count": len(actual),
        "systems": sorted(actual),
        "recorded_file_count": int(len(detail)),
        "missing_file_count": int((~detail["exists"]).sum()),
        "byte_mismatch_count": int((~detail["bytes_match"]).sum()),
        "sha256_mismatch_count": int((~detail["sha256_match"]).sum()),
        "all_pass": bool(
            actual == expected
            and detail["exists"].all()
            and detail["bytes_match"].all()
            and detail["sha256_match"].all()
        ),
        "launched_oggm": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
