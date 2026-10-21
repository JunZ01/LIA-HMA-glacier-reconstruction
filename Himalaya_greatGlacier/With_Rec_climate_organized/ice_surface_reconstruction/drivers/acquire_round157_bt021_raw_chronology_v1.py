"""Acquire the public NOAA ITRDB BT021 chronology without decoding values."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


FILES = {
    "bt021-crn-noaa.txt": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/asia/bt021-crn-noaa.txt",
    "bt021-noaa.crn": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/asia/bt021-noaa.crn",
    "bt021.crn": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/asia/bt021.crn",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def main() -> int:
    root = project_root()
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_bt021_raw_chronology_acquisition_20260819_v1"
    )
    if output.exists():
        raise FileExistsError(output)
    raw = output / "raw_unopened"
    raw.mkdir(parents=True)

    records = []
    for name, url in FILES.items():
        request = urllib.request.Request(
            url, headers={"User-Agent": "LIA-HMA-reconstruction-audit/1.0"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
            status = int(response.status)
            content_type = response.headers.get("Content-Type", "")
        if status != 200 or not payload:
            raise RuntimeError(f"download-failed:{name}:status={status}:bytes={len(payload)}")
        destination = raw / name
        destination.write_bytes(payload)
        records.append(
            {
                "name": name,
                "url": url,
                "path": str(destination),
                "http_status": status,
                "content_type": content_type,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "value_content_decoded": False,
            }
        )

    manifest = {
        "status": "complete",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA NCEI International Tree-Ring Data Bank",
        "site_code": "BT021",
        "source_url_index": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/asia/",
        "files": records,
        "value_content_decoded": False,
        "locked_1921_1939_values_accessed": False,
        "next_step": "safe_year_first_identity_and_coverage_audit",
    }
    (output / "acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    lines = [
        "# BT021 Raw Chronology Acquisition",
        "",
        "The three public NOAA ITRDB files were downloaded anonymously and fixed by SHA256.",
        "No chronology value was decoded in this acquisition step.",
        "",
    ]
    lines.extend(
        f"- `{row['name']}`: {row['size_bytes']} bytes, SHA256 `{row['sha256']}`."
        for row in records
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
