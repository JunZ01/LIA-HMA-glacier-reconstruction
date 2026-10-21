"""Fetch metadata only for the frozen Round 121 NCEI candidate inventory."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_proxy_spatial_gap_discovery_20260818_v1"
    )
    inventory = output / "ncei_hma_tree_ring_recent_headers.json"
    destination = output / "ncei_candidate_metadata"
    if not inventory.exists():
        raise FileNotFoundError(inventory)
    destination.mkdir(exist_ok=False)
    payload = json.loads(inventory.read_text(encoding="utf-8-sig"))
    identifiers = [str(item["xmlId"]) for item in payload.get("study", [])]
    if len(identifiers) != 20 or len(set(identifiers)) != 20:
        raise RuntimeError(f"candidate-inventory-contract-failed:{len(identifiers)}")
    rows = []
    for index, xml_id in enumerate(identifiers, start=1):
        url = f"https://www.ncei.noaa.gov/access/paleo-search/study/search.json?xmlId={xml_id}"
        request = urllib.request.Request(url, headers={"User-Agent": "LIA-HMA-metadata-audit/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
            status = int(response.status)
        path = destination / f"ncei_xml_{xml_id}.json"
        path.write_bytes(content)
        parsed = json.loads(content.decode("utf-8"))
        if status != 200 or len(parsed.get("study", [])) != 1:
            raise RuntimeError(f"ncei-response-contract-failed:{xml_id}:{status}")
        rows.append(
            {
                "sequence": index,
                "xml_id": xml_id,
                "url": url,
                "http_status": status,
                "size_bytes": len(content),
                "sha256": sha256(path),
            }
        )
        time.sleep(0.2)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request_kind": "NCEI study metadata only",
        "candidate_count": len(rows),
        "time_series_files_downloaded": False,
        "candidate_values_opened": False,
        "locked_1921_1939_candidate_values_opened": False,
        "requests": rows,
    }
    (output / "ncei_candidate_metadata_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
